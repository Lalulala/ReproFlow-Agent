from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
import yaml
from openai import OpenAIError

from .approval import UnsafePlanError, approve_plan
from .context import build_context_pack
from .evaluation import run_agent_evals
from .evidence import (
    approve_claim,
    audit_claim_staleness,
    mark_claim_stale,
    sync_evidence,
)
from .human_views import render_evidence_markdown, render_plan_markdown
from .planner import get_planner
from .preflight import run_preflight
from .rag import ChromaKnowledgeBase, LexicalKnowledgeBase, get_knowledge_base
from .repo_agent import (
    RepoPlanStore,
    approve_repo_plan,
    create_repo_plan,
    create_repo_repair_plan,
    inspect_repository,
    render_dependency_report_markdown,
    render_manifest_markdown,
    render_repo_plan_markdown,
    run_repo_plan,
)
from .reporting import generate_report
from .runner import SimulatedCrashError
from .storage import Store
from .workflow import resume_workflow, start_workflow

app = typer.Typer(no_args_is_help=True, help="ReproFlow reproducible experiment workflow.")
evidence_app = typer.Typer(no_args_is_help=True, help="Review and sync evidence claims.")
knowledge_app = typer.Typer(no_args_is_help=True, help="Index and search local research sources.")
repo_app = typer.Typer(no_args_is_help=True, help="Understand and execute an existing repository.")
app.add_typer(evidence_app, name="evidence")
app.add_typer(knowledge_app, name="knowledge")
app.add_typer(repo_app, name="repo")


def _root(project: Path) -> Path:
    root = project.resolve()
    if not (root / "pyproject.toml").is_file():
        raise typer.BadParameter(f"Not a ReproFlow project: {root}")
    return root


@app.command("init")
def init_project(project: Annotated[Path, typer.Argument()] = Path(".")) -> None:
    """Initialize ReproFlow state in an existing project."""
    root = _root(project)
    Store(root)
    typer.echo(f"Initialized: {root / '.reproflow'}")


@app.command("plan")
def create_plan(
    goal: Annotated[str, typer.Option("--goal", help="Natural-language experiment goal.")],
    planner: Annotated[str, typer.Option("--planner", help="mock or api")] = "mock",
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    """Create and persist a draft experiment plan."""
    root = _root(project)
    context = build_context_pack(root, "planner", goal)
    plan = get_planner(planner).create_plan(goal, context, root)
    store = Store(root)
    store.save_plan(plan)
    store.add_plan_event(plan.plan_id, "created", planner, "Generated from natural-language goal")
    typer.echo(f"Plan: {plan.plan_id}")
    typer.echo(f"YAML: {store.plan_path(plan.plan_id)}")
    typer.echo(f"Readable plan: {store.plan_markdown_path(plan.plan_id)}")
    typer.echo(render_plan_markdown(plan))


@repo_app.command("inspect")
def inspect_repo_command(
    repository: Annotated[Path, typer.Argument()],
    goal: Annotated[str, typer.Option("--goal")] = "",
) -> None:
    """Inspect a Git repository without executing or changing it."""
    try:
        manifest = inspect_repository(repository, goal)
    except ValueError as error:
        typer.echo(f"Repository inspection failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(render_manifest_markdown(manifest))


@repo_app.command("plan")
def create_repo_plan_command(
    repository: Annotated[Path, typer.Argument()],
    goal: Annotated[str, typer.Option("--goal")],
    agent: Annotated[str, typer.Option("--agent", help="mock or api")] = "mock",
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    """Let the Agent inspect a repository and propose code plus execution decisions."""
    root = _root(project)
    try:
        plan = create_repo_plan(root, repository, goal, agent=agent)
    except (KeyError, OpenAIError, RuntimeError, ValueError) as error:
        typer.echo(f"Repository planning failed: {error}", err=True)
        raise typer.Exit(1) from error
    store = RepoPlanStore(root)
    typer.echo(f"Repository plan: {plan.repo_plan_id}")
    typer.echo(f"Review: {store.markdown_path(plan.repo_plan_id)}")
    typer.echo(render_repo_plan_markdown(plan))


@repo_app.command("list")
def list_repo_plans_command(
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    for plan in RepoPlanStore(_root(project)).list():
        typer.echo(
            f"{plan.repo_plan_id}\t{plan.status.value}\t"
            f"{Path(plan.repository_path).name}\t{plan.title}"
        )


@repo_app.command("show")
def show_repo_plan_command(
    repo_plan_id: Annotated[str, typer.Argument()],
    raw: Annotated[bool, typer.Option("--raw")] = False,
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    root = _root(project)
    try:
        plan = RepoPlanStore(root).get(repo_plan_id)
    except KeyError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    typer.echo(plan.model_dump_json(indent=2) if raw else render_repo_plan_markdown(plan))


@repo_app.command("dependencies")
def show_repo_dependencies_command(
    repo_plan_id: Annotated[str, typer.Argument()],
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    """Show dependency compatibility and approved isolated-environment commands."""
    try:
        plan = RepoPlanStore(_root(project)).get(repo_plan_id)
    except KeyError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    typer.echo(render_dependency_report_markdown(plan))


@repo_app.command("repair")
def create_repo_repair_command(
    repo_plan_id: Annotated[str, typer.Argument()],
    feedback: Annotated[str, typer.Option("--feedback")] = "",
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    """Diagnose a failed repository workflow and create a new draft repair plan."""
    root = _root(project)
    try:
        plan = create_repo_repair_plan(root, repo_plan_id, feedback=feedback)
    except (KeyError, OpenAIError, RuntimeError, ValueError) as error:
        typer.echo(f"Repository repair planning failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Repair plan: {plan.repo_plan_id}")
    typer.echo(f"Parent plan: {plan.parent_plan_id}")
    typer.echo(f"Review: {RepoPlanStore(root).markdown_path(plan.repo_plan_id)}")
    typer.echo(render_repo_plan_markdown(plan))


@repo_app.command("approve")
def approve_repo_plan_command(
    repo_plan_id: Annotated[str, typer.Argument()],
    actor: Annotated[str, typer.Option("--actor")] = "human",
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    """Approve the displayed code diff and every planned command together."""
    try:
        plan = approve_repo_plan(_root(project), repo_plan_id, actor)
    except (KeyError, ValueError) as error:
        typer.echo(f"Repository plan approval blocked: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Approved repository plan: {plan.repo_plan_id} by {actor}")


def _run_repo(repo_plan_id: str, project: Path, *, resume: bool) -> None:
    try:
        state = run_repo_plan(_root(project), repo_plan_id, resume=resume)
    except (KeyError, OSError, ValueError) as error:
        typer.echo(f"Repository workflow blocked: {error}", err=True)
        raise typer.Exit(1) from error
    _print_workflow_result(state)
    if state["stage"] != "completed":
        raise typer.Exit(1)


@repo_app.command("run")
def run_repo_plan_command(
    repo_plan_id: Annotated[str, typer.Argument()],
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    """Apply approved code changes and run the approved repository workflow."""
    _run_repo(repo_plan_id, project, resume=False)


@repo_app.command("resume")
def resume_repo_plan_command(
    repo_plan_id: Annotated[str, typer.Argument()],
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    """Retry failed repository runs without repeating successful runs."""
    _run_repo(repo_plan_id, project, resume=True)


@app.command("plans")
def list_plans(project: Annotated[Path, typer.Option("--project")] = Path(".")) -> None:
    """List persisted plans."""
    store = Store(_root(project))
    for plan in store.list_plans():
        typer.echo(f"{plan.plan_id}\t{plan.status.value}\t{plan.title}")


@app.command("plan-show")
def show_plan(
    plan_id: Annotated[str, typer.Argument()],
    raw: Annotated[bool, typer.Option("--raw", help="Show the machine-readable YAML.")] = False,
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    store = Store(_root(project))
    plan = store.get_plan(plan_id)
    if raw:
        typer.echo(
            yaml.safe_dump(plan.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
        )
        typer.echo("Events:")
        typer.echo(json.dumps(store.list_plan_events(plan_id), ensure_ascii=False, indent=2))
        return
    readable = render_plan_markdown(plan)
    store.plan_markdown_path(plan_id).write_text(readable, encoding="utf-8")
    typer.echo(readable)
    typer.echo(f"\n书面计划已保存至：{store.plan_markdown_path(plan_id)}")


@app.command("preflight")
def preflight(
    plan_id: Annotated[str, typer.Argument()],
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    root = _root(project)
    report = run_preflight(Store(root).get_plan(plan_id), root)
    for check in report.checks:
        marker = "PASS" if check.passed else ("WARN" if not check.blocking else "FAIL")
        typer.echo(f"[{marker}] {check.name}: {check.detail}")
    typer.echo(f"Safe to approve: {report.safe}")
    if not report.safe:
        raise typer.Exit(1)


@app.command("approve")
def approve(
    plan_id: Annotated[str, typer.Argument()],
    actor: Annotated[str, typer.Option("--actor")] = "human",
    reason: Annotated[str, typer.Option("--reason")] = "Reviewed experiment plan",
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    root = _root(project)
    try:
        plan, report = approve_plan(root, plan_id, actor, reason)
    except UnsafePlanError as error:
        typer.echo(f"Approval blocked: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Approved: {plan.plan_id} by {plan.approved_by}")
    typer.echo(f"Preflight checks: {len(report.checks)}; safe={report.safe}")


@app.command("reject")
def reject(
    plan_id: Annotated[str, typer.Argument()],
    reason: Annotated[str, typer.Option("--reason")],
    actor: Annotated[str, typer.Option("--actor")] = "human",
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    store = Store(_root(project))
    plan = store.reject_plan(plan_id, actor, reason)
    typer.echo(f"Rejected: {plan.plan_id} by {actor}")


def _run_ids(values: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        if ":" in value:
            variant, seed = value.rsplit(":", 1)
            normalized.append(f"{variant}-seed-{seed}")
        else:
            normalized.append(value)
    return normalized


def _print_workflow_result(state: dict) -> None:
    records = state.get("run_records", [])
    succeeded = sum(record["status"] == "succeeded" for record in records)
    typer.echo(f"Workflow: {state['workflow_id']}")
    typer.echo(f"Stage: {state['stage']}")
    typer.echo(f"Runs: {succeeded}/{len(records)} succeeded")
    for key in ("summary_path", "aggregate_path", "failures_path", "plot_path", "report_path"):
        if state.get(key):
            typer.echo(f"{key}: {state[key]}")
    if state.get("proposed_claims"):
        typer.echo(f"Evidence proposals: {len(state['proposed_claims'])}")


@app.command("run")
def run_plan(
    plan_id: Annotated[str, typer.Argument()],
    project: Annotated[Path, typer.Option("--project")] = Path("."),
    simulate_failure: Annotated[
        list[str] | None,
        typer.Option("--simulate-failure", help="Demo run as variant:seed; repeatable."),
    ] = None,
    simulate_timeout: Annotated[
        list[str] | None,
        typer.Option("--simulate-timeout", help="Demo run as variant:seed; repeatable."),
    ] = None,
    timeout_seconds: Annotated[
        float | None,
        typer.Option("--timeout-seconds", help="Timeout for injected timeout runs only."),
    ] = None,
    crash_after: Annotated[
        int | None,
        typer.Option("--crash-after", help="One-shot recovery demo after N executed runs."),
    ] = None,
) -> None:
    """Execute an approved experiment plan through the checkpointed workflow."""
    root = _root(project)
    try:
        state = start_workflow(
            root,
            plan_id,
            failure_runs=_run_ids(simulate_failure),
            timeout_runs=_run_ids(simulate_timeout),
            timeout_seconds=timeout_seconds,
            crash_after=crash_after,
        )
    except SimulatedCrashError as error:
        typer.echo(f"Workflow interrupted safely: {error}", err=True)
        typer.echo(f"Resume with: reproflow resume {plan_id}", err=True)
        raise typer.Exit(2) from error
    except (KeyError, ValueError) as error:
        typer.echo(f"Run blocked: {error}", err=True)
        raise typer.Exit(1) from error
    _print_workflow_result(state)
    if state["stage"] != "complete":
        raise typer.Exit(1)


@app.command("resume")
def resume(
    workflow_id: Annotated[str, typer.Argument()],
    project: Annotated[Path, typer.Option("--project")] = Path("."),
    keep_simulations: Annotated[
        bool,
        typer.Option("--keep-simulations", help="Repeat injected demo failures/timeouts."),
    ] = False,
) -> None:
    """Resume a workflow without re-running successful tasks."""
    root = _root(project)
    try:
        state = resume_workflow(root, workflow_id, keep_simulations=keep_simulations)
    except SimulatedCrashError as error:
        typer.echo(f"Workflow interrupted safely: {error}", err=True)
        raise typer.Exit(2) from error
    except (KeyError, ValueError) as error:
        typer.echo(f"Resume blocked: {error}", err=True)
        raise typer.Exit(1) from error
    _print_workflow_result(state)
    if state["stage"] != "complete":
        raise typer.Exit(1)


@app.command("workflow-show")
def show_workflow(
    workflow_id: Annotated[str, typer.Argument()],
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    """Show persisted workflow state and the agent trace."""
    store = Store(_root(project))
    try:
        state = store.get_workflow(workflow_id)
    except KeyError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    _print_workflow_result(state)
    typer.echo("Trace:")
    typer.echo(json.dumps(store.list_traces(workflow_id), ensure_ascii=False, indent=2))


@app.command("context-show")
def show_context(
    task: Annotated[str, typer.Option("--task")],
    stage: Annotated[str, typer.Option("--stage")] = "planner",
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    """Show the minimal ContextPack for an agent stage."""
    try:
        context = build_context_pack(_root(project), stage, task)
    except KeyError as error:
        typer.echo(f"Unknown context stage: {stage}", err=True)
        raise typer.Exit(1) from error
    typer.echo(yaml.safe_dump(context.model_dump(mode="json"), sort_keys=False, allow_unicode=True))


@app.command("memories")
def list_memories(
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    """List experiment, failure, and lesson memories."""
    for memory in Store(_root(project)).list_memories():
        typer.echo(f"{memory.memory_id}\t{memory.kind}\t{memory.workflow_id or '-'}\t{memory.text}")


@app.command("report")
def report_command(
    workflow_id: Annotated[str, typer.Argument()],
    narrator: Annotated[str, typer.Option("--narrator", help="mock or api")] = "mock",
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    """Regenerate a traceable Markdown report from verified metrics."""
    try:
        path = generate_report(_root(project), workflow_id, narrator)
    except (KeyError, RuntimeError, ValueError) as error:
        typer.echo(f"Report failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Report: {path}")


def _knowledge_base(root: Path, backend: str):
    if backend == "lexical":
        return LexicalKnowledgeBase(root)
    if backend == "chroma":
        return ChromaKnowledgeBase(root)
    if backend != "auto":
        raise ValueError(f"Unsupported RAG backend: {backend}")
    return get_knowledge_base(root)


@knowledge_app.command("index")
def index_knowledge(
    backend: Annotated[str, typer.Option("--backend")] = "auto",
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    """Index project protocols, papers, and historical reports."""
    root = _root(project)
    try:
        count = _knowledge_base(root, backend).index()
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        typer.echo(f"Knowledge indexing failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Indexed chunks: {count} ({backend})")


@knowledge_app.command("search")
def search_knowledge(
    query: Annotated[str, typer.Argument()],
    backend: Annotated[str, typer.Option("--backend")] = "auto",
    limit: Annotated[int, typer.Option("--limit")] = 5,
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    """Search local knowledge with explicit source attribution."""
    root = _root(project)
    try:
        items = _knowledge_base(root, backend).search(query, limit=limit)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        typer.echo(f"Knowledge search failed: {error}", err=True)
        raise typer.Exit(1) from error
    if not items:
        typer.echo("No evidence found in the local knowledge index.")
        return
    for item in items:
        location = f"page {item.page}" if item.page else (item.section or "document")
        typer.echo(f"[{item.score:.4f}] {item.path} ({location}) #{item.content_hash[:12]}")
        typer.echo(item.content[:500].replace("\n", " "))


@evidence_app.command("list")
def list_evidence(
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    for claim in Store(_root(project)).list_claims():
        typer.echo(render_evidence_markdown(claim, include_artifacts=False))
        typer.echo("\n" + "-" * 72 + "\n")


@evidence_app.command("show")
def show_evidence(
    claim_id: Annotated[str, typer.Argument()],
    raw: Annotated[bool, typer.Option("--raw", help="Show the machine-readable YAML.")] = False,
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    try:
        claim = Store(_root(project)).get_claim(claim_id)
    except KeyError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    if raw:
        typer.echo(
            yaml.safe_dump(claim.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
        )
        return
    typer.echo(render_evidence_markdown(claim))


@evidence_app.command("approve")
def approve_evidence(
    claim_id: Annotated[str, typer.Argument()],
    actor: Annotated[str, typer.Option("--actor")] = "human",
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    try:
        claim = approve_claim(_root(project), claim_id, actor)
    except (KeyError, ValueError) as error:
        typer.echo(f"Evidence approval blocked: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Approved: {claim.claim_id} -> {claim.status.value} by {actor}")


@evidence_app.command("stale")
def stale_evidence(
    claim_id: Annotated[str, typer.Argument()],
    reason: Annotated[str, typer.Option("--reason")],
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    try:
        claim = mark_claim_stale(_root(project), claim_id, reason)
    except (KeyError, ValueError) as error:
        typer.echo(f"Cannot mark stale: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Stale: {claim.claim_id}")


@evidence_app.command("audit-stale")
def audit_evidence(
    claim_id: Annotated[str, typer.Argument()],
    plan_id: Annotated[str, typer.Option("--plan-id")],
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    try:
        claim = audit_claim_staleness(_root(project), claim_id, plan_id)
    except (KeyError, ValueError) as error:
        typer.echo(f"Staleness audit failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Evidence status: {claim.claim_id} -> {claim.status.value}")


@evidence_app.command("sync")
def sync_evidence_command(
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    try:
        registry, results = sync_evidence(_root(project))
    except ValueError as error:
        typer.echo(f"Evidence sync blocked: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Registry: {registry}")
    typer.echo(f"Generated results: {results}")


@app.command("eval")
def run_evals_command(
    cases: Annotated[Path, typer.Option("--cases")] = Path("evals/agent_cases.jsonl"),
    output: Annotated[Path, typer.Option("--output")] = Path("evals/latest_results.json"),
    minimum_passes: Annotated[int, typer.Option("--minimum-passes")] = 18,
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    """Run the deterministic 20-case Agent acceptance suite."""
    root = _root(project)
    cases_path = cases if cases.is_absolute() else root / cases
    output_path = output if output.is_absolute() else root / output
    try:
        report = run_agent_evals(
            cases_path,
            output_path=output_path,
            minimum_passes=minimum_passes,
        )
    except (OSError, ValueError) as error:
        typer.echo(f"Agent eval failed to start: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(
        f"Agent evals: {report['passed']}/{report['total']} passed "
        f"({report['pass_rate']:.0%})"
    )
    typer.echo(f"Report: {output_path}")
    if not report["threshold_met"]:
        raise typer.Exit(1)


@app.command("ui")
def launch_ui(
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    """Launch the conversational Streamlit interface."""
    root = _root(project)
    ui_path = Path(__file__).resolve().parent / "ui.py"
    completed = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ui_path), "--", str(root)],
        cwd=root,
        check=False,
    )
    raise typer.Exit(completed.returncode)


if __name__ == "__main__":
    app()
