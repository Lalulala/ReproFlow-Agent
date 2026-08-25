from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from .approval import UnsafePlanError, approve_plan
from .context import build_context_pack
from .planner import get_planner
from .preflight import run_preflight
from .runner import SimulatedCrashError
from .storage import Store
from .workflow import resume_workflow, start_workflow

app = typer.Typer(no_args_is_help=True, help="ReproFlow reproducible experiment workflow.")


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
    typer.echo(yaml.safe_dump(plan.model_dump(mode="json"), sort_keys=False, allow_unicode=True))


@app.command("plans")
def list_plans(project: Annotated[Path, typer.Option("--project")] = Path(".")) -> None:
    """List persisted plans."""
    store = Store(_root(project))
    for plan in store.list_plans():
        typer.echo(f"{plan.plan_id}\t{plan.status.value}\t{plan.title}")


@app.command("plan-show")
def show_plan(
    plan_id: Annotated[str, typer.Argument()],
    project: Annotated[Path, typer.Option("--project")] = Path("."),
) -> None:
    store = Store(_root(project))
    plan = store.get_plan(plan_id)
    typer.echo(yaml.safe_dump(plan.model_dump(mode="json"), sort_keys=False, allow_unicode=True))
    typer.echo("Events:")
    typer.echo(json.dumps(store.list_plan_events(plan_id), ensure_ascii=False, indent=2))


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
    for key in ("summary_path", "aggregate_path", "failures_path", "plot_path"):
        if state.get(key):
            typer.echo(f"{key}: {state[key]}")


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


if __name__ == "__main__":
    app()
