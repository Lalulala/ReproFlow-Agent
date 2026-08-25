from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import reproflow.reporting as reporting_module
from reproflow.cli import app
from reproflow.context import build_context_pack
from reproflow.evidence import (
    approve_claim,
    audit_claim_staleness,
    sync_evidence,
)
from reproflow.models import (
    ExperimentPlan,
    MetricSpec,
    PlanStatus,
    VariantSpec,
)
from reproflow.planner import MockPlanner
from reproflow.rag import LexicalKnowledgeBase
from reproflow.reporting import APINarrator
from reproflow.storage import Store
from reproflow.ui import PAGES, project_root
from reproflow.workflow import start_workflow

FAST_EXPERIMENT = '''
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--simulate-failure", action="store_true")
parser.add_argument("--simulate-timeout", type=float, default=0.0)
args = parser.parse_args()
score = 0.82 if args.model == "beta" else 0.74
payload = {"accuracy": score, "f1": score - 0.01, "roc_auc": score + 0.03}
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload), encoding="utf-8")
print("METRICS_JSON=" + json.dumps(payload))
'''


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("REPROFLOW_RAG_BACKEND", "lexical")
    (tmp_path / "examples").mkdir()
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "paper").mkdir()
    (tmp_path / "examples" / "experiment.py").write_text(FAST_EXPERIMENT, encoding="utf-8")
    (tmp_path / "knowledge" / "protocol.md").write_text(
        "# Protocol\nROC-AUC is the primary metric. Keep fixed seeds.\n", encoding="utf-8"
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "memory-fixture"\nversion = "0.0.0"\n', encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def approved_plan(project: Path) -> ExperimentPlan:
    plan = ExperimentPlan(
        title="Memory and evidence matrix",
        goal="compare alpha and beta with fixed seeds",
        hypothesis="beta improves ROC-AUC",
        command=[sys.executable, "examples/experiment.py"],
        variants=[
            VariantSpec(name="alpha", args=["--model", "alpha"]),
            VariantSpec(name="beta", args=["--model", "beta"]),
        ],
        seeds=[1],
        metrics=[MetricSpec(name=name) for name in ("accuracy", "f1", "roc_auc")],
        baseline="alpha",
        timeout_seconds=5,
        script_path="examples/experiment.py",
        status=PlanStatus.APPROVED,
        approved_by="Ethan",
    )
    Store(project).save_plan(plan)
    return plan


@pytest.fixture
def completed(project: Path) -> tuple[ExperimentPlan, dict]:
    plan = approved_plan(project)
    return plan, start_workflow(project, plan.plan_id)


def test_workflow_creates_memories_report_evidence_and_stage_contexts(
    project: Path, completed: tuple[ExperimentPlan, dict]
) -> None:
    plan, state = completed
    assert state["stage"] == "complete"
    assert (project / state["report_path"]).is_file()
    memories = Store(project).list_memories()
    assert {memory.kind for memory in memories} == {"experiment", "lesson"}
    assert all(memory.workflow_id == plan.plan_id for memory in memories)
    claims = Store(project).list_claims()
    assert len(claims) == 1
    assert claims[0].status.value == "proposed"
    assert claims[0].proposed_status == "supported"

    planner_context = build_context_pack(project, "planner", plan.goal)
    assert planner_context.retrieved_memories
    assert planner_context.retrieved_knowledge
    next_plan = MockPlanner().create_plan(plan.goal, planner_context, project)
    assert any(source["kind"] == "memory" for source in next_plan.context_sources)

    runner_context = build_context_pack(project, "runner", plan.goal)
    assert not runner_context.retrieved_memories
    assert not runner_context.retrieved_knowledge
    assert not runner_context.verified_evidence
    reporter_context = build_context_pack(
        project, "reporter", plan.goal, verified_evidence=[{"verified": True}]
    )
    assert reporter_context.verified_evidence == [{"verified": True}]
    assert not reporter_context.retrieved_memories


def test_report_numbers_and_sources_come_from_verified_csv(
    project: Path, completed: tuple[ExperimentPlan, dict]
) -> None:
    _, state = completed
    report = (project / state["report_path"]).read_text(encoding="utf-8")
    with (project / state["aggregate_path"]).open(encoding="utf-8", newline="") as handle:
        aggregate = list(csv.DictReader(handle))
    for row in aggregate:
        assert row["roc_auc_mean"] in report
    with (project / state["summary_path"]).open(encoding="utf-8", newline="") as handle:
        run_rows = list(csv.DictReader(handle))
    for row in run_rows:
        assert row["metrics_path"] in report
        assert row["manifest_path"] in report
    assert "ContextPack" in report


def test_unapproved_evidence_cannot_sync_then_reviewed_evidence_can(
    project: Path, completed: tuple[ExperimentPlan, dict]
) -> None:
    _, _ = completed
    claim = Store(project).list_claims()[0]
    with pytest.raises(ValueError, match="No reviewed evidence"):
        sync_evidence(project)
    assert not (project / "paper" / "evidence_registry.jsonl").exists()

    approved = approve_claim(project, claim.claim_id, "Ethan")
    assert approved.status.value == "supported"
    registry, results = sync_evidence(project)
    payload = json.loads(registry.read_text(encoding="utf-8").strip())
    assert payload["claim_id"] == claim.claim_id
    assert payload["reviewed_by"] == "Ethan"
    assert claim.claim_id in results.read_text(encoding="utf-8")


def test_config_change_marks_reviewed_claim_stale(
    project: Path, completed: tuple[ExperimentPlan, dict]
) -> None:
    plan, _ = completed
    claim = approve_claim(project, Store(project).list_claims()[0].claim_id, "Ethan")
    plan.timeout_seconds = 4
    Store(project).save_plan(plan)
    stale = audit_claim_staleness(project, claim.claim_id, plan.plan_id)
    assert stale.status.value == "stale"
    assert "configuration hash changed" in stale.stale_reason


def test_api_narrator_rejects_unverified_numbers(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {
                            "interpretation": "Beta improves the score by 99 percent.",
                            "limitations": "None.",
                            "next_steps": "Deploy immediately.",
                        }
                    )
                )
            )
        ]
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response))
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(reporting_module, "OpenAI", lambda **_: fake_client)
    plan = approved_plan(project)
    narrative = APINarrator().narrate(
        plan,
        [{"variant": "beta", "roc_auc_mean": "0.85"}],
        [{"status": "succeeded"}],
    )
    assert "99" not in narrative.interpretation
    assert "verified" in narrative.interpretation


def test_lexical_rag_has_sources_and_returns_empty_without_evidence(project: Path) -> None:
    knowledge = LexicalKnowledgeBase(project)
    assert knowledge.index() >= 1
    items = knowledge.search("primary ROC-AUC metric")
    assert items
    assert items[0].section == "Protocol"
    assert items[0].content_hash
    assert items[0].path.endswith("protocol.md")
    assert knowledge.search("zzzz-no-local-evidence-zzzz") == []


def test_streamlit_exposes_four_pages(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["ui.py", str(project)])
    assert PAGES == ("Workflow", "Runs", "Evidence", "Knowledge")
    assert project_root() == project.resolve()


def test_day5_day6_cli_surface(
    project: Path, completed: tuple[ExperimentPlan, dict]
) -> None:
    plan, state = completed
    runner = CliRunner()
    common = ["--project", str(project)]

    context = runner.invoke(
        app,
        ["context-show", "--task", plan.goal, "--stage", "planner", *common],
    )
    assert context.exit_code == 0, context.output
    assert "retrieved_memories:" in context.output
    assert plan.plan_id in context.output

    memories = runner.invoke(app, ["memories", *common])
    assert memories.exit_code == 0
    assert "lesson" in memories.output

    report = runner.invoke(app, ["report", plan.plan_id, "--narrator", "mock", *common])
    assert report.exit_code == 0, report.output
    assert state["report_path"] in report.output

    indexed = runner.invoke(app, ["knowledge", "index", "--backend", "lexical", *common])
    assert indexed.exit_code == 0, indexed.output
    assert "Indexed chunks:" in indexed.output
    found = runner.invoke(
        app,
        ["knowledge", "search", "primary ROC-AUC", "--backend", "lexical", *common],
    )
    assert found.exit_code == 0
    assert "protocol.md" in found.output
    missing = runner.invoke(
        app,
        ["knowledge", "search", "no-such-evidence-xyz", "--backend", "lexical", *common],
    )
    assert missing.exit_code == 0
    assert "No evidence found" in missing.output

    claim = Store(project).list_claims()[0]
    listed = runner.invoke(app, ["evidence", "list", *common])
    assert listed.exit_code == 0
    assert claim.claim_id in listed.output
    shown = runner.invoke(app, ["evidence", "show", claim.claim_id, *common])
    assert shown.exit_code == 0
    assert "proposed_status:" in shown.output
    blocked_sync = runner.invoke(app, ["evidence", "sync", *common])
    assert blocked_sync.exit_code == 1
    approved = runner.invoke(
        app, ["evidence", "approve", claim.claim_id, "--actor", "Ethan", *common]
    )
    assert approved.exit_code == 0, approved.output
    synced = runner.invoke(app, ["evidence", "sync", *common])
    assert synced.exit_code == 0, synced.output
    assert "evidence_registry.jsonl" in synced.output
