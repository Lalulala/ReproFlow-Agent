from __future__ import annotations

import asyncio
import csv
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from reproflow.cli import app
from reproflow.metrics import parse_csv_metrics, parse_regex_metrics
from reproflow.models import (
    ExperimentPlan,
    MetricSpec,
    PlanStatus,
    RunStatus,
    VariantSpec,
)
from reproflow.runner import SimulatedCrashError, load_manifest, run_one
from reproflow.storage import Store
from reproflow.workflow import resume_workflow, start_workflow

FAST_EXPERIMENT = '''
import argparse
import json
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--simulate-failure", action="store_true")
parser.add_argument("--simulate-timeout", type=float, default=0.0)
args = parser.parse_args()
if args.simulate_failure:
    raise RuntimeError("intentional failure")
if args.simulate_timeout:
    time.sleep(args.simulate_timeout)
base = 0.7 + (sum(map(ord, args.model)) % 10) / 100 + (args.seed % 3) / 1000
payload = {"accuracy": base, "f1": base - 0.01, "roc_auc": base + 0.02}
if args.model == "missing":
    payload.pop("f1")
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload), encoding="utf-8")
print(f"accuracy={payload['accuracy']}")
'''


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "experiment.py").write_text(FAST_EXPERIMENT, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "runner-fixture"\nversion = "0.0.0"\n', encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def approved_plan(
    project: Path,
    variants: tuple[str, ...] = ("alpha", "beta"),
    seeds: tuple[int, ...] = (1, 2),
    metrics: list[MetricSpec] | None = None,
) -> ExperimentPlan:
    plan = ExperimentPlan(
        title="Fast matrix",
        goal="test the runner",
        hypothesis="variants differ",
        command=[sys.executable, "examples/experiment.py"],
        variants=[VariantSpec(name=name, args=["--model", name]) for name in variants],
        seeds=list(seeds),
        metrics=metrics
        or [MetricSpec(name=name) for name in ("accuracy", "f1", "roc_auc")],
        baseline=variants[0],
        timeout_seconds=5,
        script_path="examples/experiment.py",
        status=PlanStatus.APPROVED,
        approved_by="Ethan",
    )
    Store(project).save_plan(plan)
    return plan


def test_successful_workflow_writes_checkpoints_csv_and_plot(project: Path) -> None:
    plan = approved_plan(project)
    state = start_workflow(project, plan.plan_id)
    assert state["stage"] == "complete"
    assert len(state["run_records"]) == 4
    assert all(record["status"] == "succeeded" for record in state["run_records"])
    workflow_dir = project / "runs" / plan.plan_id
    assert (project / ".reproflow" / "checkpoints.sqlite").is_file()
    assert len(list(workflow_dir.glob("*/manifest.json"))) == 4
    with (workflow_dir / "summary.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 4
    with (workflow_dir / "aggregate.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["accuracy_mean"]
    assert (workflow_dir / "plots" / "metrics.png").read_bytes().startswith(b"\x89PNG")
    assert Store(project).get_plan(plan.plan_id).status == PlanStatus.COMPLETED
    trace_nodes = {event["node"] for event in Store(project).list_traces(plan.plan_id)}
    assert {"execute_runs", "parse_metrics", "aggregate_results", "complete"} <= trace_nodes


def test_crash_resume_skips_successful_attempts(project: Path) -> None:
    plan = approved_plan(project)
    with pytest.raises(SimulatedCrashError):
        start_workflow(project, plan.plan_id, crash_after=2)
    before = Store(project).list_runs(plan.plan_id)
    assert len(before) == 2
    assert all(record.status == RunStatus.SUCCEEDED for record in before)

    state = resume_workflow(project, plan.plan_id)
    assert state["stage"] == "complete"
    assert len(state["run_records"]) == 4
    for record in state["run_records"]:
        loaded = load_manifest(Path(record["run_dir"]) / "manifest.json")
        assert loaded is not None
        assert loaded[1] == 1
        assert not (Path(record["run_dir"]) / "attempts.jsonl").exists()


def test_failure_timeout_are_aggregated_then_retried_on_resume(project: Path) -> None:
    plan = approved_plan(project, variants=("alpha", "beta", "gamma"), seeds=(1,))
    state = start_workflow(
        project,
        plan.plan_id,
        failure_runs=["beta-seed-1"],
        timeout_runs=["gamma-seed-1"],
        timeout_seconds=0.05,
    )
    assert state["stage"] == "partial_failure"
    statuses = {record["run_id"]: record["status"] for record in state["run_records"]}
    assert statuses == {
        "alpha-seed-1": "succeeded",
        "beta-seed-1": "failed",
        "gamma-seed-1": "timed_out",
    }
    failures_path = project / state["failures_path"]
    with failures_path.open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2

    resumed = resume_workflow(project, plan.plan_id)
    assert resumed["stage"] == "complete"
    assert all(record["status"] == "succeeded" for record in resumed["run_records"])
    attempts = {
        record["run_id"]: load_manifest(Path(record["run_dir"]) / "manifest.json")[1]
        for record in resumed["run_records"]
    }
    assert attempts == {"alpha-seed-1": 1, "beta-seed-1": 2, "gamma-seed-1": 2}


def test_missing_metric_is_not_treated_as_success(project: Path) -> None:
    plan = approved_plan(project, variants=("alpha", "missing"), seeds=(1,))
    state = start_workflow(project, plan.plan_id)
    assert state["stage"] == "partial_failure"
    missing = next(record for record in state["run_records"] if record["variant"] == "missing")
    assert missing["status"] == "failed"
    assert "Metric parsing failed" in missing["error"]


def test_csv_and_regex_metric_parsers(project: Path) -> None:
    csv_path = project / "metrics.csv"
    csv_path.write_text("loss,accuracy\n0.4,0.8\n0.3,0.9\n", encoding="utf-8")
    csv_specs = [
        MetricSpec(name="loss", parser="csv", direction="minimize"),
        MetricSpec(name="accuracy", parser="csv"),
    ]
    assert parse_csv_metrics(csv_path, csv_specs) == {"loss": 0.3, "accuracy": 0.9}
    regex_specs = [
        MetricSpec(name="f1", parser="regex", pattern=r"F1=(?P<value>\d+\.\d+)")
    ]
    assert parse_regex_metrics("epoch done F1=0.88", regex_specs) == {"f1": 0.88}


def test_async_run_can_be_cancelled_and_is_recorded(project: Path) -> None:
    plan = approved_plan(project, variants=("alpha",), seeds=(1,))

    async def cancel_run() -> None:
        task = asyncio.create_task(
            run_one(
                project,
                plan.plan_id,
                plan,
                plan.variants[0],
                1,
                simulate_timeout=True,
                timeout_seconds=4,
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_run())
    loaded = load_manifest(project / "runs" / plan.plan_id / "alpha-seed-1" / "manifest.json")
    assert loaded is not None
    assert loaded[0].status == RunStatus.CANCELLED


def test_cli_run_requires_approval_and_prints_artifacts(project: Path) -> None:
    plan = approved_plan(project, variants=("alpha",), seeds=(1,))
    store = Store(project)
    plan.status = PlanStatus.DRAFT
    store.save_plan(plan)
    runner = CliRunner()
    blocked = runner.invoke(app, ["run", plan.plan_id, "--project", str(project)])
    assert blocked.exit_code == 1
    assert "Plan must be approved" in blocked.output

    plan.status = PlanStatus.APPROVED
    store.save_plan(plan)
    completed = runner.invoke(app, ["run", plan.plan_id, "--project", str(project)])
    assert completed.exit_code == 0, completed.output
    assert "Runs: 1/1 succeeded" in completed.output
    assert "aggregate_path:" in completed.output
