from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .evidence import propose_evidence as create_evidence_proposals
from .memory import remember_workflow
from .metrics import (
    aggregate_rows,
    parse_run_metrics,
    update_manifest,
    write_aggregate_csv,
    write_failures_csv,
    write_metric_plot,
    write_summary_csv,
)
from .models import ExperimentPlan, PlanStatus, RunRecord, RunStatus, WorkflowState
from .preflight import run_preflight
from .rag import get_knowledge_base
from .reporting import generate_report as render_report
from .runner import SimulatedCrashError, execute_matrix
from .storage import Store


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _validate_faults(plan: ExperimentPlan, faults: dict[str, Any]) -> dict[str, Any]:
    valid_run_ids = {
        f"{variant.name}-seed-{seed}" for variant in plan.variants for seed in plan.seeds
    }
    failure_runs = set(faults.get("failure_runs", []))
    timeout_runs = set(faults.get("timeout_runs", []))
    unknown = (failure_runs | timeout_runs) - valid_run_ids
    if unknown:
        raise ValueError(f"Unknown simulated run(s): {', '.join(sorted(unknown))}")
    overlap = failure_runs & timeout_runs
    if overlap:
        raise ValueError(f"A run cannot have two simulations: {', '.join(sorted(overlap))}")
    crash_after = faults.get("crash_after")
    total_runs = len(valid_run_ids)
    if crash_after is not None and not 1 <= int(crash_after) <= total_runs:
        raise ValueError(f"crash_after must be between 1 and {total_runs}")
    timeout_seconds = faults.get("timeout_seconds")
    if timeout_seconds is not None and not timeout_runs:
        raise ValueError("timeout_seconds requires at least one simulated timeout run")
    if timeout_seconds is not None:
        timeout_seconds = float(timeout_seconds)
        if timeout_seconds <= 0 or timeout_seconds > plan.timeout_seconds:
            raise ValueError(
                f"timeout_seconds must be > 0 and <= approved timeout {plan.timeout_seconds}"
            )
    elif timeout_runs:
        timeout_seconds = min(1.0, float(plan.timeout_seconds))
    return {
        "failure_runs": sorted(failure_runs),
        "timeout_runs": sorted(timeout_runs),
        "timeout_seconds": timeout_seconds,
        "crash_after": int(crash_after) if crash_after is not None else None,
    }


class WorkflowNodes:
    def __init__(self, project_root: Path):
        self.root = project_root.resolve()
        self.store = Store(self.root)

    def _save(self, state: WorkflowState, stage: str) -> None:
        state["stage"] = stage
        self.store.save_workflow(state["workflow_id"], state["plan_id"], stage, dict(state))

    def execute_runs(self, state: WorkflowState) -> dict[str, Any]:
        plan = ExperimentPlan.model_validate(state["plan"])
        workflow_id = state["workflow_id"]
        records_by_id = {record.run_id: record for record in self.store.list_runs(workflow_id)}
        working_state: WorkflowState = dict(state)
        self._save(working_state, "execute_runs")
        self.store.add_trace(workflow_id, "execute_runs", "started", {"plan_id": plan.plan_id})

        def on_update(record: RunRecord, event: str) -> None:
            records_by_id[record.run_id] = record
            self.store.save_run(record)
            working_state["run_records"] = [
                item.model_dump(mode="json")
                for item in sorted(records_by_id.values(), key=lambda value: value.run_id)
            ]
            self._save(working_state, "execute_runs")
            self.store.add_trace(
                workflow_id,
                "execute_runs",
                event,
                {"run_id": record.run_id, "status": record.status.value},
            )

        try:
            records = asyncio.run(
                execute_matrix(
                    self.root,
                    workflow_id,
                    plan,
                    faults=state.get("faults", {}),
                    on_update=on_update,
                )
            )
        except SimulatedCrashError as error:
            working_state.setdefault("errors", []).append(str(error))
            self._save(working_state, "interrupted")
            self.store.add_trace(workflow_id, "execute_runs", "interrupted", {"error": str(error)})
            raise
        self.store.add_trace(
            workflow_id,
            "execute_runs",
            "completed",
            {"run_count": len(records)},
        )
        return {
            "run_records": [record.model_dump(mode="json") for record in records],
            "stage": "execute_runs",
        }

    def parse_metrics(self, state: WorkflowState) -> dict[str, Any]:
        plan = ExperimentPlan.model_validate(state["plan"])
        workflow_id = state["workflow_id"]
        records: list[RunRecord] = []
        for payload in state.get("run_records", []):
            record = parse_run_metrics(RunRecord.model_validate(payload), plan.metrics)
            update_manifest(record)
            self.store.save_run(record)
            records.append(record)
        failures = [record.run_id for record in records if record.status != RunStatus.SUCCEEDED]
        updated: WorkflowState = dict(state)
        updated["run_records"] = [record.model_dump(mode="json") for record in records]
        self._save(updated, "parse_metrics")
        self.store.add_trace(
            workflow_id,
            "parse_metrics",
            "completed",
            {"verified": len(records) - len(failures), "failures": failures},
        )
        return {"run_records": updated["run_records"], "stage": "parse_metrics"}

    def aggregate_results(self, state: WorkflowState) -> dict[str, Any]:
        plan = ExperimentPlan.model_validate(state["plan"])
        records = [RunRecord.model_validate(payload) for payload in state["run_records"]]
        workflow_dir = self.root / plan.artifact_root / state["workflow_id"]
        summary_path = write_summary_csv(workflow_dir, records, plan.metrics)
        rows = aggregate_rows(records, plan.metrics, plan.baseline)
        aggregate_path = write_aggregate_csv(workflow_dir, rows, plan.metrics)
        failures_path = write_failures_csv(workflow_dir, records)
        plot_path = write_metric_plot(workflow_dir, rows, plan.metrics)
        result = {
            "summary_path": _relative(self.root, summary_path),
            "aggregate_path": _relative(self.root, aggregate_path),
            "failures_path": _relative(self.root, failures_path),
            "plot_path": _relative(self.root, plot_path),
            "stage": "aggregate_results",
        }
        updated: WorkflowState = {**state, **result}
        self._save(updated, "aggregate_results")
        self.store.add_trace(
            state["workflow_id"],
            "aggregate_results",
            "completed",
            {key: value for key, value in result.items() if key.endswith("_path")},
        )
        return result

    def analyze_results(self, state: WorkflowState) -> dict[str, Any]:
        plan = ExperimentPlan.model_validate(state["plan"])
        records = [RunRecord.model_validate(payload) for payload in state["run_records"]]
        memories = remember_workflow(self.root, state["workflow_id"], plan, records)
        updated: WorkflowState = dict(state)
        self._save(updated, "analyze_results")
        self.store.add_trace(
            state["workflow_id"],
            "analyze_results",
            "completed",
            {"memory_ids": [memory.memory_id for memory in memories]},
        )
        return {"stage": "analyze_results"}

    def generate_report(self, state: WorkflowState) -> dict[str, Any]:
        path = render_report(self.root, state["workflow_id"], narrator_mode="mock")
        get_knowledge_base(self.root).index()
        report_path = _relative(self.root, path)
        updated: WorkflowState = dict(state)
        updated["report_path"] = report_path
        self._save(updated, "generate_report")
        return {"report_path": report_path, "stage": "generate_report"}

    def propose_evidence(self, state: WorkflowState) -> dict[str, Any]:
        claims = create_evidence_proposals(self.root, state["workflow_id"])
        proposed = [claim.model_dump(mode="json") for claim in claims]
        updated: WorkflowState = dict(state)
        updated["proposed_claims"] = proposed
        self._save(updated, "propose_evidence")
        return {"proposed_claims": proposed, "stage": "propose_evidence"}

    def complete(self, state: WorkflowState) -> dict[str, Any]:
        records = [RunRecord.model_validate(payload) for payload in state["run_records"]]
        all_succeeded = bool(records) and all(
            record.status == RunStatus.SUCCEEDED for record in records
        )
        stage = "complete" if all_succeeded else "partial_failure"
        updated: WorkflowState = dict(state)
        self._save(updated, stage)
        plan = self.store.get_plan(state["plan_id"])
        if all_succeeded and plan.status != PlanStatus.COMPLETED:
            self.store.complete_plan(plan.plan_id)
        self.store.add_trace(
            state["workflow_id"],
            "complete",
            stage,
            {
                "succeeded": sum(record.status == RunStatus.SUCCEEDED for record in records),
                "total": len(records),
            },
        )
        return {"stage": stage}


def build_graph(project_root: Path, checkpointer: SqliteSaver):
    nodes = WorkflowNodes(project_root)
    builder = StateGraph(WorkflowState)
    builder.add_node("execute_runs", nodes.execute_runs)
    builder.add_node("parse_metrics", nodes.parse_metrics)
    builder.add_node("aggregate_results", nodes.aggregate_results)
    builder.add_node("analyze_results", nodes.analyze_results)
    builder.add_node("generate_report", nodes.generate_report)
    builder.add_node("propose_evidence", nodes.propose_evidence)
    builder.add_node("complete", nodes.complete)
    builder.add_edge(START, "execute_runs")
    builder.add_edge("execute_runs", "parse_metrics")
    builder.add_edge("parse_metrics", "aggregate_results")
    builder.add_edge("aggregate_results", "analyze_results")
    builder.add_edge("analyze_results", "generate_report")
    builder.add_edge("generate_report", "propose_evidence")
    builder.add_edge("propose_evidence", "complete")
    builder.add_edge("complete", END)
    return builder.compile(checkpointer=checkpointer, name="reproflow-day3-day4")


def invoke_workflow(project_root: Path, state: WorkflowState) -> WorkflowState:
    root = project_root.resolve()
    checkpoint_path = root / ".reproflow" / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": state["workflow_id"]}}
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        graph = build_graph(root, checkpointer)
        result = graph.invoke(state, config=config)
    return WorkflowState(**result)


def start_workflow(
    project_root: Path,
    plan_id: str,
    *,
    failure_runs: list[str] | None = None,
    timeout_runs: list[str] | None = None,
    timeout_seconds: float | None = None,
    crash_after: int | None = None,
) -> WorkflowState:
    root = project_root.resolve()
    store = Store(root)
    plan = store.get_plan(plan_id)
    if plan.status != PlanStatus.APPROVED:
        raise ValueError(f"Plan must be approved before running; status={plan.status.value}")
    report = run_preflight(plan, root)
    if not report.safe:
        details = "; ".join(f"{check.name}: {check.detail}" for check in report.blocking_failures)
        raise ValueError(f"Preflight failed: {details}")
    faults = _validate_faults(
        plan,
        {
            "failure_runs": failure_runs or [],
            "timeout_runs": timeout_runs or [],
            "timeout_seconds": timeout_seconds,
            "crash_after": crash_after,
        },
    )
    workflow_id = plan.plan_id
    state = WorkflowState(
        workflow_id=workflow_id,
        project_root=str(root),
        goal=plan.goal,
        stage="prepared",
        plan=plan.model_dump(mode="json"),
        plan_id=plan.plan_id,
        plan_approved=True,
        run_records=[],
        errors=[],
        trace=[],
        faults=faults,
    )
    store.save_workflow(workflow_id, plan.plan_id, "prepared", dict(state))
    store.add_trace(workflow_id, "workflow", "created", {"faults": faults})
    return invoke_workflow(root, state)


def resume_workflow(
    project_root: Path, workflow_id: str, *, keep_simulations: bool = False
) -> WorkflowState:
    root = project_root.resolve()
    store = Store(root)
    payload = store.get_workflow(workflow_id)
    state = WorkflowState(**payload)
    plan = store.get_plan(state["plan_id"])
    if plan.status not in {PlanStatus.APPROVED, PlanStatus.COMPLETED}:
        raise ValueError(f"Workflow plan is not runnable; status={plan.status.value}")
    if not keep_simulations:
        state["faults"] = {
            "failure_runs": [],
            "timeout_runs": [],
            "timeout_seconds": None,
            "crash_after": None,
        }
    state["errors"] = []
    store.add_trace(
        workflow_id,
        "workflow",
        "resumed",
        {"keep_simulations": keep_simulations},
    )
    return invoke_workflow(root, state)
