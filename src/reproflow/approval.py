from __future__ import annotations

from pathlib import Path

from .models import ExperimentPlan, PlanStatus, PreflightReport
from .preflight import run_preflight
from .storage import Store


class UnsafePlanError(RuntimeError):
    pass


def approve_plan(
    project_root: str | Path,
    plan_id: str,
    actor: str,
    reason: str | None = None,
) -> tuple[ExperimentPlan, PreflightReport]:
    store = Store(project_root)
    plan = store.get_plan(plan_id)
    if plan.status != PlanStatus.DRAFT:
        raise ValueError(f"Only draft plans can be approved; current status={plan.status.value}")
    report = run_preflight(plan, project_root)
    if not report.safe:
        details = "; ".join(f"{item.name}: {item.detail}" for item in report.blocking_failures)
        store.add_plan_event(plan_id, "approval_blocked", actor, details)
        raise UnsafePlanError(details)
    return store.approve_plan(plan_id, actor, reason), report

