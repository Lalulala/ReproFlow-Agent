from __future__ import annotations

import json
import statistics
from pathlib import Path

from .models import ExperimentPlan, MemoryItem, RunRecord, RunStatus
from .storage import Store


def _previous_failures(records: list[RunRecord]) -> list[str]:
    failures: list[str] = []
    for record in records:
        attempts_path = Path(record.run_dir) / "attempts.jsonl"
        if not attempts_path.is_file():
            continue
        for line in attempts_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("status") != RunStatus.SUCCEEDED.value:
                failures.append(
                    f"{record.run_id}:{payload.get('status')}:{payload.get('error') or 'unknown'}"
                )
    return failures


def remember_workflow(
    project_root: str | Path,
    workflow_id: str,
    plan: ExperimentPlan,
    records: list[RunRecord],
) -> list[MemoryItem]:
    store = Store(project_root)
    store.delete_workflow_memories(workflow_id)
    succeeded = [record for record in records if record.status == RunStatus.SUCCEEDED]
    primary = (
        "roc_auc" if any(spec.name == "roc_auc" for spec in plan.metrics) else plan.metrics[0].name
    )
    means: dict[str, float] = {}
    for variant in plan.variants:
        values = [
            record.metrics[primary]
            for record in succeeded
            if record.variant == variant.name and primary in record.metrics
        ]
        if values:
            means[variant.name] = statistics.fmean(values)
    mean_text = ", ".join(f"{name}={value:.6f}" for name, value in means.items()) or "none"
    status = "completed" if len(succeeded) == len(records) else "partial"
    memories = [
        MemoryItem(
            memory_id=f"mem-exp-{workflow_id}",
            kind="experiment",
            workflow_id=workflow_id,
            text=(
                f"Workflow {workflow_id} {status} for goal '{plan.goal}'. "
                f"Verified {len(succeeded)}/{len(records)} runs. Mean {primary}: {mean_text}. "
                f"Baseline: {plan.baseline}."
            ),
            tags=[
                "experiment",
                primary,
                plan.baseline,
                *[variant.name for variant in plan.variants],
            ],
        )
    ]

    current_failures = [
        f"{record.run_id}:{record.status.value}:{record.error or 'unknown'}"
        for record in records
        if record.status != RunStatus.SUCCEEDED
    ]
    failure_history = [*current_failures, *_previous_failures(records)]
    if failure_history:
        memories.append(
            MemoryItem(
                memory_id=f"mem-failure-{workflow_id}",
                kind="failure",
                workflow_id=workflow_id,
                text=(f"Failure history for workflow {workflow_id}: " + "; ".join(failure_history)),
                tags=["failure", "timeout", "retry", *[record.variant for record in records]],
            )
        )

    if means:
        best_variant = max(means, key=means.get)
        memories.append(
            MemoryItem(
                memory_id=f"mem-lesson-{workflow_id}",
                kind="lesson",
                workflow_id=workflow_id,
                text=(
                    f"Lesson from workflow {workflow_id}: {best_variant} produced the strongest "
                    f"verified mean {primary} ({means[best_variant]:.6f}) for goal '{plan.goal}'. "
                    "Treat this as dataset-specific evidence and preserve the same seeds for "
                    "follow-up."
                ),
                tags=["lesson", "best-variant", best_variant, primary],
            )
        )

    for memory in memories:
        store.add_memory(memory)
    return memories
