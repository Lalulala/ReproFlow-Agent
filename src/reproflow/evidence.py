from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from .models import ClaimStatus, EvidenceClaim, RunRecord, RunStatus
from .runner import config_hash, git_commit
from .storage import Store


def _read_aggregate(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _claim_id(workflow_id: str, variant: str, metric: str) -> str:
    digest = hashlib.sha256(f"{workflow_id}:{variant}:{metric}".encode()).hexdigest()
    return f"C-{digest[:8].upper()}"


def _environment(record: RunRecord) -> dict:
    path = Path(record.run_dir) / "environment.json"
    return json.loads(path.read_text(encoding="utf-8"))


def propose_evidence(project_root: str | Path, workflow_id: str) -> list[EvidenceClaim]:
    root = Path(project_root).resolve()
    store = Store(root)
    state = store.get_workflow(workflow_id)
    if not state.get("aggregate_path"):
        raise ValueError("Workflow has no aggregate results")
    plan = store.get_plan(state["plan_id"])
    records = store.list_runs(workflow_id)
    rows = _read_aggregate(root / state["aggregate_path"])
    by_variant = {row["variant"]: row for row in rows}
    baseline_row = by_variant.get(plan.baseline)
    if baseline_row is None:
        raise ValueError("Aggregate results do not contain the approved baseline")
    primary = (
        "roc_auc"
        if any(spec.name == "roc_auc" for spec in plan.metrics)
        else plan.metrics[0].name
    )
    metric_spec = next(spec for spec in plan.metrics if spec.name == primary)
    claims: list[EvidenceClaim] = []

    for variant in plan.variants:
        if variant.name == plan.baseline:
            continue
        row = by_variant.get(variant.name)
        if row is None or not row.get(f"{primary}_mean") or not baseline_row.get(f"{primary}_mean"):
            continue
        baseline_value = float(baseline_row[f"{primary}_mean"])
        observed_value = float(row[f"{primary}_mean"])
        improvement = (
            observed_value - baseline_value
            if metric_spec.direction == "maximize"
            else baseline_value - observed_value
        )
        complete = (
            int(row["successful_runs"]) == len(plan.seeds)
            and int(baseline_row["successful_runs"]) == len(plan.seeds)
        )
        if not complete or improvement == 0:
            proposed_status = "inconclusive"
        elif improvement > 0:
            proposed_status = "supported"
        else:
            proposed_status = "contradicted"
        relevant = [
            record
            for record in records
            if record.variant in {plan.baseline, variant.name}
            and record.status == RunStatus.SUCCEEDED
        ]
        environments = [_environment(record) for record in relevant]
        commits = {item["git_commit"] for item in environments}
        hashes = {item["config_sha256"] for item in environments}
        if len(commits) != 1 or len(hashes) != 1:
            proposed_status = "inconclusive"
        common_seeds = sorted(
            {record.seed for record in relevant if record.variant == plan.baseline}
            & {record.seed for record in relevant if record.variant == variant.name}
        )
        artifacts = [
            str(Path(record.run_dir).relative_to(root) / "metrics.json") for record in relevant
        ]
        artifacts.append(state["aggregate_path"])
        if state.get("report_path"):
            artifacts.append(state["report_path"])
        claim_id = _claim_id(workflow_id, variant.name, primary)
        claim = EvidenceClaim(
            claim_id=claim_id,
            claim=(
                f"{variant.name} improves mean {primary} over the approved "
                f"{plan.baseline} baseline for this experiment."
            ),
            proposed_status=proposed_status,
            workflow_id=workflow_id,
            experiment_ids=[record.run_id for record in relevant],
            metric=primary,
            baseline_variant=plan.baseline,
            baseline_value=baseline_value,
            observed_variant=variant.name,
            observed_value=observed_value,
            delta=improvement,
            seeds=common_seeds,
            code_commit=next(iter(commits), "unknown"),
            config_hash=next(iter(hashes), "unknown"),
            artifacts=artifacts,
        )
        try:
            existing = store.get_claim(claim_id)
        except KeyError:
            existing = None
        if existing is not None and existing.status != ClaimStatus.PROPOSED:
            claim = existing
        else:
            store.save_claim(claim)
        claims.append(claim)

    state["proposed_claims"] = [claim.model_dump(mode="json") for claim in claims]
    store.save_workflow(workflow_id, plan.plan_id, state["stage"], state)
    store.add_trace(
        workflow_id,
        "propose_evidence",
        "completed",
        {"claim_ids": [claim.claim_id for claim in claims]},
    )
    return claims


def approve_claim(
    project_root: str | Path, claim_id: str, actor: str
) -> EvidenceClaim:
    store = Store(project_root)
    claim = store.get_claim(claim_id)
    if claim.status != ClaimStatus.PROPOSED:
        raise ValueError(f"Only proposed evidence can be approved; status={claim.status.value}")
    claim.status = ClaimStatus(claim.proposed_status)
    claim.reviewed_by = actor
    claim.reviewed_at = datetime.now(UTC)
    store.save_claim(claim)
    store.add_trace(
        claim.workflow_id,
        "wait_evidence_approval",
        "approved",
        {"claim_id": claim_id, "actor": actor, "status": claim.status.value},
    )
    return claim


def mark_claim_stale(
    project_root: str | Path, claim_id: str, reason: str
) -> EvidenceClaim:
    store = Store(project_root)
    claim = store.get_claim(claim_id)
    if claim.status == ClaimStatus.PROPOSED:
        raise ValueError("Proposed evidence must be reviewed before it can become stale")
    claim.status = ClaimStatus.STALE
    claim.stale_reason = reason
    claim.invalidated_at = datetime.now(UTC)
    store.save_claim(claim)
    return claim


def audit_claim_staleness(
    project_root: str | Path, claim_id: str, plan_id: str
) -> EvidenceClaim:
    root = Path(project_root).resolve()
    store = Store(root)
    claim = store.get_claim(claim_id)
    plan = store.get_plan(plan_id)
    reasons: list[str] = []
    current_commit = git_commit(root)
    current_config = config_hash(plan)
    if claim.code_commit != current_commit:
        reasons.append(f"Git commit changed from {claim.code_commit} to {current_commit}")
    if claim.config_hash != current_config:
        reasons.append("Approved experiment configuration hash changed")
    if reasons:
        return mark_claim_stale(root, claim_id, "; ".join(reasons))
    return claim


def sync_evidence(project_root: str | Path) -> tuple[Path, Path]:
    root = Path(project_root).resolve()
    claims = [claim for claim in Store(root).list_claims() if claim.status != ClaimStatus.PROPOSED]
    if not claims:
        raise ValueError("No reviewed evidence is available to sync")
    paper_dir = root / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    registry_path = paper_dir / "evidence_registry.jsonl"
    results_path = paper_dir / "generated_results.md"
    registry_text = "".join(claim.model_dump_json() + "\n" for claim in claims)
    registry_temp = paper_dir / ".evidence_registry.jsonl.tmp"
    registry_temp.write_text(registry_text, encoding="utf-8")
    registry_temp.replace(registry_path)

    lines = [
        "# Generated Evidence Results",
        "",
        "> Generated from reviewed Evidence Registry entries. Do not edit manually.",
        "",
        "| Claim ID | Status | Claim | Metric | Baseline | Observed | Delta | Workflow |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for claim in claims:
        lines.append(
            f"| {claim.claim_id} | {claim.status.value} | {claim.claim} | {claim.metric} | "
            f"{claim.baseline_value:.10g} | {claim.observed_value:.10g} | "
            f"{claim.delta:.10g} | {claim.workflow_id} |"
        )
    results_temp = paper_dir / ".generated_results.md.tmp"
    results_temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    results_temp.replace(results_path)
    return registry_path, results_path
