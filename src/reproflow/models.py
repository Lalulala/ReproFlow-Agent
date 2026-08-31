from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class PlanStatus(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class ClaimStatus(StrEnum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"
    STALE = "stale"


class RepoPlanStatus(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL_FAILURE = "partial_failure"


class MetricSpec(BaseModel):
    name: str
    direction: Literal["maximize", "minimize"] = "maximize"
    parser: Literal["json", "csv", "regex"] = "json"
    pattern: str | None = None


class VariantSpec(BaseModel):
    name: str
    args: list[str] = Field(default_factory=list)


class ExperimentPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: f"plan-{uuid4().hex[:10]}")
    title: str
    goal: str
    hypothesis: str
    command: list[str]
    variants: list[VariantSpec]
    seeds: list[int] = Field(min_length=1)
    metrics: list[MetricSpec]
    baseline: str
    timeout_seconds: int = Field(default=120, ge=1, le=86400)
    data_path: str | None = None
    script_path: str
    artifact_root: str = "runs"
    status: PlanStatus = PlanStatus.DRAFT
    approved_by: str | None = None
    approved_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    context_sources: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("command")
    @classmethod
    def command_must_be_argv(cls, value: list[str]) -> list[str]:
        if not value or any(not isinstance(item, str) or not item for item in value):
            raise ValueError("command must be a non-empty argv list")
        return value

    @field_validator("baseline")
    @classmethod
    def baseline_must_exist(cls, value: str, info: Any) -> str:
        variants = info.data.get("variants", [])
        if variants and value not in {variant.name for variant in variants}:
            raise ValueError("baseline must match a variant name")
        return value

    @model_validator(mode="after")
    def matrix_must_be_unambiguous(self) -> ExperimentPlan:
        variant_names = [variant.name for variant in self.variants]
        metric_names = [metric.name for metric in self.metrics]
        if not variant_names:
            raise ValueError("at least one variant is required")
        if len(set(variant_names)) != len(variant_names):
            raise ValueError("variant names must be unique")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if not metric_names:
            raise ValueError("at least one metric is required")
        if len(set(metric_names)) != len(metric_names):
            raise ValueError("metric names must be unique")
        return self


class PreflightCheck(BaseModel):
    name: str
    passed: bool
    detail: str
    blocking: bool = True


class PreflightReport(BaseModel):
    plan_id: str
    checks: list[PreflightCheck]

    @property
    def safe(self) -> bool:
        return all(check.passed or not check.blocking for check in self.checks)

    @property
    def blocking_failures(self) -> list[PreflightCheck]:
        return [check for check in self.checks if check.blocking and not check.passed]


class RunRecord(BaseModel):
    run_id: str
    workflow_id: str
    plan_id: str
    variant: str
    seed: int
    command: list[str]
    status: RunStatus = RunStatus.PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None
    duration_seconds: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    run_dir: str
    error: str | None = None


class RetrievedItem(BaseModel):
    source_id: str
    title: str
    content: str
    source_type: str = "document"
    path: str
    section: str | None = None
    page: int | None = None
    tags: list[str] = Field(default_factory=list)
    content_hash: str
    score: float = 0.0


class MemoryItem(BaseModel):
    memory_id: str = Field(default_factory=lambda: f"mem-{uuid4().hex[:10]}")
    kind: Literal["experiment", "failure", "lesson"]
    text: str
    workflow_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class ContextPack(BaseModel):
    stage: str
    task: str
    constraints: list[str]
    retrieved_memories: list[MemoryItem] = Field(default_factory=list)
    retrieved_knowledge: list[RetrievedItem] = Field(default_factory=list)
    verified_evidence: list[dict[str, Any]] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    token_budget: int = 4000


class EvidenceClaim(BaseModel):
    claim_id: str = Field(default_factory=lambda: f"C-{uuid4().hex[:8].upper()}")
    claim: str
    status: ClaimStatus = ClaimStatus.PROPOSED
    proposed_status: Literal["supported", "contradicted", "inconclusive"]
    workflow_id: str
    experiment_ids: list[str]
    metric: str
    baseline_variant: str
    baseline_value: float
    observed_variant: str
    observed_value: float
    delta: float
    seeds: list[int]
    code_commit: str
    config_hash: str
    artifacts: list[str]
    paper_sections: list[str] = Field(default_factory=lambda: ["experiments.results"])
    created_at: datetime = Field(default_factory=utc_now)
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    invalidated_at: datetime | None = None
    stale_reason: str | None = None


class RepoMetricSpec(BaseModel):
    name: str
    direction: Literal["maximize", "minimize"] = "maximize"
    parser: Literal["json", "regex"] = "json"
    key: str | None = None
    pattern: str | None = None


class RepoRunSpec(BaseModel):
    run_id: str
    purpose: str
    variant: str
    seed: int
    argv: list[str]
    cwd: str = "."
    timeout_seconds: int = Field(default=600, ge=1, le=86400)
    metrics_file: str | None = None


class RepoCodeChange(BaseModel):
    path: str
    content: str
    reason: str
    before_sha256: str | None = None
    diff: str = ""


class DependencyItem(BaseModel):
    requirement: str
    distribution: str
    installed_version: str | None = None
    status: Literal["satisfied", "missing", "conflict", "unsafe"]
    detail: str


class DependencyReport(BaseModel):
    manifest_files: list[str] = Field(default_factory=list)
    requested_requirements: list[str] = Field(default_factory=list)
    resolved_requirements: list[str] = Field(default_factory=list)
    checks: list[DependencyItem] = Field(default_factory=list)
    blocked_entries: list[str] = Field(default_factory=list)
    needs_isolation: bool = False


class RepoEnvironmentSpec(BaseModel):
    mode: Literal["current", "isolated"] = "current"
    python_version: str
    environment_id: str
    requirements: list[str] = Field(default_factory=list)
    install_commands: list[list[str]] = Field(default_factory=list)
    rationale: str


class RepoExecutionPlan(BaseModel):
    repo_plan_id: str = Field(default_factory=lambda: f"repo-plan-{uuid4().hex[:10]}")
    repository_path: str
    repository_commit: str
    goal: str
    title: str
    rationale: str
    inspected_files: list[str] = Field(default_factory=list)
    source_hashes: dict[str, str] = Field(default_factory=dict)
    code_changes: list[RepoCodeChange] = Field(default_factory=list, max_length=8)
    runs: list[RepoRunSpec] = Field(min_length=1, max_length=50)
    metrics: list[RepoMetricSpec] = Field(default_factory=list)
    baseline: str | None = None
    dependency_report: DependencyReport | None = None
    environment: RepoEnvironmentSpec | None = None
    parent_plan_id: str | None = None
    repair_attempt: int = Field(default=0, ge=0, le=3)
    status: RepoPlanStatus = RepoPlanStatus.DRAFT
    created_at: datetime = Field(default_factory=utc_now)
    approved_by: str | None = None
    approved_at: datetime | None = None


class WorkflowState(TypedDict, total=False):
    workflow_id: str
    project_root: str
    goal: str
    stage: str
    plan: dict[str, Any]
    plan_id: str
    plan_approved: bool
    evidence_approved: bool
    context_pack: dict[str, Any]
    preflight: list[dict[str, Any]]
    run_records: list[dict[str, Any]]
    summary_path: str
    aggregate_path: str
    failures_path: str
    plot_path: str
    report_path: str
    proposed_claims: list[dict[str, Any]]
    errors: list[str]
    trace: list[dict[str, Any]]
    faults: dict[str, Any]


def project_path(root: str | Path, relative: str) -> Path:
    return (Path(root).resolve() / relative).resolve()
