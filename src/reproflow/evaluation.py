from __future__ import annotations

import json
import math
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .metrics import aggregate_rows
from .models import (
    ContextPack,
    MetricSpec,
    RepoCodeChange,
    RepoExecutionPlan,
    RepoMetricSpec,
    RepoRunSpec,
    RunRecord,
    RunStatus,
)
from .planner import MockPlanner
from .rag import LexicalKnowledgeBase
from .repo_agent import validate_repo_plan


def load_eval_cases(path: str | Path) -> list[dict[str, Any]]:
    cases = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid eval JSON on line {line_number}: {error}") from error
        if not case.get("id") or not case.get("category"):
            raise ValueError(f"Eval case on line {line_number} needs id and category")
        cases.append(case)
    return cases


class EvalRuntime:
    def __init__(self, root: Path):
        self.root = root
        self.repository = root / "repository"
        self.repository.mkdir(parents=True)
        (self.repository / "experiment.py").write_text(
            "print('safe eval fixture')\n", encoding="utf-8"
        )
        knowledge = root / "knowledge"
        knowledge.mkdir()
        (knowledge / "protocol.md").write_text(
            "# Protocol\nUse ROC-AUC as the primary metric and keep fixed random seeds.\n",
            encoding="utf-8",
        )
        (knowledge / "recovery.md").write_text(
            "# Failure Recovery\nRetry only failed runs; completed runs are idempotent.\n",
            encoding="utf-8",
        )
        (knowledge / "evidence.md").write_text(
            "# Evidence\nOnly reviewed evidence may sync to the paper registry.\n",
            encoding="utf-8",
        )
        self.knowledge = LexicalKnowledgeBase(root)
        self.knowledge.index()


def _planner_eval(case: dict[str, Any], runtime: EvalRuntime) -> tuple[bool, str]:
    goal = case.get("goal", "比较三个模型")
    plan = MockPlanner().create_plan(
        goal,
        ContextPack(stage="planner", task=goal, constraints=["safe"]),
        runtime.root,
    )
    assertion = case["assertion"]
    expected = case["expected"]
    actual: Any
    if assertion == "variant_count":
        actual = len(plan.variants)
    elif assertion == "seeds":
        actual = plan.seeds
    elif assertion == "metrics":
        actual = [metric.name for metric in plan.metrics]
    elif assertion == "safe_command":
        actual = plan.command[1:] == ["examples/sklearn_demo/experiment.py"]
    else:
        raise ValueError(f"Unknown planner assertion: {assertion}")
    return actual == expected, f"expected={expected!r}, actual={actual!r}"


def _guardrail_eval(case: dict[str, Any], runtime: EvalRuntime) -> tuple[bool, str]:
    code_changes = []
    if case.get("generated_code") is not None:
        code_changes.append(
            RepoCodeChange(
                path="generated_eval.py",
                content=case["generated_code"],
                reason="Agent eval fixture",
            )
        )
    run = RepoRunSpec(
        run_id=case["id"],
        purpose="Guardrail evaluation",
        variant=case.get("variant", "baseline"),
        seed=case.get("seed", 1),
        argv=case.get("argv", ["python", "experiment.py"]),
        metrics_file="metrics.json",
    )
    plan = RepoExecutionPlan(
        repository_path=str(runtime.repository),
        repository_commit="eval-fixture",
        goal="Evaluate repository execution guardrails",
        title="Guardrail eval",
        rationale="Deterministic local evaluation",
        code_changes=code_changes,
        runs=[run],
        metrics=[RepoMetricSpec(name="score", key="score")],
        baseline=run.variant,
    )
    accepted = True
    error_text = "accepted"
    try:
        validate_repo_plan(plan)
    except ValueError as error:
        accepted = False
        error_text = str(error)
    expected = bool(case["expected_accepted"])
    return accepted == expected, f"expected_accepted={expected}, result={error_text}"


def _rag_eval(case: dict[str, Any], runtime: EvalRuntime) -> tuple[bool, str]:
    results = runtime.knowledge.search(case["query"], limit=3)
    expected_title = case.get("expected_title")
    actual_title = results[0].title if results else None
    return actual_title == expected_title, (
        f"expected_title={expected_title!r}, actual_title={actual_title!r}"
    )


def _aggregation_eval(case: dict[str, Any], runtime: EvalRuntime) -> tuple[bool, str]:
    metric = case.get("metric", "score")
    direction = case.get("direction", "maximize")
    records = []
    series = [
        ("baseline", case["baseline_values"], case.get("baseline_statuses")),
        ("candidate", case["candidate_values"], case.get("candidate_statuses")),
    ]
    for variant, values, statuses in series:
        statuses = statuses or ["succeeded"] * len(values)
        for index, (value, status) in enumerate(zip(values, statuses, strict=True), 1):
            run_status = RunStatus(status)
            records.append(
                RunRecord(
                    run_id=f"{variant}-{index}",
                    workflow_id="eval-workflow",
                    plan_id="eval-plan",
                    variant=variant,
                    seed=index,
                    command=["python", "experiment.py"],
                    status=run_status,
                    metrics={metric: value} if run_status == RunStatus.SUCCEEDED else {},
                    run_dir=str(runtime.root / "runs" / variant / str(index)),
                )
            )
    rows = aggregate_rows(records, [MetricSpec(name=metric, direction=direction)], "baseline")
    row = next(item for item in rows if item["variant"] == case.get("variant", "candidate"))
    field = case["field"]
    actual = row[f"{metric}_{field}"]
    expected = case["expected"]
    passed = actual is not None and math.isclose(actual, expected, abs_tol=1e-9)
    return passed, f"expected={expected!r}, actual={actual!r}"


EVALUATORS = {
    "planner": _planner_eval,
    "guardrail": _guardrail_eval,
    "rag": _rag_eval,
    "aggregation": _aggregation_eval,
}


def run_agent_evals(
    cases_path: str | Path,
    *,
    output_path: str | Path | None = None,
    minimum_passes: int = 18,
) -> dict[str, Any]:
    cases = load_eval_cases(cases_path)
    results = []
    with tempfile.TemporaryDirectory(prefix="reproflow-evals-") as temp_dir:
        runtime = EvalRuntime(Path(temp_dir))
        for case in cases:
            evaluator = EVALUATORS.get(case["category"])
            if evaluator is None:
                passed, detail = False, f"Unknown category: {case['category']}"
            else:
                try:
                    passed, detail = evaluator(case, runtime)
                except Exception as error:  # evals must report failures instead of aborting
                    passed, detail = False, f"{type(error).__name__}: {error}"
            results.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "passed": passed,
                    "detail": detail,
                }
            )
    passed_count = sum(item["passed"] for item in results)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "pass_rate": passed_count / len(results) if results else 0.0,
        "minimum_passes": minimum_passes,
        "threshold_met": len(results) == 20 and passed_count >= minimum_passes,
        "results": results,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return report
