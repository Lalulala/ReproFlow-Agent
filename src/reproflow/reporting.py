from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Protocol

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from openai import OpenAI
from pydantic import BaseModel

from .context import build_context_pack
from .models import ExperimentPlan, RunStatus
from .storage import Store


class ReportNarrative(BaseModel):
    interpretation: str
    limitations: str
    next_steps: str


class Narrator(Protocol):
    def narrate(
        self,
        plan: ExperimentPlan,
        aggregate_rows: list[dict[str, str]],
        run_rows: list[dict[str, str]],
    ) -> ReportNarrative: ...


def _primary_metric(plan: ExperimentPlan) -> str:
    return (
        "roc_auc"
        if any(spec.name == "roc_auc" for spec in plan.metrics)
        else plan.metrics[0].name
    )


class MockNarrator:
    def narrate(
        self,
        plan: ExperimentPlan,
        aggregate_rows: list[dict[str, str]],
        run_rows: list[dict[str, str]],
    ) -> ReportNarrative:
        primary = _primary_metric(plan)
        successful = [row for row in aggregate_rows if row.get(f"{primary}_mean")]
        if successful:
            direction = next(spec.direction for spec in plan.metrics if spec.name == primary)
            best = (max if direction == "maximize" else min)(
                successful, key=lambda row: float(row[f"{primary}_mean"])
            )
            interpretation = (
                f"Among verified runs, {best['variant']} produced the strongest mean {primary}. "
                "The aggregate table should be used for the exact values and baseline deltas."
            )
        else:
            interpretation = "No variant has enough verified metrics for a directional conclusion."
        failed = [row for row in run_rows if row.get("status") != RunStatus.SUCCEEDED.value]
        limitations = (
            "This result is limited to the bundled dataset, approved variants, and fixed split "
            "seeds. Failed or missing-metric runs are excluded from numerical conclusions."
        )
        if failed:
            limitations += (
                " The failure table must be resolved before treating the matrix as complete."
            )
        return ReportNarrative(
            interpretation=interpretation,
            limitations=limitations,
            next_steps=(
                "Repeat the comparison on an external dataset and inspect calibration, runtime, "
                "and error slices before promoting the result into a broader paper claim."
            ),
        )


class APINarrator:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for API narrator")
        self.client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL"))
        self.model = os.getenv("REPROFLOW_MODEL", "gpt-4o-mini")

    def narrate(
        self,
        plan: ExperimentPlan,
        aggregate_rows: list[dict[str, str]],
        run_rows: list[dict[str, str]],
    ) -> ReportNarrative:
        prompt = {
            "goal": plan.goal,
            "hypothesis": plan.hypothesis,
            "verified_aggregate": aggregate_rows,
            "verified_runs": run_rows,
            "instruction": (
                "Return JSON with interpretation, limitations, and next_steps. Do not include any "
                "digits or numerical values; the report template renders verified numbers itself."
            ),
        }
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Interpret verified ML results without inventing evidence.",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Narrator returned no content")
        narrative = ReportNarrative.model_validate_json(content)
        if re.search(r"\d", " ".join(narrative.model_dump().values())):
            return MockNarrator().narrate(plan, aggregate_rows, run_rows)
        return narrative


def get_narrator(mode: str = "mock") -> Narrator:
    if mode == "api":
        return APINarrator()
    if mode != "mock":
        raise ValueError(f"Unsupported narrator mode: {mode}")
    return MockNarrator()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def generate_report(
    project_root: str | Path, workflow_id: str, narrator_mode: str = "mock"
) -> Path:
    root = Path(project_root).resolve()
    store = Store(root)
    state = store.get_workflow(workflow_id)
    plan = store.get_plan(state["plan_id"])
    if not state.get("summary_path") or not state.get("aggregate_path"):
        raise ValueError("Workflow has no verified summary and aggregate artifacts")
    summary_path = root / state["summary_path"]
    aggregate_path = root / state["aggregate_path"]
    run_rows = _read_csv(summary_path)
    aggregate_rows = _read_csv(aggregate_path)
    records = store.list_runs(workflow_id)
    if not records:
        raise ValueError("Workflow has no run records")
    first_environment = Path(records[0].run_dir) / "environment.json"
    environment = json.loads(first_environment.read_text(encoding="utf-8"))
    verified = {
        "summary": run_rows,
        "aggregate": aggregate_rows,
        "artifacts": [
            str(Path(record.run_dir).relative_to(root) / "metrics.json") for record in records
        ],
    }
    context = build_context_pack(root, "reporter", plan.goal, verified_evidence=[verified])
    narrative = get_narrator(narrator_mode).narrate(plan, aggregate_rows, run_rows)
    template_dir = Path(__file__).resolve().parent / "templates"
    environment_loader = Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    template = environment_loader.get_template("report.md.j2")
    report = template.render(
        plan=plan,
        workflow_id=workflow_id,
        run_rows=run_rows,
        aggregate_rows=aggregate_rows,
        aggregate_headers=list(aggregate_rows[0]) if aggregate_rows else [],
        narrative=narrative,
        git_commit=environment["git_commit"],
        config_hash=environment["config_sha256"],
        script_hash=environment["script_sha256"],
        summary_path=state["summary_path"],
        aggregate_path=state["aggregate_path"],
        failures_path=state["failures_path"],
        plot_path=state["plot_path"],
        context=context,
    )
    report_path = root / plan.artifact_root / workflow_id / "report.md"
    report_path.write_text(report, encoding="utf-8")
    state["report_path"] = str(report_path.relative_to(root))
    store.save_workflow(workflow_id, plan.plan_id, state["stage"], state)
    store.add_trace(
        workflow_id,
        "generate_report",
        "completed",
        {"path": state["report_path"], "narrator": narrator_mode},
    )
    return report_path
