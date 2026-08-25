from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Protocol

from openai import OpenAI

from .models import ContextPack, ExperimentPlan, MetricSpec, VariantSpec


class Planner(Protocol):
    def create_plan(
        self, goal: str, context: ContextPack, project_root: Path
    ) -> ExperimentPlan: ...


class MockPlanner:
    """Deterministic planner used for offline demos and tests."""

    def create_plan(self, goal: str, context: ContextPack, project_root: Path) -> ExperimentPlan:
        script = "examples/sklearn_demo/experiment.py"
        return ExperimentPlan(
            title="Three-model breast-cancer classification comparison",
            goal=goal,
            hypothesis=(
                "At least one non-linear model will improve mean ROC-AUC over logistic regression "
                "across three fixed train/test splits."
            ),
            command=[sys.executable, script],
            variants=[
                VariantSpec(name="logistic_regression", args=["--model", "logistic_regression"]),
                VariantSpec(name="random_forest", args=["--model", "random_forest"]),
                VariantSpec(name="svm", args=["--model", "svm"]),
            ],
            seeds=[42, 43, 44],
            metrics=[
                MetricSpec(name="accuracy"),
                MetricSpec(name="f1"),
                MetricSpec(name="roc_auc"),
            ],
            baseline="logistic_regression",
            timeout_seconds=120,
            script_path=script,
            context_sources=[
                *[
                    {"kind": "knowledge", **item.model_dump(mode="json")}
                    for item in context.retrieved_knowledge
                ],
                *[
                    {
                        "kind": "memory",
                        "memory_id": item.memory_id,
                        "memory_type": item.kind,
                        "workflow_id": item.workflow_id,
                        "text": item.text,
                        "tags": item.tags,
                    }
                    for item in context.retrieved_memories
                ],
            ],
        )


class APIPlanner:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for API planner")
        self.client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL"))
        self.model = os.getenv("REPROFLOW_MODEL", "gpt-4o-mini")

    def create_plan(self, goal: str, context: ContextPack, project_root: Path) -> ExperimentPlan:
        example = MockPlanner().create_plan(goal, context, project_root).model_dump(mode="json")
        prompt = {
            "goal": goal,
            "constraints": context.constraints,
            "knowledge": [
                {"title": item.title, "content": item.content[:800], "path": item.path}
                for item in context.retrieved_knowledge
            ],
            "memories": [item.model_dump(mode="json") for item in context.retrieved_memories],
            "required_shape_example": example,
            "rule": (
                "Return only one JSON object. Keep the exact safe sklearn command, "
                "variants, seeds, metrics and script_path from the example. "
                "You may refine title, hypothesis and goal."
            ),
        }
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You create safe, reproducible ML experiment plans."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Planner returned no content")
        candidate = ExperimentPlan.model_validate_json(content)
        safe = MockPlanner().create_plan(goal, context, project_root)
        # The model may improve scientific prose, but never controls executable fields.
        safe.title = candidate.title
        safe.hypothesis = candidate.hypothesis
        safe.goal = goal
        return safe


def get_planner(mode: str | None = None) -> Planner:
    selected = (mode or os.getenv("REPROFLOW_PLANNER", "mock")).lower()
    if selected == "api":
        return APIPlanner()
    if selected != "mock":
        raise ValueError(f"Unsupported planner mode: {selected}")
    return MockPlanner()
