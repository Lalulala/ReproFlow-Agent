from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import ContextPack
from .rag import get_knowledge_base
from .storage import Store

STAGE_POLICIES = {
    "planner": {
        "tools": ["search_knowledge", "search_memories", "create_plan"],
        "constraints": [
            "Use only the bundled sklearn experiment entrypoint.",
            "Create a bounded plan with three fixed seeds.",
            "Do not invent metrics or execute commands.",
        ],
        "budget": 5000,
    },
    "runner": {
        "tools": ["preflight", "execute_approved_plan"],
        "constraints": [
            "Execute only the approved argv list with shell disabled.",
            "Stay inside the project root and enforce timeout.",
        ],
        "budget": 1200,
    },
    "reporter": {
        "tools": ["aggregate_metrics", "generate_report", "propose_evidence"],
        "constraints": [
            "Use only verified metrics and traceable artifact paths.",
            "Never invent numerical results.",
        ],
        "budget": 5000,
    },
}


def build_context_pack(
    project_root: str | Path,
    stage: str,
    task: str,
    verified_evidence: list[dict[str, Any]] | None = None,
) -> ContextPack:
    root = Path(project_root).resolve()
    policy = STAGE_POLICIES[stage]
    store = Store(root)
    knowledge = get_knowledge_base(root).search(task, limit=5) if stage == "planner" else []
    memories = store.search_memories(task, limit=5) if stage == "planner" else []
    evidence = verified_evidence or []
    if stage == "planner" and not evidence:
        evidence = [
            claim.model_dump(mode="json")
            for claim in store.list_claims()
            if claim.status.value not in {"proposed", "stale"}
        ][:5]
    return ContextPack(
        stage=stage,
        task=task,
        constraints=policy["constraints"],
        retrieved_memories=memories,
        retrieved_knowledge=knowledge,
        verified_evidence=evidence if stage != "runner" else [],
        allowed_tools=policy["tools"],
        token_budget=policy["budget"],
    )
