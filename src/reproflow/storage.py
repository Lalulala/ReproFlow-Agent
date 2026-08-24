from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .models import EvidenceClaim, ExperimentPlan, MemoryItem, PlanStatus, utc_now


SCHEMA = """
CREATE TABLE IF NOT EXISTS plans (
    plan_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS workflows (
    workflow_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    payload TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    workflow_id TEXT,
    tags TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id TEXT NOT NULL,
    node TEXT NOT NULL,
    event TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class Store:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self.state_dir = self.project_root / ".reproflow"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / "reproflow.db"
        with self.connection() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def save_plan(self, plan: ExperimentPlan) -> None:
        payload = plan.model_dump_json()
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO plans(plan_id, payload, status, created_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(plan_id) DO UPDATE SET payload=excluded.payload, status=excluded.status""",
                (plan.plan_id, payload, plan.status.value, plan.created_at.isoformat()),
            )

    def get_plan(self, plan_id: str) -> ExperimentPlan:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT payload FROM plans WHERE plan_id = ?", (plan_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown plan: {plan_id}")
        return ExperimentPlan.model_validate_json(row["payload"])

    def list_plans(self) -> list[ExperimentPlan]:
        with self.connection() as connection:
            rows = connection.execute("SELECT payload FROM plans ORDER BY created_at DESC").fetchall()
        return [ExperimentPlan.model_validate_json(row["payload"]) for row in rows]

    def approve_plan(self, plan_id: str, actor: str) -> ExperimentPlan:
        plan = self.get_plan(plan_id)
        plan.status = PlanStatus.APPROVED
        plan.approved_by = actor
        plan.approved_at = utc_now()
        self.save_plan(plan)
        return plan

    def save_workflow(self, workflow_id: str, plan_id: str, stage: str, payload: dict[str, Any]) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO workflows(workflow_id, plan_id, stage, payload, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(workflow_id) DO UPDATE SET
                   stage=excluded.stage, payload=excluded.payload, updated_at=excluded.updated_at""",
                (workflow_id, plan_id, stage, json.dumps(payload, default=str), utc_now().isoformat()),
            )

    def get_workflow(self, workflow_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown workflow: {workflow_id}")
        payload = json.loads(row["payload"])
        payload["stage"] = row["stage"]
        return payload

    def list_workflows(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM workflows ORDER BY updated_at DESC"
            ).fetchall()
        return [
            {
                "workflow_id": row["workflow_id"],
                "plan_id": row["plan_id"],
                "stage": row["stage"],
                "updated_at": row["updated_at"],
                **json.loads(row["payload"]),
            }
            for row in rows
        ]

    def add_memory(self, memory: MemoryItem) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO memories
                   (memory_id, kind, text, workflow_id, tags, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    memory.memory_id,
                    memory.kind,
                    memory.text,
                    memory.workflow_id,
                    json.dumps(memory.tags),
                    memory.created_at.isoformat(),
                ),
            )

    def search_memories(self, query: str, limit: int = 5) -> list[MemoryItem]:
        terms = {term.lower() for term in query.split() if len(term) > 2}
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM memories ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
        scored: list[tuple[int, MemoryItem]] = []
        for row in rows:
            item = MemoryItem(
                memory_id=row["memory_id"],
                kind=row["kind"],
                text=row["text"],
                workflow_id=row["workflow_id"],
                tags=json.loads(row["tags"]),
                created_at=row["created_at"],
            )
            haystack = (item.text + " " + " ".join(item.tags)).lower()
            score = sum(term in haystack for term in terms)
            scored.append((score, item))
        scored.sort(key=lambda pair: (pair[0], pair[1].created_at), reverse=True)
        return [item for score, item in scored[:limit] if score > 0 or not terms]

    def save_claim(self, claim: EvidenceClaim) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO claims(claim_id, workflow_id, status, payload, created_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(claim_id) DO UPDATE SET status=excluded.status, payload=excluded.payload""",
                (
                    claim.claim_id,
                    claim.workflow_id,
                    claim.status.value,
                    claim.model_dump_json(),
                    claim.created_at.isoformat(),
                ),
            )

    def get_claim(self, claim_id: str) -> EvidenceClaim:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT payload FROM claims WHERE claim_id = ?", (claim_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown claim: {claim_id}")
        return EvidenceClaim.model_validate_json(row["payload"])

    def list_claims(self) -> list[EvidenceClaim]:
        with self.connection() as connection:
            rows = connection.execute("SELECT payload FROM claims ORDER BY created_at DESC").fetchall()
        return [EvidenceClaim.model_validate_json(row["payload"]) for row in rows]

    def add_trace(self, workflow_id: str, node: str, event: str, payload: dict[str, Any]) -> None:
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO traces(workflow_id, node, event, payload, created_at) VALUES (?, ?, ?, ?, ?)",
                (workflow_id, node, event, json.dumps(payload, default=str), utc_now().isoformat()),
            )

    def list_traces(self, workflow_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM traces WHERE workflow_id = ? ORDER BY id", (workflow_id,)
            ).fetchall()
        return [
            {
                "node": row["node"],
                "event": row["event"],
                "payload": json.loads(row["payload"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

