from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from .human_views import render_plan_markdown
from .models import EvidenceClaim, ExperimentPlan, MemoryItem, PlanStatus, RunRecord, utc_now

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
CREATE TABLE IF NOT EXISTS plan_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id TEXT NOT NULL,
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    workflow_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (workflow_id, run_id)
);
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    repository_path TEXT NOT NULL,
    agent_mode TEXT NOT NULL,
    current_plan_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    plan_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session
ON chat_messages(session_id, id);
"""


class Store:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self.state_dir = self.project_root / ".reproflow"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / "reproflow.db"
        self.plans_dir = self.state_dir / "plans"
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        self.readable_plans_dir = self.project_root / "plans"
        self.readable_plans_dir.mkdir(parents=True, exist_ok=True)
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
                   ON CONFLICT(plan_id) DO UPDATE SET
                   payload=excluded.payload, status=excluded.status""",
                (plan.plan_id, payload, plan.status.value, plan.created_at.isoformat()),
            )
        self.plan_path(plan.plan_id).write_text(
            yaml.safe_dump(
                plan.model_dump(mode="json"),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        self.plan_markdown_path(plan.plan_id).write_text(
            render_plan_markdown(plan), encoding="utf-8"
        )

    def plan_path(self, plan_id: str) -> Path:
        return self.plans_dir / f"{plan_id}.yaml"

    def plan_markdown_path(self, plan_id: str) -> Path:
        return self.readable_plans_dir / f"{plan_id}.md"

    def load_plan_yaml(self, path: str | Path) -> ExperimentPlan:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return ExperimentPlan.model_validate(payload)

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
            rows = connection.execute(
                "SELECT payload FROM plans ORDER BY created_at DESC"
            ).fetchall()
        return [ExperimentPlan.model_validate_json(row["payload"]) for row in rows]

    def approve_plan(self, plan_id: str, actor: str, reason: str | None = None) -> ExperimentPlan:
        plan = self.get_plan(plan_id)
        plan.status = PlanStatus.APPROVED
        plan.approved_by = actor
        plan.approved_at = utc_now()
        self.save_plan(plan)
        self.add_plan_event(plan_id, "approved", actor, reason)
        return plan

    def reject_plan(self, plan_id: str, actor: str, reason: str) -> ExperimentPlan:
        plan = self.get_plan(plan_id)
        plan.status = PlanStatus.REJECTED
        self.save_plan(plan)
        self.add_plan_event(plan_id, "rejected", actor, reason)
        return plan

    def complete_plan(self, plan_id: str) -> ExperimentPlan:
        plan = self.get_plan(plan_id)
        plan.status = PlanStatus.COMPLETED
        self.save_plan(plan)
        self.add_plan_event(plan_id, "completed", "workflow", "All runs succeeded")
        return plan

    def add_plan_event(
        self, plan_id: str, action: str, actor: str, reason: str | None = None
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO plan_events(plan_id, action, actor, reason, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (plan_id, action, actor, reason, utc_now().isoformat()),
            )

    def list_plan_events(self, plan_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM plan_events WHERE plan_id = ? ORDER BY id", (plan_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def save_workflow(
        self, workflow_id: str, plan_id: str, stage: str, payload: dict[str, Any]
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO workflows(workflow_id, plan_id, stage, payload, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(workflow_id) DO UPDATE SET
                   stage=excluded.stage, payload=excluded.payload,
                   updated_at=excluded.updated_at""",
                (
                    workflow_id,
                    plan_id,
                    stage,
                    json.dumps(payload, default=str),
                    utc_now().isoformat(),
                ),
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
            rows = connection.execute("SELECT * FROM workflows ORDER BY updated_at DESC").fetchall()
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

    def save_run(self, record: RunRecord) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO runs(workflow_id, run_id, status, payload, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(workflow_id, run_id) DO UPDATE SET
                   status=excluded.status, payload=excluded.payload,
                   updated_at=excluded.updated_at""",
                (
                    record.workflow_id,
                    record.run_id,
                    record.status.value,
                    record.model_dump_json(),
                    utc_now().isoformat(),
                ),
            )

    def list_runs(self, workflow_id: str) -> list[RunRecord]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT payload FROM runs WHERE workflow_id = ? ORDER BY run_id", (workflow_id,)
            ).fetchall()
        return [RunRecord.model_validate_json(row["payload"]) for row in rows]

    def create_chat_session(
        self,
        repository_path: str,
        agent_mode: str,
        title: str = "新实验对话",
    ) -> dict[str, Any]:
        session_id = f"chat-{uuid4().hex[:12]}"
        now = utc_now().isoformat()
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO chat_sessions
                   (session_id, title, repository_path, agent_mode,
                    current_plan_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, NULL, ?, ?)""",
                (session_id, title, repository_path, agent_mode, now, now),
            )
        return self.get_chat_session(session_id)

    def get_chat_session(self, session_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown chat session: {session_id}")
        return dict(row)

    def list_chat_sessions(self, limit: int = 30) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT * FROM chat_sessions AS session
                   WHERE EXISTS (
                       SELECT 1 FROM chat_messages AS message
                       WHERE message.session_id = session.session_id
                         AND message.role = 'user'
                   )
                   ORDER BY updated_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def prune_empty_chat_sessions(self) -> int:
        """Remove legacy chats that never received a user message."""
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT session_id FROM chat_sessions AS session
                   WHERE NOT EXISTS (
                       SELECT 1 FROM chat_messages AS message
                       WHERE message.session_id = session.session_id
                         AND message.role = 'user'
                   )"""
            ).fetchall()
            session_ids = [row["session_id"] for row in rows]
            if not session_ids:
                return 0
            placeholders = ", ".join("?" for _ in session_ids)
            connection.execute(
                f"DELETE FROM chat_messages WHERE session_id IN ({placeholders})",
                session_ids,
            )
            connection.execute(
                f"DELETE FROM chat_sessions WHERE session_id IN ({placeholders})",
                session_ids,
            )
        return len(session_ids)

    def update_chat_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        repository_path: str | None = None,
        agent_mode: str | None = None,
        current_plan_id: str | None = None,
        update_plan: bool = False,
    ) -> dict[str, Any]:
        current = self.get_chat_session(session_id)
        with self.connection() as connection:
            connection.execute(
                """UPDATE chat_sessions SET
                   title = ?, repository_path = ?, agent_mode = ?, current_plan_id = ?,
                   updated_at = ? WHERE session_id = ?""",
                (
                    title if title is not None else current["title"],
                    repository_path if repository_path is not None else current["repository_path"],
                    agent_mode if agent_mode is not None else current["agent_mode"],
                    current_plan_id if update_plan else current["current_plan_id"],
                    utc_now().isoformat(),
                    session_id,
                ),
            )
        return self.get_chat_session(session_id)

    def add_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        plan_id: str | None = None,
    ) -> dict[str, Any]:
        if role not in {"assistant", "user"}:
            raise ValueError(f"Unsupported chat role: {role}")
        now = utc_now().isoformat()
        with self.connection() as connection:
            cursor = connection.execute(
                """INSERT INTO chat_messages(session_id, role, content, plan_id, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, role, content, plan_id, now),
            )
            connection.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
        return {
            "id": cursor.lastrowid,
            "session_id": session_id,
            "role": role,
            "content": content,
            "plan_id": plan_id,
            "created_at": now,
        }

    def list_chat_messages(self, session_id: str) -> list[dict[str, Any]]:
        self.get_chat_session(session_id)
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
        return [dict(row) for row in rows]

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

    def list_memories(
        self, limit: int = 100, *, workflow_id: str | None = None
    ) -> list[MemoryItem]:
        with self.connection() as connection:
            if workflow_id is None:
                rows = connection.execute(
                    "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT * FROM memories WHERE workflow_id = ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (workflow_id, limit),
                ).fetchall()
        return [
            MemoryItem(
                memory_id=row["memory_id"],
                kind=row["kind"],
                text=row["text"],
                workflow_id=row["workflow_id"],
                tags=json.loads(row["tags"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def delete_workflow_memories(self, workflow_id: str) -> None:
        with self.connection() as connection:
            connection.execute("DELETE FROM memories WHERE workflow_id = ?", (workflow_id,))

    def save_claim(self, claim: EvidenceClaim) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO claims(claim_id, workflow_id, status, payload, created_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(claim_id) DO UPDATE SET
                   status=excluded.status, payload=excluded.payload""",
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

    def list_claims(self, *, workflow_id: str | None = None) -> list[EvidenceClaim]:
        with self.connection() as connection:
            if workflow_id is None:
                rows = connection.execute(
                    "SELECT payload FROM claims ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT payload FROM claims WHERE workflow_id = ?
                       ORDER BY created_at DESC""",
                    (workflow_id,),
                ).fetchall()
        return [EvidenceClaim.model_validate_json(row["payload"]) for row in rows]

    def add_trace(self, workflow_id: str, node: str, event: str, payload: dict[str, Any]) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO traces(workflow_id, node, event, payload, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
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
