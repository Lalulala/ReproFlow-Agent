from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .models import ExperimentPlan, RunRecord, RunStatus, VariantSpec

LOG_LIMIT_BYTES = 1_000_000
ENV_ALLOWLIST = {
    "ALL_PROXY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LANG",
    "LC_ALL",
    "NO_PROXY",
    "PATH",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TMPDIR",
}


class SimulatedCrashError(RuntimeError):
    """Raised after durable progress has been saved, for the recovery demo."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def config_hash(plan: ExperimentPlan) -> str:
    payload = plan.model_dump_json(exclude={"approved_at", "approved_by", "status"})
    return hashlib.sha256(payload.encode()).hexdigest()


def git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _manifest(record: RunRecord, project_root: Path, attempt: int) -> dict[str, Any]:
    run_dir = Path(record.run_dir)
    return {
        **record.model_dump(mode="json"),
        "attempt": attempt,
        "artifacts": {
            "metrics": str((run_dir / "metrics.json").relative_to(project_root)),
            "stdout": str((run_dir / "stdout.log").relative_to(project_root)),
            "stderr": str((run_dir / "stderr.log").relative_to(project_root)),
            "environment": str((run_dir / "environment.json").relative_to(project_root)),
        },
    }


def load_manifest(path: Path) -> tuple[RunRecord, int] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RunRecord.model_validate(payload), int(payload.get("attempt", 1))


def _safe_environment(seed: int) -> dict[str, str]:
    environment = {name: value for name, value in os.environ.items() if name in ENV_ALLOWLIST}
    environment["PYTHONHASHSEED"] = str(seed)
    return environment


async def _read_limited(stream: asyncio.StreamReader | None) -> tuple[bytes, bool]:
    if stream is None:
        return b"", False
    chunks: list[bytes] = []
    retained = 0
    truncated = False
    while chunk := await stream.read(65536):
        remaining = LOG_LIMIT_BYTES - retained
        if remaining > 0:
            chunks.append(chunk[:remaining])
            retained += min(len(chunk), remaining)
        if len(chunk) > remaining:
            truncated = True
    return b"".join(chunks), truncated


def _write_log(path: Path, content: bytes, truncated: bool) -> None:
    text = content.decode("utf-8", errors="replace")
    if truncated:
        text += "\n[ReproFlow: log truncated at 1000000 bytes]\n"
    path.write_text(text, encoding="utf-8")


async def run_one(
    project_root: Path,
    workflow_id: str,
    plan: ExperimentPlan,
    variant: VariantSpec,
    seed: int,
    *,
    simulate_failure: bool = False,
    simulate_timeout: bool = False,
    timeout_seconds: float | None = None,
    on_update: Callable[[RunRecord, str], None] | None = None,
) -> tuple[RunRecord, str]:
    run_id = f"{variant.name}-seed-{seed}"
    run_dir = project_root / plan.artifact_root / workflow_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.json"
    existing = load_manifest(manifest_path)
    if existing and existing[0].status == RunStatus.SUCCEEDED and metrics_path.is_file():
        if on_update:
            on_update(existing[0], "skipped_succeeded")
        return existing[0], "skipped_succeeded"

    attempt = (existing[1] + 1) if existing else 1
    if existing:
        with (run_dir / "attempts.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_manifest(existing[0], project_root, existing[1])))
            handle.write("\n")

    command = [
        *plan.command,
        *variant.args,
        "--seed",
        str(seed),
        "--output",
        str(metrics_path),
    ]
    if simulate_failure:
        command.append("--simulate-failure")
    effective_timeout = (
        float(timeout_seconds)
        if simulate_timeout and timeout_seconds is not None
        else float(plan.timeout_seconds)
    )
    if simulate_timeout:
        command.extend(["--simulate-timeout", str(effective_timeout + 5.0)])

    record = RunRecord(
        run_id=run_id,
        workflow_id=workflow_id,
        plan_id=plan.plan_id,
        variant=variant.name,
        seed=seed,
        command=command,
        status=RunStatus.RUNNING,
        started_at=datetime.now(UTC),
        run_dir=str(run_dir),
    )
    yaml_snapshot = {
        "plan": plan.model_dump(mode="json"),
        "run": {"run_id": run_id, "variant": variant.name, "seed": seed, "attempt": attempt},
    }
    (run_dir / "plan_snapshot.yaml").write_text(
        yaml.safe_dump(yaml_snapshot, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    script = (project_root / plan.script_path).resolve()
    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "command": command,
        "git_commit": git_commit(project_root),
        "config_sha256": config_hash(plan),
        "script_sha256": sha256_file(script),
        "packages": {
            name: importlib.metadata.version(name) for name in ("scikit-learn", "reproflow-agent")
        },
        "forwarded_environment_names": sorted(_safe_environment(seed)),
    }
    _write_json(run_dir / "environment.json", environment)
    _write_json(manifest_path, _manifest(record, project_root, attempt))
    if on_update:
        on_update(record, "started")

    started = time.perf_counter()
    stdout = b""
    stderr = b""
    stdout_truncated = False
    stderr_truncated = False
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=project_root,
            env=_safe_environment(seed),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_task = asyncio.create_task(_read_limited(process.stdout))
        stderr_task = asyncio.create_task(_read_limited(process.stderr))
        try:
            await asyncio.wait_for(process.wait(), timeout=effective_timeout)
            stdout, stdout_truncated = await stdout_task
            stderr, stderr_truncated = await stderr_task
            record.exit_code = process.returncode
            record.status = RunStatus.SUCCEEDED if process.returncode == 0 else RunStatus.FAILED
            if record.status == RunStatus.FAILED:
                record.error = f"Process exited with code {process.returncode}"
        except TimeoutError:
            process.kill()
            await process.wait()
            stdout, stdout_truncated = await stdout_task
            stderr, stderr_truncated = await stderr_task
            record.exit_code = process.returncode
            record.status = RunStatus.TIMED_OUT
            record.error = f"Timed out after {effective_timeout} seconds"
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            stdout, stdout_truncated = await stdout_task
            stderr, stderr_truncated = await stderr_task
            record.exit_code = process.returncode
            record.status = RunStatus.CANCELLED
            record.error = "Run cancelled"
            raise
    finally:
        record.finished_at = datetime.now(UTC)
        record.duration_seconds = time.perf_counter() - started
        _write_log(run_dir / "stdout.log", stdout, stdout_truncated)
        _write_log(run_dir / "stderr.log", stderr, stderr_truncated)
        _write_json(manifest_path, _manifest(record, project_root, attempt))
        if on_update:
            on_update(record, "finished")
    return record, "executed"


async def execute_matrix(
    project_root: Path,
    workflow_id: str,
    plan: ExperimentPlan,
    faults: dict[str, Any] | None = None,
    on_update: Callable[[RunRecord, str], None] | None = None,
) -> list[RunRecord]:
    settings = faults or {}
    failure_runs = set(settings.get("failure_runs", []))
    timeout_runs = set(settings.get("timeout_runs", []))
    timeout_seconds = settings.get("timeout_seconds")
    crash_after = settings.get("crash_after")
    workflow_dir = project_root / plan.artifact_root / workflow_id
    workflow_dir.mkdir(parents=True, exist_ok=True)
    crash_marker = workflow_dir / ".simulated-crash-triggered"
    records: list[RunRecord] = []
    executed = 0

    for variant in plan.variants:
        for seed in plan.seeds:
            run_id = f"{variant.name}-seed-{seed}"
            record, action = await run_one(
                project_root,
                workflow_id,
                plan,
                variant,
                seed,
                simulate_failure=run_id in failure_runs,
                simulate_timeout=run_id in timeout_runs,
                timeout_seconds=timeout_seconds,
                on_update=on_update,
            )
            records.append(record)
            if action == "executed":
                executed += 1
            if crash_after and executed >= int(crash_after) and not crash_marker.exists():
                crash_marker.write_text(datetime.now(UTC).isoformat(), encoding="utf-8")
                raise SimulatedCrashError(
                    f"Simulated crash after {executed} executed run(s); use reproflow resume"
                )
    return records
