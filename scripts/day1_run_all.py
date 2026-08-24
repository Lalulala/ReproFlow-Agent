"""Run the Day 1 reproducible experiment matrix (3 models x 3 seeds)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

MODELS = ("logistic_regression", "random_forest", "svm")
SEEDS = (42, 43, 44)
METRICS = ("accuracy", "f1", "roc_auc")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def run_one(project_root: Path, workflow_dir: Path, model: str, seed: int) -> dict:
    run_id = f"{model}-seed-{seed}"
    run_dir = workflow_dir / run_id
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=False)
    script = project_root / "examples" / "sklearn_demo" / "experiment.py"
    metrics_path = run_dir / "metrics.json"
    command = [
        sys.executable,
        str(script),
        "--model",
        model,
        "--seed",
        str(seed),
        "--output",
        str(metrics_path),
    ]
    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "command": command,
        "git_commit": git_commit(project_root),
        "script_sha256": sha256(script),
    }
    (run_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "plan_snapshot.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model": model,
                "seed": seed,
                "metrics": list(METRICS),
                "baseline": "logistic_regression",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONHASHSEED": str(seed)},
    )
    duration = time.perf_counter() - started
    (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    status = "succeeded" if completed.returncode == 0 and metrics_path.exists() else "failed"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    manifest = {
        "run_id": run_id,
        "model": model,
        "seed": seed,
        "status": status,
        "exit_code": completed.returncode,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "duration_seconds": duration,
        "metrics": {name: metrics.get(name) for name in METRICS},
        "artifacts": {
            "metrics": str(metrics_path.relative_to(project_root)),
            "stdout": str((run_dir / "stdout.log").relative_to(project_root)),
            "stderr": str((run_dir / "stderr.log").relative_to(project_root)),
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def write_results(workflow_dir: Path, rows: list[dict]) -> None:
    with (workflow_dir / "results.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run_id", "model", "seed", "status", *METRICS, "duration_seconds"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "run_id": row["run_id"],
                    "model": row["model"],
                    "seed": row["seed"],
                    "status": row["status"],
                    **row["metrics"],
                    "duration_seconds": f"{row['duration_seconds']:.6f}",
                }
            )


def write_baseline(workflow_dir: Path, rows: list[dict]) -> None:
    baseline_rows = [row for row in rows if row["model"] == "logistic_regression"]
    payload = {
        "variant": "logistic_regression",
        "seeds": list(SEEDS),
        "successful_runs": len([row for row in baseline_rows if row["status"] == "succeeded"]),
        "metrics_mean": {
            metric: sum(row["metrics"][metric] for row in baseline_rows) / len(baseline_rows)
            for metric in METRICS
        },
        "source_runs": [row["run_id"] for row in baseline_rows],
    }
    (workflow_dir / "baseline.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Run tag; defaults to a UTC timestamp")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    tag = args.tag or datetime.now(UTC).strftime("day1-%Y%m%dT%H%M%SZ")
    workflow_dir = project_root / "runs" / tag
    workflow_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict] = []
    for model in MODELS:
        for seed in SEEDS:
            print(f"[run] {model} seed={seed}", flush=True)
            row = run_one(project_root, workflow_dir, model, seed)
            rows.append(row)
            print(f"      {row['status']} {row['metrics']}", flush=True)

    write_results(workflow_dir, rows)
    write_baseline(workflow_dir, rows)
    succeeded = sum(row["status"] == "succeeded" for row in rows)
    summary = {
        "tag": tag,
        "created_at": datetime.now(UTC).isoformat(),
        "total_runs": len(rows),
        "succeeded": succeeded,
        "failed": len(rows) - succeeded,
        "results": str((workflow_dir / "results.tsv").relative_to(project_root)),
        "baseline": str((workflow_dir / "baseline.json").relative_to(project_root)),
    }
    (workflow_dir / "day1_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nDay 1 matrix complete: {succeeded}/{len(rows)} succeeded")
    print(f"Artifacts: {workflow_dir}")
    return 0 if succeeded == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
