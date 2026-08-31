from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any

from .models import MetricSpec, RunRecord, RunStatus


class MetricParseError(ValueError):
    pass


def _as_number(value: Any, metric: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise MetricParseError(f"Metric {metric!r} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise MetricParseError(f"Metric {metric!r} is not finite: {number}")
    return number


def parse_json_metrics(path: Path, specs: list[MetricSpec]) -> dict[str, float]:
    if not path.is_file():
        raise MetricParseError(f"Missing JSON metrics file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        spec.name: _as_number(payload.get(spec.name), spec.name)
        for spec in specs
        if spec.parser == "json"
    }


def parse_csv_metrics(path: Path, specs: list[MetricSpec]) -> dict[str, float]:
    if not path.is_file():
        raise MetricParseError(f"Missing CSV metrics file: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise MetricParseError(f"CSV metrics file has no rows: {path}")
    row = rows[-1]
    return {
        spec.name: _as_number(row.get(spec.name), spec.name)
        for spec in specs
        if spec.parser == "csv"
    }


def parse_regex_metrics(text: str, specs: list[MetricSpec]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for spec in specs:
        if spec.parser != "regex":
            continue
        pattern = spec.pattern or rf"{re.escape(spec.name)}\s*[=:]\s*(-?\d+(?:\.\d+)?)"
        match = re.search(pattern, text, flags=re.MULTILINE)
        if not match:
            raise MetricParseError(f"Regex did not find metric {spec.name!r}")
        value = match.groupdict().get("value") if match.groupdict() else None
        if value is None:
            value = match.group(1) if match.lastindex else match.group(0)
        metrics[spec.name] = _as_number(value, spec.name)
    return metrics


def parse_run_metrics(record: RunRecord, specs: list[MetricSpec]) -> RunRecord:
    if record.status != RunStatus.SUCCEEDED:
        return record
    run_dir = Path(record.run_dir)
    try:
        metrics: dict[str, float] = {}
        if any(spec.parser == "json" for spec in specs):
            metrics.update(parse_json_metrics(run_dir / "metrics.json", specs))
        if any(spec.parser == "csv" for spec in specs):
            csv_path = run_dir / "metrics.csv"
            if not csv_path.exists():
                csv_path = run_dir / "artifacts" / "metrics.csv"
            metrics.update(parse_csv_metrics(csv_path, specs))
        if any(spec.parser == "regex" for spec in specs):
            text = "\n".join(
                path.read_text(encoding="utf-8", errors="replace")
                for path in (run_dir / "stdout.log", run_dir / "stderr.log")
                if path.exists()
            )
            metrics.update(parse_regex_metrics(text, specs))
        missing = [spec.name for spec in specs if spec.name not in metrics]
        if missing:
            raise MetricParseError(f"Missing metrics: {', '.join(missing)}")
        record.metrics = metrics
    except (json.JSONDecodeError, MetricParseError) as error:
        record.status = RunStatus.FAILED
        record.error = f"Metric parsing failed: {error}"
        record.metrics = {}
    return record


def update_manifest(record: RunRecord) -> None:
    path = Path(record.run_dir) / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update(record.model_dump(mode="json"))
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _format_number(value: float | None) -> str:
    return "" if value is None else f"{value:.10g}"


def write_summary_csv(
    workflow_dir: Path, records: list[RunRecord], specs: list[MetricSpec]
) -> Path:
    path = workflow_dir / "summary.csv"
    fields = [
        "workflow_id",
        "run_id",
        "variant",
        "seed",
        "status",
        "exit_code",
        "duration_seconds",
        *[spec.name for spec in specs],
        "metrics_path",
        "manifest_path",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            run_dir = Path(record.run_dir)
            writer.writerow(
                {
                    "workflow_id": record.workflow_id,
                    "run_id": record.run_id,
                    "variant": record.variant,
                    "seed": record.seed,
                    "status": record.status.value,
                    "exit_code": record.exit_code,
                    "duration_seconds": _format_number(record.duration_seconds),
                    **{spec.name: _format_number(record.metrics.get(spec.name)) for spec in specs},
                    "metrics_path": str(run_dir / "metrics.json"),
                    "manifest_path": str(run_dir / "manifest.json"),
                    "error": record.error or "",
                }
            )
    return path


def aggregate_rows(
    records: list[RunRecord], specs: list[MetricSpec], baseline: str
) -> list[dict[str, Any]]:
    successful = [record for record in records if record.status == RunStatus.SUCCEEDED]
    variants = sorted({record.variant for record in records})
    baseline_means: dict[str, float] = {}
    for spec in specs:
        values = [
            record.metrics[spec.name]
            for record in successful
            if record.variant == baseline and spec.name in record.metrics
        ]
        if values:
            baseline_means[spec.name] = statistics.fmean(values)

    rows: list[dict[str, Any]] = []
    for variant in variants:
        variant_runs = [record for record in records if record.variant == variant]
        row: dict[str, Any] = {
            "variant": variant,
            "total_runs": len(variant_runs),
            "successful_runs": sum(record.status == RunStatus.SUCCEEDED for record in variant_runs),
            "failed_runs": sum(record.status != RunStatus.SUCCEEDED for record in variant_runs),
        }
        for spec in specs:
            values = [
                record.metrics[spec.name]
                for record in variant_runs
                if record.status == RunStatus.SUCCEEDED and spec.name in record.metrics
            ]
            mean = statistics.fmean(values) if values else None
            std = statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)
            best = (
                (max(values) if spec.direction == "maximize" else min(values)) if values else None
            )
            baseline_mean = baseline_means.get(spec.name)
            if mean is None or baseline_mean is None:
                delta = None
            elif spec.direction == "maximize":
                delta = mean - baseline_mean
            else:
                delta = baseline_mean - mean
            row.update(
                {
                    f"{spec.name}_mean": mean,
                    f"{spec.name}_std": std,
                    f"{spec.name}_best": best,
                    f"{spec.name}_delta_vs_baseline": delta,
                }
            )
        rows.append(row)
    return rows


def write_aggregate_csv(
    workflow_dir: Path, rows: list[dict[str, Any]], specs: list[MetricSpec]
) -> Path:
    path = workflow_dir / "aggregate.csv"
    fields = ["variant", "total_runs", "successful_runs", "failed_runs"]
    for spec in specs:
        fields.extend(
            [
                f"{spec.name}_mean",
                f"{spec.name}_std",
                f"{spec.name}_best",
                f"{spec.name}_delta_vs_baseline",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _format_number(value) if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )
    return path


def write_failures_csv(workflow_dir: Path, records: list[RunRecord]) -> Path:
    path = workflow_dir / "failures.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["run_id", "variant", "seed", "status", "exit_code", "error", "run_dir"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            if record.status != RunStatus.SUCCEEDED:
                writer.writerow(
                    {
                        "run_id": record.run_id,
                        "variant": record.variant,
                        "seed": record.seed,
                        "status": record.status.value,
                        "exit_code": record.exit_code,
                        "error": record.error or "",
                        "run_dir": record.run_dir,
                    }
                )
    return path


def write_metric_plot(
    workflow_dir: Path, rows: list[dict[str, Any]], specs: list[MetricSpec]
) -> Path:
    config_dir = workflow_dir / ".matplotlib"
    config_dir.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_dir)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plot_dir = workflow_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    path = plot_dir / "metrics.png"
    figure, axes = plt.subplots(1, len(specs), figsize=(5 * len(specs), 4), squeeze=False)
    variants = [row["variant"] for row in rows]
    for axis, spec in zip(axes[0], specs, strict=True):
        means = [row.get(f"{spec.name}_mean") for row in rows]
        stds = [row.get(f"{spec.name}_std") for row in rows]
        axis.bar(
            variants,
            [value if value is not None else 0 for value in means],
            yerr=[value if value is not None else 0 for value in stds],
            capsize=4,
        )
        axis.set_title(spec.name)
        axis.set_ylabel("mean ± std")
        axis.tick_params(axis="x", rotation=25)
    figure.suptitle("Verified experiment metrics")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path
