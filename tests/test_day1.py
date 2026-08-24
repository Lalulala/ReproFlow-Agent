from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = PROJECT_ROOT / "examples" / "sklearn_demo" / "experiment.py"


@pytest.mark.parametrize("model", ["logistic_regression", "random_forest", "svm"])
def test_bundled_models_emit_fixed_metrics(tmp_path: Path, model: str) -> None:
    output = tmp_path / f"{model}.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT),
            "--model",
            model,
            "--seed",
            "42",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["model"] == model
    assert payload["seed"] == 42
    for metric in ("accuracy", "f1", "roc_auc"):
        assert 0.0 <= payload[metric] <= 1.0
    assert "METRICS_JSON=" in completed.stdout


def test_unknown_model_fails_without_metrics(tmp_path: Path) -> None:
    output = tmp_path / "unknown.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT),
            "--model",
            "unknown",
            "--seed",
            "42",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode != 0
    assert "Unsupported model" in completed.stderr
    assert not output.exists()


def test_intentional_failure_is_observable(tmp_path: Path) -> None:
    output = tmp_path / "failed.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT),
            "--model",
            "logistic_regression",
            "--seed",
            "42",
            "--output",
            str(output),
            "--simulate-failure",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode != 0
    assert "Intentional demo failure" in completed.stderr
    assert not output.exists()
