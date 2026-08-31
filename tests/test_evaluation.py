from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from reproflow.cli import app
from reproflow.evaluation import load_eval_cases, run_agent_evals

SOURCE_ROOT = Path(__file__).resolve().parents[1]


def test_agent_eval_suite_has_twenty_passing_cases(tmp_path: Path) -> None:
    cases = SOURCE_ROOT / "evals" / "agent_cases.jsonl"
    output = tmp_path / "results.json"
    report = run_agent_evals(cases, output_path=output)

    assert len(load_eval_cases(cases)) == 20
    assert report["total"] == 20
    assert report["passed"] == 20
    assert report["threshold_met"] is True
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] == 20


def test_agent_eval_cli_writes_report(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="eval-fixture"\nversion="0.0.0"\n', encoding="utf-8"
    )
    output = tmp_path / "eval-results.json"
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "--project",
            str(tmp_path),
            "--cases",
            str(SOURCE_ROOT / "evals" / "agent_cases.jsonl"),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "20/20" in result.output
    assert output.is_file()
