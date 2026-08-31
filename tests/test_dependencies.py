from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import reproflow.dependencies as dependency_module
from reproflow.dependencies import (
    analyze_dependencies,
    environment_commands,
    prepare_environment,
)
from reproflow.models import RepoEnvironmentSpec


def test_dependency_preflight_recommends_isolated_environment(tmp_path: Path) -> None:
    repository = tmp_path / "dependency-repository"
    repository.mkdir()
    (repository / "requirements.txt").write_text(
        "reproflow-definitely-missing-package>=1\n"
        "-e .\n"
        "https://example.com/unsafe.whl\n",
        encoding="utf-8",
    )

    report, environment = analyze_dependencies(
        repository, ["reproflow-definitely-missing-package"]
    )

    assert report.needs_isolation
    assert report.checks[0].status == "missing"
    assert report.resolved_requirements == ["reproflow-definitely-missing-package>=1"]
    assert len(report.blocked_entries) == 2
    assert environment.mode == "isolated"
    assert environment.environment_id.startswith("repo-env-")

    _, alternate_python = analyze_dependencies(repository, [], python_version="3.10")
    assert alternate_python.mode == "isolated"
    assert "3.10" in alternate_python.install_commands[0]


def test_prepare_environment_runs_only_approved_uv_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    environment = RepoEnvironmentSpec(
        mode="isolated",
        python_version="3.12",
        environment_id="repo-env-123456789abc",
        requirements=["fixture-package>=1"],
        rationale="test",
    )
    environment.install_commands = environment_commands(project, environment)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(command)
        if "venv" in command:
            interpreter = (
                project
                / ".reproflow"
                / "environments"
                / environment.environment_id
                / "bin"
                / "python"
            )
            interpreter.parent.mkdir(parents=True, exist_ok=True)
            interpreter.write_text("", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(dependency_module.shutil, "which", lambda _: "/usr/local/bin/uv")
    monkeypatch.setattr(subprocess, "run", fake_run)

    interpreter = prepare_environment(project, environment)
    assert interpreter.is_file()
    assert ["venv" in command for command in calls] == [True, False]
    assert "pip" in calls[1]
    assert ".reproflow/uv-cache" in calls[0]
    assert "fixture-package>=1" in calls[1]

    calls.clear()
    assert prepare_environment(project, environment) == interpreter
    assert calls == []
