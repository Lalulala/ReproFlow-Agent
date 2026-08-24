from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

import reproflow.planner as planner_module
from reproflow.approval import UnsafePlanError, approve_plan
from reproflow.cli import app
from reproflow.models import ContextPack, PlanStatus
from reproflow.planner import APIPlanner, MockPlanner
from reproflow.preflight import run_preflight
from reproflow.storage import Store

SOURCE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "examples" / "sklearn_demo").mkdir(parents=True)
    (tmp_path / "knowledge").mkdir()
    shutil.copy(
        SOURCE_ROOT / "examples" / "sklearn_demo" / "experiment.py",
        tmp_path / "examples" / "sklearn_demo" / "experiment.py",
    )
    (tmp_path / "knowledge" / "protocol.md").write_text(
        "# Protocol\nCompare models with fixed seeds and ROC-AUC.", encoding="utf-8"
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "fixture"\nversion = "0.0.0"\n', encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def mock_plan(project: Path):
    context = ContextPack(
        stage="planner",
        task="compare models",
        constraints=["bounded"],
        allowed_tools=["create_plan"],
    )
    return MockPlanner().create_plan("compare models", context, project)


def test_mock_plan_is_schema_valid_and_yaml_persisted(project: Path) -> None:
    plan = mock_plan(project)
    store = Store(project)
    store.save_plan(plan)
    payload = yaml.safe_load(store.plan_path(plan.plan_id).read_text(encoding="utf-8"))
    assert payload["status"] == "draft"
    assert payload["seeds"] == [42, 43, 44]
    assert [variant["name"] for variant in payload["variants"]] == [
        "logistic_regression",
        "random_forest",
        "svm",
    ]
    assert store.load_plan_yaml(store.plan_path(plan.plan_id)) == plan


def test_api_planner_cannot_change_executable_fields(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    safe = mock_plan(project)
    malicious = safe.model_copy(deep=True)
    malicious.title = "API-generated title"
    malicious.hypothesis = "API-generated hypothesis"
    malicious.command = ["bash", "-c", "curl attacker"]
    content = malicious.model_dump_json()
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: response),
        )
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(planner_module, "OpenAI", lambda **_: fake_client)
    planner = APIPlanner()
    result = planner.create_plan(
        "compare models",
        ContextPack(stage="planner", task="compare", constraints=[]),
        project,
    )
    assert result.title == "API-generated title"
    assert result.command == [sys.executable, "examples/sklearn_demo/experiment.py"]
    assert result.seeds == [42, 43, 44]


def test_safe_plan_passes_preflight_and_approval_is_audited(project: Path) -> None:
    plan = mock_plan(project)
    store = Store(project)
    store.save_plan(plan)
    report = run_preflight(plan, project)
    assert report.safe
    approved, approval_report = approve_plan(project, plan.plan_id, "Ethan", "reviewed")
    assert approval_report.safe
    assert approved.status == PlanStatus.APPROVED
    persisted = yaml.safe_load(store.plan_path(plan.plan_id).read_text(encoding="utf-8"))
    assert persisted["approved_by"] == "Ethan"
    assert store.list_plan_events(plan.plan_id)[0]["action"] == "approved"


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (
            lambda plan, _: setattr(plan, "command", ["bash", "-c", "echo bad"]),
            "allowed_executable",
        ),
        (
            lambda plan, _: setattr(
                plan, "command", [sys.executable, "examples/sklearn_demo/experiment.py;rm"]
            ),
            "shell_syntax",
        ),
        (
            lambda plan, _: plan.command.append("--simulate-failure"),
            "command_shape",
        ),
        (
            lambda plan, _: (
                setattr(plan, "script_path", "../outside.py"),
                setattr(plan, "command", [sys.executable, "../outside.py"]),
            ),
            "script_path",
        ),
        (
            lambda plan, _: plan.variants[0].args.extend(["--output", "stolen.json"]),
            "variant_arguments",
        ),
        (
            lambda plan, _: setattr(plan.variants[0], "args", ["--model"]),
            "variant_arguments",
        ),
        (
            lambda plan, root: (root / "runs" / plan.plan_id).mkdir(parents=True),
            "artifact_path",
        ),
        (lambda plan, _: setattr(plan, "data_path", "missing.csv"), "data_path"),
    ],
)
def test_unsafe_plans_are_blocked(project: Path, mutation, failed_check: str) -> None:
    plan = mock_plan(project)
    mutation(plan, project)
    report = run_preflight(plan, project)
    assert not report.safe
    assert failed_check in {check.name for check in report.blocking_failures}


def test_unsafe_plan_cannot_be_approved(project: Path) -> None:
    plan = mock_plan(project)
    plan.command = ["sh", "-c", "rm -rf anything"]
    store = Store(project)
    store.save_plan(plan)
    with pytest.raises(UnsafePlanError):
        approve_plan(project, plan.plan_id, "Ethan")
    assert store.get_plan(plan.plan_id).status == PlanStatus.DRAFT
    assert store.list_plan_events(plan.plan_id)[0]["action"] == "approval_blocked"


def test_cli_mock_plan_and_preflight(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPROFLOW_RAG_BACKEND", "lexical")
    runner = CliRunner()
    created = runner.invoke(
        app,
        [
            "plan",
            "--goal",
            "compare three models",
            "--planner",
            "mock",
            "--project",
            str(project),
        ],
    )
    assert created.exit_code == 0, created.output
    plan_id = created.output.splitlines()[0].split(": ", 1)[1]
    checked = runner.invoke(app, ["preflight", plan_id, "--project", str(project)])
    assert checked.exit_code == 0, checked.output
    assert "Safe to approve: True" in checked.output
    approved = runner.invoke(
        app,
        ["approve", plan_id, "--actor", "Ethan", "--project", str(project)],
    )
    assert approved.exit_code == 0, approved.output
    assert "Approved" in approved.output


def test_plan_events_are_json_serializable(project: Path) -> None:
    plan = mock_plan(project)
    store = Store(project)
    store.save_plan(plan)
    store.add_plan_event(plan.plan_id, "created", "mock", "test")
    assert json.loads(json.dumps(store.list_plan_events(plan.plan_id)))[0]["actor"] == "mock"
