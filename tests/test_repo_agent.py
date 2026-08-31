from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import reproflow.repo_agent as repo_agent_module
from reproflow.cli import app
from reproflow.models import RepoRunSpec
from reproflow.repo_agent import (
    GeneratedCodeChange,
    GeneratedRepoPlan,
    RepoPlanStore,
    approve_repo_plan,
    create_repo_plan,
    create_repo_repair_plan,
    inspect_repository,
    run_repo_plan,
    validate_repo_plan,
)
from reproflow.storage import Store

SOURCE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "reproflow-project"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "fixture"\nversion = "0.0.0"\n', encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    Store(root)
    return root


@pytest.fixture
def sklearn_repository(tmp_path: Path) -> Path:
    root = tmp_path / "sklearn-repository"
    target = root / "examples" / "sklearn_demo"
    target.mkdir(parents=True)
    shutil.copy(SOURCE_ROOT / "examples/sklearn_demo/experiment.py", target / "experiment.py")
    (root / "README.md").write_text(
        "# Demo\nRun examples/sklearn_demo/experiment.py to compare models.\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    return root


def test_mock_repo_agent_discovers_and_runs_existing_experiment(
    project: Path, sklearn_repository: Path
) -> None:
    manifest = inspect_repository(sklearn_repository, "比较三种模型")
    assert "examples/sklearn_demo/experiment.py" in {item.path for item in manifest.files}

    plan = create_repo_plan(project, sklearn_repository, "比较三种模型", agent="mock")
    assert len(plan.runs) == 9
    assert plan.code_changes == []
    assert plan.status.value == "draft"
    assert "拟执行实验" in RepoPlanStore(project).markdown_path(plan.repo_plan_id).read_text(
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="not runnable"):
        run_repo_plan(project, plan.repo_plan_id)
    approve_repo_plan(project, plan.repo_plan_id, "Ethan")
    state = run_repo_plan(project, plan.repo_plan_id)

    assert state["stage"] == "completed"
    assert len(state["run_records"]) == 9
    assert all(record["status"] == "succeeded" for record in state["run_records"])
    assert (project / state["summary_path"]).is_file()
    assert (project / state["aggregate_path"]).is_file()
    assert (project / state["report_path"]).is_file()
    assert len(Store(project).list_claims()) == 2


def test_api_repo_agent_writes_code_only_after_approval(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "empty-research-repository"
    repository.mkdir()
    (repository / "README.md").write_text(
        "# Research repository\nCreate a deterministic experiment for the requested score.\n",
        encoding="utf-8",
    )
    (repository / "local_feature.py").write_text("BASE_SCORE = 0.8\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)

    generated_code = """from __future__ import annotations

import argparse
import json
from pathlib import Path
from local_feature import BASE_SCORE

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
Path(args.output).write_text(
    json.dumps({"score": BASE_SCORE + args.seed / 1000}), encoding="utf-8"
)
"""
    generated = GeneratedRepoPlan(
        title="自动生成的可复现实验",
        rationale="Agent 判断仓库缺少可执行入口，因此生成一个最小实验脚本。",
        code_changes=[
            GeneratedCodeChange(
                path="experiments/generated_experiment.py",
                content=generated_code,
                reason="新增满足实验目标的可复现入口。",
            )
        ],
        runs=[
            RepoRunSpec(
                run_id="generated-seed-7",
                purpose="运行 Agent 生成的实验",
                variant="generated",
                seed=7,
                argv=[
                    "python",
                    "experiments/generated_experiment.py",
                    "--seed",
                    "7",
                    "--output",
                    "generated_metrics.json",
                ],
                metrics_file="generated_metrics.json",
            )
        ],
        metrics=[{"name": "score", "parser": "json", "key": "score"}],
        baseline="generated",
    )
    responses = iter(
        [
            json.dumps({"inspect_files": ["README.md"], "reasoning": "Need repository goal."}),
            generated.model_dump_json(),
        ]
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))]
                )
            )
        )
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(repo_agent_module, "OpenAI", lambda **_: fake_client)

    plan = create_repo_plan(
        project,
        repository,
        "根据要求生成实验代码并运行",
        agent="api",
    )
    target = repository / "experiments" / "generated_experiment.py"
    assert not target.exists()
    readable = RepoPlanStore(project).markdown_path(plan.repo_plan_id).read_text(encoding="utf-8")
    assert "```diff" in readable
    assert "+parser.add_argument" in readable

    approve_repo_plan(project, plan.repo_plan_id, "Ethan")
    assert not target.exists()
    state = run_repo_plan(project, plan.repo_plan_id)
    assert target.is_file()
    assert state["stage"] == "completed"
    record = state["run_records"][0]
    assert record["metrics"]["score"] == pytest.approx(0.807)


def test_repo_agent_blocks_source_changes_after_planning(
    project: Path, sklearn_repository: Path
) -> None:
    plan = create_repo_plan(project, sklearn_repository, "compare models", agent="mock")
    script = sklearn_repository / "examples" / "sklearn_demo" / "experiment.py"
    script.write_text(script.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        approve_repo_plan(project, plan.repo_plan_id, "Ethan")


def test_repository_inspection_excludes_secrets_and_redacts_source(tmp_path: Path) -> None:
    repository = tmp_path / "sensitive-repository"
    repository.mkdir()
    (repository / "README.md").write_text("# Safe repository\n", encoding="utf-8")
    (repository / ".env").write_text("OPENAI_API_KEY=must-not-leak\n", encoding="utf-8")
    (repository / "credentials.json").write_text('{"token": "must-not-leak"}', encoding="utf-8")
    (repository / "config.py").write_text(
        'API_TOKEN = "must-not-leak"\nBATCH_SIZE = 32\n', encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)

    manifest = inspect_repository(repository, "inspect configuration")
    visible_paths = {item.path for item in manifest.files}
    assert ".env" not in visible_paths
    assert "credentials.json" not in visible_paths
    assert "must-not-leak" not in json.dumps(manifest.model_dump(), ensure_ascii=False)
    assert "API_TOKEN=<REDACTED>" in manifest.initial_context["config.py"]


def test_repo_cli_inspect_plan_show_and_approve(
    project: Path, sklearn_repository: Path
) -> None:
    runner = CliRunner()
    inspected = runner.invoke(
        app,
        ["repo", "inspect", str(sklearn_repository), "--goal", "比较模型"],
    )
    assert inspected.exit_code == 0
    assert "examples/sklearn_demo/experiment.py" in inspected.stdout

    planned = runner.invoke(
        app,
        [
            "repo",
            "plan",
            str(sklearn_repository),
            "--goal",
            "比较模型",
            "--agent",
            "mock",
            "--project",
            str(project),
        ],
    )
    assert planned.exit_code == 0
    repo_plan_id = planned.stdout.split("Repository plan: ", 1)[1].splitlines()[0]

    shown = runner.invoke(
        app, ["repo", "show", repo_plan_id, "--project", str(project)]
    )
    assert shown.exit_code == 0
    assert "拟执行实验" in shown.stdout

    dependencies = runner.invoke(
        app, ["repo", "dependencies", repo_plan_id, "--project", str(project)]
    )
    assert dependencies.exit_code == 0
    assert "Dependency Preflight" in dependencies.stdout

    approved = runner.invoke(
        app,
        [
            "repo",
            "approve",
            repo_plan_id,
            "--actor",
            "Ethan",
            "--project",
            str(project),
        ],
    )
    assert approved.exit_code == 0
    assert "by Ethan" in approved.stdout


def test_repo_plan_blocks_install_modules_and_escaping_arguments(
    project: Path, sklearn_repository: Path
) -> None:
    plan = create_repo_plan(project, sklearn_repository, "compare models", agent="mock")

    install_plan = plan.model_copy(deep=True)
    install_plan.runs[0].argv = ["python", "-m", "pip", "install", "package"]
    with pytest.raises(ValueError, match="Python module is not allowed"):
        validate_repo_plan(install_plan)

    escaping_plan = plan.model_copy(deep=True)
    escaping_plan.runs[0].argv[-1] = "../../outside.json"
    with pytest.raises(ValueError, match="argument escapes repository"):
        validate_repo_plan(escaping_plan)

    seed_variant_plan = plan.model_copy(deep=True)
    seed_variant_plan.runs[0].variant = f"seed-{seed_variant_plan.runs[0].seed}"
    with pytest.raises(ValueError, match="not the seed"):
        validate_repo_plan(seed_variant_plan)


def test_repair_agent_creates_new_approved_plan_after_real_failure(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repair-repository"
    repository.mkdir()
    (repository / "README.md").write_text("# Repair fixture\n", encoding="utf-8")
    experiment = repository / "experiment.py"
    experiment.write_text('raise RuntimeError("intentional failure")\n', encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)

    initial = GeneratedRepoPlan(
        title="会失败的初始实验",
        rationale="运行仓库现有实验入口。",
        runs=[
            RepoRunSpec(
                run_id="score-seed-1",
                purpose="验证失败修复闭环",
                variant="baseline",
                seed=1,
                argv=["python", "experiment.py", "metrics.json"],
                metrics_file="metrics.json",
            )
        ],
        metrics=[{"name": "score", "parser": "json", "key": "score"}],
    )
    repaired_code = """from __future__ import annotations

import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({"score": 0.91}), encoding="utf-8")
"""
    repaired = GeneratedRepoPlan(
        title="修复后的实验",
        rationale="失败日志表明入口主动抛出异常，因此替换为确定性指标写入。",
        code_changes=[
            GeneratedCodeChange(
                path="experiment.py",
                content=repaired_code,
                reason="移除导致运行中断的异常并写出已声明指标。",
            )
        ],
        runs=initial.runs,
        metrics=initial.metrics,
    )
    responses = iter(
        [
            json.dumps({"inspect_files": ["experiment.py"], "reasoning": "Read entrypoint."}),
            initial.model_dump_json(),
            repaired.model_dump_json(),
        ]
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))]
                )
            )
        )
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(repo_agent_module, "OpenAI", lambda **_: fake_client)

    failed_plan = create_repo_plan(project, repository, "运行并修复实验", agent="api")
    approve_repo_plan(project, failed_plan.repo_plan_id, "Ethan")
    failed_state = run_repo_plan(project, failed_plan.repo_plan_id)
    assert failed_state["stage"] == "partial_failure"
    assert "intentional failure" in (
        project / "runs" / failed_plan.repo_plan_id / "score-seed-1" / "stderr.log"
    ).read_text(encoding="utf-8")

    repair_plan = create_repo_repair_plan(project, failed_plan.repo_plan_id)
    assert repair_plan.parent_plan_id == failed_plan.repo_plan_id
    assert repair_plan.repair_attempt == 1
    assert "intentional failure" in experiment.read_text(encoding="utf-8")
    assert repair_plan.status.value == "draft"

    approve_repo_plan(project, repair_plan.repo_plan_id, "Ethan")
    repaired_state = run_repo_plan(project, repair_plan.repo_plan_id)
    assert repaired_state["stage"] == "completed"
    assert repaired_state["run_records"][0]["metrics"]["score"] == pytest.approx(0.91)
