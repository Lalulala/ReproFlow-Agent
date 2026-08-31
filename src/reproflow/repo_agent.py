from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from .dependencies import (
    EnvironmentSetupError,
    analyze_dependencies,
    environment_commands,
    environment_root,
    normalize_requested_requirement,
    prepare_environment,
)
from .metrics import aggregate_rows, write_aggregate_csv, write_failures_csv, write_summary_csv
from .models import (
    ClaimStatus,
    EvidenceClaim,
    MemoryItem,
    MetricSpec,
    RepoCodeChange,
    RepoExecutionPlan,
    RepoMetricSpec,
    RepoPlanStatus,
    RepoRunSpec,
    RunRecord,
    RunStatus,
)
from .storage import Store

EXCLUDED_DIRS = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".reproflow",
    ".tox",
    ".uv-cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "runs",
}
SAFE_TEXT_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".json", ".txt"}
SENSITIVE_NAMES = {".env", "credentials.json", "secrets.json", "id_rsa", "id_ed25519"}
ENTRY_HINTS = ("train", "experiment", "evaluate", "eval", "main", "run", "test")
SHELL_TOKENS = (";", "&&", "||", "|", "`", "$(", ">", "<", "\n", "\r")
LOG_LIMIT = 1_000_000
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
BLOCKED_PYTHON_MODULES = {
    "ensurepip",
    "http.server",
    "pip",
    "pydoc",
    "venv",
    "webbrowser",
}


class RepoFileInfo(BaseModel):
    path: str
    size: int
    role: str


class RepoManifest(BaseModel):
    repository_path: str
    git_commit: str
    file_count: int
    files: list[RepoFileInfo]
    initial_context: dict[str, str]


class InspectionRequest(BaseModel):
    inspect_files: list[str] = Field(default_factory=list, max_length=16)
    reasoning: str = ""


class GeneratedCodeChange(BaseModel):
    path: str
    content: str
    reason: str


class GeneratedRepoPlan(BaseModel):
    title: str
    rationale: str
    code_changes: list[GeneratedCodeChange] = Field(default_factory=list, max_length=8)
    runs: list[RepoRunSpec] = Field(min_length=1, max_length=50)
    metrics: list[RepoMetricSpec] = Field(default_factory=list)
    baseline: str | None = None
    required_packages: list[str] = Field(default_factory=list, max_length=32)
    python_version: str | None = None


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def _is_git_repository(root: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _role(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("readme"):
        return "readme"
    if name in {"pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"}:
        return "dependency_or_build"
    stem = path.stem.lower()
    if path.suffix == ".py" and any(
        stem == hint or stem.startswith(f"{hint}_") or stem.endswith(f"_{hint}")
        for hint in ENTRY_HINTS
    ):
        return "candidate_entrypoint"
    if "config" in name or path.suffix in {".yaml", ".yml", ".json"}:
        return "configuration"
    if path.suffix == ".py":
        return "python_source"
    return "documentation"


def _safe_relative(root: Path, relative: str) -> Path:
    path = Path(relative)
    if relative in {"", "."}:
        return root.resolve()
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Path must stay relative to the repository: {relative}")
    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Path escapes the repository: {relative}") from error
    return target


def _safe_to_share(path: Path) -> bool:
    lower = path.name.lower()
    return (
        lower not in SENSITIVE_NAMES
        and not any(term in lower for term in ("secret", "credential", "private_key"))
        and path.suffix.lower() in SAFE_TEXT_SUFFIXES
    )


def _redact(text: str) -> str:
    pattern = re.compile(r"(?im)^([A-Z0-9_]*(?:KEY|TOKEN|PASSWORD|SECRET)[A-Z0-9_]*)\s*[:=]\s*.+$")
    return pattern.sub(r"\1=<REDACTED>", text)


def _read_context(root: Path, relative: str, limit: int = 20_000) -> str:
    target = _safe_relative(root, relative)
    if not target.is_file() or not _safe_to_share(target):
        return ""
    return _redact(target.read_text(encoding="utf-8", errors="replace")[:limit])


def _subprocess_text(value: str | bytes | None) -> str:
    """Normalize partial output carried by TimeoutExpired across Python versions."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def inspect_repository(repository_path: str | Path, goal: str = "") -> RepoManifest:
    root = Path(repository_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Repository does not exist: {root}")
    if not _is_git_repository(root):
        raise ValueError(f"Not a Git repository: {root}")
    files: list[RepoFileInfo] = []
    for path in sorted(root.rglob("*")):
        parts = path.relative_to(root).parts
        excluded = any(part in EXCLUDED_DIRS or part.startswith(".") for part in parts)
        if not path.is_file() or excluded:
            continue
        if path.is_symlink():
            continue
        relative = path.relative_to(root)
        if _safe_to_share(path):
            files.append(
                RepoFileInfo(path=relative.as_posix(), size=path.stat().st_size, role=_role(path))
            )
        if len(files) >= 3000:
            break
    goal_terms = {term.lower() for term in re.findall(r"[\w-]+", goal) if len(term) > 1}

    def priority(item: RepoFileInfo) -> tuple[int, int, str]:
        matched = sum(term in item.path.lower() for term in goal_terms)
        role_score = {
            "readme": 5,
            "dependency_or_build": 4,
            "candidate_entrypoint": 3,
            "configuration": 2,
            "python_source": 1,
        }.get(item.role, 0)
        return (-matched, -role_score, item.path)

    initial_context: dict[str, str] = {}
    for item in sorted(files, key=priority)[:24]:
        content = _read_context(root, item.path, 8000)
        if content:
            initial_context[item.path] = content
    return RepoManifest(
        repository_path=str(root),
        git_commit=_git_commit(root),
        file_count=len(files),
        files=files,
        initial_context=initial_context,
    )


def render_manifest_markdown(manifest: RepoManifest) -> str:
    candidates = [item for item in manifest.files if item.role == "candidate_entrypoint"]
    configs = [
        item
        for item in manifest.files
        if item.role in {"dependency_or_build", "configuration", "readme"}
    ]
    lines = [
        "# 仓库自动探索报告",
        "",
        f"- 仓库：`{manifest.repository_path}`",
        f"- Git commit：`{manifest.git_commit}`",
        f"- 可读文件：{manifest.file_count}",
        f"- 候选训练/评测入口：{len(candidates)}",
        "",
        "## Agent 优先检查的文件",
        "",
    ]
    for item in [*configs[:12], *candidates[:20]]:
        lines.append(f"- `{item.path}`（{item.role}，{item.size} bytes）")
    return "\n".join(lines).rstrip() + "\n"


def render_dependency_report_markdown(plan: RepoExecutionPlan) -> str:
    lines = [f"# Dependency Preflight：{plan.repo_plan_id}", ""]
    if plan.environment is None or plan.dependency_report is None:
        return "\n".join([*lines, "该计划没有依赖预检快照。", ""])
    lines.extend(
        [
            f"- 建议环境：`{plan.environment.mode}`",
            f"- Python：`{plan.environment.python_version}`",
            f"- 环境编号：`{plan.environment.environment_id}`",
            "- 依赖清单来源："
            f"{', '.join(plan.dependency_report.manifest_files) or 'Agent/源码导入'}",
            "",
            "| Requirement | 当前版本 | 状态 | 说明 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in plan.dependency_report.checks:
        lines.append(
            f"| `{item.requirement}` | {item.installed_version or '-'} | "
            f"{item.status} | {item.detail} |"
        )
    if plan.environment.install_commands:
        lines.extend(["", "## 批准后执行", ""])
        lines.extend(f"- `{' '.join(command)}`" for command in plan.environment.install_commands)
    if plan.dependency_report.blocked_entries:
        lines.extend(["", "## 未采用的不安全/不支持条目", ""])
        lines.extend(f"- `{item}`" for item in plan.dependency_report.blocked_entries)
    return "\n".join(lines).rstrip() + "\n"


def _chat_json(client: OpenAI, model: str, system: str, payload: dict[str, Any]) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Repository Agent returned no content")
    return content


class APIRepoPlanner:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for API repository agent")
        self.client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL"))
        self.model = os.getenv("REPROFLOW_MODEL", "gpt-4o-mini")
        self.inspected_files: list[str] = []

    def create_plan(self, manifest: RepoManifest, goal: str) -> GeneratedRepoPlan:
        inventory = [item.model_dump() for item in manifest.files]
        first = _chat_json(
            self.client,
            self.model,
            (
                "You inspect ML repositories. Select the smallest set of additional files needed "
                "to decide what code should be changed and run. Include the implementation files "
                "for every repository-local API that generated code may import; filenames alone "
                "are not evidence that an API is understood. Return JSON only."
            ),
            {
                "goal": goal,
                "inventory": inventory,
                "initial_context": manifest.initial_context,
                "response_shape": {"inspect_files": ["path"], "reasoning": "string"},
                "constraints": "Never request secret files, .env, credentials, data, or binaries.",
            },
        )
        request = InspectionRequest.model_validate_json(first)
        repository_root = Path(manifest.repository_path)
        extra_context = {
            path: content
            for path in request.inspect_files
            if (content := _read_context(repository_root, path))
        }
        self.inspected_files = sorted({*manifest.initial_context, *extra_context})
        second = _chat_json(
            self.client,
            self.model,
            (
                "You are a repository-level research coding agent. Based on inspected code, decide "
                "which existing Python entrypoint to run, whether a small code change is required, "
                "the experiment matrix, and how verified metrics will be parsed. Return JSON only. "
                "Do not propose shell commands, dependency installation, network access, deletion, "
                "or edits to secrets. A variant identifies a model/method/configuration and MUST "
                "stay identical across random seeds; never put a seed in the variant name. Use a "
                "baseline only when comparing distinct variants. Only claim to have read files "
                "present in the supplied context. List the minimum PyPI requirements needed by the "
                "chosen execution path in required_packages; do not include optional notebook, "
                "plotting, or development dependencies. ReproFlow, not you, will decide whether an "
                "isolated environment is needed. Use Simplified Chinese for explanations."
            ),
            {
                "goal": goal,
                "repository_commit": manifest.git_commit,
                "inventory": inventory,
                "context": {**manifest.initial_context, **extra_context},
                "run_contract": {
                    "argv": "shell-free argv; executable must be python/python3/pytest",
                    "cwd": "repository-relative directory",
                    "metrics": (
                        "Use parser=json with metrics_file and key, or parser=regex with pattern. "
                        "Every run must use concrete arguments and unique run_id."
                    ),
                    "required_packages": (
                        "PEP 508 registry requirements only, such as numpy>=1.26 or progressbar33; "
                        "never URLs, local paths, editable installs, or installer commands."
                    ),
                    "python_version": (
                        "Optional major.minor from 3.8 through 3.13 when repository compatibility "
                        "requires a different interpreter; otherwise null."
                    ),
                },
                "response_schema": GeneratedRepoPlan.model_json_schema(),
            },
        )
        return GeneratedRepoPlan.model_validate_json(second)


class APIRepairPlanner(APIRepoPlanner):
    def create_repair_plan(
        self,
        manifest: RepoManifest,
        previous: RepoExecutionPlan,
        failure_context: dict[str, str],
        feedback: str,
    ) -> GeneratedRepoPlan:
        root = Path(manifest.repository_path)
        relevant = sorted(
            {
                *previous.inspected_files,
                *[change.path for change in previous.code_changes],
                *manifest.initial_context,
            }
        )
        source_context = {
            path: content
            for path in relevant[:40]
            if (content := _read_context(root, path, 12_000))
        }
        self.inspected_files = sorted(source_context)
        response = _chat_json(
            self.client,
            self.model,
            (
                "You are a repair agent for a failed ML experiment workflow. Diagnose the root "
                "cause from bounded logs and inspected source, then return a NEW auditable plan. "
                "Return full replacement contents for every file you change. Preserve the research "
                "goal and successful experimental semantics, but fix code, command, metric, or "
                "minimum registry dependency requirements as needed. Never install dependencies "
                "yourself, use shell syntax, access the network from experiment code, delete "
                "files, or weaken safety checks. A variant names a method/configuration and must "
                "remain identical across seeds. Use Simplified Chinese for explanations and "
                "JSON only."
            ),
            {
                "original_plan": previous.model_dump(mode="json"),
                "failure_logs": failure_context,
                "user_feedback": feedback[:8000],
                "current_repository_commit": manifest.git_commit,
                "current_source_context": source_context,
                "constraints": {
                    "required_packages": (
                        "Minimum PEP 508 PyPI registry requirements only; no URL, local path, "
                        "editable install, or installer command. Explicit version constraints may "
                        "replace incompatible repository pins when the logs justify it."
                    ),
                    "python_version": (
                        "Optional 3.8-3.13 major.minor when logs prove interpreter incompatibility."
                    ),
                    "approval": "All repaired code and dependencies require a new human approval.",
                },
                "response_schema": GeneratedRepoPlan.model_json_schema(),
            },
        )
        return GeneratedRepoPlan.model_validate_json(response)


class MockRepoPlanner:
    def create_plan(self, manifest: RepoManifest, goal: str) -> GeneratedRepoPlan:
        paths = {item.path for item in manifest.files}
        if "examples/sklearn_demo/experiment.py" in paths:
            runs = []
            for variant in ("logistic_regression", "random_forest", "svm"):
                for seed in (42, 43, 44):
                    run_id = f"{variant}-seed-{seed}"
                    metrics_path = f".reproflow_repo_metrics/{run_id}.json"
                    runs.append(
                        RepoRunSpec(
                            run_id=run_id,
                            purpose=f"运行 {variant} 的可复现对比",
                            variant=variant,
                            seed=seed,
                            argv=[
                                "python",
                                "examples/sklearn_demo/experiment.py",
                                "--model",
                                variant,
                                "--seed",
                                str(seed),
                                "--output",
                                metrics_path,
                            ],
                            metrics_file=metrics_path,
                        )
                    )
            return GeneratedRepoPlan(
                title="Agent 自动识别的 sklearn 对比实验",
                rationale=(
                    "Agent 从仓库中识别出现有实验入口及其命令行参数，"
                    "无需修改代码即可运行三模型、三种子对比。"
                ),
                runs=runs,
                metrics=[
                    RepoMetricSpec(name="accuracy", key="accuracy"),
                    RepoMetricSpec(name="f1", key="f1"),
                    RepoMetricSpec(name="roc_auc", key="roc_auc"),
                ],
                baseline="logistic_regression",
            )
        return GeneratedRepoPlan(
            title="仓库基础测试工作流",
            rationale="Mock 模式未推测未知训练参数，仅选择仓库现有测试作为安全验证。",
            runs=[
                RepoRunSpec(
                    run_id="repository-tests",
                    purpose="运行仓库现有 Python 测试",
                    variant="repository",
                    seed=0,
                    argv=["python", "-m", "pytest", "-q"],
                )
            ],
        )


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _has_shell_syntax(value: str) -> bool:
    return any(token in value for token in SHELL_TOKENS)


def _validate_python_change(content: str, relative: str) -> None:
    try:
        tree = ast.parse(content, filename=relative)
    except SyntaxError as error:
        raise ValueError(f"Generated Python is invalid in {relative}: {error}") from error
    banned_imports = {"requests", "shutil", "socket", "subprocess", "urllib"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [item.name for item in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            modules = []
        for module in modules:
            if any(module == item or module.startswith(f"{item}.") for item in banned_imports):
                raise ValueError(f"Blocked import in generated code {relative}: {module}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec", "compile", "__import__"}:
                raise ValueError(f"Blocked dynamic execution in {relative}: {node.func.id}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"system", "popen", "remove", "unlink", "rmdir", "rmtree"}:
                raise ValueError(f"Blocked destructive call in {relative}: {node.func.attr}")


def validate_repo_plan(plan: RepoExecutionPlan) -> RepoExecutionPlan:
    root = Path(plan.repository_path).resolve()
    if plan.environment is not None:
        if not re.fullmatch(r"repo-env-[0-9a-f]{12}", plan.environment.environment_id):
            raise ValueError("Repository environment id is invalid")
        normalized = [
            normalize_requested_requirement(item) for item in plan.environment.requirements
        ]
        if normalized != plan.environment.requirements:
            raise ValueError("Repository environment requirements are not normalized")
        expected_commands = environment_commands(Path("."), plan.environment)
        if plan.environment.install_commands != expected_commands:
            raise ValueError("Repository environment commands do not match the approved spec")
    proposed_paths = {item.path for item in plan.code_changes}
    for change in plan.code_changes:
        target = _safe_relative(root, change.path)
        if target.name.lower() in SENSITIVE_NAMES or target.suffix.lower() not in {
            ".py",
            ".json",
            ".md",
            ".toml",
            ".yaml",
            ".yml",
        }:
            raise ValueError(f"Code change file type is not allowed: {change.path}")
        if len(change.content.encode("utf-8")) > 150_000:
            raise ValueError(f"Code change is too large: {change.path}")
        if target.suffix == ".py":
            _validate_python_change(change.content, change.path)
    run_ids: set[str] = set()
    for run in plan.runs:
        if run.run_id in run_ids:
            raise ValueError(f"Duplicate run_id: {run.run_id}")
        run_ids.add(run.run_id)
        if not run.argv or any(_has_shell_syntax(item) for item in run.argv):
            raise ValueError(f"Run must use a shell-free argv list: {run.run_id}")
        seed_variant = re.search(
            rf"(?i)\bseed\s*[-_=:]?\s*{re.escape(str(run.seed))}\b", run.variant
        )
        if seed_variant:
            raise ValueError(
                f"Variant must describe a method/configuration, not the seed: {run.run_id}"
            )
        executable = Path(run.argv[0]).name
        if executable not in {"python", "python3", "python3.12", "pytest"}:
            raise ValueError(f"Executable is not allowed in {run.run_id}: {run.argv[0]}")
        if "-c" in run.argv:
            raise ValueError(f"Inline Python is not allowed: {run.run_id}")
        for argument in run.argv[1:]:
            argument_path = Path(argument.split("=", 1)[-1])
            if argument_path.is_absolute() or ".." in argument_path.parts:
                raise ValueError(f"Command argument escapes repository in {run.run_id}: {argument}")
        if executable.startswith("python") and len(run.argv) > 1 and run.argv[1] == "-m":
            if len(run.argv) < 3 or run.argv[2] in BLOCKED_PYTHON_MODULES:
                module = run.argv[2] if len(run.argv) >= 3 else "<missing>"
                raise ValueError(f"Python module is not allowed in {run.run_id}: {module}")
        _safe_relative(root, run.cwd)
        if executable.startswith("python") and len(run.argv) > 1 and run.argv[1] != "-m":
            script = run.argv[1]
            target = _safe_relative(root, str(Path(run.cwd) / script))
            relative = target.relative_to(root).as_posix()
            if not target.is_file() and relative not in proposed_paths:
                raise ValueError(f"Selected Python entrypoint does not exist: {relative}")
        if run.metrics_file:
            _safe_relative(root, str(Path(run.cwd) / run.metrics_file))
    if plan.baseline and plan.baseline not in {run.variant for run in plan.runs}:
        raise ValueError("Baseline must match at least one planned run variant")
    return plan


def _config_hash(plan: RepoExecutionPlan) -> str:
    payload = plan.model_dump_json(
        exclude={"status", "approved_by", "approved_at"}, exclude_none=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class RepoPlanStore:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self.state_dir = self.project_root / ".reproflow" / "repo_plans"
        self.readable_dir = self.project_root / "repo_plans"
        self.backup_dir = self.project_root / ".reproflow" / "repo_backups"
        for path in (self.state_dir, self.readable_dir, self.backup_dir):
            path.mkdir(parents=True, exist_ok=True)

    def save(self, plan: RepoExecutionPlan) -> None:
        self.path(plan.repo_plan_id).write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        self.markdown_path(plan.repo_plan_id).write_text(
            render_repo_plan_markdown(plan), encoding="utf-8"
        )

    def get(self, repo_plan_id: str) -> RepoExecutionPlan:
        path = self.path(repo_plan_id)
        if not path.is_file():
            raise KeyError(f"Unknown repository plan: {repo_plan_id}")
        return RepoExecutionPlan.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[RepoExecutionPlan]:
        return [
            RepoExecutionPlan.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self.state_dir.glob("repo-plan-*.json"))
        ]

    def path(self, repo_plan_id: str) -> Path:
        return self.state_dir / f"{repo_plan_id}.json"

    def markdown_path(self, repo_plan_id: str) -> Path:
        return self.readable_dir / f"{repo_plan_id}.md"


def _diff_for_change(root: Path, change: RepoCodeChange) -> str:
    target = _safe_relative(root, change.path)
    before = (
        target.read_text(encoding="utf-8", errors="replace").splitlines()
        if target.is_file()
        else []
    )
    return "\n".join(
        difflib.unified_diff(
            before,
            change.content.splitlines(),
            fromfile=f"a/{change.path}",
            tofile=f"b/{change.path}",
            lineterm="",
        )
    )


def _source_hashes(root: Path, paths: set[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in sorted(paths):
        target = _safe_relative(root, relative)
        digest = _sha256(target)
        if digest:
            hashes[relative] = digest
    return hashes


def render_repo_plan_markdown(plan: RepoExecutionPlan) -> str:
    lines = [
        f"# 仓库级 Agent 执行计划：{plan.title}",
        "",
        f"> 状态：`{plan.status.value}`  ",
        f"> 计划编号：`{plan.repo_plan_id}`  ",
        f"> 目标仓库：`{plan.repository_path}`",
        "",
        "## 用户目标",
        "",
        plan.goal,
        "",
        "## Agent 的判断",
        "",
        plan.rationale,
        "",
        "## 依赖与运行环境",
        "",
    ]
    if plan.environment is None:
        lines.append("尚未生成依赖预检结果。")
    else:
        lines.extend(
            [
                f"- 模式：`{plan.environment.mode}`",
                f"- Python：`{plan.environment.python_version}`",
                f"- 环境编号：`{plan.environment.environment_id}`",
                f"- 判断：{plan.environment.rationale}",
            ]
        )
        if plan.dependency_report and plan.dependency_report.checks:
            lines.extend(
                [
                    "",
                    "| 依赖 | 当前版本 | 状态 | 说明 |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for item in plan.dependency_report.checks:
                lines.append(
                    f"| `{item.requirement}` | {item.installed_version or '-'} | "
                    f"{item.status} | {item.detail} |"
                )
        if plan.environment.install_commands:
            lines.extend(["", "批准后执行以下环境命令：", ""])
            for command in plan.environment.install_commands:
                lines.append(f"- `{' '.join(command)}`")
    if plan.parent_plan_id:
        lines.extend(
            [
                "",
                "## 修复来源",
                "",
                f"- 原失败计划：`{plan.parent_plan_id}`",
                f"- 修复轮次：{plan.repair_attempt}/3",
            ]
        )
    lines.extend(
        [
            "",
            "## Agent 阅读了什么",
            "",
            *[f"- `{path}`" for path in plan.inspected_files],
            "",
            "## 拟执行实验",
            "",
            "| Run | 用途 | Variant | Seed | 命令 | 超时 |",
            "| --- | --- | --- | ---: | --- | ---: |",
        ]
    )
    for run in plan.runs:
        command = " ".join(run.argv)
        lines.append(
            f"| `{run.run_id}` | {run.purpose} | {run.variant} | {run.seed} | "
            f"`{command}` | {run.timeout_seconds}s |"
        )
    lines.extend(["", "## 拟修改代码", ""])
    if not plan.code_changes:
        lines.append("Agent 判断无需修改代码，将直接复用仓库现有入口。")
    for change in plan.code_changes:
        lines.extend(
            [
                f"### `{change.path}`",
                "",
                change.reason,
                "",
                "```diff",
                change.diff or _diff_for_change(Path(plan.repository_path), change),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## 审批边界",
            "",
            "审批前不会写入代码，也不会执行任何命令。",
            "执行时不使用 Shell；隔离环境依赖仅按上方已批准命令安装，不允许 URL、"
            "本地路径、editable 依赖、内联 Python 和目录越界。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _inferred_requirements(root: Path, changes: list[GeneratedCodeChange]) -> list[str]:
    requirements: set[str] = set()
    for change in changes:
        if not change.path.endswith(".py"):
            continue
        tree = ast.parse(change.content, filename=change.path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [item.name.split(".", 1)[0] for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module.split(".", 1)[0]]
            else:
                modules = []
            for module in modules:
                if module in sys.stdlib_module_names:
                    continue
                if (root / f"{module}.py").is_file() or (root / module / "__init__.py").is_file():
                    continue
                requirements.add(normalize_requested_requirement(module))
    return sorted(requirements)


def _build_repo_execution_plan(
    project: Path,
    manifest: RepoManifest,
    goal: str,
    generated: GeneratedRepoPlan,
    inspected_files: list[str],
    *,
    parent_plan_id: str | None = None,
    repair_attempt: int = 0,
) -> RepoExecutionPlan:
    root = Path(manifest.repository_path)
    changes: list[RepoCodeChange] = []
    for item in generated.code_changes:
        change = RepoCodeChange(
            path=item.path,
            content=item.content,
            reason=item.reason,
            before_sha256=_sha256(_safe_relative(root, item.path)),
        )
        change.diff = _diff_for_change(root, change)
        changes.append(change)
    requested_requirements = sorted(
        {
            *[normalize_requested_requirement(item) for item in generated.required_packages],
            *_inferred_requirements(root, generated.code_changes),
        }
    )
    dependency_report, environment = analyze_dependencies(
        root, requested_requirements, python_version=generated.python_version
    )
    environment.install_commands = environment_commands(project, environment)
    run_sources: set[str] = set(inspected_files)
    for run in generated.runs:
        if len(run.argv) > 1 and run.argv[1] != "-m":
            target = _safe_relative(root, str(Path(run.cwd) / run.argv[1]))
            if target.is_file():
                run_sources.add(target.relative_to(root).as_posix())
    plan = RepoExecutionPlan(
        repository_path=manifest.repository_path,
        repository_commit=manifest.git_commit,
        goal=goal,
        title=generated.title,
        rationale=generated.rationale,
        inspected_files=inspected_files,
        source_hashes=_source_hashes(root, run_sources),
        code_changes=changes,
        runs=generated.runs,
        metrics=generated.metrics,
        baseline=generated.baseline,
        dependency_report=dependency_report,
        environment=environment,
        parent_plan_id=parent_plan_id,
        repair_attempt=repair_attempt,
    )
    validate_repo_plan(plan)
    RepoPlanStore(project).save(plan)
    return plan


def create_repo_plan(
    project_root: str | Path,
    repository_path: str | Path,
    goal: str,
    *,
    agent: Literal["mock", "api"] = "mock",
) -> RepoExecutionPlan:
    project = Path(project_root).resolve()
    manifest = inspect_repository(repository_path, goal)
    planner = APIRepoPlanner() if agent == "api" else MockRepoPlanner()
    generated = planner.create_plan(manifest, goal)
    inspected_files = sorted(
        set(getattr(planner, "inspected_files", [])) or set(manifest.initial_context)
    )
    return _build_repo_execution_plan(project, manifest, goal, generated, inspected_files)


def _collect_failure_context(
    project: Path, previous: RepoExecutionPlan
) -> dict[str, str]:
    context: dict[str, str] = {}
    store = Store(project)
    try:
        workflow = store.get_workflow(previous.repo_plan_id)
    except KeyError:
        workflow = {}
    if workflow.get("errors"):
        context["workflow/errors.json"] = json.dumps(
            workflow["errors"], ensure_ascii=False, indent=2
        )[-12_000:]
    allowed_runs = (project / "runs" / previous.repo_plan_id).resolve()
    for record in store.list_runs(previous.repo_plan_id):
        if record.status == RunStatus.SUCCEEDED:
            continue
        run_dir = Path(record.run_dir).resolve()
        try:
            run_dir.relative_to(allowed_runs)
        except ValueError:
            continue
        for name in ("stderr.log", "stdout.log"):
            path = run_dir / name
            if path.is_file():
                context[f"{record.run_id}/{name}"] = path.read_text(
                    encoding="utf-8", errors="replace"
                )[-12_000:]
        if len(context) >= 12:
            break
    if previous.environment is not None:
        env_root = environment_root(project, previous.environment)
        for name in ("install.stderr.log", "install.stdout.log"):
            path = env_root / name
            if path.is_file():
                context[f"environment/{name}"] = path.read_text(
                    encoding="utf-8", errors="replace"
                )[-16_000:]
    return context


def create_repo_repair_plan(
    project_root: str | Path,
    failed_repo_plan_id: str,
    *,
    feedback: str = "",
    agent: Literal["api"] = "api",
) -> RepoExecutionPlan:
    if agent != "api":
        raise ValueError("Repair Agent currently requires the API mode")
    project = Path(project_root).resolve()
    store = RepoPlanStore(project)
    previous = store.get(failed_repo_plan_id)
    if previous.status != RepoPlanStatus.PARTIAL_FAILURE:
        raise ValueError(
            f"Only a partial-failure plan can be repaired; status={previous.status.value}"
        )
    if previous.repair_attempt >= 3:
        raise ValueError("Repair attempt limit reached (3)")
    failures = _collect_failure_context(project, previous)
    if not failures:
        raise ValueError("No bounded failure logs are available for the Repair Agent")
    goal = previous.goal + (
        f"\n\n修复目标：诊断计划 {previous.repo_plan_id} 的失败并生成新的可审批方案。"
    )
    manifest = inspect_repository(previous.repository_path, goal)
    planner = APIRepairPlanner()
    generated = planner.create_repair_plan(manifest, previous, failures, feedback)
    return _build_repo_execution_plan(
        project,
        manifest,
        goal,
        generated,
        planner.inspected_files,
        parent_plan_id=previous.repo_plan_id,
        repair_attempt=previous.repair_attempt + 1,
    )


def approve_repo_plan(project_root: str | Path, repo_plan_id: str, actor: str) -> RepoExecutionPlan:
    store = RepoPlanStore(project_root)
    plan = store.get(repo_plan_id)
    if plan.status != RepoPlanStatus.DRAFT:
        raise ValueError(f"Only draft repository plans can be approved; status={plan.status.value}")
    validate_repo_plan(plan)
    _verify_sources(plan)
    plan.status = RepoPlanStatus.APPROVED
    plan.approved_by = actor
    plan.approved_at = datetime.now(UTC)
    store.save(plan)
    return plan


def _verify_sources(plan: RepoExecutionPlan) -> None:
    root = Path(plan.repository_path)
    desired = {
        change.path: hashlib.sha256(change.content.encode()).hexdigest()
        for change in plan.code_changes
    }
    for relative, expected in plan.source_hashes.items():
        current = _sha256(_safe_relative(root, relative))
        if current not in {expected, desired.get(relative)}:
            raise ValueError(f"Inspected source changed after planning: {relative}")


def _apply_changes(plan: RepoExecutionPlan, store: RepoPlanStore) -> None:
    root = Path(plan.repository_path)
    if _git_commit(root) != plan.repository_commit:
        raise ValueError("Repository commit changed after planning; create a fresh plan")
    # Verify every destination before writing the first file so a drift error cannot
    # leave a multi-file change only partially applied.
    for change in plan.code_changes:
        target = _safe_relative(root, change.path)
        current_hash = _sha256(target)
        desired_hash = hashlib.sha256(change.content.encode()).hexdigest()
        if current_hash not in {change.before_sha256, desired_hash}:
            raise ValueError(f"File changed after planning: {change.path}")
    for change in plan.code_changes:
        target = _safe_relative(root, change.path)
        current_hash = _sha256(target)
        desired_hash = hashlib.sha256(change.content.encode()).hexdigest()
        if current_hash == desired_hash:
            continue
        if current_hash != change.before_sha256:
            raise ValueError(f"File changed after planning: {change.path}")
        if target.is_file():
            backup = store.backup_dir / plan.repo_plan_id / change.path
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(target.read_bytes())
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f".{target.name}.{plan.repo_plan_id}.tmp")
        temp.write_text(change.content, encoding="utf-8")
        temp.replace(target)


def _safe_environment(seed: int, repository_root: Path) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if key in ENV_ALLOWLIST}
    environment["PYTHONHASHSEED"] = str(seed)
    environment["PYTHONPATH"] = str(repository_root.resolve())
    environment["MPLBACKEND"] = "Agg"
    return environment


def _actual_argv(argv: list[str], python_executable: Path) -> list[str]:
    if Path(argv[0]).name in {"python", "python3", "python3.12"}:
        return [str(python_executable), *argv[1:]]
    if Path(argv[0]).name == "pytest":
        return [str(python_executable), "-m", "pytest", *argv[1:]]
    return argv


def _parse_repo_metrics(
    plan: RepoExecutionPlan,
    run: RepoRunSpec,
    stdout: str,
    stderr: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    json_payload: dict[str, Any] | None = None
    if run.metrics_file:
        metrics_path = _safe_relative(
            Path(plan.repository_path), str(Path(run.cwd) / run.metrics_file)
        )
        if metrics_path.is_file():
            json_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    for spec in plan.metrics:
        if spec.parser == "json":
            if json_payload is None:
                raise ValueError(f"Missing metrics JSON for {run.run_id}: {run.metrics_file}")
            value = json_payload.get(spec.key or spec.name)
        else:
            pattern = spec.pattern or rf"{re.escape(spec.name)}\s*[=:]\s*(-?\d+(?:\.\d+)?)"
            match = re.search(pattern, f"{stdout}\n{stderr}", re.MULTILINE)
            if not match:
                raise ValueError(f"Metric regex did not match {spec.name} in {run.run_id}")
            value = match.groupdict().get("value") if match.groupdict() else None
            value = value if value is not None else match.group(1)
        number = float(value)
        if not (float("-inf") < number < float("inf")):
            raise ValueError(f"Metric is not finite: {spec.name}")
        metrics[spec.name] = number
    return metrics


def _load_existing_record(manifest_path: Path) -> RunRecord | None:
    if not manifest_path.is_file():
        return None
    return RunRecord.model_validate_json(manifest_path.read_text(encoding="utf-8"))


def _execute_repo_run(
    project_root: Path,
    plan: RepoExecutionPlan,
    run: RepoRunSpec,
    python_executable: Path,
) -> RunRecord:
    run_dir = project_root / "runs" / plan.repo_plan_id / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    existing = _load_existing_record(manifest_path)
    if existing and existing.status == RunStatus.SUCCEEDED:
        return existing
    repository_root = Path(plan.repository_path)
    cwd = _safe_relative(repository_root, run.cwd)
    command = _actual_argv(run.argv, python_executable)
    source_metrics_path = (
        _safe_relative(repository_root, str(Path(run.cwd) / run.metrics_file))
        if run.metrics_file
        else None
    )
    previous_metrics_mtime = (
        source_metrics_path.stat().st_mtime_ns
        if source_metrics_path is not None and source_metrics_path.is_file()
        else None
    )
    record = RunRecord(
        run_id=run.run_id,
        workflow_id=plan.repo_plan_id,
        plan_id=plan.repo_plan_id,
        variant=run.variant,
        seed=run.seed,
        command=command,
        status=RunStatus.RUNNING,
        started_at=datetime.now(UTC),
        run_dir=str(run_dir),
    )
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=_safe_environment(run.seed, repository_root),
            capture_output=True,
            text=True,
            timeout=run.timeout_seconds,
            check=False,
        )
        stdout = result.stdout[:LOG_LIMIT]
        stderr = result.stderr[:LOG_LIMIT]
        record.exit_code = result.returncode
        if result.returncode == 0:
            try:
                if (
                    source_metrics_path is not None
                    and previous_metrics_mtime is not None
                    and source_metrics_path.is_file()
                    and source_metrics_path.stat().st_mtime_ns == previous_metrics_mtime
                ):
                    raise ValueError(f"Metrics file was not refreshed: {run.metrics_file}")
                record.metrics = _parse_repo_metrics(plan, run, stdout, stderr)
                record.status = RunStatus.SUCCEEDED
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                record.status = RunStatus.FAILED
                record.error = f"Metric parsing failed: {error}"
        else:
            record.status = RunStatus.FAILED
            record.error = f"Process exited with code {result.returncode}"
    except subprocess.TimeoutExpired as error:
        stdout = _subprocess_text(error.stdout)[:LOG_LIMIT]
        stderr = _subprocess_text(error.stderr)[:LOG_LIMIT]
        record.status = RunStatus.TIMED_OUT
        record.error = f"Timed out after {run.timeout_seconds} seconds"
    record.finished_at = datetime.now(UTC)
    record.duration_seconds = time.perf_counter() - started
    (run_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(record.metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return record


def _repo_report(
    project_root: Path,
    plan: RepoExecutionPlan,
    records: list[RunRecord],
    aggregate_path: Path,
) -> Path:
    succeeded = [record for record in records if record.status == RunStatus.SUCCEEDED]
    lines = [
        f"# 仓库级 Agent 实验报告：{plan.title}",
        "",
        "## 任务",
        "",
        plan.goal,
        "",
        "## Agent 的执行决策",
        "",
        plan.rationale,
        "",
        "## 执行结果",
        "",
        f"- 成功：{len(succeeded)}/{len(records)}",
        f"- 目标仓库 commit：`{plan.repository_commit}`",
        f"- 计划配置哈希：`{_config_hash(plan)}`",
        f"- Python 环境：`{plan.environment.environment_id if plan.environment else 'current'}`",
        f"- 汇总表：`{aggregate_path.relative_to(project_root)}`",
        "",
        "| Run | Variant | Seed | 状态 | 指标 | 错误 |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for record in records:
        metrics = ", ".join(f"{key}={value:.6g}" for key, value in record.metrics.items())
        lines.append(
            f"| `{record.run_id}` | {record.variant} | {record.seed} | "
            f"{record.status.value} | {metrics or '-'} | {record.error or '-'} |"
        )
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "报告数值仅来自已执行命令生成的 metrics 文件或已声明的日志解析规则。",
            "Agent 选择的代码与命令均保存在书面计划中，可进行审计。",
        ]
    )
    report_path = project_root / "runs" / plan.repo_plan_id / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _propose_repo_evidence(
    project_root: Path,
    plan: RepoExecutionPlan,
    records: list[RunRecord],
    aggregate_rows_payload: list[dict[str, Any]],
    report_path: Path,
) -> list[EvidenceClaim]:
    if not plan.metrics or not plan.baseline:
        return []
    metric = plan.metrics[0]
    by_variant = {row["variant"]: row for row in aggregate_rows_payload}
    baseline_row = by_variant.get(plan.baseline)
    if not baseline_row or baseline_row.get(f"{metric.name}_mean") is None:
        return []
    baseline_value = float(baseline_row[f"{metric.name}_mean"])
    claims: list[EvidenceClaim] = []
    store = Store(project_root)
    for variant, row in by_variant.items():
        if variant == plan.baseline or row.get(f"{metric.name}_mean") is None:
            continue
        observed = float(row[f"{metric.name}_mean"])
        delta = (
            observed - baseline_value
            if metric.direction == "maximize"
            else baseline_value - observed
        )
        proposed_status = (
            "supported" if delta > 0 else "contradicted" if delta < 0 else "inconclusive"
        )
        digest = (
            hashlib.sha256(f"{plan.repo_plan_id}:{variant}:{metric.name}".encode())
            .hexdigest()[:8]
            .upper()
        )
        relevant = [record for record in records if record.variant in {plan.baseline, variant}]
        claim = EvidenceClaim(
            claim_id=f"C-{digest}",
            claim=(
                f"{variant} improves mean {metric.name} over the approved "
                f"{plan.baseline} baseline for this repository experiment."
            ),
            proposed_status=proposed_status,
            workflow_id=plan.repo_plan_id,
            experiment_ids=[record.run_id for record in relevant],
            metric=metric.name,
            baseline_variant=plan.baseline,
            baseline_value=baseline_value,
            observed_variant=variant,
            observed_value=observed,
            delta=delta,
            seeds=sorted({record.seed for record in relevant}),
            code_commit=plan.repository_commit,
            config_hash=_config_hash(plan),
            artifacts=[
                *[
                    str(Path(record.run_dir).relative_to(project_root) / "metrics.json")
                    for record in relevant
                ],
                str(report_path.relative_to(project_root)),
            ],
        )
        try:
            existing = store.get_claim(claim.claim_id)
        except KeyError:
            existing = None
        if existing is None or existing.status == ClaimStatus.PROPOSED:
            store.save_claim(claim)
        else:
            claim = existing
        claims.append(claim)
    return claims


def run_repo_plan(
    project_root: str | Path, repo_plan_id: str, *, resume: bool = False
) -> dict[str, Any]:
    project = Path(project_root).resolve()
    plan_store = RepoPlanStore(project)
    plan = plan_store.get(repo_plan_id)
    allowed_statuses = (
        {RepoPlanStatus.APPROVED, RepoPlanStatus.PARTIAL_FAILURE}
        if resume
        else {RepoPlanStatus.APPROVED}
    )
    if plan.status not in allowed_statuses:
        raise ValueError(f"Repository plan is not runnable; status={plan.status.value}")
    validate_repo_plan(plan)
    _verify_sources(plan)
    plan.status = RepoPlanStatus.RUNNING
    plan_store.save(plan)
    store = Store(project)
    store.add_trace(plan.repo_plan_id, "repo_execute", "started", {"resume": resume})
    try:
        python_executable = (
            prepare_environment(project, plan.environment)
            if plan.environment is not None
            else Path(sys.executable)
        )
    except EnvironmentSetupError as error:
        plan.status = RepoPlanStatus.PARTIAL_FAILURE
        plan_store.save(plan)
        workflow_dir = project / "runs" / plan.repo_plan_id
        workflow_dir.mkdir(parents=True, exist_ok=True)
        report_path = workflow_dir / "report.md"
        report_path.write_text(
            "\n".join(
                [
                    f"# 仓库级 Agent 环境预检失败：{plan.title}",
                    "",
                    "实验代码尚未写入，实验命令尚未执行。",
                    "",
                    f"- 错误：{error}",
                    f"- 环境：`{plan.environment.environment_id if plan.environment else '-'}`",
                    "",
                    "请使用 Repair Agent 根据安装日志生成新的依赖或代码修复计划。",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        state = {
            "workflow_id": plan.repo_plan_id,
            "plan_id": plan.repo_plan_id,
            "stage": plan.status.value,
            "goal": plan.goal,
            "run_records": [],
            "report_path": str(report_path.relative_to(project)),
            "errors": [str(error)],
            "proposed_claims": [],
        }
        store.save_workflow(plan.repo_plan_id, plan.repo_plan_id, plan.status.value, state)
        store.add_trace(
            plan.repo_plan_id,
            "dependency_preflight",
            "partial_failure",
            {"error": str(error)},
        )
        store.add_memory(
            MemoryItem(
                kind="failure",
                workflow_id=plan.repo_plan_id,
                text=f"Repository environment setup failed: {error}",
                tags=["repository-agent", "dependency-preflight"],
            )
        )
        return state
    _apply_changes(plan, plan_store)
    store.add_trace(
        plan.repo_plan_id,
        "dependency_preflight",
        "ready",
        {
            "mode": plan.environment.mode if plan.environment else "current",
            "python": str(python_executable),
        },
    )
    records: list[RunRecord] = []
    for run in plan.runs:
        record = _execute_repo_run(project, plan, run, python_executable)
        store.save_run(record)
        records.append(record)
        store.add_trace(
            plan.repo_plan_id,
            "repo_execute",
            "run_finished",
            {"run_id": run.run_id, "status": record.status.value},
        )
    metric_specs = [MetricSpec(name=item.name, direction=item.direction) for item in plan.metrics]
    workflow_dir = project / "runs" / plan.repo_plan_id
    summary_path = write_summary_csv(workflow_dir, records, metric_specs)
    rows = aggregate_rows(records, metric_specs, plan.baseline or records[0].variant)
    aggregate_path = write_aggregate_csv(workflow_dir, rows, metric_specs)
    failures_path = write_failures_csv(workflow_dir, records)
    report_path = _repo_report(project, plan, records, aggregate_path)
    claims = _propose_repo_evidence(project, plan, records, rows, report_path)
    all_succeeded = all(record.status == RunStatus.SUCCEEDED for record in records)
    plan.status = RepoPlanStatus.COMPLETED if all_succeeded else RepoPlanStatus.PARTIAL_FAILURE
    plan_store.save(plan)
    state = {
        "workflow_id": plan.repo_plan_id,
        "plan_id": plan.repo_plan_id,
        "stage": plan.status.value,
        "goal": plan.goal,
        "run_records": [record.model_dump(mode="json") for record in records],
        "summary_path": str(summary_path.relative_to(project)),
        "aggregate_path": str(aggregate_path.relative_to(project)),
        "failures_path": str(failures_path.relative_to(project)),
        "report_path": str(report_path.relative_to(project)),
        "environment_path": str(python_executable),
        "proposed_claims": [claim.model_dump(mode="json") for claim in claims],
    }
    store.save_workflow(plan.repo_plan_id, plan.repo_plan_id, plan.status.value, state)
    store.add_memory(
        MemoryItem(
            kind="experiment",
            workflow_id=plan.repo_plan_id,
            text=(
                f"Repository Agent executed {len(records)} runs for {plan.goal}; "
                f"{sum(record.status == RunStatus.SUCCEEDED for record in records)} succeeded."
            ),
            tags=["repository-agent", Path(plan.repository_path).name],
        )
    )
    store.add_trace(
        plan.repo_plan_id,
        "repo_execute",
        "completed" if all_succeeded else "partial_failure",
        {"evidence": [claim.claim_id for claim in claims]},
    )
    return state
