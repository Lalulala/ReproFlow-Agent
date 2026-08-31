from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

from .models import DependencyItem, DependencyReport, RepoEnvironmentSpec

INSTALL_LOG_LIMIT = 1_000_000
REQUIREMENT_FILES = ("requirements.txt", "requirements/base.txt", "requirements/runtime.txt")
IMPORT_TO_DISTRIBUTION = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "progressbar": "progressbar33",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
}


class EnvironmentSetupError(RuntimeError):
    pass


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _safe_requirement(value: str) -> Requirement | None:
    if not value or value.startswith(("-", ".", "/")):
        return None
    try:
        requirement = Requirement(value)
    except InvalidRequirement:
        return None
    if requirement.url is not None:
        return None
    return requirement


def _declared_requirements(root: Path) -> tuple[dict[str, str], list[str], list[str]]:
    declared: dict[str, str] = {}
    files: list[str] = []
    blocked: list[str] = []
    for relative in REQUIREMENT_FILES:
        path = root / relative
        if not path.is_file():
            continue
        files.append(relative)
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            value = raw.split("#", 1)[0].strip()
            if not value:
                continue
            requirement = _safe_requirement(value)
            if requirement is None:
                blocked.append(f"{relative}: {value}")
                continue
            declared[_canonical(requirement.name)] = str(requirement)
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            payload = {}
        dependencies = payload.get("project", {}).get("dependencies", [])
        if isinstance(dependencies, list):
            files.append("pyproject.toml")
            for value in dependencies:
                requirement = _safe_requirement(str(value))
                if requirement is None:
                    blocked.append(f"pyproject.toml: {value}")
                    continue
                declared[_canonical(requirement.name)] = str(requirement)
    return declared, sorted(set(files)), blocked


def normalize_requested_requirement(value: str) -> str:
    requirement = _safe_requirement(value)
    if requirement is None:
        raise ValueError(f"Unsafe or unsupported dependency requirement: {value}")
    mapped = IMPORT_TO_DISTRIBUTION.get(requirement.name, requirement.name)
    if mapped != requirement.name:
        suffix = str(requirement)[len(requirement.name) :]
        requirement = Requirement(f"{mapped}{suffix}")
    return str(requirement)


def analyze_dependencies(
    repository_root: str | Path,
    requested_requirements: list[str],
    *,
    python_version: str | None = None,
) -> tuple[DependencyReport, RepoEnvironmentSpec]:
    root = Path(repository_root).resolve()
    declared, manifest_files, blocked_entries = _declared_requirements(root)
    requested = [normalize_requested_requirement(item) for item in requested_requirements]
    resolved: list[str] = []
    for value in requested:
        requirement = Requirement(value)
        key = _canonical(requirement.name)
        # A bare model-proposed package adopts the repository's declared constraint.
        resolved.append(declared[key] if not requirement.specifier and key in declared else value)
    resolved = sorted(set(resolved), key=lambda item: _canonical(Requirement(item).name))

    checks: list[DependencyItem] = []
    for value in resolved:
        requirement = Requirement(value)
        try:
            installed = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed is None:
            status = "missing"
            detail = "not installed in the current ReproFlow environment"
        elif requirement.specifier and installed not in requirement.specifier:
            status = "conflict"
            detail = f"installed {installed} does not satisfy {requirement.specifier}"
        else:
            status = "satisfied"
            detail = f"installed {installed} satisfies the requested requirement"
        checks.append(
            DependencyItem(
                requirement=value,
                distribution=requirement.name,
                installed_version=installed,
                status=status,
                detail=detail,
            )
        )
    needs_isolation = any(item.status in {"missing", "conflict"} for item in checks)
    report = DependencyReport(
        manifest_files=manifest_files,
        requested_requirements=requested,
        resolved_requirements=resolved,
        checks=checks,
        blocked_entries=blocked_entries,
        needs_isolation=needs_isolation,
    )
    current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    python_version = python_version or current_python
    if python_version not in {"3.8", "3.9", "3.10", "3.11", "3.12", "3.13"}:
        raise ValueError(f"Unsupported isolated environment Python version: {python_version}")
    if python_version != current_python:
        needs_isolation = True
        report.needs_isolation = True
    digest_payload = json.dumps(
        {
            "repository": str(root),
            "python": python_version,
            "requirements": resolved,
        },
        sort_keys=True,
    )
    environment_id = f"repo-env-{hashlib.sha256(digest_payload.encode()).hexdigest()[:12]}"
    environment = RepoEnvironmentSpec(
        mode="isolated" if needs_isolation else "current",
        python_version=python_version,
        environment_id=environment_id,
        requirements=resolved,
        rationale=(
            "当前环境存在缺失或冲突依赖，批准后创建项目隔离的 uv 虚拟环境。"
            if needs_isolation
            else "计划所需依赖与当前 ReproFlow 环境兼容，无需创建额外环境。"
        ),
    )
    environment.install_commands = environment_commands(root, environment)
    return report, environment


def environment_root(project_root: str | Path, environment: RepoEnvironmentSpec) -> Path:
    return Path(project_root).resolve() / ".reproflow" / "environments" / environment.environment_id


def environment_python(path: Path) -> Path:
    windows = path / "Scripts" / "python.exe"
    return windows if windows.is_file() else path / "bin" / "python"


def environment_commands(
    project_root: str | Path, environment: RepoEnvironmentSpec
) -> list[list[str]]:
    if environment.mode == "current":
        return []
    del project_root
    relative_root = Path(".reproflow") / "environments" / environment.environment_id
    cache_root = Path(".reproflow") / "uv-cache"
    python_path = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    interpreter = relative_root / python_path
    commands = [
        [
            "uv",
            "--cache-dir",
            str(cache_root),
            "venv",
            "--allow-existing",
            "--python",
            environment.python_version,
            str(relative_root),
        ]
    ]
    if environment.requirements:
        commands.append(
            [
                "uv",
                "--cache-dir",
                str(cache_root),
                "pip",
                "install",
                "--python",
                str(interpreter),
                *environment.requirements,
            ]
        )
    return commands


def prepare_environment(project_root: str | Path, environment: RepoEnvironmentSpec) -> Path:
    if environment.mode == "current":
        return Path(sys.executable)
    root = environment_root(project_root, environment)
    root.mkdir(parents=True, exist_ok=True)
    interpreter = environment_python(root)
    marker = root / "reproflow-environment.json"
    expected = {
        "python_version": environment.python_version,
        "requirements": environment.requirements,
    }
    if interpreter.is_file() and marker.is_file():
        try:
            if json.loads(marker.read_text(encoding="utf-8")) == expected:
                return interpreter
        except json.JSONDecodeError:
            pass
    uv = shutil.which("uv")
    if uv is None:
        raise EnvironmentSetupError("uv is required to create an isolated repository environment")
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    for command in environment_commands(project_root, environment):
        actual = [uv, *command[1:]]
        result = subprocess.run(
            actual,
            cwd=Path(project_root).resolve(),
            capture_output=True,
            text=True,
            check=False,
            timeout=1200,
        )
        stdout_parts.append(result.stdout)
        stderr_parts.append(result.stderr)
        if result.returncode != 0:
            (root / "install.stdout.log").write_text(
                "".join(stdout_parts)[-INSTALL_LOG_LIMIT:], encoding="utf-8"
            )
            (root / "install.stderr.log").write_text(
                "".join(stderr_parts)[-INSTALL_LOG_LIMIT:], encoding="utf-8"
            )
            raise EnvironmentSetupError(
                f"Isolated environment setup failed with code {result.returncode}; "
                f"see {root / 'install.stderr.log'}"
            )
    if not interpreter.is_file():
        raise EnvironmentSetupError(f"Environment interpreter was not created: {interpreter}")
    marker.write_text(json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "install.stdout.log").write_text(
        "".join(stdout_parts)[-INSTALL_LOG_LIMIT:], encoding="utf-8"
    )
    (root / "install.stderr.log").write_text(
        "".join(stderr_parts)[-INSTALL_LOG_LIMIT:], encoding="utf-8"
    )
    return interpreter
