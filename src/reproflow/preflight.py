from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .models import ExperimentPlan, PreflightCheck, PreflightReport

SHELL_TOKENS = (";", "&&", "||", "|", "`", "$(", ">", "<", "\n", "\r")
RESERVED_VARIANT_FLAGS = {"--seed", "--output", "--simulate-failure", "--simulate-timeout"}
SAFE_FLAG = re.compile(r"^--[a-z][a-z0-9-]*$")


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _has_shell_syntax(value: str) -> bool:
    return any(token in value for token in SHELL_TOKENS)


def _git_check(root: Path) -> PreflightCheck:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return PreflightCheck(name="git_repository", passed=False, detail="Not a Git repository")
    dirty = bool(result.stdout.strip())
    return PreflightCheck(
        name="git_status",
        passed=not dirty,
        blocking=False,
        detail="Working tree is clean" if not dirty else "Working tree has uncommitted changes",
    )


def run_preflight(plan: ExperimentPlan, project_root: str | Path) -> PreflightReport:
    root = Path(project_root).resolve()
    checks: list[PreflightCheck] = []

    executable = Path(plan.command[0]).resolve()
    executable_name_allowed = executable.name in {"python", "python3", "python3.12"}
    executable_exists = (
        executable == Path(sys.executable).resolve() or shutil.which(plan.command[0]) is not None
    )
    executable_allowed = executable_name_allowed and executable_exists
    checks.append(
        PreflightCheck(
            name="allowed_executable",
            passed=executable_allowed,
            detail=f"Executable: {plan.command[0]}",
        )
    )
    command_safe = all(not _has_shell_syntax(argument) for argument in plan.command)
    checks.append(
        PreflightCheck(
            name="shell_syntax",
            passed=command_safe,
            detail="Command is a shell-free argv list" if command_safe else "Shell syntax detected",
        )
    )
    checks.append(
        PreflightCheck(
            name="command_shape",
            passed=len(plan.command) == 2,
            detail="Base command must contain only Python and the experiment script",
        )
    )

    script = (root / plan.script_path).resolve()
    script_matches = len(plan.command) > 1 and Path(plan.command[1]).as_posix() == plan.script_path
    checks.append(
        PreflightCheck(
            name="script_path",
            passed=_inside(root, script) and script.is_file() and script_matches,
            detail=f"Script: {script}",
        )
    )

    data_ok = True
    data_detail = "No external data path required"
    if plan.data_path:
        data = (root / plan.data_path).resolve()
        data_ok = _inside(root, data) and data.exists()
        data_detail = f"Data: {data}"
    checks.append(PreflightCheck(name="data_path", passed=data_ok, detail=data_detail))

    artifact_root = (root / plan.artifact_root).resolve()
    output_dir = artifact_root / plan.plan_id
    checks.append(
        PreflightCheck(
            name="artifact_path",
            passed=_inside(root, artifact_root) and not output_dir.exists(),
            detail=f"Reserved output: {output_dir}",
        )
    )

    variant_args_safe = True
    variant_detail = "Variant arguments are safe"
    for variant in plan.variants:
        if len(variant.args) % 2 != 0:
            variant_args_safe = False
            variant_detail = f"Arguments must be flag/value pairs: {variant.name}"
            break
        for index in range(0, len(variant.args), 2):
            flag, value = variant.args[index : index + 2]
            if _has_shell_syntax(flag) or _has_shell_syntax(value):
                variant_args_safe = False
                variant_detail = f"Shell syntax in variant: {variant.name}"
                break
            if flag in RESERVED_VARIANT_FLAGS:
                variant_args_safe = False
                variant_detail = f"Reserved argument in {variant.name}: {flag}"
                break
            if not SAFE_FLAG.fullmatch(flag) or value.startswith("--"):
                variant_args_safe = False
                variant_detail = f"Invalid flag/value pair in {variant.name}: {flag} {value}"
                break
        if variant.args[:2] != ["--model", variant.name]:
            variant_args_safe = False
            variant_detail = f"Variant must declare its own --model value: {variant.name}"
        if not variant_args_safe:
            break
    checks.append(
        PreflightCheck(name="variant_arguments", passed=variant_args_safe, detail=variant_detail)
    )

    python_ok = sys.version_info[:2] >= (3, 12) and sys.version_info[:2] < (3, 14)
    checks.append(
        PreflightCheck(
            name="python_version",
            passed=python_ok,
            detail=f"Python {sys.version.split()[0]}",
        )
    )
    sklearn_ok = importlib.util.find_spec("sklearn") is not None
    checks.append(
        PreflightCheck(
            name="dependencies",
            passed=sklearn_ok,
            detail="scikit-learn is installed" if sklearn_ok else "scikit-learn is missing",
        )
    )
    checks.append(_git_check(root))
    return PreflightReport(plan_id=plan.plan_id, checks=checks)
