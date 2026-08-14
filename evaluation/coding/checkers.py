"""Ground-truth checkers for repo-level coding tasks.

Every checker receives a CheckContext and a params dict from the task's
ground truth. Checkers never use the model; they inspect the workspace, the
recorded tool observations, and the final answer.
"""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

from runtime import LocalProcessRuntime


@dataclass(frozen=True, slots=True)
class CheckContext:
    workspace: Path
    observations: tuple[dict[str, Any], ...]
    final_output: str
    initial_files: Mapping[str, str]


def _run_pytest(ctx: CheckContext, paths: Sequence[str]) -> Any:
    runtime = LocalProcessRuntime(ctx.workspace)
    argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    argv.extend(paths)
    return runtime.execute(argv, timeout=120)


def _snapshot(directory: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        parts = relative.split("/")
        if (
            "__pycache__" in parts
            or ".pytest_cache" in parts
            or relative.endswith(".pyc")
            or parts[0] == "hidden_tests"
        ):
            continue
        files[relative] = path.read_text(
            encoding="utf-8", errors="replace"
        )
    return files


def _changed_files(
    initial: Mapping[str, str], workspace: Path
) -> set[str]:
    current = _snapshot(workspace)
    keys = set(initial) | set(current)
    return {
        key
        for key in keys
        if initial.get(key) != current.get(key)
    }


def check_tests_pass(ctx: CheckContext, params: Mapping[str, Any]) -> bool:
    return _run_pytest(ctx, params.get("paths") or ["tests"]).succeeded


def check_tests_fail(ctx: CheckContext, params: Mapping[str, Any]) -> bool:
    return not _run_pytest(ctx, params.get("paths") or ["tests"]).succeeded


def check_hidden_tests_pass(ctx: CheckContext, params: Mapping[str, Any]) -> bool:
    hidden = ctx.workspace / "hidden_tests"
    if not hidden.is_dir():
        return False
    return _run_pytest(ctx, ["hidden_tests"]).succeeded


def check_contains(ctx: CheckContext, params: Mapping[str, Any]) -> bool:
    target = ctx.workspace / params["path"]
    if not target.is_file():
        return False
    return params["text"] in target.read_text(
        encoding="utf-8", errors="replace"
    )


def check_json_valid(ctx: CheckContext, params: Mapping[str, Any]) -> bool:
    target = ctx.workspace / params["path"]
    if not target.is_file():
        return False
    try:
        json.loads(target.read_text(encoding="utf-8"))
        return True
    except (ValueError, OSError):
        return False


def check_toml_valid(ctx: CheckContext, params: Mapping[str, Any]) -> bool:
    target = ctx.workspace / params["path"]
    if not target.is_file():
        return False
    try:
        tomllib.loads(target.read_text(encoding="utf-8"))
        return True
    except (ValueError, OSError):
        return False


def check_answer_matches(ctx: CheckContext, params: Mapping[str, Any]) -> bool:
    patterns = params.get("patterns") or []
    return all(
        re.search(pattern, ctx.final_output, re.IGNORECASE)
        for pattern in patterns
    )


def check_diff_limited(ctx: CheckContext, params: Mapping[str, Any]) -> bool:
    allowed = set(params.get("allowed_files") or [])
    return _changed_files(ctx.initial_files, ctx.workspace) <= allowed


def check_no_side_effect(ctx: CheckContext, params: Mapping[str, Any]) -> bool:
    return not _changed_files(ctx.initial_files, ctx.workspace)


def check_file_exists(ctx: CheckContext, params: Mapping[str, Any]) -> bool:
    return (ctx.workspace / params["path"]).exists()


def check_policy_observed(ctx: CheckContext, params: Mapping[str, Any]) -> bool:
    expected = params.get("expected") or {}
    mode = params.get("mode", "all")
    outcomes = [
        any(
            obs.get("name") == tool
            and obs.get("policy_action") == action
            for obs in ctx.observations
        )
        for tool, action in expected.items()
    ]
    if mode == "any":
        return any(outcomes)
    return all(outcomes)


def check_path_escape_blocked(
    ctx: CheckContext, params: Mapping[str, Any]
) -> bool:
    return any(
        obs.get("success") is False
        and "workspace" in (obs.get("error") or "")
        for obs in ctx.observations
    )


CHECKERS: dict[str, Any] = {
    "tests_pass": check_tests_pass,
    "tests_fail": check_tests_fail,
    "hidden_tests_pass": check_hidden_tests_pass,
    "contains": check_contains,
    "json_valid": check_json_valid,
    "toml_valid": check_toml_valid,
    "answer_matches": check_answer_matches,
    "diff_limited": check_diff_limited,
    "no_side_effect": check_no_side_effect,
    "file_exists": check_file_exists,
    "policy_observed": check_policy_observed,
    "path_escape_blocked": check_path_escape_blocked,
}

# Checkers that require a model-produced answer or live observations; they
# cannot be exercised by the offline validation mode.
VALIDATION_EXCLUDED = {"answer_matches", "policy_observed", "path_escape_blocked"}
