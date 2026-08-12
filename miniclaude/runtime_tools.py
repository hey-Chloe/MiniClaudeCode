"""Coding tools backed by an injected runtime."""

import fnmatch
import re
from collections.abc import Sequence
from pathlib import Path

from miniclaude.tools import ToolDefinition
from runtime.base import Runtime
from security.policy import ToolRisk


def create_runtime_tools(runtime: Runtime) -> list[ToolDefinition]:
    """Create host-capable tools for an explicitly supplied runtime."""

    def execute_command(argv: Sequence[str], cwd: str = ".", timeout: int = 120):
        return runtime.execute(argv, cwd=cwd, timeout=timeout).to_dict()

    workspace = runtime.info.workspace
    git_workspace = "/workspace" if runtime.info.name == "docker" else str(workspace)

    def git_command(*arguments: str):
        return ["git", "-c", f"safe.directory={git_workspace}", *arguments]

    def list_directory(path: str = "."):
        directory = _resolve(runtime, path)
        if not directory.is_dir():
            raise ValueError(f"directory does not exist: {path}")
        return [
            {"name": child.name, "type": "directory" if child.is_dir() else "file"}
            for child in sorted(directory.iterdir(), key=lambda item: item.name.lower())
        ]

    def glob_files(pattern: str, path: str = ".", limit: int = 500):
        root = _resolve(runtime, path)
        matches = []
        for candidate in root.rglob("*"):
            relative = candidate.relative_to(workspace).as_posix()
            if candidate.is_file() and fnmatch.fnmatch(relative, pattern):
                matches.append(relative)
                if len(matches) >= limit:
                    break
        return matches

    def grep_files(pattern: str, path: str = ".", limit: int = 200):
        expression = re.compile(pattern)
        root = _resolve(runtime, path)
        matches = []
        candidates = [root] if root.is_file() else root.rglob("*")
        for candidate in candidates:
            if not candidate.is_file() or candidate.stat().st_size > 2_000_000:
                continue
            try:
                lines = candidate.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(lines, 1):
                if expression.search(line):
                    matches.append({
                        "path": candidate.relative_to(workspace).as_posix(),
                        "line": number,
                        "text": line[:500],
                    })
                    if len(matches) >= limit:
                        return matches
        return matches

    def replace_text(path: str, old: str, new: str):
        if not old:
            raise ValueError("old text must not be empty")
        content = runtime.read_text(path)
        count = content.count(old)
        if count != 1:
            raise ValueError(f"old text must match exactly once; found {count}")
        runtime.write_text(path, content.replace(old, new, 1))
        return {"path": path, "replacements": 1}

    def git_status():
        return _require_success(
            runtime.execute(git_command("status", "--short"), timeout=30),
            "git status",
        )

    def git_diff():
        return _require_success(
            runtime.execute(git_command("diff", "--no-ext-diff"), timeout=30),
            "git diff",
        )

    return [
        ToolDefinition(
            name="list_directory",
            description="List direct children of a workspace directory.",
            parameters=_object_schema({"path": {"type": "string"}}),
            handler=list_directory,
        ),
        ToolDefinition(
            name="glob_files",
            description="Find workspace files matching a glob pattern.",
            parameters=_object_schema(
                {"pattern": {"type": "string"}, "path": {"type": "string"}, "limit": {"type": "integer"}},
                ["pattern"],
            ),
            handler=glob_files,
        ),
        ToolDefinition(
            name="grep_files",
            description="Search UTF-8 workspace files with a regular expression.",
            parameters=_object_schema(
                {"pattern": {"type": "string"}, "path": {"type": "string"}, "limit": {"type": "integer"}},
                ["pattern"],
            ),
            handler=grep_files,
        ),
        ToolDefinition(
            name="read_file",
            description="Read a UTF-8 text file inside the workspace.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=runtime.read_text,
            risk=ToolRisk.READ_ONLY,
        ),
        ToolDefinition(
            name="write_file",
            description="Atomically write a UTF-8 text file inside the workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            handler=runtime.write_text,
            risk=ToolRisk.MUTATING,
        ),
        ToolDefinition(
            name="replace_text",
            description="Replace one exact, uniquely matching text block in a workspace file.",
            parameters=_object_schema(
                {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
                ["path", "old", "new"],
            ),
            handler=replace_text,
            risk=ToolRisk.MUTATING,
        ),
        ToolDefinition(
            name="execute_command",
            description=(
                "Execute an argument vector without a shell in the workspace. "
                "The local backend is not OS-isolated."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "argv": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "cwd": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["argv"],
                "additionalProperties": False,
            },
            handler=execute_command,
            risk=ToolRisk.MUTATING,
        ),
        ToolDefinition(
            name="git_status",
            description="Return git status --short for the workspace.",
            parameters=_object_schema({}),
            handler=git_status,
        ),
        ToolDefinition(
            name="git_diff",
            description="Return the current unstaged git diff.",
            parameters=_object_schema({}),
            handler=git_diff,
        ),
    ]


def _resolve(runtime: Runtime, path: str) -> Path:
    paths = getattr(runtime, "paths", None)
    if paths is None:
        candidate = (runtime.info.workspace / path).resolve()
        candidate.relative_to(runtime.info.workspace)
        return candidate
    return paths.resolve(path)


def _require_success(result, operation: str):
    if not result.succeeded:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"{operation} failed with exit code {result.exit_code}: {detail}")
    return result.to_dict()


def _object_schema(properties, required=None):
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }
