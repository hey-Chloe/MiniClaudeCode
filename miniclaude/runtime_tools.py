"""Coding tools backed by an injected runtime."""

import fnmatch
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from git.diff import generate_diff
from miniclaude.memory import FileCache
from miniclaude.tools import ToolDefinition
from runtime.base import Runtime
from security.policy import ToolRisk


def create_runtime_tools(
    runtime: Runtime,
    *,
    cache_enabled: bool = True,
) -> list[ToolDefinition]:
    """Create host-capable tools for an explicitly supplied runtime."""

    def execute_command(argv: Sequence[str], cwd: str = ".", timeout: int = 120):
        return runtime.execute(argv, cwd=cwd, timeout=timeout).to_dict()

    workspace = runtime.info.workspace
    initial_files = _snapshot_workspace(workspace)
    git_workspace = "/workspace" if runtime.info.name == "docker" else str(workspace)
    cache = FileCache() if cache_enabled else None

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

    def file_tree(path: str = ".", depth: int = 3, limit: int = 500):
        """Return a depth-limited, sorted recursive listing of the workspace."""
        if depth < 1 or limit < 1:
            raise ValueError("depth and limit must be positive")
        root = _resolve(runtime, path)
        if not root.is_dir():
            raise ValueError(f"directory does not exist: {path}")
        entries: list[dict[str, Any]] = []
        stack = [(root, 0)]
        while stack:
            directory, level = stack.pop()
            if level >= depth:
                continue
            children = sorted(
                directory.iterdir(), key=lambda item: item.name.lower()
            )
            for child in children:
                if len(entries) >= limit:
                    break
                relative = child.relative_to(workspace).as_posix()
                entries.append(
                    {
                        "path": relative,
                        "type": "directory" if child.is_dir() else "file",
                        "depth": level + 1,
                    }
                )
                if child.is_dir():
                    stack.append((child, level + 1))
        return entries

    def todo_scan(
        pattern: str = r"TODO|FIXME|HACK",
        path: str = ".",
        limit: int = 100,
    ):
        """Scan workspace text files for TODO/FIXME-style markers."""
        expression = re.compile(pattern)
        root = _resolve(runtime, path)
        matches: list[dict[str, Any]] = []
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
                    matches.append(
                        {
                            "path": candidate.relative_to(workspace).as_posix(),
                            "line": number,
                            "text": line.strip()[:300],
                        }
                    )
                    if len(matches) >= limit:
                        return matches
        return matches

    def file_stat(path: str):
        """Return metadata for one workspace file or directory."""
        target = _resolve(runtime, path)
        if not target.exists():
            raise ValueError(f"path does not exist: {path}")
        stat_result = target.stat()
        line_count = None
        if target.is_file():
            try:
                line_count = len(
                    target.read_text(encoding="utf-8", errors="replace").splitlines()
                )
            except OSError:
                line_count = None
        return {
            "path": target.relative_to(workspace).as_posix(),
            "type": "directory" if target.is_dir() else "file",
            "size_bytes": stat_result.st_size,
            "mtime": stat_result.st_mtime,
            "line_count": line_count,
        }

    def replace_text(path: str, old: str, new: str):
        if not old:
            raise ValueError("old text must not be empty")
        content = runtime.read_text(path)
        count = content.count(old)
        if count != 1:
            raise ValueError(f"old text must match exactly once; found {count}")
        runtime.write_text(path, content.replace(old, new, 1))
        cache.invalidate(path)
        return {"path": path, "replacements": 1}

    def read_file(path: str):
        """Read with a freshness-checked cache; reports ``cache_hit``."""
        target = _resolve(runtime, path)
        stat_result = None
        if cache is not None:
            try:
                stat_result = target.stat()
                cached = cache.get(
                    path, stat_result.st_mtime, stat_result.st_size
                )
                if cached is not None:
                    return {
                        "path": path,
                        "content": cached,
                        "cache_hit": True,
                    }
            except OSError:
                stat_result = None
        content = runtime.read_text(path)
        if cache is not None and stat_result is not None:
            cache.put(path, stat_result.st_mtime, stat_result.st_size, content)
        return {
            "path": path,
            "content": content,
            "cache_hit": False,
        }

    def write_file(path: str, content: str):
        result = runtime.write_text(path, content)
        if cache is not None:
            cache.invalidate(path)
        return result

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

    def workspace_diff():
        current = _snapshot_workspace(workspace)
        changed = [
            relative
            for relative in sorted(set(initial_files) | set(current))
            if initial_files.get(relative) != current.get(relative)
        ]
        sections = [
            generate_diff(
                initial_files.get(relative, ""),
                current.get(relative, ""),
                path=relative,
            )
            for relative in changed
        ]
        return {
            "changed_files": changed,
            "diff": "\n".join(sections),
        }

    return [
        ToolDefinition(
            name="list_directory",
            description="List direct children of a workspace directory.",
            parameters=_object_schema({"path": {"type": "string"}}),
            handler=list_directory,
            activation_keywords=(
                "list",
                "directory",
                "structure",
                "explore",
                "tree",
            ),
        ),
        ToolDefinition(
            name="glob_files",
            description="Find workspace files matching a glob pattern.",
            parameters=_object_schema(
                {"pattern": {"type": "string"}, "path": {"type": "string"}, "limit": {"type": "integer"}},
                ["pattern"],
            ),
            handler=glob_files,
            activation_keywords=("glob", "pattern", "find", "search"),
        ),
        ToolDefinition(
            name="grep_files",
            description="Search UTF-8 workspace files with a regular expression.",
            parameters=_object_schema(
                {"pattern": {"type": "string"}, "path": {"type": "string"}, "limit": {"type": "integer"}},
                ["pattern"],
            ),
            handler=grep_files,
            activation_keywords=("grep", "search", "pattern", "find", "regex"),
        ),
        ToolDefinition(
            name="read_file",
            description=(
                "Read a UTF-8 text file inside the workspace; repeated reads "
                "of an unchanged file are served from a freshness-checked cache."
            ),
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=read_file,
            risk=ToolRisk.READ_ONLY,
            activation_keywords=("read", "inspect", "view", "show", "open"),
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
            handler=write_file,
            risk=ToolRisk.MUTATING,
            activation_keywords=("write", "create", "new", "save", "add"),
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
            activation_keywords=(
                "replace",
                "edit",
                "modify",
                "change",
                "refactor",
                "fix",
            ),
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
            activation_keywords=(
                "run",
                "execute",
                "command",
                "test",
                "pytest",
                "install",
                "build",
                "python",
            ),
        ),
        ToolDefinition(
            name="git_status",
            description="Return git status --short for the workspace.",
            parameters=_object_schema({}),
            handler=git_status,
            activation_keywords=("git", "status"),
        ),
        ToolDefinition(
            name="git_diff",
            description="Return the current unstaged git diff.",
            parameters=_object_schema({}),
            handler=git_diff,
            activation_keywords=("git", "diff"),
        ),
        ToolDefinition(
            name="workspace_diff",
            description=(
                "Return the unified diff of files changed since this session "
                "started, excluding caches."
            ),
            parameters=_object_schema({}),
            handler=workspace_diff,
            activation_keywords=("diff", "change", "workspace", "modified"),
        ),
        ToolDefinition(
            name="file_tree",
            description=(
                "Return a depth-limited, sorted recursive listing of a "
                "workspace directory (paths, types, depth)."
            ),
            parameters=_object_schema(
                {
                    "path": {"type": "string"},
                    "depth": {"type": "integer"},
                    "limit": {"type": "integer"},
                }
            ),
            handler=file_tree,
            activation_keywords=(
                "tree",
                "structure",
                "hierarchy",
                "layout",
                "explore",
            ),
        ),
        ToolDefinition(
            name="todo_scan",
            description=(
                "Scan workspace text files for TODO/FIXME/HACK markers and "
                "return file, line, and text for each hit."
            ),
            parameters=_object_schema(
                {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                ["pattern"],
            ),
            handler=todo_scan,
            activation_keywords=("todo", "fixme", "hack", "marker", "pending"),
        ),
        ToolDefinition(
            name="file_stat",
            description=(
                "Return metadata for one workspace file or directory "
                "(type, size, mtime, line count)."
            ),
            parameters=_object_schema({"path": {"type": "string"}}, ["path"]),
            handler=file_stat,
            activation_keywords=(
                "stat",
                "size",
                "mtime",
                "metadata",
                "lines",
            ),
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


def _snapshot_workspace(workspace: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(workspace).as_posix()
        parts = relative.split("/")
        if (
            "__pycache__" in parts
            or ".pytest_cache" in parts
            or relative.endswith(".pyc")
            or parts[0] == "hidden_tests"
            or parts[0] in {".git", ".venv", "venv", "node_modules"}
        ):
            continue
        files[relative] = path.read_text(
            encoding="utf-8", errors="replace"
        )
    return files
