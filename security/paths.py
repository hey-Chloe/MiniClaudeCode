"""Workspace path-boundary validation."""

from pathlib import Path


class PathSecurityError(ValueError):
    """Raised when a requested path escapes the configured workspace."""


class WorkspacePathPolicy:
    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()
        if not self.workspace.is_dir():
            raise ValueError("workspace must be an existing directory")

    def resolve(self, requested: str | Path) -> Path:
        path = Path(requested)
        candidate = path.resolve() if path.is_absolute() else (self.workspace / path).resolve()
        try:
            candidate.relative_to(self.workspace)
        except ValueError as exc:
            raise PathSecurityError(
                f"path escapes workspace: {requested}"
            ) from exc
        return candidate

