"""Backward-compatible name for the local runtime backend."""

from pathlib import Path

from runtime.local import LocalProcessRuntime


class SandboxRuntime(LocalProcessRuntime):
    """Deprecated compatibility facade.

    Despite the historic name, this backend is a local process and is not an
    operating-system sandbox. Use ``info.isolated`` to inspect this explicitly.
    """

    def __init__(self, workspace: str | Path = ".", **options):
        super().__init__(workspace, **options)
