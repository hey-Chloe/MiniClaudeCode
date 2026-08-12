"""Runtime contracts shared by local and future isolated backends."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence


class RuntimeErrorBase(RuntimeError):
    """Base error raised by a runtime backend."""


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    name: str
    isolated: bool
    workspace: Path


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: tuple[str, ...]
    cwd: str
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float
    isolated: bool
    output_truncated: bool = False

    @property
    def succeeded(self) -> bool:
        return not self.timed_out and self.exit_code == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "argv": list(self.argv),
            "cwd": self.cwd,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "duration_seconds": self.duration_seconds,
            "isolated": self.isolated,
            "output_truncated": self.output_truncated,
            "succeeded": self.succeeded,
        }


class Runtime(Protocol):
    @property
    def info(self) -> RuntimeInfo:
        """Describe the backend and its isolation boundary."""

    def execute(
        self,
        argv: Sequence[str],
        *,
        cwd: str = ".",
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        """Execute an argument vector without invoking a shell."""

    def read_text(self, path: str, *, encoding: str = "utf-8") -> str:
        """Read a text file inside the workspace."""

    def write_text(self, path: str, content: str, *, encoding: str = "utf-8") -> int:
        """Atomically write a text file inside the workspace."""

