"""Workspace-confined local process runtime.

This backend does not provide operating-system isolation. It is intended for
controlled development environments and reports that limitation explicitly.
"""

import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from runtime.base import CommandResult, RuntimeErrorBase, RuntimeInfo
from security.paths import WorkspacePathPolicy


_DEFAULT_ENV_KEYS = {
    "COMSPEC",
    "LANG",
    "LC_ALL",
    "PATH",
    "PATHEXT",
    "PYTHONPATH",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "WINDIR",
}


class LocalProcessRuntime:
    """Runs argument vectors locally with workspace and output limits."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        default_timeout: float = 120.0,
        max_output_chars: int = 100_000,
        allowed_env_keys: set[str] | None = None,
    ):
        if default_timeout <= 0:
            raise ValueError("default_timeout must be greater than zero")
        if max_output_chars < 1:
            raise ValueError("max_output_chars must be at least 1")
        self.paths = WorkspacePathPolicy(workspace)
        self.default_timeout = default_timeout
        self.max_output_chars = max_output_chars
        self.allowed_env_keys = {
            key.upper() for key in (allowed_env_keys or _DEFAULT_ENV_KEYS)
        }

    @property
    def info(self) -> RuntimeInfo:
        return RuntimeInfo("local-process", False, self.paths.workspace)

    def execute(
        self,
        argv: Sequence[str],
        *,
        cwd: str = ".",
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        arguments = self._validate_argv(argv)
        working_directory = self.paths.resolve(cwd)
        if not working_directory.is_dir():
            raise RuntimeErrorBase(f"working directory does not exist: {cwd}")

        execution_timeout = self.default_timeout if timeout is None else timeout
        if execution_timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        process_env = self._build_environment(env)
        started = time.monotonic()

        try:
            process = subprocess.Popen(
                arguments,
                cwd=working_directory,
                env=process_env,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                start_new_session=os.name != "nt",
            )
            stdout_value, stderr_value = process.communicate(timeout=execution_timeout)
            stdout, stdout_cut = self._truncate(stdout_value)
            stderr, stderr_cut = self._truncate(stderr_value)
            return CommandResult(
                argv=arguments,
                cwd=str(working_directory),
                exit_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                timed_out=False,
                duration_seconds=time.monotonic() - started,
                isolated=False,
                output_truncated=stdout_cut or stderr_cut,
            )
        except subprocess.TimeoutExpired as exc:
            self._terminate_tree(process)
            final_stdout, final_stderr = process.communicate()
            stdout, stdout_cut = self._truncate(self._as_text(final_stdout or exc.stdout))
            stderr, stderr_cut = self._truncate(self._as_text(final_stderr or exc.stderr))
            return CommandResult(
                argv=arguments,
                cwd=str(working_directory),
                exit_code=None,
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
                duration_seconds=time.monotonic() - started,
                isolated=False,
                output_truncated=stdout_cut or stderr_cut,
            )
        except OSError as exc:
            raise RuntimeErrorBase(f"process could not start: {exc}") from exc

    def read_text(self, path: str, *, encoding: str = "utf-8") -> str:
        target = self.paths.resolve(path)
        if not target.is_file():
            raise RuntimeErrorBase(f"file does not exist: {path}")
        return target.read_text(encoding=encoding)

    def write_text(self, path: str, content: str, *, encoding: str = "utf-8") -> int:
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        target = self.paths.resolve(path)
        if not target.parent.is_dir():
            raise RuntimeErrorBase(f"parent directory does not exist: {target.parent}")

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding=encoding,
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, target)
        except OSError as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise RuntimeErrorBase(f"file could not be written: {exc}") from exc
        return len(content.encode(encoding))

    def _build_environment(self, additions: Mapping[str, str] | None) -> dict[str, str]:
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in self.allowed_env_keys
        }
        for key, value in (additions or {}).items():
            if key.upper() not in self.allowed_env_keys:
                raise RuntimeErrorBase(f"environment variable is not allowed: {key}")
            if not isinstance(value, str):
                raise TypeError("environment variable values must be strings")
            environment[key] = value
        return environment

    def _truncate(self, value: str) -> tuple[str, bool]:
        if len(value) <= self.max_output_chars:
            return value, False
        return value[: self.max_output_chars], True

    @staticmethod
    def _validate_argv(argv: Sequence[str]) -> tuple[str, ...]:
        if isinstance(argv, (str, bytes)) or not argv:
            raise ValueError("argv must be a non-empty sequence of strings")
        if any(not isinstance(argument, str) or "\x00" in argument for argument in argv):
            raise ValueError("argv entries must be strings without null bytes")
        return tuple(argv)

    @staticmethod
    def _as_text(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        return value

    @staticmethod
    def _terminate_tree(process: subprocess.Popen) -> None:
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                    timeout=0.5,
                )
            except subprocess.TimeoutExpired:
                process.kill()
            if process.poll() is None:
                process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
