"""Docker-backed command runtime with explicit resource controls."""

import shutil
from pathlib import Path

from runtime.base import CommandResult, RuntimeErrorBase, RuntimeInfo
from runtime.local import LocalProcessRuntime


class DockerRuntime(LocalProcessRuntime):
    def __init__(self, workspace: str | Path, *, image="python:3.12-slim", network="none", memory="1g", cpus="1.0", pids_limit=256, **options):
        super().__init__(workspace, **options)
        self.image, self.network, self.memory, self.cpus = image, network, memory, cpus
        self.pids_limit = pids_limit
        self.docker = shutil.which("docker")
        if self.docker is None:
            raise RuntimeErrorBase("Docker executable was not found")

    @property
    def info(self) -> RuntimeInfo:
        return RuntimeInfo("docker", True, self.paths.workspace)

    def execute(self, argv, *, cwd=".", timeout=None, env=None) -> CommandResult:
        working = self.paths.resolve(cwd)
        relative = working.relative_to(self.paths.workspace).as_posix()
        container_cwd = "/workspace" if relative == "." else f"/workspace/{relative}"
        command = [self.docker, "run", "--rm", "--network", self.network,
                   "--memory", self.memory, "--cpus", self.cpus, "--pids-limit", str(self.pids_limit),
                   "--user", "65534:65534", "--mount", f"type=bind,src={self.paths.workspace},dst=/workspace",
                   "--workdir", container_cwd]
        for key, value in (env or {}).items():
            if key.upper() not in self.allowed_env_keys:
                raise RuntimeErrorBase(f"environment variable is not allowed: {key}")
            command.extend(["--env", f"{key}={value}"])
        command.extend([self.image, *self._validate_argv(argv)])
        host = super().execute(command, timeout=timeout)
        return CommandResult(tuple(argv), container_cwd, host.exit_code, host.stdout, host.stderr,
                             host.timed_out, host.duration_seconds, True, host.output_truncated)
