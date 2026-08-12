"""Conservative, execution-free shell command risk analysis."""

import re
import shlex
from dataclasses import dataclass

from security.policy import PolicyAction


@dataclass(frozen=True, slots=True)
class CommandAssessment:
    action: PolicyAction
    reason: str


_SHELL_OPERATORS = re.compile(r"(?:&&|\|\||[|;<>`]|\$\(|\r|\n)")
_DESTRUCTIVE = {
    "rm",
    "rmdir",
    "del",
    "erase",
    "format",
    "mkfs",
    "shutdown",
    "reboot",
}
_MUTATING = {
    "cp",
    "copy",
    "mv",
    "move",
    "mkdir",
    "touch",
    "chmod",
    "chown",
    "docker",
    "pip",
    "npm",
}
_READ_ONLY = {"pwd", "ls", "dir", "type", "cat", "head", "tail", "rg", "grep"}


def assess_command(command: str) -> CommandAssessment:
    """Classify a command without executing or expanding it."""
    if not isinstance(command, str) or not command.strip():
        return CommandAssessment(PolicyAction.DENY, "command must not be empty")
    if _SHELL_OPERATORS.search(command):
        return CommandAssessment(
            PolicyAction.ASK, "shell operators require explicit approval"
        )

    try:
        tokens = shlex.split(command, posix=False)
    except ValueError:
        return CommandAssessment(PolicyAction.DENY, "command could not be parsed")

    executable = tokens[0].strip('"\'').lower()
    if executable in _DESTRUCTIVE:
        return CommandAssessment(PolicyAction.DENY, "destructive command denied")
    if executable == "git":
        subcommand = tokens[1].lower() if len(tokens) > 1 else ""
        if subcommand in {"status", "diff", "log", "show", "branch"}:
            return CommandAssessment(PolicyAction.ALLOW, "read-only git command")
        return CommandAssessment(PolicyAction.ASK, "mutating git command requires approval")
    if executable in _READ_ONLY:
        return CommandAssessment(PolicyAction.ALLOW, "recognized read-only command")
    if executable in _MUTATING:
        return CommandAssessment(PolicyAction.ASK, "mutating command requires approval")
    return CommandAssessment(PolicyAction.ASK, "unrecognized command requires approval")


def assess_argv(argv: object) -> CommandAssessment:
    """Classify a shell-free argument vector."""
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) for x in argv):
        return CommandAssessment(PolicyAction.DENY, "argv must be a non-empty string array")
    executable = PathLikeName(argv[0])
    if executable in _DESTRUCTIVE:
        return CommandAssessment(PolicyAction.DENY, "destructive command denied")
    if executable == "git":
        subcommand = argv[1].lower() if len(argv) > 1 else ""
        if subcommand in {"status", "diff", "log", "show", "branch"}:
            return CommandAssessment(PolicyAction.ALLOW, "read-only git command")
        return CommandAssessment(PolicyAction.ASK, "mutating git command requires approval")
    if executable in _READ_ONLY:
        return CommandAssessment(PolicyAction.ALLOW, "recognized read-only command")
    if executable in _MUTATING:
        return CommandAssessment(PolicyAction.ASK, "mutating command requires approval")
    return CommandAssessment(PolicyAction.ASK, "unrecognized command requires approval")


def PathLikeName(value: str) -> str:
    normalized = value.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return normalized[:-4] if normalized.endswith(".exe") else normalized
