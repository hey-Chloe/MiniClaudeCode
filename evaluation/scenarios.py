"""Built-in, offline evaluation scenarios."""

import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from miniclaude.agent import Agent
from miniclaude.context import ContextConfig, ContextManager
from miniclaude.tools import ToolDefinition, ToolRegistry
from runtime import LocalProcessRuntime
from security.command_analysis import assess_command


Scenario = Callable[[dict[str, Any]], dict[str, Any]]


def agent_completion(params: dict[str, Any]) -> dict[str, Any]:
    result = Agent().run_result(str(params.get("task", "benchmark task")))
    return {"status": result.status.value, "turns": result.turns}


def tool_dispatch(params: dict[str, Any]) -> dict[str, Any]:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo text for evaluation.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=lambda text: text,
        )
    )
    import json

    observation = registry.dispatch(
        "evaluation-call", "echo", json.dumps({"text": params.get("text", "")})
    )
    return {"success": observation.success, "output": observation.output}


def command_policy(params: dict[str, Any]) -> dict[str, Any]:
    assessment = assess_command(str(params.get("command", "")))
    return {"action": assessment.action.value}


def runtime_python(params: dict[str, Any]) -> dict[str, Any]:
    output = str(params.get("output", "runtime-ok"))
    with tempfile.TemporaryDirectory() as directory:
        result = LocalProcessRuntime(directory).execute(
            [sys.executable, "-c", "import sys; print(sys.argv[1])", output],
            timeout=10,
        )
    return {
        "succeeded": result.succeeded,
        "isolated": result.isolated,
        "stdout": result.stdout.strip(),
    }


def context_project_instructions(params: dict[str, Any]) -> dict[str, Any]:
    instructions = str(params.get("instructions", ""))
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        (workspace / "AGENTS.md").write_text(instructions, encoding="utf-8")
        snapshot = ContextManager(ContextConfig(workspace=workspace)).start("evaluate")
    return {"loaded": instructions in snapshot.instructions}


DEFAULT_SCENARIOS: dict[str, Scenario] = {
    "agent_completion": agent_completion,
    "tool_dispatch": tool_dispatch,
    "command_policy": command_policy,
    "runtime_python": runtime_python,
    "context_project_instructions": context_project_instructions,
}

