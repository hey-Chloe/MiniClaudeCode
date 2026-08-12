import tempfile
import unittest
from pathlib import Path

from miniclaude.agent import Agent
from miniclaude.context import (
    ContextConfig,
    ContextManager,
    ContextMessage,
)
from miniclaude.llm import LLMResponse, LLMToolCall
from miniclaude.tools import ToolDefinition


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


def echo_tool():
    return ToolDefinition(
        name="echo",
        description="Echo text.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        handler=lambda text: text,
    )


class ContextManagerTests(unittest.TestCase):
    def test_system_instructions_and_task_are_preserved(self):
        manager = ContextManager(
            ContextConfig(system_instructions="system rule", max_chars=100)
        )

        snapshot = manager.start("fix bug")

        self.assertEqual(snapshot.instructions, "system rule")
        self.assertEqual(snapshot.task, "fix bug")
        self.assertEqual(snapshot.messages, (ContextMessage("user", "fix bug"),))
        self.assertFalse(snapshot.truncated)

    def test_loads_supported_project_instructions_from_workspace_root(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "AGENTS.md").write_text("run focused tests", encoding="utf-8")
            (workspace / "IGNORED.md").write_text("do not load", encoding="utf-8")
            manager = ContextManager(
                ContextConfig(
                    workspace=workspace,
                    system_instructions="system",
                    max_chars=1000,
                )
            )

            snapshot = manager.start("task")

            self.assertIn("run focused tests", snapshot.instructions)
            self.assertNotIn("do not load", snapshot.instructions)

    def test_project_instructions_are_deterministically_limited(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "AGENTS.md").write_text("abcdefghij", encoding="utf-8")
            manager = ContextManager(
                ContextConfig(
                    workspace=workspace,
                    system_instructions="system",
                    max_chars=100,
                    max_project_instruction_chars=4,
                )
            )

            snapshot = manager.start("task")

            self.assertIn("abcd", snapshot.instructions)
            self.assertNotIn("abcde", snapshot.instructions)

    def test_oldest_messages_are_dropped_when_budget_is_exceeded(self):
        manager = ContextManager(
            ContextConfig(system_instructions="system", max_chars=40)
        )
        manager.start("task")
        manager.add_assistant("old-message-12345")
        manager.add_tool("latest")

        snapshot = manager.snapshot()

        self.assertTrue(snapshot.truncated)
        self.assertIn(ContextMessage("tool", "latest"), snapshot.messages)
        self.assertNotIn(
            ContextMessage("assistant", "old-message-12345"), snapshot.messages
        )

    def test_start_resets_history_for_new_run(self):
        manager = ContextManager(ContextConfig(system_instructions="system"))
        manager.start("first")
        manager.add_assistant("old answer")

        snapshot = manager.start("second")

        self.assertEqual(snapshot.messages, (ContextMessage("user", "second"),))


class ContextLoopIntegrationTests(unittest.TestCase):
    def test_context_is_attached_to_provider_request(self):
        provider = ScriptedProvider([LLMResponse(text="done")])
        config = ContextConfig(system_instructions="follow project rules")

        Agent(provider=provider, context_config=config).run_result("fix bug")

        request = provider.requests[0]
        self.assertEqual(request.instructions, "follow project rules")
        self.assertEqual(
            request.messages,
            ({"role": "user", "content": "fix bug"},),
        )

    def test_tool_history_is_available_on_follow_up_turn(self):
        provider = ScriptedProvider(
            [
                LLMResponse(
                    response_id="resp_1",
                    tool_calls=(LLMToolCall("call_1", "echo", '{"text":"hi"}'),),
                ),
                LLMResponse(response_id="resp_2", text="done"),
            ]
        )

        Agent(provider=provider, tools=[echo_tool()]).run_result("echo")

        second = provider.requests[1]
        roles = [message["role"] for message in second.messages]
        self.assertEqual(roles, ["user", "assistant", "tool"])
        self.assertEqual(second.previous_response_id, "resp_1")


if __name__ == "__main__":
    unittest.main()
