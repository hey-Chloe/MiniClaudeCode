import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from miniclaude.agent import Agent
from miniclaude.llm import (
    LLMResponse,
    LLMToolCall,
    OpenAIProvider,
    OpenAIProviderConfig,
)
from miniclaude.models import RunStatus
from miniclaude.session import SessionCheckpoint, SessionStore
from miniclaude.tools import ToolDefinition


def echo_tool():
    return ToolDefinition(
        name="echo",
        description="Return supplied text.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        handler=lambda text: {"text": text},
    )


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


class SessionResumeTests(unittest.TestCase):
    def test_checkpoint_resume_continues_budget_and_history(self):
        first = ScriptedProvider(
            [
                LLMResponse(
                    response_id="resp_1",
                    tool_calls=(
                        LLMToolCall("call_1", "echo", '{"text":"hi"}'),
                    ),
                )
            ]
        )
        agent = Agent(provider=first, tools=[echo_tool()], max_turns=20)
        interrupted = agent.run_result("echo hi")

        self.assertEqual(interrupted.status, RunStatus.FAILED)
        self.assertEqual(interrupted.turns, 1)
        checkpoint = agent.build_checkpoint(interrupted)
        self.assertEqual(checkpoint.turn_count, 1)
        self.assertEqual(checkpoint.provider_response_id, "resp_1")
        self.assertEqual(
            [message["role"] for message in checkpoint.messages],
            ["user", "assistant", "tool"],
        )

        second = ScriptedProvider(
            [LLMResponse(response_id="resp_2", text="done")]
        )
        resumed_agent = Agent(provider=second, tools=[echo_tool()], max_turns=20)
        result = resumed_agent.resume("echo hi", checkpoint)

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.turns, 2)
        self.assertEqual(second.requests[0].previous_response_id, "resp_1")
        self.assertTrue(
            any(event.get("event") == "resumed" for event in result.events)
        )
        self.assertEqual(
            [message["role"] for message in second.requests[0].messages],
            ["user", "assistant", "tool"],
        )

    def test_store_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(directory)
            checkpoint = SessionCheckpoint(
                task="task",
                max_turns=20,
                turn_count=3,
                status="max_turns",
                provider_response_id="resp_9",
                usage_input_tokens=5,
                messages=({"role": "user", "content": "task"},),
            )
            path = store.save_checkpoint("sess-1", checkpoint)
            loaded = store.load_checkpoint("sess-1")

            self.assertTrue(path.is_file())
            self.assertEqual(loaded, checkpoint)

    def test_resume_rejects_mismatched_task(self):
        agent = Agent()
        checkpoint = SessionCheckpoint(
            task="other", max_turns=20, turn_count=0, status="running"
        )
        with self.assertRaises(ValueError):
            agent.resume("task", checkpoint)

    def test_invalid_checkpoint_session_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(directory)
            with self.assertRaises(ValueError):
                store.load_checkpoint("../escape")

    def test_provider_state_round_trip_preserves_tool_call_id(self):
        provider = OpenAIProvider(
            OpenAIProviderConfig(
                model="deepseek-chat",
                base_url="https://api.deepseek.com",
            ),
            client=SimpleNamespace(
                responses=object(),
                chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: None)),
            ),
        )
        provider._chat_messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "hello",
            },
        ]

        restored = OpenAIProvider(
            OpenAIProviderConfig(
                model="deepseek-chat",
                base_url="https://api.deepseek.com",
            ),
            client=SimpleNamespace(
                responses=object(),
                chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: None)),
            ),
        )
        restored.restore_state(provider.export_state())

        self.assertEqual(restored._chat_messages, provider._chat_messages)
        self.assertEqual(restored._chat_messages[1]["tool_call_id"], "call_1")

    def test_checkpoint_carries_provider_state(self):
        class StatefulProvider(ScriptedProvider):
            def export_state(self):
                return {
                    "chat_messages": [
                        {"role": "tool", "tool_call_id": "call_1", "content": "hi"}
                    ]
                }

        first = StatefulProvider(
            [
                LLMResponse(
                    response_id="resp_1",
                    tool_calls=(
                        LLMToolCall("call_1", "echo", '{"text":"hi"}'),
                    ),
                )
            ]
        )
        agent = Agent(provider=first, tools=[echo_tool()], max_turns=20)
        interrupted = agent.run_result("echo hi")
        checkpoint = agent.build_checkpoint(interrupted)

        self.assertEqual(
            checkpoint.provider_state,
            {"chat_messages": [{"role": "tool", "tool_call_id": "call_1", "content": "hi"}]},
        )

    def test_resume_uses_provider_state_when_available(self):
        class StatefulProvider(ScriptedProvider):
            def __init__(self, responses):
                super().__init__(responses)
                self.restored_state = None

            def export_state(self):
                return {"chat_messages": [{"role": "assistant", "content": "x"}]}

            def restore_state(self, state):
                self.restored_state = state

        first = StatefulProvider(
            [
                LLMResponse(
                    response_id="resp_1",
                    tool_calls=(
                        LLMToolCall("call_1", "echo", '{"text":"hi"}'),
                    ),
                )
            ]
        )
        agent = Agent(provider=first, tools=[echo_tool()], max_turns=20)
        interrupted = agent.run_result("echo hi")
        checkpoint = agent.build_checkpoint(interrupted)

        second = StatefulProvider(
            [LLMResponse(response_id="resp_2", text="done")]
        )
        resumed_agent = Agent(provider=second, tools=[echo_tool()], max_turns=20)
        result = resumed_agent.resume("echo hi", checkpoint)

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(
            second.restored_state,
            {"chat_messages": [{"role": "assistant", "content": "x"}]},
        )


if __name__ == "__main__":
    unittest.main()
