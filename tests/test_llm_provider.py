import unittest
from types import SimpleNamespace
from unittest.mock import patch

from miniclaude.agent import Agent
from miniclaude.llm import (
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMToolCall,
    OpenAIProvider,
    OpenAIProviderConfig,
)
from miniclaude.models import RunStatus


class RecordingResponses:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **parameters):
        self.calls.append(parameters)
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.responses = RecordingResponses(response=response, error=error)


class RecordingChatCompletions(RecordingResponses):
    pass


class FakeChatClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=RecordingChatCompletions())
        self._responses = list(responses)

        def create(**parameters):
            self.chat.completions.calls.append(parameters)
            return self._responses.pop(0)

        self.chat.completions.create = create


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


class OpenAIProviderTests(unittest.TestCase):
    def test_text_response_is_normalized(self):
        sdk_response = SimpleNamespace(
            id="resp_123",
            model="test-model",
            output_text="finished",
            output=[],
            usage=SimpleNamespace(input_tokens=10, output_tokens=4, total_tokens=14),
        )
        client = FakeClient(response=sdk_response)
        provider = OpenAIProvider(
            OpenAIProviderConfig(
                model="test-model",
                instructions="act as a coding agent",
            ),
            client=client,
        )

        response = provider.complete(LLMRequest(task="fix bug"))

        self.assertEqual(response.text, "finished")
        self.assertEqual(response.response_id, "resp_123")
        self.assertEqual(response.usage.total_tokens, 14)
        self.assertEqual(
            client.responses.calls,
            [
                {
                    "model": "test-model",
                    "input": "fix bug",
                    "instructions": "act as a coding agent",
                }
            ],
        )

    def test_function_calls_are_normalized(self):
        sdk_response = SimpleNamespace(
            id="resp_tools",
            model="test-model",
            output_text="",
            output=[
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": '{"path":"README.md"}',
                }
            ],
            usage={"input_tokens": 8, "output_tokens": 3},
        )
        provider = OpenAIProvider(
            OpenAIProviderConfig(model="test-model"),
            client=FakeClient(response=sdk_response),
        )

        response = provider.complete(LLMRequest(task="inspect"))

        self.assertEqual(
            response.tool_calls,
            (
                LLMToolCall(
                    call_id="call_1",
                    name="read_file",
                    arguments='{"path":"README.md"}',
                ),
            ),
        )
        self.assertEqual(response.usage.total_tokens, 11)

    def test_request_context_instructions_are_merged_without_replaying_messages(self):
        sdk_response = SimpleNamespace(
            output_text="done",
            output=[],
            usage=None,
        )
        client = FakeClient(response=sdk_response)
        provider = OpenAIProvider(
            OpenAIProviderConfig(
                model="test-model",
                instructions="provider instruction",
            ),
            client=client,
        )

        provider.complete(
            LLMRequest(
                task="fix bug",
                instructions="context instruction",
                messages=({"role": "user", "content": "audit history"},),
            )
        )

        parameters = client.responses.calls[0]
        self.assertEqual(parameters["input"], "fix bug")
        self.assertEqual(
            parameters["instructions"],
            "provider instruction\n\ncontext instruction",
        )

    def test_provider_errors_are_wrapped(self):
        provider = OpenAIProvider(
            OpenAIProviderConfig(model="test-model"),
            client=FakeClient(error=TimeoutError("timed out")),
        )

        with self.assertRaisesRegex(LLMProviderError, "timed out"):
            provider.complete(LLMRequest(task="inspect"))

    def test_empty_provider_response_is_rejected(self):
        provider = OpenAIProvider(
            OpenAIProviderConfig(model="test-model"),
            client=FakeClient(
                response=SimpleNamespace(output_text="", output=[], usage=None)
            ),
        )

        with self.assertRaisesRegex(LLMProviderError, "empty response"):
            provider.complete(LLMRequest(task="inspect"))

    def test_missing_sdk_has_actionable_error(self):
        config = OpenAIProviderConfig(model="test-model", api_key="test")

        with patch.dict("sys.modules", {"openai": None}):
            with self.assertRaisesRegex(LLMProviderError, "not installed"):
                OpenAIProvider(config)

    def test_deepseek_uses_chat_completions_and_replays_tool_history(self):
        tool_call = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="read_file", arguments='{"path":"README.md"}'),
        )
        first = SimpleNamespace(
            id="chat_1",
            model="deepseek-chat",
            choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tool_call]))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        )
        second = SimpleNamespace(
            id="chat_2",
            model="deepseek-chat",
            choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=[]))],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=3, total_tokens=23),
        )
        client = FakeChatClient([first, second])
        provider = OpenAIProvider(
            OpenAIProviderConfig(
                model="deepseek-chat",
                api_key="test",
                base_url="https://api.deepseek.com",
            ),
            client=client,
        )
        tool_schema = {
            "type": "function",
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {"type": "object", "properties": {}},
            "strict": True,
        }

        response = provider.complete(
            LLMRequest(task="inspect", tools=(tool_schema,), instructions="system")
        )
        final = provider.complete(
            LLMRequest(
                task="inspect",
                turn=1,
                tools=(tool_schema,),
                tool_outputs=({"call_id": "call_1", "output": "contents"},),
            )
        )

        self.assertEqual(response.tool_calls[0].name, "read_file")
        self.assertEqual(final.text, "done")
        first_call = client.chat.completions.calls[0]
        self.assertEqual(first_call["messages"][0], {"role": "system", "content": "system"})
        self.assertIn("function", first_call["tools"][0])
        second_messages = client.chat.completions.calls[1]["messages"]
        self.assertEqual(second_messages[-1]["role"], "tool")
        self.assertEqual(second_messages[-1]["tool_call_id"], "call_1")

    def test_deepseek_errors_are_distinct_from_responses_errors(self):
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_: (_ for _ in ()).throw(RuntimeError("401 Unauthorized"))
                )
            )
        )
        provider = OpenAIProvider(
            OpenAIProviderConfig(
                model="deepseek-chat",
                base_url="https://api.deepseek.com",
            ),
            client=client,
        )

        with self.assertRaisesRegex(LLMProviderError, "DeepSeek.*401"):
            provider.complete(LLMRequest(task="smoke"))


class LLMDriverIntegrationTests(unittest.TestCase):
    def test_provider_text_completes_agent_loop(self):
        provider = ScriptedProvider(
            [
                LLMResponse(text="I will fix the bug."),
                LLMResponse(text="implemented"),
            ]
        )

        result = Agent(provider=provider).run_result("fix bug")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.output, "implemented")
        self.assertEqual(result.turns, 2)
        self.assertEqual(provider.requests[0].task, "fix bug")
        self.assertEqual(provider.requests[0].turn, 0)
        self.assertTrue(provider.requests[0].instructions)
        self.assertEqual(provider.requests[0].messages[0]["role"], "user")
        self.assertEqual(result.phases, ("plan", "finalize"))

    def test_provider_text_completes_immediately_without_plan_first(self):
        provider = ScriptedProvider([LLMResponse(text="implemented")])

        result = Agent(provider=provider, plan_first=False).run_result("fix bug")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.output, "implemented")
        self.assertEqual(result.turns, 1)
        self.assertEqual(result.phases, ("finalize",))

    def test_unregistered_provider_tool_call_is_not_executed(self):
        provider = ScriptedProvider(
            [
                LLMResponse(
                    tool_calls=(LLMToolCall("call_1", "read_file", "{}"),)
                )
            ]
        )

        result = Agent(provider=provider, max_turns=1).run_result("inspect")

        self.assertEqual(result.status, RunStatus.MAX_TURNS)
        self.assertEqual(result.events[1]["event"], "tool_results")
        observation = result.events[1]["detail"][0]
        self.assertEqual(observation["name"], "read_file")
        self.assertFalse(observation["success"])
        self.assertIn("unknown tool", observation["error"])

    def test_driver_and_provider_are_mutually_exclusive(self):
        provider = ScriptedProvider([LLMResponse(text="done")])

        with self.assertRaisesRegex(ValueError, "either driver or provider"):
            Agent(driver=object(), provider=provider)


class StreamingTests(unittest.TestCase):
    def test_chat_stream_yields_deltas_and_keeps_history(self):
        def create(**parameters):
            self.assertTrue(parameters["stream"])
            return iter(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content="hel")
                            )
                        ]
                    ),
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content="lo")
                            )
                        ]
                    ),
                ]
            )

        client = SimpleNamespace(
            responses=object(),
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )
        provider = OpenAIProvider(
            OpenAIProviderConfig(
                model="deepseek-chat",
                base_url="https://api.deepseek.com",
            ),
            client=client,
        )

        chunks = list(
            provider.complete_stream(LLMRequest(task="t", turn=0))
        )

        self.assertEqual(chunks, ["hel", "lo"])
        self.assertEqual(
            provider._chat_messages[-1],
            {"role": "assistant", "content": "hello"},
        )

    def test_responses_stream_yields_text_deltas(self):
        def create(**parameters):
            self.assertTrue(parameters["stream"])
            return iter(
                [
                    SimpleNamespace(
                        type="response.output_text.delta", delta="hi"
                    ),
                    SimpleNamespace(
                        type="response.output_text.delta", delta="!"
                    ),
                    SimpleNamespace(type="response.completed"),
                ]
            )

        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        provider = OpenAIProvider(
            OpenAIProviderConfig(model="m"),
            client=client,
        )

        chunks = list(provider.complete_stream(LLMRequest(task="t")))

        self.assertEqual(chunks, ["hi", "!"])


if __name__ == "__main__":
    unittest.main()
