import unittest
from types import SimpleNamespace

from miniclaude.llm import (
    AnthropicProvider,
    AnthropicProviderConfig,
    LLMProviderError,
    LLMRequest,
)


class AnthropicProviderTests(unittest.TestCase):
    def _provider(self, responses):
        calls = []

        class Messages:
            def create(self, **parameters):
                calls.append(parameters)
                return responses.pop(0)

        client = SimpleNamespace(messages=Messages())
        return (
            AnthropicProvider(
                AnthropicProviderConfig(model="claude-x", api_key="k"),
                client=client,
            ),
            calls,
        )

    def test_text_response_is_normalized(self):
        provider, _ = self._provider(
            [
                SimpleNamespace(
                    id="msg_1",
                    model="claude-x",
                    content=[{"type": "text", "text": "hello"}],
                    usage={"input_tokens": 10, "output_tokens": 5},
                    stop_reason="end_turn",
                )
            ]
        )

        response = provider.complete(
            LLMRequest(task="hi", instructions="be brief")
        )

        self.assertEqual(response.text, "hello")
        self.assertEqual(response.tool_calls, ())
        self.assertEqual(response.usage.input_tokens, 10)
        self.assertEqual(response.usage.output_tokens, 5)

    def test_tool_use_response_is_normalized(self):
        provider, _ = self._provider(
            [
                SimpleNamespace(
                    id="msg_1",
                    model="claude-x",
                    content=[
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "echo",
                            "input": {"text": "hi"},
                        }
                    ],
                    usage={"input_tokens": 3, "output_tokens": 2},
                    stop_reason="tool_use",
                )
            ]
        )

        response = provider.complete(LLMRequest(task="echo"))

        self.assertEqual(response.text, "")
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].call_id, "call_1")
        self.assertEqual(response.tool_calls[0].name, "echo")
        self.assertIn('"text"', response.tool_calls[0].arguments)

    def test_complete_stream_yields_text_deltas(self):
        provider, calls = self._provider(
            [
                iter(
                    [
                        {"type": "message_start"},
                        {
                            "type": "content_block_delta",
                            "delta": {"type": "text_delta", "text": "hel"},
                        },
                        {
                            "type": "content_block_delta",
                            "delta": {"type": "text_delta", "text": "lo"},
                        },
                        {
                            "type": "content_block_delta",
                            "delta": {"type": "input_json_delta", "partial_json": "{}"},
                        },
                        {"type": "message_stop"},
                    ]
                )
            ]
        )

        deltas = list(
            provider.complete_stream(LLMRequest(task="hi"))
        )

        self.assertEqual(deltas, ["hel", "lo"])
        self.assertTrue(calls[0]["stream"])

    def test_tool_results_are_attached_to_tool_use_blocks(self):
        provider, calls = self._provider(
            [
                SimpleNamespace(
                    id="msg_1",
                    model="claude-x",
                    content=[
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "echo",
                            "input": {"text": "hi"},
                        }
                    ],
                    usage={"input_tokens": 3, "output_tokens": 2},
                    stop_reason="tool_use",
                ),
                SimpleNamespace(
                    id="msg_2",
                    model="claude-x",
                    content=[{"type": "text", "text": "done"}],
                    usage={"input_tokens": 5, "output_tokens": 1},
                    stop_reason="end_turn",
                ),
            ]
        )

        provider.complete(LLMRequest(task="echo"))
        response = provider.complete(
            LLMRequest(
                task="echo",
                turn=1,
                tool_outputs=(
                    {"call_id": "call_1", "output": "hello"},
                ),
            )
        )

        self.assertEqual(response.text, "done")
        messages = calls[1]["messages"]
        self.assertEqual(messages[-2]["role"], "assistant")
        self.assertEqual(messages[-2]["content"][0]["type"], "tool_use")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"][0]["type"], "tool_result")
        self.assertEqual(
            messages[-1]["content"][0]["tool_use_id"], "call_1"
        )

    def test_transient_error_is_retried(self):
        attempts = []

        class Messages:
            def create(self, **parameters):
                attempts.append(1)
                if len(attempts) == 1:
                    error = RuntimeError("rate limited")
                    error.status_code = 429
                    raise error
                return SimpleNamespace(
                    id="msg_1",
                    model="claude-x",
                    content=[{"type": "text", "text": "ok"}],
                    usage={"input_tokens": 1, "output_tokens": 1},
                    stop_reason="end_turn",
                )

        from unittest.mock import patch

        client = SimpleNamespace(messages=Messages())
        with patch("miniclaude.llm.anthropic_provider.time.sleep"):
            provider = AnthropicProvider(
                AnthropicProviderConfig(
                    model="claude-x",
                    max_retries=2,
                    retry_base_delay=0.01,
                    retry_max_delay=0.01,
                    retry_jitter=False,
                ),
                client=client,
            )
            response = provider.complete(LLMRequest(task="t"))

        self.assertEqual(len(attempts), 2)
        self.assertEqual(response.text, "ok")

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            AnthropicProviderConfig(model="  ")
        with self.assertRaises(ValueError):
            AnthropicProviderConfig(model="m", max_retries=-1)

    def test_missing_sdk_raises_provider_error(self):
        from unittest.mock import patch

        with patch(
            "builtins.__import__",
            side_effect=ImportError("no module named anthropic"),
        ):
            with self.assertRaises(LLMProviderError):
                AnthropicProvider._create_client(
                    AnthropicProviderConfig(model="m")
                )


if __name__ == "__main__":
    unittest.main()
