import asyncio
import unittest
from types import SimpleNamespace

from miniclaude.llm import (
    AsyncOpenAIProvider,
    OpenAIProviderConfig,
    RunInLoopProvider,
)
from miniclaude.llm.base import LLMRequest


def responses_response(text="hello"):
    return SimpleNamespace(
        id="resp_1",
        model="m",
        output_text=text,
        output=[],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        ),
    )


def chat_response(text="hello", tool_calls=()):
    return SimpleNamespace(
        id="chat_1",
        model="m",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=tool_calls)
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=3,
            completion_tokens=2,
            total_tokens=5,
        ),
    )


class AsyncIterator:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


class AsyncOpenAIProviderTests(unittest.TestCase):
    def test_responses_path_acomplete(self):
        async def create(**parameters):
            return responses_response()

        client = SimpleNamespace(
            responses=SimpleNamespace(create=create)
        )
        provider = AsyncOpenAIProvider(
            OpenAIProviderConfig(model="m", api_key="k"),
            client=client,
        )

        response = asyncio.run(provider.acomplete(LLMRequest(task="hi")))

        self.assertEqual(response.text, "hello")
        self.assertEqual(response.usage.input_tokens, 10)
        self.assertEqual(response.usage.output_tokens, 5)

    def test_chat_path_acomplete(self):
        async def create(**parameters):
            return chat_response()

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )
        provider = AsyncOpenAIProvider(
            OpenAIProviderConfig(
                model="m",
                api_key="k",
                base_url="https://api.deepseek.com/v1",
            ),
            client=client,
        )

        response = asyncio.run(provider.acomplete(LLMRequest(task="hi")))

        self.assertEqual(response.text, "hello")
        self.assertEqual(response.tool_calls, ())

    def test_chat_stream_yields_deltas(self):
        async def create(**parameters):
            return AsyncIterator(
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
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )
        provider = AsyncOpenAIProvider(
            OpenAIProviderConfig(
                model="m",
                api_key="k",
                base_url="https://api.deepseek.com/v1",
            ),
            client=client,
        )

        async def collect():
            return [
                delta
                async for delta in provider.acomplete_stream(
                    LLMRequest(task="hi")
                )
            ]

        self.assertEqual(asyncio.run(collect()), ["hel", "lo"])


class RunInLoopProviderTests(unittest.TestCase):
    def test_bridges_async_complete_to_sync_contract(self):
        async def create(**parameters):
            return responses_response()

        client = SimpleNamespace(
            responses=SimpleNamespace(create=create)
        )
        async_provider = AsyncOpenAIProvider(
            OpenAIProviderConfig(model="m", api_key="k"),
            client=client,
        )
        bridge = RunInLoopProvider(async_provider)

        response = bridge.complete(LLMRequest(task="hi"))

        self.assertEqual(response.text, "hello")
        self.assertEqual(response.usage.total_tokens, 15)

    def test_bridges_stream(self):
        async def create(**parameters):
            return AsyncIterator(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content="ab")
                            )
                        ]
                    )
                ]
            )

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )
        async_provider = AsyncOpenAIProvider(
            OpenAIProviderConfig(
                model="m",
                api_key="k",
                base_url="https://api.deepseek.com/v1",
            ),
            client=client,
        )
        bridge = RunInLoopProvider(async_provider)

        self.assertEqual(
            list(bridge.complete_stream(LLMRequest(task="hi"))),
            ["ab"],
        )

    def test_state_helpers_delegate(self):
        class Stateful:
            def __init__(self):
                self.state = None

            async def acomplete(self, request):
                return responses_response()

            def export_state(self):
                return {"chat_messages": ["x"]}

            def restore_state(self, state):
                self.state = state

        stateful = Stateful()
        bridge = RunInLoopProvider(stateful)

        self.assertEqual(bridge.export_state(), {"chat_messages": ["x"]})
        bridge.restore_state({"chat_messages": ["y"]})
        self.assertEqual(stateful.state, {"chat_messages": ["y"]})


if __name__ == "__main__":
    unittest.main()
