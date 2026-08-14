import unittest
from types import SimpleNamespace
from unittest.mock import patch

from miniclaude.llm import (
    LLMProviderError,
    LLMRequest,
    OpenAIProvider,
    OpenAIProviderConfig,
)


class RetryTests(unittest.TestCase):
    def test_transient_error_is_retried_and_succeeds(self):
        attempts = []

        class FlakyResponses:
            def create(self, **parameters):
                attempts.append(parameters["model"])
                if len(attempts) == 1:
                    raise TimeoutError("boom")
                return SimpleNamespace(
                    id="resp_1",
                    model="m",
                    output_text="ok",
                    output=[],
                    usage=SimpleNamespace(
                        input_tokens=1, output_tokens=1, total_tokens=2
                    ),
                )

        client = SimpleNamespace(responses=FlakyResponses())
        with patch("miniclaude.llm.openai_provider.time.sleep") as sleep:
            provider = OpenAIProvider(
                OpenAIProviderConfig(
                    model="m",
                    max_retries=2,
                    retry_base_delay=0.01,
                    retry_max_delay=0.01,
                ),
                client=client,
            )
            response = provider.complete(LLMRequest(task="t"))

        self.assertEqual(len(attempts), 2)
        self.assertEqual(response.text, "ok")
        sleep.assert_called_once()

    def test_retries_exhausted_raises_provider_error(self):
        attempts = []

        class FailingResponses:
            def create(self, **parameters):
                attempts.append(1)
                raise TimeoutError("still down")

        client = SimpleNamespace(responses=FailingResponses())
        with patch("miniclaude.llm.openai_provider.time.sleep"):
            provider = OpenAIProvider(
                OpenAIProviderConfig(model="m", max_retries=2),
                client=client,
            )
            with self.assertRaises(LLMProviderError):
                provider.complete(LLMRequest(task="t"))

        self.assertEqual(len(attempts), 3)

    def test_non_retryable_error_raises_immediately(self):
        attempts = []

        class FailingResponses:
            def create(self, **parameters):
                attempts.append(1)
                raise RuntimeError("401 Unauthorized")

        client = SimpleNamespace(responses=FailingResponses())
        provider = OpenAIProvider(
            OpenAIProviderConfig(model="m", max_retries=2),
            client=client,
        )
        with self.assertRaises(LLMProviderError):
            provider.complete(LLMRequest(task="t"))
        self.assertEqual(len(attempts), 1)

    def test_max_retries_zero_disables_retry(self):
        attempts = []

        class FailingResponses:
            def create(self, **parameters):
                attempts.append(1)
                raise TimeoutError("boom")

        client = SimpleNamespace(responses=FailingResponses())
        provider = OpenAIProvider(
            OpenAIProviderConfig(model="m", max_retries=0),
            client=client,
        )
        with self.assertRaises(LLMProviderError):
            provider.complete(LLMRequest(task="t"))
        self.assertEqual(len(attempts), 1)

    def test_jitter_randomizes_delay_within_cap(self):
        attempts = []

        class FlakyResponses:
            def create(self, **parameters):
                attempts.append(1)
                if len(attempts) == 1:
                    raise TimeoutError("boom")
                return SimpleNamespace(
                    id="resp_1",
                    model="m",
                    output_text="ok",
                    output=[],
                    usage=SimpleNamespace(
                        input_tokens=1, output_tokens=1, total_tokens=2
                    ),
                )

        client = SimpleNamespace(responses=FlakyResponses())
        with (
            patch("miniclaude.llm.openai_provider.time.sleep") as sleep,
            patch(
                "miniclaude.llm.openai_provider.random.uniform",
                side_effect=lambda low, high: (low + high) / 2,
            ) as uniform,
        ):
            provider = OpenAIProvider(
                OpenAIProviderConfig(
                    model="m",
                    max_retries=2,
                    retry_base_delay=1.0,
                    retry_max_delay=8.0,
                ),
                client=client,
            )
            response = provider.complete(LLMRequest(task="t"))

        self.assertEqual(response.text, "ok")
        uniform.assert_called_once_with(0, 1.0)
        sleep.assert_called_once_with(0.5)

    def test_jitter_disabled_uses_deterministic_backoff(self):
        attempts = []

        class FailingResponses:
            def create(self, **parameters):
                attempts.append(1)
                raise TimeoutError("still down")

        client = SimpleNamespace(responses=FailingResponses())
        with patch("miniclaude.llm.openai_provider.time.sleep") as sleep:
            provider = OpenAIProvider(
                OpenAIProviderConfig(
                    model="m",
                    max_retries=3,
                    retry_base_delay=0.01,
                    retry_max_delay=8.0,
                    retry_jitter=False,
                ),
                client=client,
            )
            with self.assertRaises(LLMProviderError):
                provider.complete(LLMRequest(task="t"))

        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [0.01, 0.02, 0.04],
        )

    def test_jitter_enabled_by_default(self):
        self.assertTrue(OpenAIProviderConfig(model="m").retry_jitter)

    def test_restore_seeds_chat_history(self):
        provider = OpenAIProvider(
            OpenAIProviderConfig(model="m"),
            client=SimpleNamespace(responses=object(), chat=object()),
        )
        provider.restore(
            (
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            )
        )
        self.assertEqual(
            provider._chat_messages,
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
