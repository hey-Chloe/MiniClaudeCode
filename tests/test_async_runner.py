import asyncio
import threading
import time
import unittest
from types import SimpleNamespace

from evaluation.coding.async_runner import (
    _async_provider_factory,
    run_tasks_concurrently,
)
from miniclaude.llm import RunInLoopProvider


class AsyncRunnerTests(unittest.TestCase):
    def test_concurrency_is_bounded_and_all_tasks_run(self):
        lock = threading.Lock()
        state = {"active": 0, "peak": 0}

        def fake_case(task, **kwargs):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.03)
            with lock:
                state["active"] -= 1
            return f"done:{task}"

        import evaluation.coding.async_runner as module

        original = module._run_live_case
        module._run_live_case = fake_case
        try:
            results, peak = asyncio.run(
                run_tasks_concurrently(
                    tuple(range(8)),
                    concurrency=3,
                    case_kwargs={},
                )
            )
        finally:
            module._run_live_case = original

        self.assertEqual(len(results), 8)
        self.assertEqual(peak, 3)
        self.assertEqual(state["peak"], 3)

    def test_invalid_concurrency_is_rejected(self):
        with self.assertRaises(ValueError):
            asyncio.run(
                run_tasks_concurrently(
                    (),
                    concurrency=0,
                    case_kwargs={},
                )
            )

    def test_provider_factory_returns_bridge(self):
        config = SimpleNamespace(
            model="m",
            api_key="k",
            base_url=None,
            timeout=10.0,
            max_retries=1,
        )
        factory = _async_provider_factory(config)

        self.assertIsInstance(factory(), RunInLoopProvider)


if __name__ == "__main__":
    unittest.main()
