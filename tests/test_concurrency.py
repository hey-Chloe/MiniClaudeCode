import threading
import time
import unittest

from miniclaude.agent import Agent
from miniclaude.llm import LLMResponse, LLMToolCall
from miniclaude.models import RunStatus
from miniclaude.tools import ToolDefinition
from security.policy import ToolRisk


def _tool(name, handler, risk=ToolRisk.READ_ONLY):
    return ToolDefinition(
        name=name,
        description=f"Tool {name}.",
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        handler=handler,
        risk=risk,
    )


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, request):
        return self.responses.pop(0)


class ConcurrencyProbe:
    """Tracks the peak number of handlers running at once."""

    def __init__(self):
        self.lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def __call__(self):
        with self.lock:
            self.current += 1
            self.peak = max(self.peak, self.current)
        time.sleep(0.1)
        with self.lock:
            self.current -= 1
        return "ok"


class ConcurrentDispatchTests(unittest.TestCase):
    def test_read_only_batch_runs_concurrently(self):
        probe = ConcurrencyProbe()

        provider = ScriptedProvider(
            [
                LLMResponse(
                    response_id="resp_1",
                    tool_calls=(
                        LLMToolCall("call_1", "a_tool", "{}"),
                        LLMToolCall("call_2", "b_tool", "{}"),
                    ),
                ),
                LLMResponse(response_id="resp_2", text="done"),
            ]
        )

        result = Agent(
            provider=provider,
            tools=[_tool("a_tool", probe), _tool("b_tool", probe)],
        ).run_result("parallel")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertTrue(
            all(
                observation["success"]
                for event in result.events
                if event.get("event") == "tool_results"
                for observation in event.get("detail", [])
            )
        )
        self.assertEqual(result.metrics.parallel_batches, 1)
        self.assertEqual(result.metrics.max_parallelism, 2)
        self.assertEqual(probe.peak, 2)

    def test_mutating_batch_stays_sequential(self):
        probe = ConcurrencyProbe()

        provider = ScriptedProvider(
            [
                LLMResponse(
                    response_id="resp_1",
                    tool_calls=(
                        LLMToolCall("call_1", "write_a", "{}"),
                        LLMToolCall("call_2", "write_b", "{}"),
                    ),
                ),
                LLMResponse(response_id="resp_2", text="done"),
            ]
        )

        result = Agent(
            provider=provider,
            tools=[
                _tool("write_a", probe, risk=ToolRisk.MUTATING),
                _tool("write_b", probe, risk=ToolRisk.MUTATING),
            ],
            approval_callback=lambda *_: True,
        ).run_result("sequential writes")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.metrics.parallel_batches, 0)
        self.assertEqual(result.metrics.max_parallelism, 1)
        self.assertEqual(probe.peak, 1)


if __name__ == "__main__":
    unittest.main()
