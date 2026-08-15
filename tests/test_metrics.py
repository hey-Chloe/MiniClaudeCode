import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from miniclaude.agent import Agent
from miniclaude.llm import LLMResponse, LLMToolCall, LLMUsage
from miniclaude.metrics import CostCalculator, Pricing, RunMetrics
from miniclaude.models import RunStatus
from miniclaude.tools import ToolDefinition
from miniclaude.trace import Trace


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, request):
        return self.responses.pop(0)


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


class RunMetricsTests(unittest.TestCase):
    def test_from_run_aggregates_trace_and_state(self):
        trace = Trace()
        trace.add(
            "tool_results",
            [
                {
                    "name": "read_file",
                    "success": True,
                    "policy_action": "allow",
                    "arguments": {"path": "a.py"},
                },
                {
                    "name": "read_file",
                    "success": True,
                    "policy_action": "allow",
                    "arguments": {"path": "a.py"},
                },
                {
                    "name": "replace_text",
                    "success": False,
                    "policy_action": "deny",
                    "arguments": {"path": "a.py"},
                },
            ],
        )
        state = SimpleNamespace(
            turn_count=3,
            usage_input_tokens=100,
            usage_output_tokens=20,
            model_name="test-model",
            context_truncated=True,
            skills_loaded=("bug-fix",),
        )
        metrics = RunMetrics.from_run(state, trace, started=__import__("time").time())

        self.assertEqual(metrics.turns, 3)
        self.assertEqual(metrics.tool_calls, 3)
        self.assertEqual(metrics.tool_successes, 2)
        self.assertEqual(metrics.tool_success_rate, 2 / 3)
        self.assertEqual(metrics.policy_actions, {"allow": 2, "deny": 1})
        self.assertEqual(metrics.safety_block_rate, 1 / 3)
        self.assertEqual(metrics.total_reads, 2)
        self.assertEqual(metrics.repeated_reads, 1)
        self.assertEqual(metrics.repeated_read_rate, 0.5)
        self.assertEqual(metrics.input_tokens, 100)
        self.assertEqual(metrics.output_tokens, 20)
        self.assertEqual(metrics.total_tokens, 120)
        self.assertTrue(metrics.context_truncated)
        self.assertEqual(metrics.model_name, "test-model")
        self.assertEqual(metrics.skills_loaded, ("bug-fix",))

    def test_from_run_ignores_arguments_missing_paths(self):
        trace = Trace()
        trace.add(
            "tool_results",
            [{"name": "read_file", "success": True, "policy_action": "allow"}],
        )
        state = SimpleNamespace(
            turn_count=1,
            usage_input_tokens=0,
            usage_output_tokens=0,
            model_name=None,
            context_truncated=False,
            skills_loaded=(),
        )
        metrics = RunMetrics.from_run(state, trace, started=__import__("time").time())
        self.assertEqual(metrics.total_reads, 1)
        self.assertEqual(metrics.repeated_reads, 0)

    def test_from_run_computes_recovery_rate(self):
        trace = Trace()
        trace.add(
            "tool_results",
            [
                {"name": "echo", "success": False, "policy_action": "allow"},
                {"name": "echo", "success": True, "policy_action": "allow"},
            ],
        )
        trace.add(
            "tool_results",
            [
                {"name": "grep_files", "success": False, "policy_action": "allow"},
                {"name": "read_file", "success": True, "policy_action": "allow"},
            ],
        )
        trace.add(
            "tool_results",
            [{"name": "echo", "success": False, "policy_action": "allow"}],
        )
        state = SimpleNamespace(
            turn_count=3,
            usage_input_tokens=0,
            usage_output_tokens=0,
            model_name=None,
            context_truncated=False,
            skills_loaded=(),
        )
        metrics = RunMetrics.from_run(state, trace, started=__import__("time").time())

        self.assertEqual(metrics.recoverable_failures, 2)
        self.assertEqual(metrics.recovered_failures, 1)
        self.assertEqual(metrics.recovery_rate, 0.5)

    def test_recovery_rate_is_none_without_recoverable_failures(self):
        metrics = RunMetrics()
        self.assertIsNone(metrics.recovery_rate)

    def test_from_run_counts_cache_hits_and_compression(self):
        trace = Trace()
        trace.add(
            "tool_results",
            [
                {
                    "name": "read_file",
                    "success": True,
                    "policy_action": "allow",
                    "arguments": {"path": "a.py"},
                    "output": {
                        "path": "a.py",
                        "content": "x",
                        "cache_hit": True,
                    },
                }
            ],
        )
        state = SimpleNamespace(
            turn_count=1,
            usage_input_tokens=0,
            usage_output_tokens=0,
            model_name=None,
            context_truncated=False,
            skills_loaded=(),
            context_compression={"micro_compacted": 2, "chars_removed": 100},
        )
        metrics = RunMetrics.from_run(
            state, trace, started=__import__("time").time()
        )

        self.assertEqual(metrics.cache_hits, 1)
        self.assertEqual(metrics.cache_hit_rate, 1.0)
        self.assertEqual(
            metrics.context_compression,
            {"micro_compacted": 2, "chars_removed": 100},
        )


class CostCalculatorTests(unittest.TestCase):
    def test_estimate_uses_pricing_table(self):
        calculator = CostCalculator(
            {"m1": Pricing(input_per_million=1.0, output_per_million=2.0)}
        )
        self.assertAlmostEqual(calculator.estimate("m1", 1_000_000, 500_000), 2.0)

    def test_unknown_model_returns_none(self):
        calculator = CostCalculator({"m1": Pricing(1.0, 2.0)})
        self.assertIsNone(calculator.estimate("other", 10, 10))
        self.assertIsNone(calculator.estimate(None, 10, 10))

    def test_negative_price_is_rejected(self):
        with self.assertRaises(ValueError):
            Pricing(input_per_million=-1.0, output_per_million=1.0)


class AgentMetricsIntegrationTests(unittest.TestCase):
    def test_agent_result_carries_metrics(self):
        provider = ScriptedProvider(
            [
                LLMResponse(
                    response_id="resp_1",
                    tool_calls=(
                        LLMToolCall("call_1", "echo", '{"text":"hello"}'),
                    ),
                    usage=LLMUsage(input_tokens=10, output_tokens=2, total_tokens=12),
                ),
                LLMResponse(
                    response_id="resp_2",
                    text="done",
                    usage=LLMUsage(input_tokens=8, output_tokens=1, total_tokens=9),
                ),
            ]
        )
        result = Agent(provider=provider, tools=[echo_tool()]).run_result("echo")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.turns, 2)
        metrics = result.metrics
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.tool_calls, 1)
        self.assertEqual(metrics.tool_successes, 1)
        self.assertEqual(metrics.input_tokens, 18)
        self.assertEqual(metrics.output_tokens, 3)
        self.assertEqual(metrics.policy_actions, {"allow": 1})
        self.assertEqual(metrics.model_name, None)
        self.assertEqual(metrics.tools_sent, 2)
        self.assertEqual(metrics.average_tools_per_turn, 1.0)

    def test_cost_calculator_attaches_cost_estimate(self):
        provider = ScriptedProvider(
            [
                LLMResponse(
                    text="done",
                    model="m1",
                    usage=LLMUsage(input_tokens=1_000_000, output_tokens=0),
                )
            ]
        )
        agent = Agent(
            provider=provider,
            cost_calculator=CostCalculator(
                {"m1": Pricing(input_per_million=1.0, output_per_million=1.0)}
            ),
        )
        result = agent.run_result("task")

        self.assertEqual(result.metrics.cost_usd, 1.0)


if __name__ == "__main__":
    unittest.main()
