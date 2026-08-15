import unittest

from evaluation.ab_concurrency import run_benchmark
from evaluation.ab_tool_gating import measure_task, run_ab


class ToolGatingABTests(unittest.TestCase):
    def test_gating_never_sends_more_schemas_than_full_toolset(self):
        payload = run_ab()
        self.assertEqual(payload["totals"]["tasks"], 26)
        for case in payload["cases"]:
            self.assertLessEqual(
                case["schema_chars_gated"],
                case["schema_chars_full"],
            )
            self.assertLessEqual(
                case["tools_sent_gated"],
                case["tools_total"],
            )

    def test_report_has_aggregates_and_categories(self):
        payload = run_ab()
        self.assertEqual(payload["run_type"], "ab_tool_gating")
        self.assertGreaterEqual(payload["totals"]["gated_tasks"], 0)
        self.assertIn("per_category", payload)
        self.assertGreater(len(payload["per_category"]), 0)

    def test_measure_task_reuses_real_activation_path(self):
        # measure_task is exercised through run_ab; here we only assert the
        # helper signature contract is stable for direct reuse.
        import tempfile
        from pathlib import Path

        from miniclaude.context import ContextConfig, ContextManager
        from miniclaude.runtime_tools import create_runtime_tools
        from miniclaude.tools import ToolRegistry
        from runtime.local import LocalProcessRuntime
        from security.approval import ApprovalManager

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            runtime = LocalProcessRuntime(workspace)
            registry = ToolRegistry(
                approvals=ApprovalManager(lambda *_: True)
            )
            for tool in create_runtime_tools(runtime):
                registry.register(tool)
            context = ContextManager(ContextConfig())
            result = measure_task("fix the failing test", registry, context)
            self.assertIn("tools_sent_gated", result)
            self.assertIn("schema_chars_full", result)


class ConcurrencyABTests(unittest.TestCase):
    def test_benchmark_reports_both_modes(self):
        payload = run_benchmark(repeats=2)
        self.assertEqual(payload["run_type"], "ab_concurrency")
        self.assertIn("sequential", payload["results"])
        self.assertIn("pooled", payload["results"])
        self.assertGreater(payload["batch"]["calls"], 1)
        self.assertGreater(
            payload["results"]["sequential"]["median_seconds"], 0
        )
        self.assertGreater(payload["results"]["pooled"]["median_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
