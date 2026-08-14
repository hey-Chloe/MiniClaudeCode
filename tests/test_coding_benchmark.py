import json
import tempfile
import unittest
from pathlib import Path

from evaluation.coding.checkers import (
    CHECKERS,
    VALIDATION_EXCLUDED,
    CheckContext,
)
from evaluation.coding.models import CodingCaseResult, CodingReport
from evaluation.coding.runner import (
    compute_first_pass,
    materialize,
    snapshot_files,
    validate_tasks,
)
from evaluation.coding.tasks import TASKS


class TaskCatalogTests(unittest.TestCase):
    def test_task_count_and_ids(self):
        self.assertGreaterEqual(len(TASKS), 20)
        self.assertLessEqual(len(TASKS), 30)
        ids = [task.id for task in TASKS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_categories_are_covered(self):
        categories = {task.category for task in TASKS}
        self.assertEqual(
            categories,
            {
                "failing_test_fix",
                "small_feature",
                "code_search",
                "safe_refactor",
                "config_repair",
                "dependency_issue",
                "permission_security",
            },
        )

    def test_every_task_references_valid_checkers(self):
        for task in TASKS:
            self.assertTrue(task.files, task.id)
            self.assertTrue(task.ground_truth, task.id)
            for name in task.ground_truth:
                self.assertIn(name, CHECKERS, f"{task.id}: {name}")
            for name in task.pre_checks:
                self.assertIn(name, CHECKERS, f"{task.id}: {name}")

    def test_validate_only_passes_representative_tasks(self):
        selected = tuple(
            task
            for task in TASKS
            if task.id
            in {"fix-add-negatives", "feat-chunk", "config-fix-json"}
        )
        report = validate_tasks(selected)
        self.assertEqual(report["passed"], report["total"])


class CheckerTests(unittest.TestCase):
    def test_contains_and_json_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "config.json").write_text(
                '{"retries": 5}\n', encoding="utf-8"
            )
            ctx = CheckContext(workspace, (), "", {})

            self.assertTrue(
                CHECKERS["contains"](
                    ctx, {"path": "config.json", "text": '"retries": 5'}
                )
            )
            self.assertTrue(CHECKERS["json_valid"](ctx, {"path": "config.json"}))
            self.assertFalse(
                CHECKERS["contains"](
                    ctx, {"path": "config.json", "text": "missing"}
                )
            )

    def test_diff_limited_and_no_side_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            initial = {"a.py": "old\n"}
            (workspace / "a.py").write_text("new\n", encoding="utf-8")
            ctx = CheckContext(workspace, (), "", initial)

            self.assertTrue(
                CHECKERS["diff_limited"](
                    ctx, {"allowed_files": ["a.py"]}
                )
            )
            self.assertFalse(
                CHECKERS["diff_limited"](
                    ctx, {"allowed_files": ["other.py"]}
                )
            )
            self.assertFalse(CHECKERS["no_side_effect"](ctx, {}))

    def test_policy_observed_and_path_escape(self):
        observations = (
            {"name": "write_file", "policy_action": "ask", "success": True},
            {"name": "read_file", "policy_action": "allow", "success": True},
            {
                "name": "read_file",
                "policy_action": None,
                "success": False,
                "error": "path escapes workspace: ../x",
            },
        )
        ctx = CheckContext(Path.cwd(), observations, "", {})
        self.assertTrue(
            CHECKERS["policy_observed"](
                ctx,
                {"expected": {"write_file": "ask"}, "mode": "any"},
            )
        )
        self.assertFalse(
            CHECKERS["policy_observed"](
                ctx,
                {"expected": {"write_file": "deny"}, "mode": "any"},
            )
        )
        self.assertTrue(CHECKERS["path_escape_blocked"](ctx, {}))


class FirstPassTests(unittest.TestCase):
    def test_green_then_edit_is_not_first_pass(self):
        events = [
            {
                "event": "tool_results",
                "detail": [
                    {
                        "name": "execute_command",
                        "success": True,
                        "output": {"argv": ["pytest"], "exit_code": 0},
                    }
                ],
            },
            {
                "event": "tool_results",
                "detail": [
                    {
                        "name": "replace_text",
                        "success": True,
                        "output": None,
                    }
                ],
            },
        ]
        self.assertFalse(compute_first_pass(events))

    def test_red_then_green_is_first_pass(self):
        events = [
            {
                "event": "tool_results",
                "detail": [
                    {
                        "name": "execute_command",
                        "success": False,
                        "output": {"argv": ["pytest"], "exit_code": 1},
                    }
                ],
            },
            {
                "event": "tool_results",
                "detail": [
                    {
                        "name": "execute_command",
                        "success": True,
                        "output": {"argv": ["pytest"], "exit_code": 0},
                    }
                ],
            },
        ]
        self.assertTrue(compute_first_pass(events))

    def test_no_pytest_is_not_first_pass(self):
        events = [
            {
                "event": "tool_results",
                "detail": [
                    {
                        "name": "execute_command",
                        "success": True,
                        "output": {"argv": ["git", "diff"], "exit_code": 0},
                    }
                ],
            }
        ]
        self.assertFalse(compute_first_pass(events))


class ReportAggregationTests(unittest.TestCase):
    def _case(self, passed=True, turns=3, tool_calls=4, tool_successes=4, **overrides):
        fields = dict(
            id="c1",
            category="failing_test_fix",
            status="completed" if passed else "max_turns",
            passed=passed,
            checks={"tests_pass": passed},
            turns=turns,
            tool_calls=tool_calls,
            tool_successes=tool_successes,
            tool_success_rate=tool_successes / tool_calls,
            policy_actions={"allow": 4},
            total_reads=2,
            repeated_reads=1,
            repeated_read_rate=0.5,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            context_truncated=False,
            latency_seconds=2.0,
            cost_usd=0.01,
            skills_loaded=(),
            first_pass=True,
            security_blocks=0,
            policy_checks_matched=1,
            policy_checks_total=1,
            recoverable_failures=0,
            recovered_failures=0,
            recovery_rate=None,
        )
        fields.update(overrides)
        return CodingCaseResult(**fields)

    def test_aggregate_math(self):
        cases = (self._case(), self._case(passed=False, turns=5, cost_usd=0.02))
        report = CodingReport.aggregate(cases, run_type="live", model="m1")

        self.assertEqual(report.total, 2)
        self.assertEqual(report.passed, 1)
        self.assertEqual(report.task_success_rate, 0.5)
        self.assertEqual(report.tool_success_rate, 1.0)
        self.assertEqual(report.average_turns, 4.0)
        self.assertEqual(report.average_tokens, 15.0)
        self.assertEqual(report.total_cost_usd, 0.03)
        self.assertEqual(report.approval_accuracy, 1.0)
        self.assertEqual(report.security_blocks, 0)
        self.assertIsNone(report.recovery_rate)
        self.assertEqual(report.average_tools_sent_per_turn, 0.0)

    def test_aggregate_recovery_rate(self):
        cases = (
            self._case(
                recoverable_failures=2,
                recovered_failures=1,
                recovery_rate=0.5,
            ),
            self._case(
                recoverable_failures=1,
                recovered_failures=1,
                recovery_rate=1.0,
            ),
        )
        report = CodingReport.aggregate(cases, run_type="live", model="m1")
        self.assertEqual(report.recovery_rate, 2 / 3)

    def test_materialize_and_snapshot(self):
        task = next(task for task in TASKS if task.id == "fix-add-negatives")
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            materialize(task, workspace)
            files = snapshot_files(workspace)
            self.assertIn("calculator.py", files)
            self.assertIn("tests/test_calculator.py", files)


if __name__ == "__main__":
    unittest.main()
