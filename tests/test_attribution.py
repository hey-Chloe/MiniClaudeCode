import unittest

from evaluation.attribution import (
    FailureAttribution,
    attribute_run,
    generate_attribution_candidates,
)
from evaluation.evolution import StrategyConfig, evolve
from evaluation.coding.tasks import TASKS


def observation(name, success, error=None, policy_action="allow"):
    return {
        "name": name,
        "success": success,
        "error": error,
        "policy_action": policy_action,
    }


def events_for_tool_round(*observations):
    return [{"event": "tool_results", "detail": list(observations)}]


class AttributionTests(unittest.TestCase):
    def test_attributes_failures_recovery_and_policy_denials(self):
        events = [
            *events_for_tool_round(
                observation("grep_files", False, error="timed out"),
                observation("grep_files", True),
            ),
            *events_for_tool_round(
                observation("write_file", False, error="tool blocked: denied"),
                observation("write_file", False, error="tool blocked: denied"),
            ),
        ]
        attribution = attribute_run(events, phases=("plan", "act", "observe", "reflect"))

        self.assertEqual(attribution.failed_tools, ("grep_files", "write_file"))
        self.assertIn("timeout", attribution.error_kinds)
        self.assertIn("policy_denial", attribution.error_kinds)
        self.assertEqual(attribution.policy_denials, 2)
        self.assertEqual(attribution.recoverable_failures, 2)
        self.assertEqual(attribution.recovered_failures, 1)
        self.assertEqual(attribution.recovery_rate, 0.5)
        self.assertEqual(attribution.failed_phase, "reflect")

    def test_attribution_is_empty_for_clean_run(self):
        events = events_for_tool_round(observation("read_file", True))
        attribution = attribute_run(events)

        self.assertEqual(attribution.failed_tools, ())
        self.assertIsNone(attribution.recovery_rate)

    def test_candidates_target_attributed_failures(self):
        base = StrategyConfig(version="v1")
        attribution = FailureAttribution(
            failed_tools=("execute_command",),
            error_kinds=("timeout",),
            recoverable_failures=2,
            recovered_failures=0,
        )
        candidates = generate_attribution_candidates(base, attribution)
        versions = [candidate.version for candidate in candidates]

        self.assertIn("attr-recovery-hint", versions)
        self.assertIn("attr-skill_top_k-2", versions)
        self.assertIn("attr-retry_max_retries-3", versions)
        hint = next(
            candidate for candidate in candidates
            if candidate.version == "attr-recovery-hint"
        )
        self.assertIn("Recovery guidance", hint.system_instructions)
        self.assertIn("execute_command", hint.system_instructions)

    def test_candidate_generation_is_deterministic(self):
        base = StrategyConfig(version="v1")
        attribution = FailureAttribution(
            failed_tools=("read_file",),
            error_kinds=("execution",),
            recoverable_failures=1,
            recovered_failures=0,
        )
        first = generate_attribution_candidates(base, attribution)
        second = generate_attribution_candidates(base, attribution)
        self.assertEqual(
            [candidate.version for candidate in first],
            [candidate.version for candidate in second],
        )

    def test_evolve_can_promote_attribution_candidate(self):
        base = StrategyConfig(version="v1")
        attribution = FailureAttribution(
            failed_tools=("execute_command",),
            error_kinds=("execution",),
            recoverable_failures=2,
            recovered_failures=0,
            policy_denials=0,
        )
        extra = generate_attribution_candidates(base, attribution)
        target = next(
            candidate for candidate in extra
            if candidate.version == "attr-recovery-hint"
        )
        train, holdout = [], []
        # Use a tiny fixed split so the test stays fast and deterministic.
        for index, task in enumerate(TASKS[:4]):
            (train if index % 2 == 0 else holdout).append(task.id)

        def evaluate(strategy, ids):
            if strategy.version == target.version:
                return {
                    "success_rate": 0.9,
                    "average_tokens": 700,
                    "average_latency_seconds": 5.0,
                    "recovery_rate": 1.0,
                    "average_tools_sent_per_turn": 4.0,
                }
            return {
                "success_rate": 0.5,
                "average_tokens": 1000,
                "average_latency_seconds": 10.0,
                "recovery_rate": 0.5,
                "average_tools_sent_per_turn": 4.0,
            }

        run = evolve(
            base,
            evaluate,
            tuple(train),
            tuple(holdout),
            generations=1,
            max_candidates=2,
            extra_candidates=extra,
        )

        self.assertEqual(run.generations[0].decision, "promoted")
        self.assertEqual(run.generations[0].promoted_version, target.version)


if __name__ == "__main__":
    unittest.main()
