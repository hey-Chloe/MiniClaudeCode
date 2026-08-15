import unittest

from evaluation.coding.tasks import TASKS
from evaluation.evolution import (
    StrategyConfig,
    default_splits,
    estimate_context_chars,
    evolve,
    generate_candidates,
)


def score(success, tokens, latency=10.0, recovery=0.5):
    return {
        "success_rate": success,
        "average_tokens": tokens,
        "average_latency_seconds": latency,
        "recovery_rate": recovery,
        "average_tools_sent_per_turn": 4.0,
    }


def split_aware_evaluate(train_scores, holdout_scores, holdout_ids):
    def evaluate(strategy, ids):
        table = holdout_scores if ids == holdout_ids else train_scores
        return table.get(strategy.version, score(0.1, 9999))

    return evaluate


class StrategyEvolutionTests(unittest.TestCase):
    def test_default_splits_cover_all_tasks_and_do_not_overlap(self):
        train, holdout = default_splits()
        task_ids = {task.id for task in TASKS}

        self.assertEqual(len(train) + len(holdout), len(task_ids))
        self.assertEqual(set(train) & set(holdout), set())
        self.assertEqual(set(train) | set(holdout), task_ids)
        self.assertTrue(train)
        self.assertTrue(holdout)

    def test_candidate_generation_is_deterministic(self):
        base = StrategyConfig(version="v1")
        first = generate_candidates(base)
        second = generate_candidates(base)

        self.assertEqual(
            [candidate.version for candidate in first],
            [candidate.version for candidate in second],
        )
        self.assertEqual(first[0].version, "v1")
        self.assertGreater(len(first), 1)
        self.assertNotIn(
            "v1", [candidate.version for candidate in first[1:]]
        )

    def test_mutations_cover_routing_and_cache_dimensions(self):
        base = StrategyConfig(version="v1")
        versions = {
            candidate.version
            for candidate in generate_candidates(base, max_candidates=20)
        }

        self.assertIn("cand-routing_mode-keyword", versions)
        self.assertIn("cand-routing_mode-semantic", versions)
        self.assertIn("cand-read_cache_enabled-False", versions)

    def test_invalid_routing_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            StrategyConfig(version="bad", routing_mode="vector")

    def test_evolve_promotes_candidate_that_improves_holdout(self):
        base = StrategyConfig(version="v1")
        candidate_version = "cand-skill_top_k-2"
        train, holdout = default_splits()
        evaluate = split_aware_evaluate(
            {"v1": score(0.5, 1000), candidate_version: score(0.7, 800)},
            {"v1": score(0.5, 1000), candidate_version: score(0.7, 800)},
            holdout,
        )

        run = evolve(
            base,
            evaluate,
            train,
            holdout,
        )

        self.assertEqual(len(run.generations), 1)
        generation = run.generations[0]
        self.assertEqual(generation.decision, "promoted")
        self.assertEqual(generation.promoted_version, candidate_version)
        self.assertEqual(run.final_version, candidate_version)

    def test_evolve_keeps_base_when_holdout_regresses(self):
        base = StrategyConfig(version="v1")
        candidate_version = "cand-skill_top_k-2"
        train, holdout = default_splits()
        evaluate = split_aware_evaluate(
            {"v1": score(0.6, 1000), candidate_version: score(0.9, 700)},
            {"v1": score(0.6, 1000), candidate_version: score(0.4, 700)},
            holdout,
        )

        run = evolve(
            base,
            evaluate,
            train,
            holdout,
        )

        self.assertEqual(run.generations[0].decision, "kept_base")
        self.assertEqual(run.final_version, "v1")

    def test_evolve_keeps_base_when_base_is_best(self):
        base = StrategyConfig(version="v1")
        train, holdout = default_splits()
        evaluate = split_aware_evaluate(
            {"v1": score(0.9, 500), "cand-skill_top_k-2": score(0.4, 400)},
            {"v1": score(0.9, 500), "cand-skill_top_k-2": score(0.4, 400)},
            holdout,
        )

        run = evolve(
            base,
            evaluate,
            train,
            holdout,
        )

        self.assertEqual(run.generations[0].decision, "kept_base")
        self.assertEqual(run.final_version, "v1")

    def test_multi_generation_chains_promotions(self):
        base = StrategyConfig(version="v1")
        candidate = generate_candidates(base)[1]
        train, holdout = default_splits()
        evaluate = split_aware_evaluate(
            {"v1": score(0.5, 1000), candidate.version: score(0.7, 700)},
            {"v1": score(0.5, 1000), candidate.version: score(0.7, 700)},
            holdout,
        )

        run = evolve(
            base,
            evaluate,
            train,
            holdout,
            generations=3,
        )

        self.assertEqual(run.generations[0].decision, "promoted")
        self.assertEqual(run.generations[0].promoted_version, candidate.version)
        self.assertEqual(run.generations[1].decision, "kept_base")
        self.assertEqual(len(run.generations), 2)
        self.assertEqual(run.final_version, candidate.version)

    def test_estimate_context_chars_reflects_gating(self):
        gated = StrategyConfig(version="gated", tool_gating=True)
        full = StrategyConfig(version="full", tool_gating=False)

        gated_estimate = estimate_context_chars(
            gated, "fix tests", ("read_file", "grep_files")
        )
        full_estimate = estimate_context_chars(
            full, "fix tests", ("read_file", "grep_files")
        )

        self.assertLess(gated_estimate["tools_sent"], full_estimate["tools_sent"])
        self.assertLess(
            gated_estimate["schema_chars"], full_estimate["schema_chars"]
        )

    def test_invalid_generation_count_is_rejected(self):
        base = StrategyConfig(version="v1")
        with self.assertRaises(ValueError):
            evolve(base, lambda strategy, ids: score(0.5, 1), (), (), generations=0)


if __name__ == "__main__":
    unittest.main()
