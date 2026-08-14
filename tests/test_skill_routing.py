import tempfile
import unittest
from pathlib import Path

from evaluation.coding.tasks import TASKS
from evaluation.skill_routing import CATEGORY_EXPECTED_SKILL, evaluate_routing
from miniclaude.skills import SkillRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SkillRoutingTests(unittest.TestCase):
    def test_every_task_category_has_an_expected_skill(self):
        categories = {task.category for task in TASKS}
        self.assertTrue(categories)
        for category in categories:
            self.assertIn(category, CATEGORY_EXPECTED_SKILL)

    def test_evaluate_routing_reports_hit_rates(self):
        registry = SkillRegistry(PROJECT_ROOT / "skills")
        report = evaluate_routing(TASKS, registry)

        self.assertEqual(report["run_type"], "skill_routing")
        self.assertEqual(report["total"], len(TASKS))
        self.assertEqual(len(report["cases"]), len(TASKS))
        self.assertGreaterEqual(report["keyword_hit_rate"], 0.0)
        self.assertLessEqual(report["keyword_hit_rate"], 1.0)
        self.assertGreaterEqual(report["hybrid_hit_rate"], 0.0)
        self.assertLessEqual(report["hybrid_hit_rate"], 1.0)

    def test_evaluate_routing_is_deterministic(self):
        registry = SkillRegistry(PROJECT_ROOT / "skills")
        first = evaluate_routing(TASKS, registry)
        second = evaluate_routing(TASKS, registry)
        self.assertEqual(first["cases"], second["cases"])

    def test_save_report_persists_artifact(self):
        from evaluation.reporting import load_report, save_report

        registry = SkillRegistry(PROJECT_ROOT / "skills")
        report = evaluate_routing(TASKS, registry)
        with tempfile.TemporaryDirectory() as directory:
            target = save_report("skill-routing", report, directory)
            self.assertEqual(load_report(target)["run_type"], "skill_routing")


if __name__ == "__main__":
    unittest.main()
