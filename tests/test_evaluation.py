import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from evaluation.models import BenchmarkCase
from evaluation.runner import BenchmarkConfigurationError, BenchmarkRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = PROJECT_ROOT / "evaluation" / "benchmark.json"


class BenchmarkRunnerTests(unittest.TestCase):
    def test_package_and_project_versions_match_release(self):
        from miniclaude import __version__

        project = tomllib.loads(
            (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(__version__, "5.2.0")
        self.assertEqual(project["project"]["version"], __version__)

    def test_repository_benchmark_passes(self):
        report = BenchmarkRunner().run_file(BENCHMARK)

        self.assertTrue(report.passed)
        self.assertEqual(report.total, 6)
        self.assertEqual(report.passed_cases, 6)
        self.assertTrue(all(report.threshold_results.values()))

    def test_failed_expectation_and_threshold_are_reported(self):
        runner = BenchmarkRunner({"fixed": lambda _: {"value": "actual"}})
        report = runner.run(
            "test",
            {"accuracy": 1.0},
            [BenchmarkCase("case", "accuracy", "fixed", expected={"value": "expected"})],
        )

        self.assertFalse(report.passed)
        self.assertEqual(report.metrics["accuracy"], 0.0)
        self.assertFalse(report.threshold_results["accuracy"])

    def test_unknown_scenario_becomes_failed_case(self):
        report = BenchmarkRunner({}).run(
            "test",
            {},
            [BenchmarkCase("case", "accuracy", "missing")],
        )

        self.assertFalse(report.passed)
        self.assertIn("unknown scenario", report.cases[0].error)

    def test_duplicate_ids_and_invalid_thresholds_are_rejected(self):
        definitions = [
            {
                "version": "test",
                "thresholds": {"accuracy": 2},
                "cases": [{"id": "one", "metric": "accuracy", "scenario": "fixed"}],
            },
            {
                "version": "test",
                "cases": [
                    {"id": "one", "metric": "accuracy", "scenario": "fixed"},
                    {"id": "one", "metric": "accuracy", "scenario": "fixed"},
                ],
            },
        ]
        for definition in definitions:
            with self.subTest(definition=definition):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "benchmark.json"
                    path.write_text(json.dumps(definition), encoding="utf-8")
                    with self.assertRaises(BenchmarkConfigurationError):
                        BenchmarkRunner({"fixed": lambda _: {}}).load(path)

    def test_cli_outputs_json_and_success_exit_code(self):
        completed = subprocess.run(
            [sys.executable, "-m", "evaluation.runner", str(BENCHMARK)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertTrue(report["passed"])
        self.assertEqual(report["passed_cases"], 6)


if __name__ == "__main__":
    unittest.main()
