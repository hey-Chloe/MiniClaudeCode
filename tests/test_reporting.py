import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluation.reporting import (
    compare,
    compare_markdown,
    load_report,
    save_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SaveReportTests(unittest.TestCase):
    def test_save_report_writes_versioned_and_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            reports_dir = Path(directory)
            payload = {"run_type": "live", "total": 2, "passed": 1}
            target = save_report("benchmark-live", payload, reports_dir)

            self.assertEqual(target.parent, reports_dir)
            self.assertTrue(target.name.startswith("benchmark-live-"))
            self.assertTrue(target.name.endswith(".json"))
            self.assertEqual(load_report(target), payload)

            latest = reports_dir / "latest-benchmark-live.json"
            self.assertTrue(latest.exists())
            self.assertEqual(load_report(latest), payload)

    def test_identical_payload_produces_same_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            reports_dir = Path(directory)
            payload = {"total": 1, "passed": 1}
            first = save_report("benchmark-validate", payload, reports_dir)
            second = save_report("benchmark-validate", payload, reports_dir)
            self.assertEqual(first.name, second.name)

    def test_invalid_name_is_rejected(self):
        with self.assertRaises(ValueError):
            save_report("   ", {"total": 1})


class CompareTests(unittest.TestCase):
    def test_compare_reports_numeric_deltas(self):
        delta = compare(
            {
                "total": 26,
                "passed": 20,
                "task_success_rate": 0.769,
                "cases": [{"id": "x", "passed": True}],
            },
            {
                "total": 26,
                "passed": 23,
                "task_success_rate": 0.885,
                "cases": [{"id": "x", "passed": True}],
            },
        )

        self.assertEqual(delta["changes"]["passed"]["delta"], 3)
        self.assertAlmostEqual(
            delta["changes"]["task_success_rate"]["percent"], 15.08, places=1
        )
        self.assertNotIn("cases", delta["compared"])
        self.assertNotIn("cases", delta["changes"])

    def test_compare_accepts_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.json"
            right = root / "right.json"
            left.write_text(json.dumps({"turns": 4}), encoding="utf-8")
            right.write_text(json.dumps({"turns": 5}), encoding="utf-8")

            delta = compare(left, right)

        self.assertEqual(delta["changes"]["turns"]["delta"], 1)
        self.assertEqual(delta["compared"], ["turns"])

    def test_compare_markdown_renders_table(self):
        delta = {
            "left": "a.json",
            "right": "b.json",
            "compared": ["passed"],
            "changes": {
                "passed": {"left": 20, "right": 23, "delta": 3, "percent": 15.0}
            },
        }
        rendered = compare_markdown(delta)
        self.assertIn("| metric | left | right | delta | % change |", rendered)
        self.assertIn("| passed |", rendered)
        self.assertIn("+15.00%", rendered)


class ReportingCliTests(unittest.TestCase):
    def test_cli_compare_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.json"
            right = root / "right.json"
            left.write_text(json.dumps({"a": 1}), encoding="utf-8")
            right.write_text(json.dumps({"a": 2}), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evaluation.reporting",
                    "compare",
                    "--left",
                    str(left),
                    "--right",
                    str(right),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["changes"]["a"]["delta"], 1)


if __name__ == "__main__":
    unittest.main()
