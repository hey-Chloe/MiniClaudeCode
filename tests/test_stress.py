import json
import unittest

from evaluation.stress import (
    replay,
    run_stress,
    synthesize_session,
)


class StressReplayTests(unittest.TestCase):
    def test_session_is_deterministic_for_a_seed(self):
        first = synthesize_session(20, seed=11)
        second = synthesize_session(20, seed=11)
        self.assertEqual(first, second)
        self.assertNotEqual(
            synthesize_session(20, seed=12),
            first,
        )

    def test_session_contains_compression_material(self):
        messages = synthesize_session(40, seed=7)
        tool_names = [
            json.loads(message["content"]).get("name")
            for message in messages
            if message["role"] == "tool"
        ]
        self.assertGreater(tool_names.count("workspace_diff"), 1)
        self.assertGreater(tool_names.count("grep_files"), 1)

    def test_compression_reduces_context(self):
        messages = synthesize_session(40, seed=7)
        baseline = replay(messages, compression_layers=(), max_chars=200_000)
        compressed = replay(
            messages,
            compression_layers=("stale_snip", "micro_compact"),
            max_chars=200_000,
        )

        self.assertLess(compressed.chars_out, baseline.chars_out)
        self.assertLess(compressed.messages_out, baseline.messages_out)
        self.assertGreater(compressed.compression["stale_sniped"], 0)
        self.assertGreater(compressed.compression["micro_compacted"], 0)

    def test_report_contains_real_gain_numbers(self):
        payload = run_stress(turns=24, seed=3)
        self.assertEqual(payload["run_type"], "stress_compression")
        self.assertGreater(payload["compression_gain"]["chars_removed"], 0)
        self.assertGreater(payload["compression_gain"]["chars_removed_pct"], 0)
        self.assertIn("baseline", payload["configs"])
        self.assertIn("compressed", payload["configs"])


if __name__ == "__main__":
    unittest.main()
