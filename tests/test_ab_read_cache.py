import unittest

from evaluation.ab_read_cache import run_ab


class ReadCacheABTests(unittest.TestCase):
    def test_cache_on_hits_and_off_never_hits(self):
        payload = run_ab(repeats=2)
        self.assertEqual(payload["run_type"], "ab_read_cache")
        results = payload["results"]

        self.assertEqual(results["cache_off"]["cache_hit_rate"], 0.0)
        self.assertGreater(results["cache_on"]["cache_hit_rate"], 0.0)
        # The same reads happen in both configs, so the repeated-read rate is
        # identical; the cache changes the cost, not the access pattern.
        self.assertEqual(
            results["cache_off"]["repeated_read_rate"],
            results["cache_on"]["repeated_read_rate"],
        )
        self.assertGreater(results["cache_on"]["total_reads"], 0)


if __name__ == "__main__":
    unittest.main()
