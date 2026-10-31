#!/usr/bin/env python3
"""CPU-only regression checks for expert reporting; no simulator required."""

import unittest

from eval_dexterity_expert import summarize


class ReportingTest(unittest.TestCase):
    def test_timeouts_and_censoring_are_not_failures(self):
        rows = []
        for ended, timeout, censored, success in (
            (True, False, False, 0),
            (False, True, False, 1),
            (False, False, True, 0),
        ):
            rows.append({
                "terminated": ended, "truncated": timeout, "censored": censored,
                "duration_s": 10, "return": 1,
                "metrics": {"episode_success": success, "revolutions": success * 2},
            })
        result = summarize(rows)
        self.assertAlmostEqual(result["failure_rate"], 1 / 3)
        self.assertAlmostEqual(result["timeout_rate"], 1 / 3)
        self.assertAlmostEqual(result["censored_rate"], 1 / 3)
        self.assertAlmostEqual(result["success_rate"], 1 / 3)
        self.assertAlmostEqual(result["mean_final_command_metrics"]["revolutions"], 2 / 3)

    def test_empty_evaluation_is_not_success(self):
        with self.assertRaises(ValueError):
            summarize([])


if __name__ == "__main__":
    unittest.main()
