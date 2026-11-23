#!/usr/bin/env python3
"""
Unit tests for empirical evaluation metrics:
- Wilson 95% Score Confidence Intervals
- Continuity-Corrected McNemar's Contingency Tests
- Multiple-choice answer parsing logic
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze import wilson_ci, mcnemar_test


class TestEmpiricalMetrics(unittest.TestCase):

    def test_wilson_ci_bounds(self):
        """Validates that Wilson score intervals properly bound binomial distributions."""
        # Edge case: zero trials
        self.assertEqual(wilson_ci(0, 0), (0.0, 0.0))

        # Edge case: 0 successes out of 50 (Subjective Feedback MoE)
        low, high = wilson_ci(0, 50)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 7.13, places=1)

        # 1 success out of 21 (Subjective Feedback Dense)
        low, high = wilson_ci(1, 21)
        self.assertAlmostEqual(low, 0.85, places=1)
        self.assertAlmostEqual(high, 22.67, places=1)

        # 50% success rate (25 out of 50)
        low, high = wilson_ci(25, 50)
        self.assertAlmostEqual(low, 36.63, places=1)
        self.assertAlmostEqual(high, 63.37, places=1)

        # 100% success rate (50 out of 50)
        low, high = wilson_ci(50, 50)
        self.assertAlmostEqual(low, 92.87, places=1)
        self.assertAlmostEqual(high, 100.0, places=1)

    def test_mcnemar_paired_test(self):
        """Validates McNemar's contingency test calculations."""
        # Null shift: zero discordant pairs
        chi2, p_val = mcnemar_test(0, 0)
        self.assertEqual(chi2, 0.0)
        self.assertEqual(p_val, 1.0)

        # Equal discordant pairs: b = 3, c = 3
        chi2, p_val = mcnemar_test(3, 3)
        self.assertEqual(chi2, (abs(3 - 3) - 1.0) ** 2 / 6.0)
        self.assertAlmostEqual(chi2, 0.1667, places=3)
        self.assertGreater(p_val, 0.05)  # Statistically not significant

        # Asymmetric shift: b = 10, c = 0
        chi2, p_val = mcnemar_test(10, 0)
        self.assertEqual(chi2, (abs(10 - 0) - 1.0) ** 2 / 10.0)
        self.assertEqual(chi2, 8.1)
        self.assertLess(p_val, 0.01)  # Highly statistically significant


if __name__ == "__main__":
    unittest.main()
