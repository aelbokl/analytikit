import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from scipy import stats


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytikit import stat_kit as sk


def _contains_zero(result):
    lower, upper = result["CI"]
    lower_inclusive, upper_inclusive = result["CI_inclusive"]
    above_lower = lower < 0 or (lower == 0 and lower_inclusive)
    below_upper = upper > 0 or (upper == 0 and upper_inclusive)
    return above_lower and below_upper


class NonParametricInferenceTests(unittest.TestCase):
    def test_mann_whitney_policy_preserves_scipy_auto_p_value(self):
        samples = [
            (np.array([1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10])),
            (np.array([1, 1, 2, 2, 3]), np.array([2, 2, 3, 3, 4])),
            (np.arange(20), np.arange(20) + 0.25),
        ]

        for group1, group2 in samples:
            with self.subTest(group1=group1, group2=group2):
                previous = stats.mannwhitneyu(group1, group2)
                explicit = sk._mann_whitney_test(group1, group2)
                self.assertEqual(previous.statistic, explicit.statistic)
                self.assertAlmostEqual(previous.pvalue, explicit.pvalue, places=15)

    def test_tie_aware_mann_whitney_interval_is_compatible_with_p_value(self):
        # The former no-tie rank-index interval was [-2, 0] (closed), despite
        # p=0.032. Test inversion correctly makes zero an excluded endpoint.
        group1 = np.array([5, 2, 4, 1, 4, 1, 4, 2, 3, 3, 3, 2, 2, 2, 4, 5, 2, 1, 1, 1])
        group2 = np.array([4, 5, 2, 3, 3, 3, 5, 2, 4, 3, 5, 2, 5, 5, 3, 5, 4, 2, 5, 1])

        test_result = sk._mann_whitney_test(group1, group2)
        interval = sk.compute_confidence_interval_difference(
            group1, group2, method="median", paired=False
        )

        self.assertLess(test_result.pvalue, 0.05)
        self.assertFalse(_contains_zero(interval))
        self.assertEqual(interval["CI"][1], 0)
        self.assertFalse(interval["CI_inclusive"][1])

    def test_wilcoxon_policy_and_interval_are_compatible(self):
        group1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        group2 = np.array([1, 1, 2, 3, 5, 5, 7, 7, 8, 9])

        previous = stats.wilcoxon(group1, group2)
        explicit = sk._wilcoxon_test(group1, group2)
        interval = sk.compute_confidence_interval_difference(
            group1, group2, method="median", paired=True
        )

        self.assertAlmostEqual(previous.pvalue, explicit.pvalue, places=15)
        self.assertEqual(previous.pvalue >= 0.05, _contains_zero(interval))

    def test_significance_uses_raw_p_value_while_display_remains_rounded(self):
        group1 = pd.Series([2.0, 3.0, 4.0, 5.0])
        group2 = pd.Series([1.0, 2.0, 3.0, 4.0])

        output = io.StringIO()
        with mock.patch.object(sk.stats, "ttest_ind", return_value=(2.0, 0.0496)):
            with contextlib.redirect_stdout(output):
                sk.compare_ind(
                    [group1, group2],
                    group_labels=["Group 1", "Group 2"],
                    data_type="cont",
                    force_normality=True,
                    do_graphs=False,
                )

        rendered = output.getvalue()
        self.assertIn("p-value: 0.05", rendered)
        self.assertIn("There is a significant difference between groups.", rendered)


if __name__ == "__main__":
    unittest.main()
