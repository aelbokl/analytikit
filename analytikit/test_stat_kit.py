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
        samples = [
            (
                np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                np.array([1, 1, 2, 3, 5, 5, 7, 7, 8, 9]),
            ),  # small, with ties and zeros
            (np.arange(14), np.zeros(14)),  # larger, with a zero
            (np.arange(1, 21), np.zeros(20)),  # exact, without ties or zeros
            (np.arange(1, 61), np.zeros(60)),  # asymptotic because n > 50
        ]

        for group1, group2 in samples:
            with self.subTest(n=len(group1)):
                previous = stats.wilcoxon(group1, group2)
                explicit = sk._wilcoxon_test(group1, group2)
                interval = sk.compute_confidence_interval_difference(
                    group1, group2, method="median", paired=True
                )

                self.assertAlmostEqual(previous.pvalue, explicit.pvalue, places=15)
                self.assertEqual(previous.pvalue >= 0.05, _contains_zero(interval))
                self.assertEqual(_contains_zero(interval), interval["null_value_included"])

    def test_significant_tied_wilcoxon_excludes_zero(self):
        # Ties make the lower confidence bound open at zero. Zero is therefore
        # not a member of the confidence set, consistently with p < 0.05.
        differences = np.array(
            [-2, -2, 4, 4, -3, -2, -1, 2, 1, 3, 4, 4, -3, 5, 2, -1, 3, 3, 5, 3],
            dtype=float,
        )
        group2 = np.zeros_like(differences)

        test_result = sk._wilcoxon_test(differences)
        interval = sk.compute_confidence_interval_difference(
            differences, group2, method="median", paired=True
        )

        self.assertLess(test_result.pvalue, 0.05)
        self.assertFalse(_contains_zero(interval))
        self.assertFalse(interval["null_value_included"])
        self.assertEqual(interval["CI"][0], 0)
        self.assertFalse(interval["CI_inclusive"][0])

    def test_wilcoxon_ci_holds_method_fixed_during_inversion(self):
        differences = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7], dtype=float)
        group2 = np.zeros_like(differences)
        expected_method = sk._wilcoxon_method(differences)

        with mock.patch.object(sk, "_wilcoxon_test", wraps=sk._wilcoxon_test) as wrapped:
            interval = sk.compute_confidence_interval_difference(
                differences, group2, method="median", paired=True
            )

        self.assertEqual(expected_method, "asymptotic")
        self.assertEqual(interval["test_method"], expected_method)
        self.assertTrue(wrapped.call_count > 1)
        self.assertTrue(
            all(call.kwargs.get("method") == expected_method for call in wrapped.call_args_list)
        )

    def test_wilcoxon_difference_rounding_is_applied_before_ranking(self):
        group1 = np.array([0.3, 0.3, 0.3, 0.3])
        group2 = np.array([0.1 + 0.2, 0.3, 0.3, 0.3])
        differences = sk._paired_differences(group1, group2, difference_decimals=12)

        np.testing.assert_array_equal(differences, np.zeros(4))

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
