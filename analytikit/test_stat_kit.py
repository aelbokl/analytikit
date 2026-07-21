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

    def test_asymptotic_mann_whitney_effect_ci_matches_asht_reference(self):
        # Reference values from asht::wmwTestAsymptotic 1.0.3 (LAPH variance,
        # tie adjustment, and continuity correction), reversing its x<y
        # orientation to this package's group1>group2 orientation.
        cases = [
            (
                np.array([5, 2, 4, 1, 4, 1, 4, 2, 3, 3, 3, 2, 2, 2, 4, 5, 2, 1, 1, 1]),
                np.array([4, 5, 2, 3, 3, 3, 5, 2, 4, 3, 5, 2, 5, 5, 3, 5, 4, 2, 5, 1]),
                0.305,
                (0.17529400283910690, 0.48332178507935458),
            ),
            (
                np.array([4] * 10 + [5] * 10),
                np.array([4] * 20),
                0.75,
                (0.61740755544853587, 0.84451778407009237),
            ),
            (
                np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5]),
                np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4]),
                0.73469387755102045,
                (0.51905566070715192, 0.87140866822395402),
            ),
        ]

        for group1, group2, expected_estimate, expected_ci in cases:
            with self.subTest(expected_estimate=expected_estimate):
                result = sk.mann_whitney_effect_ci(
                    group1, group2, method="asymptotic"
                )
                self.assertAlmostEqual(
                    result["probability_superiority"], expected_estimate, places=14
                )
                np.testing.assert_allclose(
                    result["probability_CI"], expected_ci, rtol=0, atol=2e-9
                )
                scipy_result = stats.mannwhitneyu(
                    group1,
                    group2,
                    alternative="two-sided",
                    use_continuity=True,
                    method="asymptotic",
                )
                self.assertAlmostEqual(result["p_value"], scipy_result.pvalue, places=15)

    def test_exact_mann_whitney_effect_ci_matches_asht_reference(self):
        group1 = np.arange(1, 6)
        group2 = np.arange(101, 106)
        result = sk.mann_whitney_effect_ci(group1, group2, method="exact")

        self.assertEqual(result["probability_superiority"], 0)
        np.testing.assert_allclose(
            result["probability_CI"],
            (0, 0.344943017598996),
            rtol=0,
            atol=2e-10,
        )
        self.assertAlmostEqual(result["p_value"], 0.007936507936507936, places=15)

    def test_exact_mann_whitney_effect_ci_rejects_ties(self):
        with self.assertRaisesRegex(ValueError, "requires tie-free data"):
            sk.mann_whitney_effect_ci(
                np.array([1, 1, 2]), np.array([2, 3, 4]), method="exact"
            )

    def test_mann_whitney_effect_ci_is_compatible_with_selected_p_value(self):
        rng = np.random.default_rng(20260721)
        samples = []
        for _ in range(50):
            samples.append(
                (
                    rng.integers(1, 6, size=20),
                    rng.integers(1, 6, size=20),
                )
            )
        samples.extend(
            [
                (np.arange(5), np.arange(5) + shift)
                for shift in (-3.0, -1.0, 0.25, 1.0, 3.0)
            ]
        )

        for group1, group2 in samples:
            result = sk.mann_whitney_effect_ci(group1, group2)
            lower, upper = result["probability_CI"]
            null_included = lower <= 0.5 <= upper
            self.assertEqual(result["p_value"] >= 0.05, null_included)
            delta_lower, delta_upper = result["cliffs_delta_CI"]
            self.assertEqual(null_included, delta_lower <= 0 <= delta_upper)

    def test_compare_ind_reports_effect_ci_and_location_shift_is_optional(self):
        group1 = pd.Series([4] * 10 + [5] * 10, name="Intervention")
        group2 = pd.Series([4] * 20, name="Control")

        default_output = io.StringIO()
        with contextlib.redirect_stdout(default_output):
            sk.compare_ind(
                [group1, group2],
                group_labels=["Intervention", "Control"],
                data_type="cont",
                force_non_normality=True,
                do_graphs=False,
            )
        rendered = default_output.getvalue()
        self.assertIn("Probability of superiority", rendered)
        self.assertIn("95.0% compatible CI: [0.617, 0.845]", rendered)
        self.assertIn("95.0% compatible CI for δ: [0.235, 0.689]", rendered)
        self.assertIn("LAPH/proportional-odds working model", rendered)
        self.assertNotIn("Hodges-Lehmann", rendered)

        optional_output = io.StringIO()
        with contextlib.redirect_stdout(optional_output):
            sk.compare_ind(
                [group1, group2],
                group_labels=["Intervention", "Control"],
                data_type="ordinal",
                do_graphs=False,
                mann_whitney_location_shift_ci=True,
            )
        rendered_optional = optional_output.getvalue()
        self.assertIn("Mann-Whitney U", rendered_optional)
        self.assertIn("Optional Hodges-Lehmann location-shift estimate", rendered_optional)
        self.assertIn("location-shift model", rendered_optional)

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

    def test_very_small_p_value_is_not_displayed_as_zero(self):
        self.assertEqual(sk._display_p_value(0.0003338), "<0.001")

    def test_compare_ind_never_displays_very_small_p_value_as_zero(self):
        group1 = pd.Series([10.0, 12.0, 14.0, 17.0])
        group2 = pd.Series([1.0, 2.0, 3.0, 4.0])
        output = io.StringIO()

        with mock.patch.object(sk.stats, "ttest_ind", return_value=(12.0, 2e-12)):
            with contextlib.redirect_stdout(output):
                sk.compare_ind(
                    [group1, group2],
                    group_labels=["Group 1", "Group 2"],
                    data_type="cont",
                    force_normality=True,
                    do_graphs=False,
                )

        rendered = output.getvalue()
        self.assertIn("p-value: <0.001", rendered)
        self.assertNotIn("p-value: 0.0", rendered)

    def test_compare_dep_never_displays_very_small_p_value_as_zero(self):
        group1 = pd.Series([10.0, 12.0, 14.0, 17.0])
        group2 = pd.Series([1.0, 2.0, 3.0, 4.0])
        output = io.StringIO()

        with mock.patch.object(sk.stats, "ttest_rel", return_value=(12.0, 2e-12)):
            with contextlib.redirect_stdout(output):
                sk.compare_dep(
                    [group1, group2],
                    group_labels=["Time 2", "Time 1"],
                    data_type="cont",
                    force_normality=True,
                    do_graphs=False,
                )

        rendered = output.getvalue()
        self.assertIn("p-value: <0.001", rendered)
        self.assertNotIn("p-value: 0.0", rendered)


if __name__ == "__main__":
    unittest.main()
