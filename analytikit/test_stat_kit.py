import contextlib
import io
from pathlib import Path
import sys
import unittest
import warnings
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
                differences = sk._paired_differences(group1, group2)
                explicit = sk._wilcoxon_test(group1, group2)
                interval = sk.compute_confidence_interval_difference(
                    group1, group2, method="median", paired=True
                )

                # SciPy's own auto policy is reproduced wherever this SciPy can
                # actually carry out the method the package resolves. Where its
                # exact path cannot represent ties or zeros, the package
                # supplies the exhaustive calculation instead, so its p-value is
                # deliberately not SciPy's here.
                if not sk._wilcoxon_needs_exact_fallback(differences):
                    previous = stats.wilcoxon(group1, group2)
                    self.assertAlmostEqual(previous.pvalue, explicit.pvalue, places=15)

                self.assertEqual(explicit.pvalue >= 0.05, _contains_zero(interval))
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

    def test_wilcoxon_exact_fallback_matches_exhaustive_sign_flipping(self):
        # The convolution used when SciPy's exact path cannot represent ties or
        # zeros must agree with exhaustive enumeration of the sign patterns,
        # which is what newer SciPy releases perform.
        exhaustive = stats.PermutationMethod(n_resamples=np.inf)
        rng = np.random.default_rng(11)

        for _ in range(20):
            differences = rng.integers(-3, 4, size=9).astype(float)
            if np.all(differences == 0):
                continue
            with self.subTest(differences=list(differences)):
                reference = stats.wilcoxon(
                    differences, method=exhaustive, **sk._WILCOXON_OPTIONS
                )
                fallback = sk._wilcoxon_exact_test(differences)

                self.assertEqual(reference.statistic, fallback.statistic)
                self.assertAlmostEqual(reference.pvalue, fallback.pvalue, places=15)

    def test_wilcoxon_exact_fallback_matches_scipy_without_ties_or_zeros(self):
        # Without ties or zeros the two exact calculations are the same
        # distribution, so the fallback must not perturb untied results.
        rng = np.random.default_rng(12)

        for _ in range(20):
            differences = rng.normal(size=12)
            with self.subTest(differences=list(differences)):
                reference = stats.wilcoxon(
                    differences, method="exact", **sk._WILCOXON_OPTIONS
                )
                fallback = sk._wilcoxon_exact_test(differences)

                self.assertEqual(reference.statistic, fallback.statistic)
                self.assertAlmostEqual(reference.pvalue, fallback.pvalue, places=15)

    def test_wilcoxon_ci_inversion_holds_the_resolved_method_in_scipy(self):
        # Candidate shifts sit on Walsh averages, so they routinely create both
        # zeros and ties among the absolute differences. Whatever reaches SciPy
        # must still be the method resolved from the observed data, and SciPy
        # must not report having substituted another one.
        differences = np.arange(1.0, 13.0)
        group2 = np.zeros_like(differences)
        expected = sk._wilcoxon_method(differences)
        self.assertEqual(expected, "exact")

        seen = []
        real_wilcoxon = sk.stats.wilcoxon

        def recording_wilcoxon(*args, **kwargs):
            seen.append(kwargs.get("method"))
            return real_wilcoxon(*args, **kwargs)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mock.patch.object(sk.stats, "wilcoxon", recording_wilcoxon):
                interval = sk.compute_confidence_interval_difference(
                    differences, group2, method="median", paired=True
                )

        self.assertEqual(interval["test_method"], expected)
        self.assertTrue(
            all(method == sk._scipy_wilcoxon_method(expected) for method in seen)
        )
        self.assertFalse(
            [entry for entry in caught if "zero" in str(entry.message).lower()]
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


class RegressionTests(unittest.TestCase):
    """Tests for regress(). Graphs are disabled throughout so the suite stays headless."""

    @staticmethod
    def _linear_frame(n=300, seed=7):
        rng = np.random.default_rng(seed)
        age = rng.normal(50, 12, n)
        sex = rng.choice(["female", "male"], n)
        smoking = rng.choice(["never", "former", "current"], n)
        outcome = (
            90.0
            + 0.6 * age
            + 5.0 * (sex == "male")
            + 8.0 * (smoking == "current")
            + rng.normal(0, 6, n)
        )
        return pd.DataFrame({"bp": outcome, "age": age, "sex": sex, "smoking": smoking})

    @staticmethod
    def _logistic_frame(n=600, seed=11):
        rng = np.random.default_rng(seed)
        age = rng.normal(60, 10, n)
        sex = rng.choice(["female", "male"], n)
        linear_predictor = -6.0 + 0.08 * age + 0.9 * (sex == "male")
        event = rng.binomial(1, 1.0 / (1.0 + np.exp(-linear_predictor)))
        return pd.DataFrame(
            {"died": np.where(event == 1, "yes", "no"), "age": age, "sex": sex}
        )

    @staticmethod
    def _run(**kwargs):
        """Call regress() with stdout suppressed, returning the result and the printed text."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = sk.regress(**kwargs)
        return result, output.getvalue()

    # --- Model selection ---

    def test_selects_linear_for_a_continuous_outcome(self):
        frame = self._linear_frame()
        result, _ = self._run(
            data=frame, outcome="bp", predictors=["age"], outcome_type="cont", do_graphs=False
        )
        self.assertEqual(result["model"], "linear")

    def test_selects_logistic_for_a_binary_outcome(self):
        frame = self._logistic_frame()
        result, _ = self._run(
            data=frame, outcome="died", predictors=["age"], outcome_type="cat", do_graphs=False
        )
        self.assertEqual(result["model"], "logistic")

    def test_infers_the_model_and_warns_when_outcome_type_is_not_set(self):
        frame = self._linear_frame()
        result, rendered = self._run(
            data=frame, outcome="bp", predictors=["age"], do_graphs=False
        )
        self.assertEqual(result["model"], "linear")
        self.assertIn("outcome_type not set", rendered)

    def test_force_model_skips_inference_and_prints_no_warning(self):
        frame = self._linear_frame()
        result, rendered = self._run(
            data=frame,
            outcome="bp",
            predictors=["age"],
            force_model="linear",
            do_graphs=False,
        )
        self.assertEqual(result["model"], "linear")
        self.assertNotIn("outcome_type not set", rendered)

    def test_force_model_distinguishes_planned_from_unknown_models(self):
        frame = self._linear_frame()
        with self.assertRaises(Exception) as planned:
            self._run(
                data=frame,
                outcome="bp",
                predictors=["age"],
                force_model="poisson",
                do_graphs=False,
            )
        self.assertIn("not implemented yet", str(planned.exception))

        with self.assertRaises(Exception) as unknown:
            self._run(
                data=frame,
                outcome="bp",
                predictors=["age"],
                force_model="banana",
                do_graphs=False,
            )
        self.assertIn("not recognised", str(unknown.exception))

    def test_a_multi_level_categorical_outcome_is_refused_rather_than_mismodelled(self):
        frame = self._linear_frame()
        frame["grade"] = np.resize(["mild", "moderate", "severe"], len(frame))
        with self.assertRaises(Exception) as error:
            self._run(
                data=frame,
                outcome="grade",
                predictors=["age"],
                outcome_type="cat",
                do_graphs=False,
            )
        self.assertIn("binary outcomes only", str(error.exception))

    # --- Estimates ---

    def test_linear_coefficients_recover_the_simulated_effects(self):
        frame = self._linear_frame()
        result, _ = self._run(
            data=frame,
            outcome="bp",
            predictors=["age", "sex", "smoking"],
            outcome_type="cont",
            reference={"smoking": "never"},
            do_graphs=False,
        )
        table = result["coefficients"].set_index("term")
        self.assertAlmostEqual(table.loc["age", "coefficient"], 0.6, delta=0.15)
        self.assertAlmostEqual(table.loc["sex [male vs female]", "coefficient"], 5.0, delta=1.5)

    def test_logistic_reports_odds_ratios_consistent_with_its_coefficients(self):
        frame = self._logistic_frame()
        result, _ = self._run(
            data=frame,
            outcome="died",
            predictors=["age", "sex"],
            outcome_type="cat",
            outcome_positive="yes",
            do_graphs=False,
        )
        table = result["coefficients"]
        np.testing.assert_allclose(
            table["odds_ratio"].to_numpy(), np.exp(table["coefficient"].to_numpy())
        )
        np.testing.assert_allclose(
            table["or_ci_lower"].to_numpy(), np.exp(table["ci_lower"].to_numpy())
        )

    def test_outcome_positive_choice_inverts_the_odds_ratios(self):
        frame = self._logistic_frame()
        as_yes, _ = self._run(
            data=frame,
            outcome="died",
            predictors=["age"],
            outcome_type="cat",
            outcome_positive="yes",
            do_graphs=False,
        )
        as_no, _ = self._run(
            data=frame,
            outcome="died",
            predictors=["age"],
            outcome_type="cat",
            outcome_positive="no",
            do_graphs=False,
        )
        self.assertEqual(as_yes["outcome_positive"], "yes")
        self.assertEqual(as_no["outcome_positive"], "no")
        np.testing.assert_allclose(
            as_yes["coefficients"]["coefficient"].to_numpy(),
            -as_no["coefficients"]["coefficient"].to_numpy(),
            atol=1e-6,
        )

    def test_reference_level_is_honoured_and_reported(self):
        frame = self._linear_frame()
        result, rendered = self._run(
            data=frame,
            outcome="bp",
            predictors=["smoking"],
            outcome_type="cont",
            reference={"smoking": "former"},
            do_graphs=False,
        )
        self.assertEqual(result["reference_levels"]["smoking"]["reference"], "former")
        self.assertIn('reference = "former"', rendered)
        self.assertIn("smoking [current vs former]", result["coefficients"]["term"].tolist())

    def test_an_unknown_reference_level_is_rejected(self):
        frame = self._linear_frame()
        with self.assertRaises(Exception) as error:
            self._run(
                data=frame,
                outcome="bp",
                predictors=["smoking"],
                outcome_type="cont",
                reference={"smoking": "vaping"},
                do_graphs=False,
            )
        self.assertIn("was not found", str(error.exception))

    def test_confidence_intervals_respond_to_alpha(self):
        frame = self._linear_frame()
        narrow, _ = self._run(
            data=frame, outcome="bp", predictors=["age"], outcome_type="cont",
            alpha=0.20, do_graphs=False,
        )
        wide, _ = self._run(
            data=frame, outcome="bp", predictors=["age"], outcome_type="cont",
            alpha=0.01, do_graphs=False,
        )
        narrow_width = (narrow["coefficients"]["ci_upper"] - narrow["coefficients"]["ci_lower"]).iloc[1]
        wide_width = (wide["coefficients"]["ci_upper"] - wide["coefficients"]["ci_lower"]).iloc[1]
        self.assertLess(narrow_width, wide_width)

    # --- Diagnostics are reported, not re-dispatched ---

    def test_heteroscedasticity_switches_to_robust_errors_without_changing_coefficients(self):
        rng = np.random.default_rng(3)
        n = 300
        predictor = rng.uniform(1, 20, n)
        outcome = 4.0 + 2.0 * predictor + rng.normal(0, 1, n) * predictor
        frame = pd.DataFrame({"y": outcome, "x": predictor})

        result, rendered = self._run(
            data=frame, outcome="y", predictors=["x"], outcome_type="cont", do_graphs=False
        )
        self.assertTrue(result["robust_standard_errors"])
        self.assertIn("HC3", rendered)

        # The model itself must be unchanged - only the standard errors are corrected.
        plain = sk.smf.ols("y ~ x", data=frame).fit()
        np.testing.assert_allclose(
            result["coefficients"]["coefficient"].to_numpy(),
            plain.params.to_numpy(),
            rtol=1e-10,
        )
        self.assertEqual(result["model"], "linear")

    def test_a_well_behaved_linear_model_keeps_model_based_errors(self):
        frame = self._linear_frame()
        result, _ = self._run(
            data=frame, outcome="bp", predictors=["age"], outcome_type="cont", do_graphs=False
        )
        self.assertFalse(result["robust_standard_errors"])

    def test_non_normal_residuals_are_reported_without_changing_the_model(self):
        rng = np.random.default_rng(19)
        n = 300
        predictor = rng.normal(0, 1, n)
        outcome = 1.0 + 2.0 * predictor + rng.standard_t(2, n) * 3
        frame = pd.DataFrame({"y": outcome, "x": predictor})
        result, rendered = self._run(
            data=frame, outcome="y", predictors=["x"], outcome_type="cont", do_graphs=False
        )
        self.assertEqual(result["model"], "linear")
        self.assertIn("Residual normality", rendered)
        self.assertIsNotNone(result["diagnostics"]["residual_normality"])

    def test_collinear_predictors_raise_the_vif_warning(self):
        rng = np.random.default_rng(23)
        n = 200
        first = rng.normal(0, 1, n)
        second = first + rng.normal(0, 0.01, n)  # almost identical to the first
        frame = pd.DataFrame({"y": first + rng.normal(0, 1, n), "a": first, "b": second})
        result, rendered = self._run(
            data=frame, outcome="y", predictors=["a", "b"], outcome_type="cont", do_graphs=False
        )
        self.assertIn("VIF", rendered)
        self.assertTrue((result["diagnostics"]["multicollinearity"]["VIF"] > 5).any())

    def test_low_events_per_variable_is_flagged(self):
        rng = np.random.default_rng(29)
        n = 60
        frame = pd.DataFrame(
            {
                "event": np.array([1] * 6 + [0] * 54),
                "a": rng.normal(0, 1, n),
                "b": rng.normal(0, 1, n),
                "c": rng.normal(0, 1, n),
            }
        )
        _, rendered = self._run(
            data=frame,
            outcome="event",
            predictors=["a", "b", "c"],
            outcome_type="cat",
            do_graphs=False,
        )
        self.assertIn("EPV", rendered)
        self.assertIn("below the conventional minimum", rendered)

    # --- Separation ---

    def test_separation_is_detected_before_fitting_and_named(self):
        rng = np.random.default_rng(31)
        n = 200
        frame = pd.DataFrame(
            {
                "event": rng.binomial(1, 0.4, n),
                "age": rng.normal(50, 10, n),
                "marker": rng.choice(["low", "high"], n),
            }
        )
        frame.loc[frame["marker"] == "high", "event"] = 1

        result, rendered = self._run(
            data=frame,
            outcome="event",
            predictors=["age", "marker"],
            outcome_type="cat",
            do_graphs=False,
        )
        self.assertIn("separation detected before fitting", rendered)
        self.assertIn("marker", rendered)
        if result is not None:
            self.assertTrue(result["diagnostics"]["separation"]["pre_fit_findings"])

    def test_complete_separation_reports_separation_not_collinearity(self):
        """
        Under complete separation statsmodels may either fail or converge to a meaningless fit,
        depending on the data. Either way the user must be told it is separation - the remedy
        differs completely from the one for collinearity, which produces the same bare
        "Singular matrix" error from statsmodels.
        """
        frame = pd.DataFrame(
            {
                "event": [0] * 50 + [1] * 50,
                "marker": [0.0] * 50 + [1.0] * 50,
                "noise": np.linspace(0, 1, 100),
            }
        )
        try:
            _, rendered = self._run(
                data=frame,
                outcome="event",
                predictors=["marker", "noise"],
                outcome_type="cat",
                do_graphs=False,
            )
        except Exception as error:  # the fit failed outright
            self.assertIn("separation", str(error).lower())
        else:  # the fit "succeeded" - the warning must still name the problem
            self.assertIn("separation", rendered.lower())
            self.assertNotIn("perfectly collinear", rendered)

    # --- Input handling ---

    def test_formula_supports_interactions(self):
        frame = self._linear_frame()
        result, _ = self._run(
            data=frame, formula="bp ~ age * sex", outcome_type="cont", do_graphs=False
        )
        self.assertIn("age:sex[T.male]", result["coefficients"]["design_term"].tolist())
        self.assertEqual(result["formula"], "bp ~ age * sex")

    def test_column_names_containing_spaces_are_handled(self):
        frame = self._linear_frame()
        frame["body mass index"] = np.linspace(20, 35, len(frame))
        result, _ = self._run(
            data=frame,
            outcome="bp",
            predictors=["body mass index"],
            outcome_type="cont",
            do_graphs=False,
        )
        self.assertIn("body mass index", result["coefficients"]["term"].tolist())

    def test_missing_values_are_dropped_listwise_and_counted(self):
        frame = self._linear_frame()
        frame.loc[frame.index[:10], "age"] = np.nan
        result, rendered = self._run(
            data=frame, outcome="bp", predictors=["age"], outcome_type="cont", do_graphs=False
        )
        self.assertEqual(result["n_dropped"], 10)
        self.assertEqual(result["n"], len(frame) - 10)
        self.assertIn("Dropping 10 row(s)", rendered)

    def test_a_constant_outcome_returns_none_rather_than_failing(self):
        frame = self._linear_frame()
        frame["flat"] = 1.0
        result, rendered = self._run(
            data=frame, outcome="flat", predictors=["age"], outcome_type="cont", do_graphs=False
        )
        self.assertIsNone(result)
        self.assertIn("constant", rendered)

    def test_a_constant_predictor_is_dropped_with_a_warning(self):
        frame = self._linear_frame()
        frame["flat"] = 1.0
        result, rendered = self._run(
            data=frame,
            outcome="bp",
            predictors=["age", "flat"],
            outcome_type="cont",
            do_graphs=False,
        )
        self.assertIsNotNone(result)
        self.assertIn("Dropping constant predictor", rendered)
        self.assertNotIn("flat", result["coefficients"]["term"].tolist())

    def test_too_few_observations_returns_none(self):
        frame = self._linear_frame().head(6)
        result, rendered = self._run(
            data=frame, outcome="bp", predictors=["age"], outcome_type="cont", do_graphs=False
        )
        self.assertIsNone(result)
        self.assertIn("Cannot fit a regression model", rendered)

    def test_a_missing_column_is_reported_by_name(self):
        frame = self._linear_frame()
        with self.assertRaises(Exception) as error:
            self._run(
                data=frame, outcome="bp", predictors=["nonexistent"], do_graphs=False
            )
        self.assertIn("nonexistent", str(error.exception))

    def test_the_outcome_cannot_also_be_a_predictor(self):
        frame = self._linear_frame()
        with self.assertRaises(Exception) as error:
            self._run(
                data=frame, outcome="bp", predictors=["bp", "age"], do_graphs=False
            )
        self.assertIn("both the outcome and a predictor", str(error.exception))

    # --- Univariable table ---

    def test_univariable_table_matches_single_predictor_models(self):
        frame = self._linear_frame()
        result, _ = self._run(
            data=frame,
            outcome="bp",
            predictors=["age", "sex"],
            outcome_type="cont",
            univariable=True,
            do_graphs=False,
        )
        crude = result["univariable"].set_index("term")
        alone, _ = self._run(
            data=frame, outcome="bp", predictors=["age"], outcome_type="cont", do_graphs=False
        )
        expected = alone["coefficients"].set_index("term").loc["age", "coefficient"]
        self.assertAlmostEqual(crude.loc["age", "coefficient"], expected, places=10)

    def test_univariable_is_omitted_by_default(self):
        frame = self._linear_frame()
        result, _ = self._run(
            data=frame, outcome="bp", predictors=["age"], outcome_type="cont", do_graphs=False
        )
        self.assertIsNone(result["univariable"])

    def test_univariable_is_ignored_with_a_formula_and_says_so(self):
        frame = self._linear_frame()
        result, rendered = self._run(
            data=frame,
            formula="bp ~ age + sex",
            outcome_type="cont",
            univariable=True,
            do_graphs=False,
        )
        self.assertIsNone(result["univariable"])
        self.assertIn("univariable=True is ignored", rendered)

    # --- Reporting ---

    def test_result_carries_the_fitted_model_and_a_tidy_table(self):
        frame = self._linear_frame()
        result, _ = self._run(
            data=frame, outcome="bp", predictors=["age"], outcome_type="cont", do_graphs=False
        )
        self.assertIsInstance(result["coefficients"], pd.DataFrame)
        for column in ("term", "coefficient", "std_error", "p_value", "ci_lower", "ci_upper"):
            self.assertIn(column, result["coefficients"].columns)
        self.assertTrue(hasattr(result["fitted_model"], "params"))
        self.assertAlmostEqual(float(result["fit"]["r_squared"]), result["fitted_model"].rsquared)

    def test_small_p_values_are_never_displayed_as_zero(self):
        frame = self._linear_frame()
        _, rendered = self._run(
            data=frame, outcome="bp", predictors=["age"], outcome_type="cont", do_graphs=False
        )
        self.assertIn("<0.001", rendered)
        self.assertNotIn("p-value: 0.0", rendered)

    def test_design_terms_are_rendered_readably(self):
        self.assertEqual(
            sk._regression_pretty_term(
                "C(Q('smoking status'), Treatment(reference='never'))[T.current]"
            ),
            "smoking status [current vs never]",
        )
        self.assertEqual(sk._regression_pretty_term("Q('body mass index')"), "body mass index")
        self.assertEqual(sk._regression_pretty_term("age"), "age")

    def test_auc_matches_a_known_value(self):
        # Perfectly ordered predictions give an AUC of exactly 1.
        self.assertAlmostEqual(
            sk._regression_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0
        )
        # Reversed predictions give 0, and a constant prediction gives 0.5.
        self.assertAlmostEqual(sk._regression_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)
        self.assertAlmostEqual(sk._regression_auc([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)

    def test_hosmer_lemeshow_does_not_flag_a_well_calibrated_model(self):
        rng = np.random.default_rng(41)
        probabilities = rng.uniform(0.05, 0.95, 500)
        observed = rng.binomial(1, probabilities)
        result = sk._regression_hosmer_lemeshow(observed, probabilities)
        self.assertIsNotNone(result)
        self.assertGreater(result["p_value"], 0.01)


if __name__ == "__main__":
    unittest.main()
