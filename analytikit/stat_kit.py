# Stat Kit
# Helper and wrapper functions for data cleaning, display and statistical analysis.
# Ahmed Elbokl (ahmed.elbokl@med.asu.edu.eg)
# Maha
# 2023-2026

# Imports
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.optimize import brentq
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scikit_posthocs import posthoc_dunn as dunn
import scikit_posthocs as sp
import copy
from collections import namedtuple
import math 
import warnings
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.stats.contingency_tables import SquareTable 
from scipy.stats import friedmanchisquare
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.tools.sm_exceptions import PerfectSeparationError
import re
import colorsys
import seaborn as sns
import matplotlib.pyplot as plt
from pingouin import pairwise_tests
from .helpers import percent, plus_minus, print_title, mean_ci, median_ci, print_mean_std, print_median_iqr

sns.set()

def printColor(string, color):
    """
    Print a string in a specific color in the console.

    Parameters:
    string (str): The string to be printed.
    color (str): The color to print the string in. Options are 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'.

    Returns:
    None
    """
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'endc': '\033[0m',
    }
    if color in colors:
        print(colors[color] + string + colors['endc'])
    else:
        print(string)


# test successful import
def version():
    """
    Returns the version of the package
    """
    return "1.1.2"


# Explicit inference policies used by the non-parametric two-sample tests.
# Keeping these options in one place prevents SciPy defaults from being applied
# differently by the test and by the confidence-interval calculation.
_MANN_WHITNEY_OPTIONS = {
    "alternative": "two-sided",
    "use_continuity": True,
}
_WILCOXON_OPTIONS = {
    "zero_method": "wilcox",
    "correction": False,
    "alternative": "two-sided",
}


def _scipy_wilcoxon_capabilities():
    """Probe how the installed SciPy spells and implements Wilcoxon inference.

    Two things vary across SciPy releases. The non-exact method was renamed
    from ``method="approx"`` to ``method="asymptotic"``, and the exact path was
    changed from the tabulated no-ties signed-rank distribution to exhaustive
    inference that is also valid when the differences contain ties or zeros.
    Both are probed by behaviour rather than by version string, so the
    package's own vocabulary ("exact" / "asymptotic") stays stable and the
    resolved policy never asks SciPy for an exact calculation it cannot
    actually perform.
    """
    # Distinct absolute values, so the only feature under test is the zero.
    probe = np.array(
        [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        asymptotic_name = None
        for candidate in ("asymptotic", "approx"):
            try:
                stats.wilcoxon(probe, method=candidate, **_WILCOXON_OPTIONS)
            except ValueError:
                continue
            asymptotic_name = candidate
            break
        if asymptotic_name is None:
            raise RuntimeError(
                "SciPy's wilcoxon accepts neither method='asymptotic' nor "
                "method='approx'; the package's Wilcoxon policy cannot be "
                "applied to this SciPy version"
            )

        # A SciPy whose exact path cannot represent zeros silently falls back
        # to the non-exact calculation and returns an identical p-value.
        zero_probe = np.append(probe, 0.0)
        exact_p = stats.wilcoxon(
            zero_probe, method="exact", **_WILCOXON_OPTIONS
        ).pvalue
        asymptotic_p = stats.wilcoxon(
            zero_probe, method=asymptotic_name, **_WILCOXON_OPTIONS
        ).pvalue

    exhaustive = not np.isclose(exact_p, asymptotic_p, rtol=1e-12, atol=0.0)
    return asymptotic_name, exhaustive


_SCIPY_WILCOXON_ASYMPTOTIC, _WILCOXON_EXACT_HANDLES_TIES = (
    _scipy_wilcoxon_capabilities()
)


def _scipy_wilcoxon_method(method):
    """Translate the package's method vocabulary into this SciPy's spelling."""
    if method in ("asymptotic", "approx"):
        return _SCIPY_WILCOXON_ASYMPTOTIC
    return method


_WilcoxonResult = namedtuple("WilcoxonResult", ["statistic", "pvalue"])


def _wilcoxon_signed_rank_statistics(differences):
    """Return ``(statistic, r_plus_ranks, absolute_midranks)`` for the test.

    Zeros are discarded before ranking, which is what ``zero_method="wilcox"``
    means, and the reported statistic is ``min(r_plus, r_minus)`` to match
    SciPy's two-sided convention.
    """
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        raise ValueError(
            "zero_method 'wilcox' and 'pratt' do not work if x - y is zero "
            "for all elements."
        )
    midranks = stats.rankdata(np.abs(nonzero))
    r_plus = float(midranks[nonzero > 0].sum())
    r_minus = float(midranks[nonzero < 0].sum())
    return min(r_plus, r_minus), r_plus, midranks


def _wilcoxon_needs_exact_fallback(differences):
    """Whether SciPy's own exact path cannot honour an exact request here.

    SciPy releases whose exact path is the tabulated no-ties signed-rank
    distribution silently substitute the asymptotic calculation when a zero is
    present, and silently discard ties when one is not.
    """
    if _WILCOXON_EXACT_HANDLES_TIES:
        return False
    absolute_nonzero = np.abs(differences[differences != 0])
    has_zeros = absolute_nonzero.size != differences.size
    has_ties = np.unique(absolute_nonzero).size != absolute_nonzero.size
    return has_zeros or has_ties


def _wilcoxon_exact_test(differences):
    """Exact two-sided signed-rank test that remains valid with ties and zeros.

    Conditional on the observed absolute midranks, the null distribution of
    ``r_plus`` is the distribution of the sum of a uniformly random subset of
    those midranks, because each non-zero difference is equally likely to carry
    either sign. That distribution is obtained exactly by convolution, so no
    enumeration of the 2**n sign patterns is needed and the calculation stays
    cheap across the whole range of sample sizes for which the package resolves
    an exact method.

    This reproduces the exhaustive calculation performed by newer SciPy
    releases. It is used only where the installed SciPy's own exact path cannot
    represent ties or zeros; there, SciPy silently discards ties and truncates
    a half-integral ``r_plus``, and silently substitutes the asymptotic
    calculation when a zero is present. Either substitution would break the
    fixed-method invariant that confidence-interval inversion depends on, since
    candidate shifts routinely create both ties and zeros.
    """
    statistic, r_plus, midranks = _wilcoxon_signed_rank_statistics(differences)

    # Midranks are integers or half-integers, so doubling them makes the whole
    # convolution exact in integer arithmetic.
    doubled = np.rint(midranks * 2.0).astype(np.int64)
    total = int(doubled.sum())
    distribution = np.zeros(total + 1)
    distribution[0] = 1.0
    for weight in doubled:
        shifted = np.zeros(total + 1)
        shifted[weight:] = distribution[: total + 1 - weight]
        distribution = 0.5 * (distribution + shifted)

    observed = int(np.rint(r_plus * 2.0))
    cumulative = distribution[: observed + 1].sum()
    survival = distribution[observed:].sum()
    p_value = min(1.0, 2.0 * min(cumulative, survival))
    return _WilcoxonResult(statistic=statistic, pvalue=p_value)


def _display_p_value(p_value):
    """Round a p-value for display without changing the value used for inference."""
    value = float(p_value)
    if value < 0.001:
        return "<0.001"
    return round(value, 3)


def _effect_ci_display_decimals(interval, null_value, minimum=3, maximum=10):
    """Avoid displaying an excluded null as if rounding put it on a CI bound."""
    lower, upper = interval
    null_is_excluded = upper < null_value or lower > null_value
    if not null_is_excluded:
        return minimum
    for decimals in range(minimum, maximum + 1):
        displayed_lower = round(lower, decimals)
        displayed_upper = round(upper, decimals)
        if not (displayed_lower <= null_value <= displayed_upper):
            return decimals
    return maximum


def _mann_whitney_method(group1, group2):
    """Resolve SciPy's current ``method='auto'`` policy once per analysis.

    The resolved method is then held fixed while the test is inverted to form
    its confidence interval. This reproduces the p-value from SciPy's current
    auto policy while preventing the method from changing as candidate shifts
    create or remove cross-group ties.
    """
    group1_arr = np.asarray(group1)
    group2_arr = np.asarray(group2)
    pooled = np.concatenate((group1_arr.ravel(), group2_arr.ravel()))
    has_ties = np.unique(pooled).size < pooled.size
    return "exact" if min(group1_arr.size, group2_arr.size) <= 8 and not has_ties else "asymptotic"


def _mann_whitney_test(group1, group2, method=None):
    """Run the package's explicit two-sided Mann-Whitney test policy."""
    if method is None:
        method = _mann_whitney_method(group1, group2)
    return stats.mannwhitneyu(
        group1,
        group2,
        method=method,
        **_MANN_WHITNEY_OPTIONS,
    )


def _mann_whitney_tie_factor(group1, group2):
    """Return the standard WMW variance multiplier for pooled ties."""
    pooled = np.concatenate((np.asarray(group1), np.asarray(group2)))
    total = pooled.size
    if total < 2:
        return 1.0
    counts = np.unique(pooled, return_counts=True)[1].astype(float)
    return 1 - np.sum(counts**3 - counts) / (total * (total + 1) * (total - 1))


def _mann_whitney_asymptotic_effect_ci(group1, group2, statistic, alpha):
    """Fay-Malinovsky compatible CI for the Mann-Whitney probability.

    This inverts the same tie-adjusted, continuity-corrected normal test used
    by SciPy's asymptotic Mann-Whitney calculation. The variance away from the
    null uses the LAPH model from Fay and Malinovsky (2018).
    """
    n1, n2 = len(group1), len(group2)
    pair_count = n1 * n2
    estimate = float(statistic) / pair_count
    tie_factor = _mann_whitney_tie_factor(group1, group2)
    if tie_factor == 0:
        return (0.0, 1.0), tie_factor

    def variance(parameter):
        base = parameter * (1 - parameter) / pair_count
        model_factor = 1 + ((n1 + n2 - 2) / 2) * (
            (1 - parameter) / (2 - parameter) + parameter / (1 + parameter)
        )
        return tie_factor * base * model_factor

    correction_less = -0.5 / pair_count
    correction_greater = 0.5 / pair_count
    epsilon = 1e-10

    def root(z_quantile, correction):
        def objective(parameter):
            return (
                (estimate - parameter - correction) / np.sqrt(variance(parameter))
                - z_quantile
            )

        lower_value = objective(epsilon)
        if lower_value <= 0:
            return epsilon
        upper_value = objective(1 - epsilon)
        if upper_value >= 0:
            return 1 - epsilon
        return brentq(objective, epsilon, 1 - epsilon, xtol=1e-12)

    if estimate == 0:
        lower = 0.0
    else:
        lower = root(stats.norm.ppf(1 - alpha / 2), correction_greater)
    if estimate == 1:
        upper = 1.0
    else:
        upper = root(stats.norm.ppf(alpha / 2), correction_less)
    return (lower, upper), tie_factor


def _mann_whitney_order_distribution(n1, n2, parameter, descending=False):
    """Distribution of U under the exact LAPH ordered-sample model.

    A frontier dynamic program avoids enumerating all ``choose(n1+n2, n1)``
    allocations. ``parameter`` is P(group1 > group2), with ties counting 1/2;
    this exact branch is used only for tie-free samples.
    """
    maximum_u = n1 * n2
    initial = np.zeros(maximum_u + 1, dtype=float)
    initial[0] = 1.0
    frontier = {0: initial}  # key: number of group-1 observations selected

    for selected_total in range(n1 + n2):
        next_frontier = {}
        for selected_1, probabilities in frontier.items():
            selected_2 = selected_total - selected_1
            remaining_1 = n1 - selected_1
            remaining_2 = n2 - selected_2
            if descending:
                denominator = parameter * remaining_1 + (1 - parameter) * remaining_2
            else:
                denominator = (1 - parameter) * remaining_1 + parameter * remaining_2

            if remaining_1:
                probability_1 = (
                    parameter * remaining_1 / denominator
                    if descending
                    else (1 - parameter) * remaining_1 / denominator
                )
                increment = remaining_2 if descending else selected_2
                destination = next_frontier.setdefault(
                    selected_1 + 1, np.zeros(maximum_u + 1, dtype=float)
                )
                destination[increment:] += probability_1 * probabilities[:maximum_u + 1 - increment]

            if remaining_2:
                probability_2 = (
                    (1 - parameter) * remaining_2 / denominator
                    if descending
                    else parameter * remaining_2 / denominator
                )
                destination = next_frontier.setdefault(
                    selected_1, np.zeros(maximum_u + 1, dtype=float)
                )
                destination += probability_2 * probabilities
        frontier = next_frontier

    return frontier[n1]


def _mann_whitney_exact_effect_ci(n1, n2, statistic, alpha):
    """Exact central Fay-Malinovsky CI for a tie-free Mann-Whitney test."""
    observed_u = int(round(float(statistic)))
    maximum_u = n1 * n2
    epsilon = 1e-10

    def tails(parameter):
        distribution = 0.5 * (
            _mann_whitney_order_distribution(n1, n2, parameter, descending=False)
            + _mann_whitney_order_distribution(n1, n2, parameter, descending=True)
        )
        return (
            float(np.sum(distribution[:observed_u + 1])),
            float(np.sum(distribution[observed_u:])),
        )

    if observed_u == 0:
        lower = 0.0
    else:
        lower = brentq(
            lambda parameter: tails(parameter)[1] - alpha / 2,
            epsilon,
            1 - epsilon,
            xtol=1e-10,
        )
    if observed_u == maximum_u:
        upper = 1.0
    else:
        upper = brentq(
            lambda parameter: tails(parameter)[0] - alpha / 2,
            epsilon,
            1 - epsilon,
            xtol=1e-10,
        )
    return (lower, upper)


def mann_whitney_effect_ci(group1, group2, alpha=0.05, method=None):
    """Estimate the Mann-Whitney probability and a test-compatible CI.

    The probability is ``P(group1 > group2) + 0.5*P(group1 == group2)``.
    Cliff's delta is its linear transformation ``2*probability - 1``.
    The reported p-value is SciPy's existing two-sided Mann-Whitney p-value;
    its resolved exact/asymptotic method is also used for the interval.

    Method: Fay MP, Malinovsky Y. Statistics in Medicine. 2018;37:3991-4006.
    doi:10.1002/sim.7890. The effect CI uses their LAPH/proportional-odds
    working model; this assumption is not required by the WMW test itself.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    first = np.asarray(group1, dtype=float)
    second = np.asarray(group2, dtype=float)
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("Mann-Whitney samples must be one-dimensional")
    if first.size == 0 or second.size == 0:
        raise ValueError("Mann-Whitney samples must both contain observations")
    if method is None:
        method = _mann_whitney_method(first, second)

    pooled = np.concatenate((first, second))
    has_ties = np.unique(pooled).size < pooled.size
    if method == "exact" and has_ties:
        raise ValueError(
            "The exact Mann-Whitney effect CI requires tie-free data; "
            "use method='asymptotic' for tied or discrete observations"
        )

    test_result = _mann_whitney_test(first, second, method=method)
    probability = float(test_result.statistic) / (first.size * second.size)
    if method == "exact":
        probability_ci = _mann_whitney_exact_effect_ci(
            first.size, second.size, test_result.statistic, alpha
        )
        tie_factor = 1.0
    elif method == "asymptotic":
        probability_ci, tie_factor = _mann_whitney_asymptotic_effect_ci(
            first, second, test_result.statistic, alpha
        )
    else:
        raise ValueError("method must resolve to 'exact' or 'asymptotic'")

    delta = 2 * probability - 1
    delta_ci = tuple(2 * bound - 1 for bound in probability_ci)
    return {
        "probability_superiority": probability,
        "probability_CI": probability_ci,
        "cliffs_delta": delta,
        "cliffs_delta_CI": delta_ci,
        "p_value": float(test_result.pvalue),
        "test_stat": float(test_result.statistic),
        "test_method": method,
        "tie_factor": tie_factor,
        "null_probability": 0.5,
        "null_delta": 0.0,
        "ci_assumption": "LAPH/proportional-odds working model",
    }


def _paired_differences(group1, group2=None, difference_decimals=None):
    """Return one clean difference vector for all paired rank inference.

    SciPy recommends calculating paired differences before calling
    ``wilcoxon`` because subtraction round-off can change ranks. Callers may
    specify the measurement precision with ``difference_decimals``; no
    rounding is applied by default.
    """
    first = np.asarray(group1, dtype=float)
    differences = first if group2 is None else first - np.asarray(group2, dtype=float)
    if differences.ndim != 1:
        raise ValueError("Wilcoxon samples must be one-dimensional")
    if difference_decimals is not None:
        if not isinstance(difference_decimals, (int, np.integer)):
            raise TypeError("difference_decimals must be an integer or None")
        differences = np.round(differences, decimals=int(difference_decimals))
    return differences


def _wilcoxon_method(differences):
    """Resolve SciPy's ``method='auto'`` policy once for an analysis.

    Holding this method fixed during CI inversion is essential: candidate
    shifts can create or remove ties and must not silently switch the
    inferential algorithm part-way through the confidence-set calculation.
    """
    differences = np.asarray(differences, dtype=float)
    n = differences.size
    has_zeros = np.any(differences == 0)
    absolute_nonzero = np.abs(differences[differences != 0])
    has_ties = np.unique(absolute_nonzero).size < absolute_nonzero.size

    # Exhaustive inference remains feasible through n=13 with ties or zeros;
    # otherwise exact inference is used through n=50 only when the data contain
    # neither ties nor zeros. The policy is deliberately independent of the
    # installed SciPy: where SciPy's own exact path cannot represent ties or
    # zeros, `_wilcoxon_test` supplies the exhaustive calculation itself rather
    # than letting the resolved method be silently substituted.
    if n <= 13 or (n <= 50 and not has_zeros and not has_ties):
        return "exact"
    return "asymptotic"


def _wilcoxon_test(group1, group2=None, method=None, difference_decimals=None):
    """Run the package's explicit two-sided Wilcoxon signed-rank policy."""
    differences = _paired_differences(group1, group2, difference_decimals)
    if differences.size == 0:
        raise ValueError("Paired samples must contain at least one observation")
    if method is None:
        method = _wilcoxon_method(differences)
    if method == "exact" and _wilcoxon_needs_exact_fallback(differences):
        return _wilcoxon_exact_test(differences)
    return stats.wilcoxon(
        differences, method=_scipy_wilcoxon_method(method), **_WILCOXON_OPTIONS
    )


def _test_inverted_interval(breakpoints, p_value_at, estimate, alpha):
    """Invert a two-sided stepwise test and return its central acceptance interval.

    Rank-test p-values change only at a finite collection of breakpoints. Both
    the breakpoints and the open regions between them are tested because ties
    can make inclusion at an endpoint differ from inclusion immediately beside
    it. The returned inclusivity flags allow callers to print ``(`` or ``)``
    when a discrete endpoint itself is rejected.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    points = np.unique(np.asarray(breakpoints, dtype=float))
    points = points[np.isfinite(points)]
    if points.size == 0:
        raise ValueError("Cannot invert a test without finite candidate breakpoints")

    number_of_points = points.size
    last_state = 2 * number_of_points
    span = max(float(np.ptp(points)), 1.0)
    cache = {}

    def value_for_state(state):
        if state == 0:
            return points[0] - span - 1.0
        if state == last_state:
            return points[-1] + span + 1.0
        if state % 2 == 1:
            return points[(state - 1) // 2]
        right = state // 2
        return points[right - 1] + (points[right] - points[right - 1]) / 2

    def is_accepted(state):
        if state not in cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                p_value = float(p_value_at(value_for_state(state)))
            cache[state] = np.isfinite(p_value) and p_value >= alpha
        return cache[state]

    insertion = int(np.searchsorted(points, estimate))
    if insertion < number_of_points and np.isclose(points[insertion], estimate, rtol=0, atol=0):
        preferred_state = 2 * insertion + 1
    else:
        preferred_state = 2 * insertion
    preferred_state = min(max(preferred_state, 0), last_state)

    # A full scan is exact and inexpensive for the small/tie-heavy datasets for
    # which these intervals are most often used. For large continuous samples,
    # locate the central accepted component with logarithmic test evaluations.
    if last_state + 1 <= 2001:
        accepted = np.array([is_accepted(state) for state in range(last_state + 1)])
        accepted_states = np.flatnonzero(accepted)
        if accepted_states.size == 0:
            raise RuntimeError("The inverted test has no accepted parameter values")
        centre = int(accepted_states[np.argmin(np.abs(accepted_states - preferred_state))])
        lower_state = centre
        while lower_state > 0 and accepted[lower_state - 1]:
            lower_state -= 1
        upper_state = centre
        while upper_state < last_state and accepted[upper_state + 1]:
            upper_state += 1
    else:
        nearby = range(max(0, preferred_state - 2), min(last_state, preferred_state + 2) + 1)
        accepted_nearby = [state for state in nearby if is_accepted(state)]
        if not accepted_nearby:
            # This is a defensive fallback for unusual discrete samples.
            accepted_nearby = [state for state in range(last_state + 1) if is_accepted(state)]
            if not accepted_nearby:
                raise RuntimeError("The inverted test has no accepted parameter values")
        centre = min(accepted_nearby, key=lambda state: abs(state - preferred_state))

        if is_accepted(0):
            lower_state = 0
        else:
            rejected, accepted_state = 0, centre
            while accepted_state - rejected > 1:
                middle = (rejected + accepted_state) // 2
                if is_accepted(middle):
                    accepted_state = middle
                else:
                    rejected = middle
            lower_state = accepted_state

        if is_accepted(last_state):
            upper_state = last_state
        else:
            accepted_state, rejected = centre, last_state
            while rejected - accepted_state > 1:
                middle = (accepted_state + rejected) // 2
                if is_accepted(middle):
                    accepted_state = middle
                else:
                    rejected = middle
            upper_state = accepted_state

    if lower_state == 0:
        lower, lower_inclusive = -np.inf, False
    elif lower_state % 2 == 1:
        lower, lower_inclusive = points[(lower_state - 1) // 2], True
    else:
        lower, lower_inclusive = points[lower_state // 2 - 1], False

    if upper_state == last_state:
        upper, upper_inclusive = np.inf, False
    elif upper_state % 2 == 1:
        upper, upper_inclusive = points[(upper_state - 1) // 2], True
    else:
        upper, upper_inclusive = points[upper_state // 2], False

    return (lower, upper), (lower_inclusive, upper_inclusive)

# drop_columns, lookup_columns, print_missing should live in cleaning_kit

## ci helpers are imported from helpers

## print_mean_std moved to helpers


## print_median_iqr moved to helpers


## cleaning-related function removed; use cleaning_kit.print_missing if needed


def make_subject_palette(index):
    """
    Create a consistent color mapping for subjects keyed by their DataFrame index.

    Parameters:
    index: A pandas Index, list, or array of subject identifiers.

    Returns:
    dict mapping each subject identifier to an RGB color tuple.
    Call this once from your DataFrame and reuse the result across all compare_ind calls
    to ensure the same subject always receives the same color.

    Example:
        palette = sk.make_subject_palette(df.index)
        sk.compare_ind([df.loc[g1, 'NIHSS'], df.loc[g2, 'NIHSS']], subject_palette=palette)
    """
    unique_ids = list(dict.fromkeys(index))  # deduplicate while preserving order
    # Golden-ratio hue stepping: each color is ~222° away from the last on the hue
    # wheel, preventing the circular wrap-around similarity that evenly-spaced
    # palettes (husl, hls) suffer from at large N.
    golden_ratio = 0.618033988749895
    h = 0.1  # starting hue
    colors = []
    for _ in range(len(unique_ids)):
        colors.append(colorsys.hsv_to_rgb(h, 0.65, 0.90))
        h = (h + golden_ratio) % 1.0
    return dict(zip(unique_ids, colors))


def plot_group_comparison(
    groups,
    group_labels,
    test_name,
    p_value,
    alpha=0.05,
    do_graphs=True,
    graphs_for_non_significance=False,
    continuous=True,
    all_normal=None,
    contingency_table=None,
    subject_palette=None,
):
    if not do_graphs or p_value is None:
        return

    if p_value >= alpha and not graphs_for_non_significance:
        return

    if continuous:
        plot_df = pd.DataFrame(
            {
                "group": np.concatenate(
                    [[label] * len(group) for label, group in zip(group_labels, groups)]
                ),
                "value": np.concatenate([np.asarray(group) for group in groups]),
                "patient_id": np.concatenate([group.index for group in groups]),
            }
        )
        value_label = groups[0].name if getattr(groups[0], "name", None) else "Value"

        plt.figure(figsize=(max(4.8, len(group_labels) * 0.95), 4.2))
        if all_normal and len(groups) == 2:
            ax = sns.violinplot(
                data=plot_df,
                x="group",
                y="value",
                inner=None,
                cut=0,
                width=0.5,
                palette="pastel",
                linewidth=1,
            )
        else:
            ax = sns.boxplot(
                data=plot_df,
                x="group",
                y="value",
                width=0.5,
                palette="pastel",
                showfliers=False,
                fliersize=3,
                linewidth=1,
            )

        if subject_palette is not None:
            sns.stripplot(
                data=plot_df,
                x="group",
                y="value",
                hue="patient_id",
                palette=subject_palette,
                alpha=0.7,
                jitter=0.12,
                size=4,
                legend=False,
            )
        else:
            sns.stripplot(
                data=plot_df,
                x="group",
                y="value",
                color="black",
                alpha=0.5,
                jitter=0.12,
                size=4,
            )
        ax.set_xlabel("Group", fontsize=12, fontweight="bold")
        ax.set_ylabel(value_label, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", labelrotation=0, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        for label in ax.get_xticklabels():
            label.set_fontweight("normal")
            label.set_ha("center")
        for label in ax.get_yticklabels():
            label.set_fontweight("normal")
        sns.despine(ax=ax)
    else:
        if contingency_table is None:
            _all_values = np.concatenate([g.values for g in groups])
            _all_labels = np.concatenate([[group_labels[i]] * len(g) for i, g in enumerate(groups)])
            contingency_table = pd.crosstab(
                _all_values, _all_labels,
                rownames=[groups[0].name if getattr(groups[0], 'name', None) else 'Category'],
                colnames=['Group']
            )

        plt.figure(
            figsize=(
                max(4.8, contingency_table.shape[1] * 0.95),
                max(3.8, contingency_table.shape[0] * 0.75),
            )
        )
        ax = sns.heatmap(
            contingency_table,
            annot=True,
            fmt="g",
            cmap="Blues",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"shrink": 0.85},
        )
        ax.set_xlabel(contingency_table.columns.name or "Group", fontsize=12, fontweight="bold")
        ax.set_ylabel(contingency_table.index.name or "Category", fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", labelrotation=0, labelsize=10)
        ax.tick_params(axis="y", labelrotation=0, labelsize=10)
        for label in ax.get_xticklabels():
            label.set_fontweight("normal")
            label.set_ha("center")
        for label in ax.get_yticklabels():
            label.set_fontweight("normal")

    plt.tight_layout()
    plt.show()


# Compare series/columns for independent groups
def compare_ind(
    groups,
    group_labels=None,
    alpha=0.05,
    categorical_limit=20,
    force_test=None,
    force_normality=False,
    force_non_normality=False,
    data_type=None,
    do_graphs=True,
    graphs_for_non_significance=False,
    subject_palette=None,
    mann_whitney_location_shift_ci=False,
):  # Depends on print_mean_std()
    # Note: this function does not handle the independent variable (grouping variable). It assumes that the series are already grouped.
    # groups: a list of pandas series/columns to be compared
    # group_labels: a list of labels for the groups. If not provided, the group names + group number will be used as labels. The list of labels has to correspond to the list of groups.
    # force_test: if provided, the function will use the test provided instead of determining it automatically. The test has to be a string and has to be one of the following: "ttest", "one-way ANOVA", "mannwhitney", "kruskalwallis", "chisquare", 'fisherexact'
    # alpha: the alpha value to be used for the test. Default is 0.05
    # data_type: 'cont', 'ordinal', 'cat', or 'nominal'. Ordinal data uses rank tests directly. If not provided, the function attempts to infer a type from the number of unique values.
    # categorical_limit: fallback unique-value threshold used only when data_type is not set. Default is 20.

    # Declare vars
    test_name = ""
    test_statistic_sign = ""
    line1 = ""
    line2 = ""
    effect_size = {}
    all_normal = None
    contingency_table = None
    
    #ci
    ci_lower = None
    ci_upper = None
    ci_inclusive = (True, True)
    mann_whitney_effect_result = None
    mann_whitney_shift_result = None

    tukey_results = None
    dunn_results = None

    # If group_labels is not provided, use the group names + group number as labels
    if group_labels is None:
        group_labels = []
        index = 0
        for group in groups:
            index += 1
            group_labels.append(f"{group.name} (Group {index})")

    # Drop isna values from each group (copies to avoid mutating caller's Series)
    clean_groups = []
    index = 0
    for group in groups:
        missing_in_group = group.isna().sum()
        if missing_in_group > 0:
            print(
                "Dropping",
                missing_in_group,
                'missing values from "' + group_labels[index] + '".',
            )
            clean_groups.append(group.dropna())
        else:
            print('No missing values in "' + group_labels[index] + '" to drop.')
            clean_groups.append(group.copy())
        index += 1
    groups = clean_groups
    print()
    print("alpha is set to", alpha)

    # Guard: abort if any group is empty after dropping NAs
    empty_groups = [group_labels[i] for i, g in enumerate(groups) if len(g) == 0]
    if empty_groups:
        printColor(f"Cannot run test: the following group(s) have no observations: {', '.join(empty_groups)}", "yellow")
        return

    # Determine if data is continuous or categorical
    ordinal = data_type == "ordinal"
    if data_type == "cont":
        continuous = True
        print("Data type explicitly set to continuous.")
    elif ordinal:
        continuous = True
        print("Data type explicitly set to ordinal; using rank-based inference.")
    elif data_type in ("cat", "nominal"):
        continuous = False
        print("Data type explicitly set to categorical.")
    else:
        nunique = max([group.nunique() for group in groups])
        continuous = nunique > categorical_limit
        printColor(
            f"Warning: data_type not set. Inferring '{'cont' if continuous else 'cat'}' "
            f"from unique-value count ({nunique}). Pass data_type='cont' or "
            f"data_type='cat' to suppress this warning.",
            "yellow"
        )

    #### IF DATA IS CONTINUOUS ####
    if continuous:

        ### Normality Test ###
        # Use Shapiro-Wilk test to check for normality of each group
        if ordinal:
            all_normal = False
            print("Ordinal data is analyzed with non-parametric rank tests.")
        elif force_non_normality:
            all_normal = False
            print("The compare function is forced to use non-parametric tests.")
        elif force_normality:
            all_normal = True
            print("The compare function is forced to use parametric tests.")
        else:
            all_normal = True

        index = 0

        for group in groups:
            if not ordinal and not force_normality and not force_non_normality:
                print_title("Test for Normality")
                statistic, p_value = stats.shapiro(group)
                statistic = round(statistic, 3)
                # print group label for this group
                print("Test for normality (Shapiro-Wilk) for", group_labels[index])
                print("-------------------------------------------------")
                print("Statistic:", statistic)
                print("p-value:", _display_p_value(p_value))
                if p_value > alpha:
                    print("Data is normally distributed.")
                else:
                    print("Data is not normally distributed.")
                    all_normal = False
                index += 1
            # # Plot QQ
            # print()
            # plt.figure(figsize=(4, 3))
            # stats.probplot(group, dist="norm", plot=plt)
            # plt.title("Normal Q-Q for " + group_labels[index])
            # plt.show()
            # print("-------------------------------------------------")
            # index += 1
            # if index < len(groups):
            #     print()

        # Print mean and standard deviation for each group if all groups are normally distributed
        print_title("Central Tendency")
        if all_normal:
            index = 0
            for group in groups:  # if you want to get index
                print_mean_std(
                    group_labels[index],
                    group,
                )

                # Print n = number of observations in each group
                print("n:", len(group))
                if index < len(groups):
                    print()
                index += 1

        # Print median and IQR for each group if not all groups are normally distributed
        else:
            index = 0
            for group in groups:
                print_median_iqr(group_labels[index], group)

                # Print n = number of observations in each group
                print("n:", len(group))
                if index < len(groups):
                    print()
                index += 1

        # If all groups are normally distributed, and we have 2 groups,use independent t-test
        if all_normal and len(groups) == 2:
            test_name = "Independent t-test"
            test_statistic_sign = "t"
            line1 = "All data is normally distributed."

            # Use Welch's t-test (equal_var=False) to match the CI calculation
            statistic, p_value = stats.ttest_ind(groups[0], groups[1], equal_var=False)
            statistic = round(statistic, 3)

            #ci difference
            ci_result = compute_confidence_interval_difference(groups[0], groups[1], method="mean", paired=False, alpha=alpha)
            ci_lower, ci_upper = ci_result["CI"]
            ci_inclusive = ci_result["CI_inclusive"]

            # Weighted pooled SD — correct for unequal group sizes
            n1 = len(groups[0])
            n2 = len(groups[1])
            pooled_std = np.sqrt(
                ((n1 - 1) * np.std(groups[0], ddof=1) ** 2 + (n2 - 1) * np.std(groups[1], ddof=1) ** 2)
                / (n1 + n2 - 2)
            )

            # Compute Cohen's d
            cohen_d = (np.mean(groups[0]) - np.mean(groups[1])) / pooled_std

            effect_size["label"] = "Cohen's d"
            effect_size["value"] = round(cohen_d, 3) 


        # If all groups are normally distributed, and we have more than 2 groups/group, use one-way ANOVA
        elif all_normal and len(groups) > 2:
            test_name = "one-way ANOVA"
            test_statistic_sign = "F"
            line1 = "Using one-way ANOVA because there are more than 2 groups."

            # Use one-way ANOVA to compare all group
            statistic, p_value = stats.f_oneway(*groups)
            statistic = round(statistic, 3)

            #MAHA# # Compute the effect size for one-way ANOVA (eta squared (η²))
            N = sum(len(group) for group in groups)  # Total sample size
            grand_mean = np.mean(np.concatenate(groups))  # Overall mean
            # Compute SST (Between-Group Sum of Squares)
            sst = sum([len(group) * (np.mean(group) - grand_mean) ** 2 for group in groups])
            # Compute SSW (Within-Group Sum of Squares) - Fixed
            ssw = sum([np.sum((group - np.mean(group)) ** 2) for group in groups])
            # Compute eta squared
            eta_squared = sst / (sst + ssw)
            effect_size["label"] = "eta squared (η²)"
            effect_size["value"] = round (eta_squared, 3) 
            # print(f"ANOVA F-statistic: {F_stat}")
            # print(f"Eta Squared (η²): {eta_squared:.3f}")

     #MAHA# #perform post-hoc test after one-way ANOVA if p-value is significant
            if p_value < alpha:
                print("Post-hoc test is needed.")
                # Use Tukey's HSD test for post-hoc analysis
                # Define df with the necessary data
                df = pd.DataFrame({
                'value': np.concatenate(groups),
                'group': np.concatenate([[label] * len(group) for label, group in zip(group_labels, groups)])
        })
                tukey_results = pairwise_tukeyhsd (endog=df['value'], groups=df['group'], alpha=alpha)
                
                
            

        # If all group are not normally distributed, and we have 2 groups, use Mann-Whitney U test
        elif not all_normal and len(groups) == 2:
            test_name = "Mann-Whitney U"
            test_statistic_sign = "Statistic"
            line1 = "Not all data is normally distributed."

            # Use Mann-Whitney U test to compare all group
            mann_whitney_method = _mann_whitney_method(groups[0], groups[1])
            statistic, p_value = _mann_whitney_test(
                groups[0], groups[1], method=mann_whitney_method
            )

            # Primary effect and confidence interval on the parameter directly
            # associated with Mann-Whitney. This works on continuous, discrete,
            # and ordinal scales and treats ties as half-wins.
            mann_whitney_effect_result = mann_whitney_effect_ci(
                groups[0],
                groups[1],
                alpha=alpha,
                method=mann_whitney_method,
            )
            if mann_whitney_location_shift_ci:
                mann_whitney_shift_result = compute_confidence_interval_difference(
                    groups[0],
                    groups[1],
                    method="median",
                    paired=False,
                    alpha=alpha,
                )

            # Round for display after using in calculations
            statistic = round(statistic, 3)
            
        # If all group are not normally distributed, and we have more than 2 groups, use Kruskal-Wallis H test
        elif not all_normal and len(groups) > 2:
            test_name = "Kruskal-Wallis H test"
            test_statistic_sign = "Statistic"
            line1 = "Not all data is normally distributed."

            # Use Kruskal-Wallis H test to compare all group
            statistic, p_value = stats.kruskal(*groups)
            statistic = round(statistic, 3)
            #MAHA# # Compute the effect size for Kruskal-Wallis H (Epsilon Squared (ε²))
            N = sum(len(group) for group in groups)  # Total number of observations
            k = len(groups)  # Number of groups
            H_stat = statistic
            epsilon_squared = H_stat / (N - 1)  # Standard formula; bounded [0, 1]
            effect_size["label"] = "Epsilon Squared (ε²)"
            effect_size["value"] = round(epsilon_squared, 3)

            #perform post-hoc test after Kruskal-Wallis H test if p-value is significant



            if p_value < alpha:
                print("Post-hoc test is needed.")
                # Use Dunn's test for post-hoc analysis
                
                 # Prepare data for Dunn's test
                data = np.concatenate(groups)
                group_labels_array = np.concatenate([[label] * len(group) for label, group in zip(group_labels, groups)])
                
                # Create a DataFrame for the Dunn test
                df = pd.DataFrame({'value': data, 'group': group_labels_array})
                
                # Use Dunn's test for post-hoc analysis
                dunn_results = dunn(df, val_col='value', group_col='group', p_adjust='bonferroni')
                
            
    #### IF DATA IS CATEGORICAL ####
    if not continuous:
        # Check Chi-squared assumptions to determine which test to use: Chi-squared or Fisher's exact test.
        _all_values = np.concatenate([g.values for g in groups])
        _all_labels = np.concatenate([[group_labels[i]] * len(g) for i, g in enumerate(groups)])
        contingency_table = pd.crosstab(
            _all_values, _all_labels,
            rownames=[groups[0].name if getattr(groups[0], 'name', None) else 'Category'],
            colnames=['Group']
        )
        
        # Let congtingency table show percentages between brackets after counts (for printing only)
        contingency_table_with_percent = (
            contingency_table.astype(str)
            + " ("
            + (contingency_table / contingency_table.sum() * 100).round(2).astype(str)
            + "%)"
        )

        # Add lines to the table
        line = "---------------------------------------------"

        # Print Table
        print()
        print(line)
        print(contingency_table_with_percent)
        print(line)
        print("* percent values are per column")

        print()

        # Calculate expected counts
        statistic, p_value, dof, expected_frequencies = stats.chi2_contingency(
            contingency_table
        )

        # Check if Chi-square assumptions are met
        # Assumption 1: At least 80% of the cells have expected frequencies >= 5
        cells_five_or_larger = np.sum(expected_frequencies >= 5)
        percent_of_five_or_greater = cells_five_or_larger / contingency_table.size

        # Assumption 2: No cell has expected frequencies less than 1
        cells_less_than_one = np.sum(expected_frequencies < 1)

        # Check assumptions
        if percent_of_five_or_greater >= 0.8 and cells_less_than_one == 0:
            print("Chi-squared assumptions are met")

            # Perform test
            test_name = "Chi-squared"
            test_statistic_sign = "Chi2"

            chi2_stat_unrounded = statistic  # Save unrounded value for effect size calculation
            statistic = round(statistic, 3)

            #MAHA# # Compute the effect size for Chi-squared (Cramer's V)
            n = contingency_table.sum().sum()  # Total sample size
            r, k = contingency_table.shape  # Rows and columns
            # Prevent division by zero
            if min(r, k) == 1:
                V = 0  # Cramer's V is undefined for 1x2 or 2x1 tables
            else:
                V = np.sqrt(chi2_stat_unrounded / (n * (min(r, k) - 1)))
                effect_size["label"] = "Cramer's V"
                effect_size["value"] = round(V, 3)
            # # Store the effect size
            # effect_size = {"label": "Cramer's V", "value": V}
            # print(f"Chi-square statistic: {chi2_stat:.3f}")
            # print(f"Cramer's V: {V:.3f}")

            # #perform post-hoc test after chi-squared test if p-value is significant
            #     if p_value < alpha:
            #         print("Post-hoc test is needed.")
            #         # Use pairwise comparisons with Bonferroni correction
            #         chi2_results = pairwise_chi2(data=df, x='group', y='value')
            #         print(chi2_results)
            #     else:
            #         print("No need for a post-hoc test.")
            
        else:
            print("Chi-squared assumptions are not met")

            # Check if it's a 2x2 contingency table
            is_2x2_table = contingency_table.shape == (2, 2)
            if is_2x2_table:
                # Perform Fisher's exact test
                statistic, p_value = stats.fisher_exact(contingency_table)
                statistic = round(statistic, 3)

                test_name = "Fisher's exact test"
                test_statistic_sign = "Odd's ratio"

                #compute the effect size for Fisher's exact (Odds Ratio)
                OR = statistic
                effect_size["label"] = "Odds Ratio"
                effect_size["value"] = OR
                
            else:
                print(
                    "The contingency table is not 2x2. Cannot perform Fisher's exact test"
                )
                print("Using Chi-squared test instead")
                # Perform test
                test_name = "Chi-squared"
                test_statistic_sign = "Chi2"

                chi2_stat_unrounded = statistic  # Save unrounded value for effect size calculation
                statistic = round(statistic, 3)

                # # Compute the effect size (Cramer's V)
                n = contingency_table.sum().sum()  # Total sample size
                r, k = contingency_table.shape  # Rows and columns
                # # Prevent division by zero
                if min(r, k) == 1:
                    V = 0  # Cramer's V is undefined for 1x2 or 2x1 tables
                else:
                    V = np.sqrt(chi2_stat_unrounded / (n * (min(r, k) - 1)))
                    effect_size["label"] = "Cramer's V"
                    effect_size["value"] = round(V, 3)
                # # Store the effect size
                # effect_size = {"label": "Cramer's V", "value": V}
                # print(f"Chi-square statistic: {chi2_stat:.3f}")
                # print(f"Cramer's V: {V:.3f}")
                
#MAHA#          #perform post-hoc test after chi-squared test if p-value is significant
                # if p_value < alpha:
                #     print("Post-hoc test is needed.")
                #     # Use pairwise comparisons with Bonferroni correction
                #     chi2_results = pairwise_chi2(data=df, x='group', y='value')
                #     print(chi2_results)
                # else:
                #     print("No need for a post-hoc test.")
                    
    #### Print Output ####
    print_title(test_name)
    if line1 != "":
        print(line1)
    if line2 != "":
        print(line2)

    print(test_statistic_sign + ":", statistic)

    print("p-value:", _display_p_value(p_value))

    if mann_whitney_effect_result is not None:
        confidence_level = 100 - (alpha * 100)
        probability = mann_whitney_effect_result["probability_superiority"]
        probability_ci = mann_whitney_effect_result["probability_CI"]
        delta = mann_whitney_effect_result["cliffs_delta"]
        delta_ci = mann_whitney_effect_result["cliffs_delta_CI"]
        probability_decimals = _effect_ci_display_decimals(
            probability_ci, null_value=0.5
        )
        delta_decimals = _effect_ci_display_decimals(delta_ci, null_value=0.0)
        print(
            f"Probability of superiority ({group_labels[0]} > {group_labels[1]}, "
            f"ties count as 0.5): {probability:.3f}"
        )
        print(
            f"{confidence_level}% compatible CI: "
            f"[{probability_ci[0]:.{probability_decimals}f}, "
            f"{probability_ci[1]:.{probability_decimals}f}]"
        )
        print(f"Cliff's Delta (δ): {delta:.3f}")
        print(
            f"{confidence_level}% compatible CI for δ: "
            f"[{delta_ci[0]:.{delta_decimals}f}, {delta_ci[1]:.{delta_decimals}f}]"
        )
        null_excluded = probability_ci[1] < 0.5 or probability_ci[0] > 0.5
        print(
            f"* Null values (probability = 0.5; δ = 0) are "
            f"{'excluded from' if null_excluded else 'included in'} these intervals."
        )
        print(
            "* Compatible effect intervals use the LAPH/proportional-odds "
            "working model; the Mann-Whitney test itself does not require it."
        )

        if mann_whitney_shift_result is not None:
            shift_ci = mann_whitney_shift_result["CI"]
            shift_inclusive = mann_whitney_shift_result["CI_inclusive"]
            left_bracket = "[" if shift_inclusive[0] else "("
            right_bracket = "]" if shift_inclusive[1] else ")"
            print(
                f"Optional Hodges-Lehmann location-shift estimate: "
                f"{mann_whitney_shift_result['difference']:.3f}"
            )
            print(
                f"{confidence_level}% location-shift confidence set: "
                f"{left_bracket}{shift_ci[0]:.3f}, {shift_ci[1]:.3f}{right_bracket}"
            )
            print("* Interpret this only when a common-shape location-shift model is defensible.")
    
    if ci_lower!=None and ci_upper!=None:
        confidence_level= 100-(alpha*100)
        left_bracket = "[" if ci_inclusive[0] else "("
        right_bracket = "]" if ci_inclusive[1] else ")"
        print(f"{confidence_level}% CI: {left_bracket}{ci_lower:.2f}, {ci_upper:.2f}{right_bracket}")
        if not all(ci_inclusive):
            print("* Parentheses indicate an excluded endpoint.")
    print()

    #

    if p_value < alpha:
        printColor("There is a significant difference between groups.", "green")
    else:
        printColor("There is no significant difference between groups.", "red")


    # Print effect size
    if len(effect_size) > 0 and mann_whitney_effect_result is None:
        print("Effect size (" + effect_size["label"] + "):", effect_size["value"])

    #Print post-hoc test results
    if tukey_results is not None:
        print(tukey_results)
    elif dunn_results is not None:
        print(dunn_results)

    plot_group_comparison(
        groups=groups,
        group_labels=group_labels,
        test_name=test_name,
        p_value=p_value,
        alpha=alpha,
        do_graphs=do_graphs,
        graphs_for_non_significance=graphs_for_non_significance,
        continuous=continuous,
        all_normal=all_normal,
        contingency_table=contingency_table,
        subject_palette=subject_palette,
    )
          


################################################################################################
def plot_dependent_group_comparison(
    groups,
    group_labels,
    p_value,
    alpha=0.05,
    do_graphs=True,
    graphs_for_non_significance=False,
    use_median=True,
):
    if not do_graphs or p_value is None:
        return

    if p_value >= alpha and not graphs_for_non_significance:
        return

    if use_median:
        statistics = [np.median(group) for group in groups]
        label = 'Median'
    else:
        statistics = [np.mean(group) for group in groups]
        label = 'Mean'

    plt.figure(figsize=(max(4.8, len(group_labels) * 0.95), 4.2))
    ax = plt.gca()
    ax.plot(group_labels, statistics, marker='o', color='#52796F', linewidth=1.8, markersize=6)
    ax.set_xlabel('Time Point', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{label} Score', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', labelrotation=0, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    for tick_label in ax.get_xticklabels():
        tick_label.set_fontweight('normal')
        tick_label.set_ha('center')
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontweight('normal')
    ax.grid(axis='y', alpha=0.25)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.show()



# def plot_categorical_statistics_over_time(groups, test_name):
#     # Convert groups to a DataFrame for easier plotting
#     data_wide = pd.DataFrame({f'Time Point {i+1}': group for i, group in enumerate(groups)})
    
#     # Ensure data is categorical
#     data_wwide = data_wide.apply(lambda x: x.astype('category'))
    
#     # Plot the frequency of each category at each time point
#     category_counts = data_wide.apply(lambda x: x.value_counts()).T.fillna(0)
    
#     # Transpose the data for the desired plot
#     category_counts = category_counts.T
    
#     category_counts.plot(kind='barh', stacked=True)
#     plt.title(f'Frequency of Categories at Each Time Point for {test_name}')
#     plt.xlabel('Time Points')
#     plt.ylabel('Categories')
#     plt.legend(title='Frequency')
#     plt.show()

# Compare series/columns for independent groups
# # #save a copy of groups with missing values:
    # groups_with_missing = copy.deepcopy(groups)

    
    
def compare_dep(
    groups,
    group_labels=None,
    alpha=0.05,
    categorical_limit=20,
    force_test=None,
    force_normality=False,
    force_non_normality=False,
    data_type=None,
    do_graphs=True,
    graphs_for_non_significance=False,
    difference_decimals=None,
):  # Depends on print_mean_std()
    # Note: this function does not handle the independent variable (grouping variable). It assumes that the series are already grouped.
    # groups: a list of pandas series/columns to be compared
    # group_labels: a list of labels for the groups. If not provided, the group names + group number will be used as labels. The list of labels has to correspond to the list of groups.
    # force_test: if provided, the function will use the test provided instead of determining it automatically. The test has to be a string and has to be one of the following: "ttest", "anova", "mannwhitney", "kruskalwallis", "chisquare", 'fisherexact'
    # alpha: the alpha value to be used for the test. Default is 0.05
    # categorical_limit: the number of unique values in a series/column that determines whether the data is continuous or categorical. If it is below this limit, data will be treated as categorical. Default is 20.

    # Declare vars
    test_name = ""
    test_statistic_sign = ""
    line1 = ""
    line2 = ""
    effect_size = {}
    post_hoc = {}
    ci_lower = None
    ci_upper = None
    ci_inclusive = (True, True)
    ci_estimand = None
    use_median_for_plot = None
    all_normal = None

    # If group_labels is not provided, use the group names + group number as labels
    if group_labels is None:
        group_labels = []
        index = 0
        for group in groups:
            index += 1
            group_labels.append(f"{group.name} (Group {index})")


    ##########################################################################

    # # Drop isna values from each group #open again
    index = 0
    df = pd.concat(groups, axis=1)

    #number of dropped rows (complete observations removed due to any missing value)
    dropped_rows = df.isna().any(axis=1).sum()
    if dropped_rows > 0:
        print(f"Dropping {dropped_rows} incomplete observations (rows with any missing values). Remaining n = {len(df) - dropped_rows}.")

    df = df.dropna()
    groups = [df[col] for col in df.columns]

    # Guard: abort if too few complete observations remain
    n_complete = len(groups[0]) if groups else 0
    if n_complete < 3:
        printColor(
            f"Cannot run test: only {n_complete} complete observation(s) remain after "
            "dropping rows with missing values. Minimum required is 3.",
            "yellow"
        )
        return

    print()
    print("alpha is set to", alpha)

    # Determine if data is continuous or categorical
    if data_type == "cont":
        continuous = True
        print("Data type explicitly set to continuous.")
    elif data_type in ("cat", "ordinal", "nominal"):
        continuous = False
        print("Data type explicitly set to categorical.")
    else:
        nunique = max([group.nunique() for group in groups])
        continuous = nunique > categorical_limit
        printColor(
            f"Warning: data_type not set. Inferring '{'cont' if continuous else 'cat'}' "
            f"from unique-value count ({nunique}). Pass data_type='cont' or "
            f"data_type='cat' to suppress this warning.",
            "yellow"
        )

    #### IF DATA IS CONTINUOUS ####
    if continuous:

        ### Normality Test ###
        # Use Shapiro-Wilk test to check for normality of each group
        if force_non_normality:
            all_normal = False
            print("The compare function is forced to use non-parametric tests.")
        elif force_normality:
            all_normal = True
            print("The compare function is forced to use parametric tests.")
        else:
            all_normal = True

        index = 0

        for group in groups:
            if not force_normality and not force_non_normality:
                print_title("Test for Normality")
                statistic, p_value = stats.shapiro(group)
                statistic = round(statistic, 3)
                # print group label for this group
                print("Test for normality (Shapiro-Wilk) for", group_labels[index])
                print("-------------------------------------------------")
                print("Statistic:", statistic)
                print("p-value:", _display_p_value(p_value))
                if p_value > alpha:
                    print("Data is normally distributed.")
                else:
                    print("Data is not normally distributed.")
                    all_normal = False
                index += 1

            # # Plot QQ
            # print()
            # plt.figure(figsize=(4, 3))
            # stats.probplot(group, dist="norm", plot=plt)
            # plt.title("Normal Q-Q Plot for " + group_labels[index])
            # plt.show()
            # print("-------------------------------------------------")
            # index += 1
            # if index < len(groups):
            #     print()

        # Print mean and standard deviation for each group if all groups are normally distributed
        print_title("Central Tendency")
        if all_normal:
            index = 0
            for group in groups:
                print_mean_std(
                    group_labels[index],
                    group,
                )

                # Print n = number of observations in each group
                print("n:", len(group))
                if index < len(groups):
                    print()
                index += 1

        # Print median and IQR for each group if not all groups are normally distributed
        else:
            index = 0
            for group in groups:
                print_median_iqr(group_labels[index], group)

                # Print n = number of observations in each group
                print("n:", len(group))
                if index < len(groups):
                    print()
                index += 1

        # If all groups are normally distributed and we have 2 groups only, use paired t-test (dependent, related)
        if all_normal and len(groups) == 2:
            test_name = "Paired t-test"
            test_statistic_sign = "t"
            line1 = "All data is normally distributed."

            # Use t-test to compare all group
            statistic, p_value = stats.ttest_rel(groups[0], groups[1])
            statistic = round(statistic, 3)

            #ci difference
            ci_result = compute_confidence_interval_difference(groups[0], groups[1], method="mean", paired=True, alpha=alpha)
            ci_lower, ci_upper = ci_result["CI"]
            ci_inclusive = ci_result["CI_inclusive"]

# # Compute the effect size (Cohen's d)
# n = len(groups[0])
# d = (np.mean(groups[0]) - np.mean(groups[1])) / np.std(
#     groups[0] - groups[1], ddof=1
# )

# effect_size["label"] = "Cohen's d"
# effect_size["value"] = d

            #MAHA# Compute the effect size for paired t-test (Cohen's d paired)
            # Compute paired differences
            differences = groups[0] - groups[1]
            # Compute Cohen’s d for paired samples
            n = len(differences)  # Number of pairs
            d = np.mean(differences) / np.std(differences, ddof=1)  # Corrected standard deviation
            effect_size["label"] = "Cohen's d (Paired)"
            effect_size["value"] = round(d, 3)
            print(f"Cohen's d (Paired): {d:.3f}")
            use_median_for_plot = False
            
        # If all groups are normally distributed, and we have more than 2 groups, use repeated measures ANOVA
        elif all_normal and len(groups) > 2: 
            test_name = "Repeated Measures ANOVA"
            test_statistic_sign = "F"
            line1 = "Using repeated measures ANOVA because there are more than 2 groups."

            #MAHA#    # Use repeated measures ANOVA to compare all group 
            # Prepare the data for repeated measures ANOVA
            # data = pd.DataFrame({
            #     "Score": np.concatenate(groups_with_missing),
            #     "Condition": np.repeat(group_labels, [len(group) for group in groups_with_missing]),
            #     "Subject": np.tile(np.arange(len(groups_with_missing[0])), len(groups_with_missing)) 
            # })

            data = pd.DataFrame({
                "Score": np.concatenate(groups),
                "Condition": np.repeat(group_labels, [len(group) for group in groups]),
                "Subject": np.tile(np.arange(len(groups[0])), len(groups)) 
            })
        
            anova_results = pg.rm_anova(dv="Score", within="Condition", subject="Subject", data=data, correction=True) #score=col values, condition=time points, subject=group

            # Check sphericity and select the appropriate p-value
            # pg.rm_anova with correction=True runs Mauchly's test and adds GG-corrected columns
            if 'p-GG-corr' in anova_results.columns and not np.isnan(anova_results.loc[0, 'p-GG-corr']):
                sphericity_ok = anova_results.loc[0, 'sphericity'] if 'sphericity' in anova_results.columns else True
                if not sphericity_ok:
                    print("Sphericity violated (Mauchly's test p < 0.05). Using Greenhouse-Geisser correction.")
                    p_col = 'p-GG-corr'
                else:
                    print("Sphericity assumption met (Mauchly's test p >= 0.05). Using uncorrected p-value.")
                    p_col = 'p_unc' if 'p_unc' in anova_results.columns else 'p-unc'
            else:
                p_col = 'p_unc' if 'p_unc' in anova_results.columns else 'p-unc'

            statistic, p_value = anova_results.loc[0, ["F", p_col]]

            # Round results
            statistic = round(statistic, 3)

            # Compute partial eta squared from F and dfs (correct formula, version-independent)
            # ηp² = (F × df1) / (F × df1 + df2)
            F_val = anova_results.loc[0, "F"]
            df1 = anova_results.loc[0, "ddof1"]
            df2 = anova_results.loc[0, "ddof2"]
            eta_p2 = (F_val * df1) / (F_val * df1 + df2) if (F_val * df1 + df2) != 0 else np.nan
            effect_size["label"] = "Partial Eta Squared (ηp²)"
            effect_size["value"] = round(eta_p2, 3)

            # print(f"Partial Eta Squared (ηp²): {eta_squared:.3f}")

            #perform post-hoc test after repeated measures ANOVA if p-value is significant
            if p_value < alpha:
                print("Post-hoc test is needed.")
                # Use pairwise comparisons with Bonferroni correction
                pairwise_results = pairwise_tests(data=data, dv="Score", within="Condition", subject="Subject", padjust="bonferroni")
                post_hoc = {"value": pairwise_results, "label": "Bonferroni"}
            use_median_for_plot = False
        
        # If all groups are not normally distributed, and we have 2 groups, use Wilcoxon signed-rank test
        elif not all_normal and len(groups) == 2:
            test_name = "Wilcoxon signed-rank test"
            test_statistic_sign = "Statistic"
            line1 = "Not all data is normally distributed."

            # Use Wilcoxon signed-rank test to compare all group
            differences = _paired_differences(
                groups[0], groups[1], difference_decimals=difference_decimals
            )
            wilcoxon_method = _wilcoxon_method(differences)
            statistic_raw, p_value = _wilcoxon_test(
                differences, method=wilcoxon_method
            )
            statistic = round(statistic_raw, 3)

            #ci difference
            ci_result = compute_confidence_interval_difference(
                groups[0],
                groups[1],
                method="median",
                paired=True,
                alpha=alpha,
                difference_decimals=difference_decimals,
            )
            ci_lower, ci_upper = ci_result["CI"]
            ci_inclusive = ci_result["CI_inclusive"]
            ci_estimand = "Hodges-Lehmann pseudomedian of paired differences"

            # Compute the effect size for Wilcoxon signed-rank (r)
            # Use unrounded statistic; divide by sqrt(2n) per standard formula
            n = len(groups[0])  # Number of pairs
            z_score = statistic_raw - (n * (n + 1) / 4)  # Approximate Z-score correction
            z_score /= np.sqrt(n * (n + 1) * (2 * n + 1) / 24)  # Standard error

            # Compute effect size r = Z / sqrt(2n)  (total observations = 2n for paired data)
            r = z_score / np.sqrt(2 * n)

            # # Store the effect size
            # effect_size = {"label": "r", "value": round(r, 3)}

            # Print results
            effect_size["label"] = "r"
            effect_size["value"] = round(r, 3)
            # print(f"Wilcoxon Statistic: {statistic}")
            # print(f"Z-score Approximation: {z_score:.3f}")
            # print(f"Effect Size (r): {r:.3f}")

# # Compute the effect size (r)
# n = len(groups[0])
# r = statistic / (n * (n + 1) / 2)
# effect_size["label"] = "r"
# effect_size["value"] = r
            use_median_for_plot = True


#MAHA#   # If all group are not normally distributed, and we have more than 2 groups, use Friedman test
        elif not all_normal and len(groups) > 2:
            test_name = "Friedman test"
            test_statistic_sign = "Statistic"
            line1 = "Not all data is normally distributed."

            # data = pd.DataFrame({
            #     "Score": np.concatenate(groups_with_missing),
            #     "Condition": np.repeat(group_labels, [len(group) for group in groups_with_missing]),
            #     "Subject": np.tile(np.arange(len(groups_with_missing[0])), len(groups_with_missing))
            # })

            
            data = pd.DataFrame({
                "Score": np.concatenate(groups),
                "Condition": np.repeat(group_labels, [len(group) for group in groups]),
                "Subject": np.tile(np.arange(len(groups[0])), len(groups))
            })

            # Use Friedman test to compare all group
            friedman_results = pg.friedman(data=data, dv='Score', within='Condition', subject='Subject')
        
            # Extract F-statistic and p-value
            statistic = friedman_results['Q'].iloc[0]
            p_col = 'p_unc' if 'p_unc' in friedman_results.columns else 'p-unc'
            p_value = friedman_results[p_col].iloc[0]
            statistic = round(statistic, 3)

            # Calculate Kendall's W
            n = len(groups[0])  # number of subjects
            k = len(groups)     # number of conditions
            chi2 = friedman_results['Q'].values[0]  # Friedman test statistic
            kendalls_w = chi2 / (n * (k - 1))

            effect_size["label"] = "Kendall's W"
            effect_size["value"] =round(friedman_results['W'].iloc[0],3)
            use_median_for_plot = True
            
            # perform post hoc test after Friedman test if p-value is significant:
            #Use Nemenyi post-hoc test for post-hoc analysis:
            if p_value < alpha:
                print("Post-hoc test is needed.")
                nemenyi_results = sp.posthoc_nemenyi_friedman (np.array(groups).T)
                post_hoc = {"value": nemenyi_results, "label": "posthoc_nemenyi_friedman"}
                
       

        #### IF DATA IS CATEGORICAL #### 
    if not continuous: 

        # Print frequencies and proportions for each group (time point)
        print_title("Frequencies")
        for group, label in zip(groups, group_labels):
            counts = group.value_counts().sort_index()
            proportions = group.value_counts(normalize=True).sort_index() * 100
            freq_table = pd.concat([counts, proportions.round(2)], axis=1)
            freq_table.columns = ['Count', 'Percent (%)']
            print(f"{label} (n={len(group)}):")
            print(freq_table.to_string())
            print()

        # Convert groups to categorical format
        groups = [group.astype("category") for group in groups]

        #check if the data is binary or ordinal
        nunique = max(group.nunique() for group in groups)

        # Binary Data (Nominal) with Two Time Points → McNemar's Test
        if len(groups) == 2 and nunique == 2:
            test_name = "McNemar's test"
            test_statistic_sign = "Statistic"
            line1 = "Data is binary and has two time points: using McNemar's test."

            # Create a contingency table
            contingency_table = pd.crosstab(groups[0], groups[1])
            # n = contingency_table.sum().sum() #ci

            # Run McNemar's test
            result = mcnemar(contingency_table, exact=True)  # exact=True for exact test
                                                # exact=False for chi-square approximation
            statistic = round(result.statistic, 3)
            p_value = result.pvalue
            
            # Compute effect size (Odds Ratio)
            OR = (contingency_table.iloc[1, 0] + 1) / (contingency_table.iloc[0, 1] + 1)
            effect_size["label"] = "Odds Ratio"
            effect_size["value"] = round(OR, 3)

            # Plot the data
            # plot_categorical_statistics_over_time(groups, test_name)

        # Binary Data (Nominal) with More Than Two Time Points → Cochran's Q Test
        elif len(groups) > 2 and nunique == 2:
            test_name = "Cochran's Q test"
            test_statistic_sign = "Statistic"
            line1 = "Data is binary and has more than two time points: using Cochran's Q test."

            # Create a DataFrame with subjects as rows and conditions as columns
            data_wide = pd.DataFrame({label: group for label, group in zip(group_labels, groups)})
            data_wide=data_wide.dropna()

            result = cochrans_q(data_wide)
            statistic = round(result.statistic, 3)
            p_value = result.pvalue

            
            # Compute Kendall's W as effect size
            W = statistic / (len(groups) - 1)
            effect_size["label"] = "Kendall's W"
            effect_size["value"] = round(W, 3)

            # Plot the data
            # plot_categorical_statistics_over_time(groups, test_name)

        # Ordinal(non_binary) Data with Two Time Points → Stuart-Maxwell Test
        elif len(groups) == 2 and nunique > 2:
            test_name = "Stuart-Maxwell test"
            test_statistic_sign = "Statistic"
            line1 = "Data is ordinal (non_binary) and has two time points: using Stuart-Maxwell Test."

            # Create a contingency table
            contingency_table = pd.crosstab(groups[0], groups[1])
            contingency_array = contingency_table.to_numpy()
            # n = contingency_table.sum().sum() #ci
            table =SquareTable(contingency_array)

            # Compute Stuart-Maxwell test from statsmodels package
            result = table.homogeneity(method='stuart_maxwell')

            # Store results
            p_value = result.pvalue
            statistic = round(result.statistic, 3)

            # Compute effect size (Cohen's W approximation)
            N = contingency_array.sum()
            W = np.sqrt(statistic / N)

    
            effect_size["label"] = "Cohen's W"
            effect_size["value"] = round(W, 3)

            # Plot the data
            # plot_categorical_statistics_over_time(groups, test_name)

        # Ordinal (non-binary) Data with More Than Two Time Points → Friedman Test
        elif len(groups) > 2 and nunique > 2:
            
            #check if user enters data type. if not, throw an error
            if data_type is None:
                raise ValueError("Please enter the data_type parameter: 'ordinal' or 'nominal'")
            
            #if data is ordinal perform Friedman test
            if data_type == 'ordinal': 

                test_name = "Friedman test"
                test_statistic_sign = "Statistic"
                line1 = "Data is ordinal (non-binary) and has more than two time points: using Friedman Test."
        
                statistic, p_value = friedmanchisquare(*groups)
            

                #Compute effect size (Kendall’s W)
                # W = friedman_results['W'].iloc[0]

                # Store results
                statistic=round(statistic,3)

            
                # effect_size["label"] = "Kendall's W"
                # effect_size["value"] = round(W, 3)

                # Plot the data
                # plot_categorical_statistics_over_time(groups, test_name)

                # #perform a post-hoc test after Friedman test if p-value is significant
                # # Use Nemenyi post-hoc test for post-hoc analysis
                # if p_value < alpha:
                #     print("Post-hoc test is needed.")
                #     # Convert groups into a 2D NumPy array (subjects x conditions)
                #     data_matrix = np.column_stack(groups)  
                #     # Run Nemenyi post-hoc test
                #     nemenyi_results = sp.posthoc_nemenyi_friedman(data_matrix)
                #     print(nemenyi_results)
                # else:
                #     print("No significant differences found.")
            else:                 
                raise ValueError("statkit does not support test for nominal data at more than two time points.")

    #### Print Output ####
    print_title(test_name)
    if line1 != "":
        print(line1)
    if line2 != "":
        print(line2)

    print(test_statistic_sign + ":", statistic)


    print("p-value:", _display_p_value(p_value))
    
    if ci_lower!=None and ci_upper!=None:
        confidence_level= 100-(alpha*100)
        left_bracket = "[" if ci_inclusive[0] else "("
        right_bracket = "]" if ci_inclusive[1] else ")"
        ci_name = f" CI for {ci_estimand}" if ci_estimand else " CI"
        print(f"{confidence_level}%{ci_name}: {left_bracket}{ci_lower:.2f}, {ci_upper:.2f}{right_bracket}")
        if not all(ci_inclusive):
            print("* Parentheses indicate an excluded endpoint.")
        if ci_estimand:
            zero_is_included = (
                (ci_lower < 0 or (ci_lower == 0 and ci_inclusive[0]))
                and (ci_upper > 0 or (ci_upper == 0 and ci_inclusive[1]))
            )
            print(f"* Null value 0 is {'included in' if zero_is_included else 'excluded from'} this confidence set.")
    print()

    if p_value < alpha:
        printColor("There is a significant difference between groups.", "green")
    else:
        printColor("There is no significant difference between groups.", "red")

    # Print effect size
    if len(effect_size) > 0:
        print()
        print("Effect size (" + effect_size["label"] + "):", effect_size["value"])

    #Print post-hoc test results
    if len(post_hoc) > 0:
        print()
        print("Post-hoc (" + post_hoc["label"] + "):")
        print(post_hoc["value"])

    if continuous and use_median_for_plot is not None:
        plot_dependent_group_comparison(
            groups=groups,
            group_labels=group_labels,
            p_value=p_value,
            alpha=alpha,
            do_graphs=do_graphs,
            graphs_for_non_significance=graphs_for_non_significance,
            use_median=use_median_for_plot,
        )


# A function to perform correlation analysis between two continuous or ordinal variables
def correlate(data, x, y, force_test=None, alpha=0.05):

    """
    Automatically selects and performs the appropriate correlation test.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        x (str): Name of the first column.
        y (str): Name of the second column.
        force_test (str, optional): Force a specific correlation test. Options:
            - 'pearson'  : Force Pearson correlation (assumes both variables are continuous and normal).
            - 'spearman' : Force Spearman correlation (use for ordinal or non-normal data).
            - None       : Auto-select by running Shapiro-Wilk on both variables. A yellow warning
                           is printed because this assumes both variables are continuous.
        alpha (float): Significance level for hypothesis testing. Default is 0.05.

    Returns:
        dict: Dictionary containing correlation coefficient, p-value, significance, and direction.
    """

    # Check if columns exist
    if x not in data.columns:
        raise Exception(f'Column "{x}" does not exist in the data.')
    if y not in data.columns:
        raise Exception(f'Column "{y}" does not exist in the data.')

    # Drop missing values without modifying the original DataFrame
    data_clean = data.dropna(subset=[x, y])

    # Guard: need at least 3 observations (Shapiro-Wilk minimum)
    if len(data_clean) < 3:
        printColor(
            f"Skipping correlation between '{x}' and '{y}': only {len(data_clean)} complete "
            f"observation(s) remain after dropping NaN — need at least 3.",
            "yellow"
        )
        return None

    # Check for constant variables (zero variance)
    if data_clean[x].std() == 0 or data_clean[y].std() == 0:
        printColor(
            f"Skipping correlation between '{x}' and '{y}': one or both variables have zero variance (all values are identical).",
            "yellow"
        )
        return None

    # --- Determine method ---
    if force_test == "spearman":
        print("force_test='spearman': using Spearman correlation.")
        method = "spearman"
    elif force_test == "pearson":
        print("force_test='pearson': using Pearson correlation.")
        method = "pearson"
    else:
        # Auto-select via Shapiro-Wilk
        printColor(
            "Warning: force_test not set. Running Shapiro-Wilk and selecting Pearson or Spearman "
            "automatically. This assumes both variables are continuous. If either variable is "
            "ordinal or categorical, use force_test='spearman'.",
            "yellow"
        )
        if len(data_clean) <= 5000:
            p_x = stats.shapiro(data_clean[x])[1]
            p_y = stats.shapiro(data_clean[y])[1]
        else:
            p_x, p_y = 1, 1  # Assume normality for large datasets
        method = "pearson" if p_x > alpha and p_y > alpha else "spearman"

    # --- Run the test ---
    if method == "pearson":
        statistic, p_value = stats.pearsonr(data_clean[x], data_clean[y])
    else:
        statistic, p_value = stats.spearmanr(data_clean[x], data_clean[y], nan_policy='omit')

    # Interpretation
    significance = "Significant" if p_value < alpha else "Not Significant"

    # **Determine correlation direction**
    if statistic > 0:
        direction = "Positive correlation"
    elif statistic < 0:
        direction = "Negative correlation"
    else:
        direction = "No correlation"

    #creat a scatter plot for the data
    # plt.figure(figsize=(6, 4))
    # sns.scatterplot(x=x, y=y, data=data_clean)
    # plt.title(f"{method.capitalize()} Correlation")
    # plt.xlabel(x)
    # plt.ylabel(y)
    # plt.show()
    # print()    

    # Return results
    # Keep full precision in the returned values; round only for display.
    result={
        "method": method,
        "correlation_coefficient": float(statistic),
        "p_value": float(p_value),
        "direction": direction,
        "significance": significance
    } 

    # # Iterate over the dictionary items
    for key, value in result.items():
        if key == "significance":
            color = "green" if str(value).lower().startswith("sign") else "red"
            printColor(f"{key}: {value}", color)
        elif key == "p_value":
            print(f"{key}:", _display_p_value(p_value))
        elif key == "correlation_coefficient":
            print(f"{key}:", round(value, 3))
        else:
            print(f"{key}:", value)
    
    return result


# A function to calculate confidence interval difference between two groups
# works both for parametric (method="mean") and non-parametric (method="median") tests
# works for both paired (paired=True) and independent (paired=False) samples.
def compute_confidence_interval_difference(
    group1,
    group2,
    method="mean",
    paired=False,
    alpha=0.05,
    n_bootstrap=10000,
    difference_decimals=None,
):
    """
    Computes a confidence interval for the difference in means (parametric) or
    the Hodges-Lehmann location shift (non-parametric), supporting both
    independent and dependent (paired) samples.

    Parameters:
        group1 (array-like): First sample.
        group2 (array-like): Second sample (paired or independent).
        method (str): "mean" for a t-test mean difference or "median" for a
            test-inverted Hodges-Lehmann location shift. The name "median" is
            retained for backward compatibility.
        paired (bool): If True, performs paired (dependent) tests instead of independent.
        alpha (float): Significance level (default is 0.05 for 95% CI).
        n_bootstrap (int): Retained for backward compatibility; test inversion
            does not use bootstrap resampling.
        difference_decimals (int or None): For paired non-parametric inference,
            round the precomputed paired differences to this many decimals
            before ranking. Use the known measurement precision when floating
            subtraction would otherwise create artificial rank differences.

    Returns:
        dict: Contains difference, confidence interval, test statistic, and p-value.
    """

    if method == "mean":
        if paired:
            # --- Parametric: Paired t-test ---
            diffs = np.array(group1) - np.array(group2)  # Compute paired differences
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs, ddof=1)
            n = len(diffs)

            # Perform paired t-test
            test_stat, p_value = stats.ttest_rel(group1, group2)

            # Compute standard error of the mean difference
            se_diff = std_diff / np.sqrt(n)

            # Compute critical t-value
            t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

            # Compute confidence interval
            ci_lower = mean_diff - t_crit * se_diff
            ci_upper = mean_diff + t_crit * se_diff

        else:
            # --- Parametric: Independent t-test ---
            mean1, mean2 = np.mean(group1), np.mean(group2)
            std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
            n1, n2 = len(group1), len(group2)

            # Compute standard error of the difference
            se_diff = np.sqrt((std1**2 / n1) + (std2**2 / n2))

            # Perform independent t-test (Welch's by default)
            test_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

            # Compute degrees of freedom (Welch-Satterthwaite equation)
            df = ((std1**2 / n1) + (std2**2 / n2))**2 / (
                ((std1**2 / n1)**2 / (n1 - 1)) + ((std2**2 / n2)**2 / (n2 - 1))
            )

            # Get critical t-value
            t_crit = stats.t.ppf(1 - alpha / 2, df)

            # Compute confidence interval
            mean_diff = mean1 - mean2
            ci_lower = mean_diff - t_crit * se_diff
            ci_upper = mean_diff + t_crit * se_diff

        ci_inclusive = (True, True)
        test_method = "t"

    elif method == "median":
        if paired:
            # --- Non-Parametric: Wilcoxon Signed-Rank Test (Paired) ---
            # Invert the same explicit Wilcoxon policy used for the p-value.
            diffs = _paired_differences(
                group1, group2, difference_decimals=difference_decimals
            )
            n = len(diffs)
            if n == 0:
                raise ValueError("Paired samples must contain at least one observation")

            test_method = _wilcoxon_method(diffs)
            test_result = _wilcoxon_test(diffs, method=test_method)
            test_stat, p_value = test_result.statistic, test_result.pvalue

            if np.all(diffs == 0):
                observed_diff = 0.0
                ci_lower, ci_upper = 0.0, 0.0
                ci_inclusive = (True, True)
            else:
                # Walsh averages: (d_i + d_j) / 2 for all i <= j
                pair_sums = np.add.outer(diffs, diffs)
                walsh = pair_sums[np.triu_indices(n)] / 2
                walsh.sort()
                observed_diff = np.median(walsh)

                def wilcoxon_p_value_at(shift):
                    if shift == 0:
                        return p_value
                    return _wilcoxon_test(
                        diffs - shift, method=test_method
                    ).pvalue

                (ci_lower, ci_upper), ci_inclusive = _test_inverted_interval(
                    walsh,
                    wilcoxon_p_value_at,
                    observed_diff,
                    alpha,
                )
        else:
            # --- Non-Parametric: Mann-Whitney U Test (Independent) ---
            # Invert the same explicit Mann-Whitney policy used for the p-value.
            group1_arr = np.asarray(group1, dtype=float)
            group2_arr = np.asarray(group2, dtype=float)
            n1, n2 = len(group1_arr), len(group2_arr)
            if n1 == 0 or n2 == 0:
                raise ValueError("Independent samples must both contain observations")

            test_method = _mann_whitney_method(group1_arr, group2_arr)
            test_result = _mann_whitney_test(group1_arr, group2_arr, method=test_method)
            test_stat, p_value = test_result.statistic, test_result.pvalue

            # All pairwise differences (Hodges-Lehmann estimator)
            all_diffs = (group1_arr[:, None] - group2_arr[None, :]).ravel()
            all_diffs.sort()
            observed_diff = np.median(all_diffs)

            def mann_whitney_p_value_at(shift):
                if shift == 0:
                    return p_value
                return _mann_whitney_test(
                    group1_arr - shift,
                    group2_arr,
                    method=test_method,
                ).pvalue

            (ci_lower, ci_upper), ci_inclusive = _test_inverted_interval(
                all_diffs,
                mann_whitney_p_value_at,
                observed_diff,
                alpha,
            )

        mean_diff = observed_diff  # Hodges-Lehmann location-shift estimate

    else:
        raise ValueError("Invalid method. Choose 'mean' (parametric) or 'median' (non-parametric).")

    return {
        "difference": mean_diff,
        "CI": (ci_lower, ci_upper),
        "CI_inclusive": ci_inclusive,
        "test_stat": test_stat,
        "p_value": p_value,
        "test_method": test_method,
        "null_value_included": (
            (ci_lower < 0 or (ci_lower == 0 and ci_inclusive[0]))
            and (ci_upper > 0 or (ci_upper == 0 and ci_inclusive[1]))
        ),
    }


def _prepare_biserial_inputs(
    data, continuous_var, binary_var, minimum_n=3, minimum_group_n=2
):
    """Validate and align one continuous and one dichotomous column.

    Shared by the biserial correlations so all of them use the same level
    ordering, and therefore the same sign convention.

    Problems with the *data* (too few observations, a collapsed binary
    variable, a group that is too small, a constant or non-numeric continuous
    variable) print a yellow explanation and return ``None`` so that a loop
    over many variable pairs reports the skip and carries on -- the same
    convention ``correlate`` uses. Problems with the *call* (a missing column,
    the same column passed twice) still raise, because no data would fix them.

    Returns:
        tuple or None: (data_clean, binary_levels, codes, group_sizes) where
            ``codes`` is 0 for the first sorted level and 1 for the second, and
            ``data_clean[continuous_var]`` has been coerced to float. ``None``
            if the pair cannot support a biserial correlation.
    """
    for variable in (continuous_var, binary_var):
        if variable not in data.columns:
            raise ValueError(f'Column "{variable}" does not exist in the data.')
    if continuous_var == binary_var:
        raise ValueError("continuous_var and binary_var must be different columns.")

    pair = f"'{continuous_var}' & '{binary_var}'"

    def skip(reason):
        printColor(f"Skipping {pair}: {reason}", "yellow")
        return None

    data_clean = data[[continuous_var, binary_var]].dropna().copy()
    if len(data_clean) < minimum_n:
        return skip(
            f"only {len(data_clean)} complete observation(s) remain after dropping "
            f"missing values — need at least {minimum_n}."
        )

    binary_levels = data_clean[binary_var].unique()
    if len(binary_levels) != 2:
        return skip(
            f'"{binary_var}" has {len(binary_levels)} observed level(s) among the '
            f"complete cases — need exactly 2. "
            + (
                "Every remaining case falls in the same category, so one group is "
                "empty."
                if len(binary_levels) < 2
                else f"Found levels {sorted(map(str, binary_levels))}."
            )
        )
    # Sort the levels so the sign of the coefficient does not depend on row order.
    try:
        binary_levels = sorted(binary_levels)
    except TypeError:
        binary_levels = list(binary_levels)

    continuous_values = pd.to_numeric(data_clean[continuous_var], errors="coerce")
    if continuous_values.isna().any():
        return skip(f'"{continuous_var}" contains non-numeric values.')
    if continuous_values.nunique() < 2:
        return skip(
            f'"{continuous_var}" has zero variance (all values are identical).'
        )
    # Store the coerced values so numeric-looking strings are not plotted as categories.
    data_clean[continuous_var] = continuous_values.astype(float)

    codes = pd.Categorical(
        data_clean[binary_var], categories=binary_levels, ordered=True
    ).codes
    group_sizes = {
        str(binary_levels[0]): int((codes == 0).sum()),
        str(binary_levels[1]): int((codes == 1).sum()),
    }
    if min(group_sizes.values()) < minimum_group_n:
        return skip(
            f"group sizes are {group_sizes}; every group needs at least "
            f"{minimum_group_n} observations for the effect size and its interval "
            "to be estimable. Pass minimum_group_n=1 to analyze it anyway."
        )
    return data_clean, binary_levels, codes, group_sizes


def _plot_biserial_groups(data_clean, continuous_var, binary_var, binary_levels, title):
    """Draw the shared box/strip plot of the continuous values by group."""
    plt.figure(figsize=(6, 4))
    sns.boxplot(
        data=data_clean,
        x=binary_var,
        y=continuous_var,
        order=binary_levels,
        color="lightblue",
    )
    sns.stripplot(
        data=data_clean,
        x=binary_var,
        y=continuous_var,
        order=binary_levels,
        color="black",
        alpha=0.65,
        jitter=True,
    )
    plt.title(title)
    plt.xlabel(binary_var)
    plt.ylabel(continuous_var)
    plt.tight_layout()
    plt.show()


def correlate_p_biserial(
    data,
    continuous_var,
    binary_var,
    alpha=0.05,
    plot=True,
    minimum_group_n=2,
    _prepared=None,
):
    """Calculate a point-biserial correlation and plot values by binary group.

    NB: normal distribution for the continuous variable is assumed.

    The point-biserial correlation is the Pearson correlation between a
    continuous variable and a 0/1 coded dichotomous variable. Its p-value is
    algebraically identical to the equal-variance (Student) two-sample t-test,
    so it inherits that test's assumptions: independent observations, roughly
    normal values within each group, and equal variances. The magnitude of r is
    also attenuated when the two groups are unbalanced, so compare it across
    datasets only with that in mind.

    Parameters:
        data (pd.DataFrame): DataFrame containing the variables to analyze.
        continuous_var (str): Name of the continuous numeric column.
        binary_var (str): Name of the column with exactly two observed levels.
        alpha (float): Significance level for hypothesis testing.
        plot (bool): Whether to draw the box/strip plot. Default is True.
        minimum_group_n (int): Smallest acceptable group size; smaller groups
            are skipped rather than analyzed. Pass 1 to analyze anyway.

    Returns:
        dict or None: Correlation coefficient with its confidence interval,
            p-value, direction, significance, and the group coding used for the
            sign. ``None`` if the pair cannot support the correlation (see
            ``_prepare_biserial_inputs``); a yellow message says why.

    Raises:
        ValueError: If either column is missing or both names are the same.
    """
    prepared = _prepared or _prepare_biserial_inputs(
        data, continuous_var, binary_var, minimum_group_n=minimum_group_n
    )
    if prepared is None:
        return None
    data_clean, binary_levels, binary_values, group_sizes = prepared
    reference_level, comparison_level = binary_levels[0], binary_levels[1]

    statistic, p_value = stats.pointbiserialr(binary_values, data_clean[continuous_var])
    statistic, p_value = float(statistic), float(p_value)

    # Fisher z interval for r (approximate; needs n > 3 for a finite standard error).
    n = len(data_clean)
    if n > 3 and abs(statistic) < 1:
        z = np.arctanh(statistic)
        se = 1.0 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_lower = float(np.tanh(z - z_crit * se))
        ci_upper = float(np.tanh(z + z_crit * se))
    else:
        ci_lower, ci_upper = float("nan"), float("nan")

    significance = "Significant" if p_value < alpha else "Not Significant"
    if statistic > 0:
        direction = "Positive correlation"
    elif statistic < 0:
        direction = "Negative correlation"
    else:
        direction = "No correlation"

    if plot:
        _plot_biserial_groups(
            data_clean,
            continuous_var,
            binary_var,
            binary_levels,
            f"Point-Biserial Correlation: r = {statistic:.3f}, "
            f"p = {_display_p_value(p_value)}",
        )

    result = {
        "method": "point-biserial",
        "correlation_coefficient": statistic,
        "CI": (ci_lower, ci_upper),
        "p_value": p_value,
        "direction": direction,
        "significance": significance,
        "n": n,
        "group_sizes": group_sizes,
        "coding": f"0 = {reference_level}, 1 = {comparison_level}",
        "null_value_included": (
            None if np.isnan(ci_lower) else bool(ci_lower <= 0 <= ci_upper)
        ),
    }

    print("method:", result["method"])
    print("coding:", result["coding"], f"(n = {n}, {group_sizes})")
    print("correlation_coefficient:", round(statistic, 3))
    if np.isnan(ci_lower):
        print(f"{int(round((1 - alpha) * 100))}% CI: not available")
    else:
        print(
            f"{int(round((1 - alpha) * 100))}% CI:",
            (round(ci_lower, 3), round(ci_upper, 3)),
        )
    print("p_value:", _display_p_value(p_value))
    print("direction:", direction)
    printColor(
        f"significance: {significance}",
        "green" if significance == "Significant" else "red",
    )

    return result


def correlate_rank_biserial(
    data,
    continuous_var,
    binary_var,
    alpha=0.05,
    plot=True,
    method=None,
    minimum_group_n=2,
    _prepared=None,
):
    """Calculate a rank-biserial correlation and plot values by binary group.

    NB: the non-parametric counterpart of ``correlate_p_biserial``. Use it when
    the continuous variable is skewed, ordinal, or has outliers; no normality
    or equal-variance assumption is made.

    The rank-biserial correlation for two independent groups is Cliff's delta,
    ``2 * P(group1 > group0) - 1``, where ties count as half a win. It is the
    effect size that accompanies the Mann-Whitney U test, and the reported
    p-value is that test's. Unlike Pearson's r it is a rank statistic, so it is
    unaffected by monotone transformations of the continuous variable and is
    not attenuated by group imbalance.

    Parameters:
        data (pd.DataFrame): DataFrame containing the variables to analyze.
        continuous_var (str): Name of the continuous, skewed or ordinal column.
        binary_var (str): Name of the column with exactly two observed levels.
        alpha (float): Significance level for hypothesis testing.
        plot (bool): Whether to draw the box/strip plot. Default is True.
        method (str, optional): 'exact' or 'asymptotic'. Defaults to the
            package's resolved Mann-Whitney policy.
        minimum_group_n (int): Smallest acceptable group size; smaller groups
            are skipped rather than analyzed. Pass 1 to analyze anyway.

    Returns:
        dict or None: Rank-biserial correlation (Cliff's delta) and the
            equivalent probability of superiority, each with a confidence
            interval, plus the Mann-Whitney p-value, direction, significance,
            and the group coding used for the sign. ``None`` if the pair cannot
            support the correlation; a yellow message says why.

    Raises:
        ValueError: If either column is missing or both names are the same.
    """
    prepared = _prepared or _prepare_biserial_inputs(
        data, continuous_var, binary_var, minimum_group_n=minimum_group_n
    )
    if prepared is None:
        return None
    data_clean, binary_levels, binary_values, group_sizes = prepared
    reference_level, comparison_level = binary_levels[0], binary_levels[1]

    values = data_clean[continuous_var].to_numpy()
    group0 = values[binary_values == 0]
    group1 = values[binary_values == 1]

    # Order matters: group1 first makes a positive coefficient mean that the
    # level coded 1 has the higher values, matching correlate_p_biserial.
    effect = mann_whitney_effect_ci(group1, group0, alpha=alpha, method=method)
    statistic = effect["cliffs_delta"]
    ci_lower, ci_upper = effect["cliffs_delta_CI"]
    p_value = effect["p_value"]

    significance = "Significant" if p_value < alpha else "Not Significant"
    if statistic > 0:
        direction = "Positive correlation"
    elif statistic < 0:
        direction = "Negative correlation"
    else:
        direction = "No correlation"

    n = len(data_clean)
    group_medians = {
        str(reference_level): float(np.median(group0)),
        str(comparison_level): float(np.median(group1)),
    }

    if plot:
        _plot_biserial_groups(
            data_clean,
            continuous_var,
            binary_var,
            binary_levels,
            f"Rank-Biserial Correlation: r_rb = {statistic:.3f}, "
            f"p = {_display_p_value(p_value)}",
        )

    result = {
        "method": "rank-biserial",
        "correlation_coefficient": statistic,
        "CI": (ci_lower, ci_upper),
        "probability_superiority": effect["probability_superiority"],
        "probability_CI": effect["probability_CI"],
        "p_value": p_value,
        "direction": direction,
        "significance": significance,
        "n": n,
        "group_sizes": group_sizes,
        "group_medians": group_medians,
        "coding": f"0 = {reference_level}, 1 = {comparison_level}",
        "test_stat": effect["test_stat"],
        "test_method": effect["test_method"],
        "ci_assumption": effect["ci_assumption"],
        "null_value_included": bool(ci_lower <= 0 <= ci_upper),
    }

    decimals = _effect_ci_display_decimals((ci_lower, ci_upper), 0.0)
    confidence = int(round((1 - alpha) * 100))
    print("method:", result["method"], f"(Mann-Whitney, {effect['test_method']})")
    print("coding:", result["coding"], f"(n = {n}, {group_sizes})")
    print("group_medians:", {k: round(v, 3) for k, v in group_medians.items()})
    print("correlation_coefficient:", round(statistic, 3), "(Cliff's delta)")
    print(
        f"{confidence}% CI:",
        (round(ci_lower, decimals), round(ci_upper, decimals)),
    )
    print(
        "probability_superiority:",
        round(effect["probability_superiority"], 3),
        f"({confidence}% CI",
        tuple(round(bound, 3) for bound in effect["probability_CI"]),
        ")",
    )
    print("p_value:", _display_p_value(p_value))
    print("direction:", direction)
    printColor(
        f"significance: {significance}",
        "green" if significance == "Significant" else "red",
    )

    return result


def correlate_biserial(
    data,
    continuous_var,
    binary_var,
    alpha=0.05,
    plot=True,
    force_test=None,
    force_normality=False,
    force_non_normality=False,
    data_type=None,
    method=None,
    minimum_group_n=2,
):
    """Choose between the point-biserial and rank-biserial correlation.

    Runs Shapiro-Wilk on the continuous variable *within each group* and then
    delegates to ``correlate_p_biserial`` when both groups are normal, or to
    ``correlate_rank_biserial`` otherwise. Normality is assessed per group
    because that -- not the normality of the pooled column -- is the assumption
    behind the point-biserial correlation. Pooled values from two groups with
    different locations form a mixture that can fail Shapiro-Wilk even when
    both groups are perfectly normal.

    Parameters:
        data (pd.DataFrame): DataFrame containing the variables to analyze.
        continuous_var (str): Name of the continuous numeric column.
        binary_var (str): Name of the column with exactly two observed levels.
        alpha (float): Significance level, used both for the normality
            screen and for the correlation itself.
        plot (bool): Whether to draw the box/strip plot. Default is True.
        force_test (str, optional): 'point-biserial' or 'rank-biserial' to skip
            the normality screen and use that test.
        force_normality (bool): Skip the screen and treat both groups as normal.
        force_non_normality (bool): Skip the screen and use rank-based inference.
        data_type (str, optional): 'cont' or 'ordinal'. Ordinal data goes
            straight to the rank-biserial correlation.
        method (str, optional): Passed to ``correlate_rank_biserial`` as its
            Mann-Whitney method; ignored by the parametric branch.
        minimum_group_n (int): Smallest acceptable group size; smaller groups
            are skipped rather than analyzed. Pass 1 to analyze anyway.

    Returns:
        dict or None: The delegate's result dictionary, plus a ``normality``
            entry recording the screen and a ``selected_method`` entry.
            ``None`` if the pair cannot support a biserial correlation, so that
            a loop over many variable pairs reports the skip and continues.

    Raises:
        ValueError: If either column is missing, both names are the same, or a
            forcing argument is contradictory or unrecognised.
    """
    if force_normality and force_non_normality:
        raise ValueError(
            "force_normality and force_non_normality cannot both be True."
        )
    if data_type is not None and data_type not in ("cont", "ordinal"):
        raise ValueError(
            "data_type must be 'cont' or 'ordinal' for a biserial correlation; "
            f"got {data_type!r}."
        )
    valid_tests = ("point-biserial", "rank-biserial")
    if force_test is not None and force_test not in valid_tests:
        raise ValueError(f"force_test must be one of {valid_tests}; got {force_test!r}.")

    # Validate once here so the normality screen runs on clean, aligned data.
    # The delegate re-runs the same preparation; it is cheap and keeps each
    # correlation function usable on its own.
    prepared = _prepare_biserial_inputs(
        data, continuous_var, binary_var, minimum_group_n=minimum_group_n
    )
    if prepared is None:
        return None
    data_clean, binary_levels, binary_values, group_sizes = prepared
    values = data_clean[continuous_var].to_numpy()
    groups = [values[binary_values == 0], values[binary_values == 1]]

    normality = {
        "test": "Shapiro-Wilk",
        "alpha": alpha,
        "by_group": {},
        "all_normal": None,
        "basis": None,
    }

    # --- Decide which test to run ---
    if force_test is not None:
        selected = force_test
        normality["basis"] = f"force_test={force_test!r}"
        print(f"force_test={force_test!r}: skipping the normality screen.")
    elif data_type == "ordinal":
        selected = "rank-biserial"
        normality["all_normal"] = False
        normality["basis"] = "data_type='ordinal'"
        print("Ordinal data is analyzed with non-parametric rank tests.")
    elif force_non_normality:
        selected = "rank-biserial"
        normality["all_normal"] = False
        normality["basis"] = "force_non_normality=True"
        print("The correlate_biserial function is forced to use non-parametric tests.")
    elif force_normality:
        selected = "point-biserial"
        normality["all_normal"] = True
        normality["basis"] = "force_normality=True"
        print("The correlate_biserial function is forced to use parametric tests.")
    else:
        ### Normality Test ###
        # Shapiro-Wilk per group; every group must pass for the parametric branch.
        print_title("Test for Normality")
        all_normal = True
        for level, group in zip(binary_levels, groups):
            print(f"Test for normality (Shapiro-Wilk) for {binary_var} = {level}")
            print("-------------------------------------------------")
            if group.size < 3:
                print(
                    f"Only {group.size} observation(s): Shapiro-Wilk needs at least 3. "
                    "Assuming non-normality."
                )
                is_normal = False
                statistic, p_value = float("nan"), float("nan")
            elif group.size > 5000:
                print(
                    f"{group.size} observations: skipping Shapiro-Wilk and assuming "
                    "normality for a large sample."
                )
                is_normal = True
                statistic, p_value = float("nan"), float("nan")
            elif np.unique(group).size == 1:
                print("All values are identical: assuming non-normality.")
                is_normal = False
                statistic, p_value = float("nan"), float("nan")
            else:
                statistic, p_value = stats.shapiro(group)
                statistic, p_value = float(statistic), float(p_value)
                print("Statistic:", round(statistic, 3))
                print("p-value:", _display_p_value(p_value))
                is_normal = p_value > alpha
                if is_normal:
                    print("Data is normally distributed.")
                else:
                    print("Data is not normally distributed.")
            normality["by_group"][str(level)] = {
                "n": int(group.size),
                "statistic": statistic,
                "p_value": p_value,
                "normal": is_normal,
            }
            all_normal = all_normal and is_normal
            print()

        normality["all_normal"] = all_normal
        normality["basis"] = "Shapiro-Wilk within each group"
        selected = "point-biserial" if all_normal else "rank-biserial"
        if all_normal:
            print("All data is normally distributed.")
        else:
            print("Not all data is normally distributed.")

    if selected == "point-biserial":
        print("Using the point-biserial correlation (parametric).")
        print()
        result = correlate_p_biserial(
            data_clean,
            continuous_var,
            binary_var,
            alpha=alpha,
            plot=plot,
            _prepared=prepared,
        )
    else:
        print("Using the rank-biserial correlation (non-parametric).")
        print()
        result = correlate_rank_biserial(
            data_clean,
            continuous_var,
            binary_var,
            alpha=alpha,
            plot=plot,
            method=method,
            _prepared=prepared,
        )

    result["selected_method"] = selected
    result["normality"] = normality
    return result


################################################################################################
#                                        REGRESSION
################################################################################################
# regress() follows the same "inspect the data, then choose" idea used by compare_ind() and
# compare_dep(), but with one deliberate difference: regression assumptions can only be checked
# after a model is fitted, and there is no single agreed-upon fallback when one fails. So the
# model family is chosen from the outcome variable up front, and the post-fit diagnostics are
# reported rather than used to silently switch models. The only automatic change is the
# covariance estimator (robust standard errors under heteroscedasticity), which leaves the
# fitted model itself untouched.


# Model families implemented so far. Named here so force_model can tell "not a model" apart
# from "a real model that is not built yet".
_REGRESSION_IMPLEMENTED_MODELS = ("linear", "logistic")
_REGRESSION_PLANNED_MODELS = ("poisson", "negativebinomial", "ordinal", "multinomial", "cox")


def _regression_quote(name):
    """Wrap a column name so patsy accepts names with spaces, dashes or leading digits."""
    if name.isidentifier():
        return name
    escaped = name.replace("\\", "\\\\").replace("'", "\\'")
    return f"Q('{escaped}')"


def _regression_is_categorical(series, categorical_limit):
    """Decide whether a predictor should enter the model as a factor rather than a number."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    if series.dtype == bool or series.dtype == object:
        return True
    # Numeric but with very few distinct values is ambiguous; treat as numeric and let the
    # user override, because silently dummy-coding a numeric dose or score would be worse.
    return False


def _regression_reference_level(series, requested):
    """Pick the reference (baseline) level for a categorical predictor."""
    levels = list(pd.Categorical(series).categories)
    if requested is None:
        return levels[0], levels
    if requested not in levels:
        raise Exception(
            f'Reference level "{requested}" was not found in "{series.name}". '
            f"Available levels: {levels}."
        )
    return requested, levels


def _regression_build_formula(outcome, predictors, data, reference, categorical_limit):
    """
    Build a patsy formula from an outcome and a list of predictors.

    Categorical predictors are wrapped in C(...) with an explicit Treatment reference so the
    baseline level is chosen deliberately rather than by accident of sort order.
    """
    reference = reference or {}
    unknown_reference = [name for name in reference if name not in predictors]
    if unknown_reference:
        printColor(
            f"Warning: reference level(s) given for variable(s) not in predictors and ignored: "
            f"{', '.join(map(str, unknown_reference))}.",
            "yellow",
        )

    terms = []
    reference_levels = {}
    for name in predictors:
        series = data[name]
        if _regression_is_categorical(series, categorical_limit):
            baseline, levels = _regression_reference_level(series, reference.get(name))
            reference_levels[name] = {"reference": baseline, "levels": levels}
            escaped = str(baseline).replace("\\", "\\\\").replace("'", "\\'")
            terms.append(f"C({_regression_quote(name)}, Treatment(reference='{escaped}'))")
        else:
            if name in reference:
                printColor(
                    f'Warning: "{name}" is numeric, so the reference level given for it is ignored.',
                    "yellow",
                )
            terms.append(_regression_quote(name))

    formula = f"{_regression_quote(outcome)} ~ " + " + ".join(terms)
    return formula, reference_levels


def _regression_formula_outcome(formula, data):
    """Extract the outcome column name from a user-supplied formula's left-hand side."""
    if "~" not in formula:
        raise Exception(
            f'The formula "{formula}" has no "~". A formula must look like "outcome ~ age + sex".'
        )
    left = formula.split("~", 1)[0].strip()
    if left in data.columns:
        return left
    # Allow an explicitly quoted name such as Q('body mass index').
    for quote in ("Q('", 'Q("'):
        if left.startswith(quote) and left.endswith(quote[-1] + ")"):
            inner = left[len(quote):-2]
            if inner in data.columns:
                return inner
    return None


def _regression_formula_columns(formula, data):
    """Find which DataFrame columns a formula refers to, so we can drop rows listwise."""

    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula))
    quoted = set(re.findall(r"Q\(\s*['\"](.+?)['\"]\s*\)", formula))
    used = [column for column in data.columns if column in tokens or column in quoted]
    return used


def _regression_select_model(
    outcome_series, outcome_name, outcome_type, force_model, categorical_limit, alpha
):
    """
    Choose the model family from the outcome variable.

    Unlike compare_ind(), the choice here does not depend on any assumption test - it depends
    only on what kind of thing the outcome is, which is why it can be settled before fitting.
    """
    if force_model is not None:
        model = str(force_model).strip().lower().replace(" ", "").replace("_", "")
        if model in _REGRESSION_IMPLEMENTED_MODELS:
            print(f"force_model='{model}': fitting a {model} model without inspecting the outcome.")
            return model
        if model in _REGRESSION_PLANNED_MODELS:
            raise Exception(
                f'force_model="{force_model}" is a recognised model family but is not implemented '
                f"yet. Currently available: {', '.join(_REGRESSION_IMPLEMENTED_MODELS)}."
            )
        raise Exception(
            f'force_model="{force_model}" is not recognised. '
            f"Options are: {', '.join(_REGRESSION_IMPLEMENTED_MODELS)}."
        )

    n_unique = outcome_series.nunique(dropna=True)

    if outcome_type == "cont":
        print("Outcome type explicitly set to continuous; fitting a linear model.")
        return "linear"
    if outcome_type in ("cat", "nominal", "binary", "ordinal"):
        if n_unique != 2:
            raise Exception(
                f'Outcome "{outcome_name}" was declared categorical but has {n_unique} levels. '
                f"Stage 1 supports binary outcomes only; multinomial and ordinal models are not "
                f"implemented yet."
            )
        print("Outcome type explicitly set to categorical with 2 levels; fitting a logistic model.")
        return "logistic"

    # Inferred - warn in the same style as compare_ind(), because a wrong guess here is costly.
    if n_unique == 2:
        model = "logistic"
        basis = "the outcome has exactly 2 levels"
    elif pd.api.types.is_numeric_dtype(outcome_series) and n_unique > categorical_limit:
        model = "linear"
        basis = f"the outcome is numeric with {n_unique} distinct values"
    elif pd.api.types.is_numeric_dtype(outcome_series):
        model = "linear"
        basis = (
            f"the outcome is numeric with only {n_unique} distinct values, which is fewer than "
            f"categorical_limit ({categorical_limit})"
        )
    else:
        raise Exception(
            f'Cannot infer a model for outcome "{outcome_name}": it is not numeric and has '
            f"{n_unique} levels. Stage 1 supports continuous and binary outcomes. Set "
            f"outcome_type or force_model explicitly."
        )

    printColor(
        f"Warning: outcome_type not set. Selecting a {model} model because {basis}. "
        f"Pass outcome_type='cont' or outcome_type='cat' (or force_model) to suppress this warning.",
        "yellow",
    )
    return model


def _regression_encode_binary_outcome(data, outcome, outcome_positive):
    """
    Recode a binary outcome to 0/1 and report which level is being modelled as the event.

    Doing this explicitly (rather than leaving it to patsy) is what makes the direction of every
    odds ratio unambiguous, which is the single most common way a logistic result is misread.
    """
    levels = list(pd.Series(data[outcome].unique()).dropna())
    try:
        levels = sorted(levels)
    except TypeError:
        levels = list(levels)

    if len(levels) != 2:
        raise Exception(
            f'A logistic model needs a binary outcome, but "{outcome}" has {len(levels)} '
            f"level(s): {levels}."
        )

    if outcome_positive is None:
        # Default to the higher/later level as the event, which makes 1 the event for 0/1 data.
        event = levels[1]
    else:
        if outcome_positive not in levels:
            raise Exception(
                f'outcome_positive="{outcome_positive}" is not a level of "{outcome}". '
                f"Levels are: {levels}."
            )
        event = outcome_positive

    baseline = [level for level in levels if level != event][0]
    encoded = data[outcome].apply(lambda value: 1 if value == event else 0).astype(float)
    return encoded, event, baseline


def _regression_vif(fitted):
    """
    Variance inflation factors for each design-matrix column.

    Reported per design column, so a categorical predictor contributes one row per dummy. Dummy
    columns from the same multi-level factor inflate each other by construction, so a high VIF
    there is expected and is not the same finding as two genuinely collinear predictors.
    """
    exog = np.asarray(fitted.model.exog, dtype=float)
    names = list(fitted.model.exog_names)
    rows = []
    for index, name in enumerate(names):
        if name == "Intercept":
            continue
        try:
            value = variance_inflation_factor(exog, index)
        except Exception:
            value = float("nan")
        rows.append({"term": name, "VIF": float(value)})
    return pd.DataFrame(rows)


def _regression_print_vif(vif_table, alpha_free_threshold=5.0):
    """Print the VIF table and flag collinearity worth investigating."""
    if vif_table.empty:
        return None
    print("Multicollinearity (variance inflation factors):")
    for _, row in vif_table.iterrows():
        value = row["VIF"]
        display = "not estimable" if not np.isfinite(value) else f"{value:.2f}"
        print(f"  {row['term']}: {display}")

    finite = vif_table[np.isfinite(vif_table["VIF"])]
    high = finite[finite["VIF"] >= alpha_free_threshold]
    if not high.empty:
        printColor(
            f"Warning: VIF >= {alpha_free_threshold:.0f} for: "
            f"{', '.join(high['term'].tolist())}. Coefficients and their standard errors for "
            f"these terms are unstable. This is expected, and not a problem, when the terms are "
            f"dummies of one multi-level categorical predictor, or an interaction or polynomial "
            f"term and the components it is built from - they are collinear by construction. "
            f"Between separate predictors it usually means they carry overlapping information.",
            "yellow",
        )
    else:
        print("  No VIF at or above the conventional threshold.")
    print()
    return high["term"].tolist() if not high.empty else []


def _regression_sample_size_check(n, n_parameters, model, events=None):
    """
    Warn when the model is being asked to estimate more than the data can support.

    Reported because reviewers of medical work routinely ask for it, and because an overfitted
    model produces confident-looking coefficients that will not replicate.
    """
    n_predictors = max(n_parameters - 1, 1)  # exclude the intercept
    if model == "logistic" and events is not None:
        limiting = min(events, n - events)
        epv = limiting / n_predictors
        print(f"Events per variable (EPV): {epv:.1f} ({limiting} limiting events / {n_predictors} model term(s)).")
        if epv < 10:
            printColor(
                f"Warning: EPV is below the conventional minimum of 10. Coefficients are likely "
                f"biased away from the null and the confidence intervals are optimistic. Consider "
                f"fewer predictors, or a penalised model.",
                "yellow",
            )
        print()
        return {"events_per_variable": float(epv), "limiting_events": int(limiting)}

    ratio = n / n_predictors
    print(f"Observations per model term: {ratio:.1f} ({n} observations / {n_predictors} term(s)).")
    if ratio < 10:
        printColor(
            "Warning: fewer than 10 observations per model term. The model is at high risk of "
            "overfitting, and the coefficients should be treated as exploratory.",
            "yellow",
        )
    print()
    return {"observations_per_term": float(ratio)}


def _regression_linear_diagnostics(fitted, alpha, n):
    """
    Post-fit checks for a linear model. These are reported, never used to switch models.

    The one exception is heteroscedasticity, which triggers a refit with HC3 robust standard
    errors. That changes only the covariance estimator - the coefficients, and the model itself,
    are identical - so it does not amount to choosing a different model behind the user's back.
    """
    diagnostics = {}
    residuals = np.asarray(fitted.resid, dtype=float)

    print_title("Model Diagnostics")

    # --- Residual normality ---
    if n <= 5000:
        shapiro_statistic, shapiro_p = stats.shapiro(residuals)
        diagnostics["residual_normality"] = {
            "test": "Shapiro-Wilk",
            "statistic": float(shapiro_statistic),
            "p_value": float(shapiro_p),
            "normal": bool(shapiro_p > alpha),
        }
        print("Residual normality (Shapiro-Wilk on the residuals):")
        print(f"  Statistic: {round(float(shapiro_statistic), 3)}")
        print(f"  p-value: {_display_p_value(shapiro_p)}")
        if shapiro_p > alpha:
            print("  Residuals are consistent with normality.")
        else:
            printColor(
                "  Residuals are not normally distributed. The coefficients remain unbiased, but "
                "small-sample p-values and confidence intervals may be inaccurate. With a large "
                "n this matters little; with a small n, or with visible skew or outliers in the "
                "residual plots, consider transforming the outcome.",
                "yellow",
            )
    else:
        diagnostics["residual_normality"] = {
            "test": "skipped", "reason": "n > 5000", "normal": None,
        }
        print("Residual normality: skipped (n > 5000; Shapiro-Wilk is unreliable at this size).")
    print()

    # --- Homoscedasticity ---
    robust = False
    try:
        bp_statistic, bp_p, _, _ = het_breuschpagan(residuals, fitted.model.exog)
        diagnostics["homoscedasticity"] = {
            "test": "Breusch-Pagan",
            "statistic": float(bp_statistic),
            "p_value": float(bp_p),
            "constant_variance": bool(bp_p > alpha),
        }
        print("Homoscedasticity (Breusch-Pagan):")
        print(f"  Statistic: {round(float(bp_statistic), 3)}")
        print(f"  p-value: {_display_p_value(bp_p)}")
        if bp_p > alpha:
            print("  Residual variance is consistent with being constant.")
        else:
            robust = True
            printColor(
                "  Residual variance is not constant (heteroscedasticity). Refitting with HC3 "
                "robust standard errors. The coefficients are unchanged; only their standard "
                "errors, p-values and confidence intervals are corrected.",
                "yellow",
            )
    except Exception as error:
        diagnostics["homoscedasticity"] = {"test": "failed", "reason": str(error)}
        print(f"Homoscedasticity: could not be tested ({error}).")
    print()

    # --- Linearity ---
    try:
        reset = linear_reset(fitted, power=2, test_type="fitted", use_f=True)
        reset_p = float(reset.pvalue)
        diagnostics["linearity"] = {
            "test": "Ramsey RESET (squared fitted values)",
            "p_value": reset_p,
            "linear": bool(reset_p > alpha),
        }
        print("Linearity (Ramsey RESET):")
        print(f"  p-value: {_display_p_value(reset_p)}")
        if reset_p > alpha:
            print("  No evidence of a missing non-linear term.")
        else:
            printColor(
                "  Evidence that the relationship is not purely linear. Consider adding a "
                "quadratic or spline term, or transforming a predictor, using the formula "
                "argument. Check the residuals-vs-fitted plot to see which predictor is "
                "responsible.",
                "yellow",
            )
    except Exception as error:
        diagnostics["linearity"] = {"test": "failed", "reason": str(error)}
        print(f"Linearity: could not be tested ({error}).")
    print()

    # --- Influential observations ---
    try:
        influence = OLSInfluence(fitted)
        cooks = np.asarray(influence.cooks_distance[0], dtype=float)
        threshold = 4.0 / n
        flagged = int(np.sum(cooks > threshold))
        diagnostics["influence"] = {
            "measure": "Cook's distance",
            "threshold": float(threshold),
            "n_flagged": flagged,
            "max": float(np.nanmax(cooks)) if cooks.size else float("nan"),
        }
        print("Influential observations (Cook's distance):")
        print(f"  Threshold (4/n): {threshold:.4f}")
        print(f"  Largest Cook's distance: {np.nanmax(cooks):.4f}")
        print(f"  Observations above the threshold: {flagged} of {n}")
        if flagged > 0:
            printColor(
                f"  {flagged} observation(s) exert disproportionate influence on the fit. Check "
                f"them for data-entry errors. Do not delete them merely because they are "
                f"influential - a genuine extreme value is information, not noise.",
                "yellow",
            )
        diagnostics["influence"]["cooks_distance"] = cooks
    except Exception as error:
        diagnostics["influence"] = {"measure": "failed", "reason": str(error)}
        print(f"Influential observations: could not be assessed ({error}).")
    print()

    # --- Independence of residuals ---
    durbin_watson_value = float(sm.stats.durbin_watson(residuals))
    diagnostics["durbin_watson"] = durbin_watson_value
    print(f"Independence of residuals (Durbin-Watson): {durbin_watson_value:.3f}")
    print(
        "  Values near 2 indicate no first-order autocorrelation. This only matters when the "
        "rows have a meaningful order, such as a time series or repeated measures."
    )
    print()

    return diagnostics, robust


def _regression_auc(y_true, probabilities):
    """
    Concordance statistic (area under the ROC curve).

    Computed from the Mann-Whitney U statistic, which is exactly equivalent to the AUC and keeps
    this function free of any dependency the package does not already declare.
    """
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    positives = probabilities[y_true == 1]
    negatives = probabilities[y_true == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    statistic = stats.mannwhitneyu(positives, negatives, alternative="two-sided").statistic
    return float(statistic / (positives.size * negatives.size))


def _regression_hosmer_lemeshow(y_true, probabilities, groups=10):
    """
    Hosmer-Lemeshow calibration test.

    A large p-value means no evidence of poor calibration, so - unusually - a non-significant
    result is the reassuring one here. The test is known to be sensitive to the number of groups,
    so the group count actually used is reported alongside it.
    """
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    n = y_true.size

    order = np.argsort(probabilities)
    y_sorted = y_true[order]
    p_sorted = probabilities[order]

    groups = int(min(groups, max(2, n // 5)))
    edges = np.array_split(np.arange(n), groups)
    edges = [chunk for chunk in edges if chunk.size > 0]
    groups = len(edges)
    if groups < 3:
        return None

    statistic = 0.0
    table = []
    for chunk in edges:
        observed = float(y_sorted[chunk].sum())
        expected = float(p_sorted[chunk].sum())
        size = float(chunk.size)
        table.append(
            {
                "n": int(size),
                "observed_events": observed,
                "expected_events": expected,
                "mean_predicted": float(p_sorted[chunk].mean()),
                "observed_rate": observed / size,
            }
        )
        denominator = expected * (1.0 - expected / size)
        if denominator <= 0:
            continue
        statistic += (observed - expected) ** 2 / denominator

    degrees_of_freedom = groups - 2
    if degrees_of_freedom < 1:
        return None
    p_value = float(stats.chi2.sf(statistic, degrees_of_freedom))
    return {
        "test": "Hosmer-Lemeshow",
        "statistic": float(statistic),
        "degrees_of_freedom": int(degrees_of_freedom),
        "p_value": p_value,
        "groups": groups,
        "table": pd.DataFrame(table),
    }


def _regression_logistic_diagnostics(fitted, y_true, alpha, n):
    """Post-fit checks for a logistic model. Reported, never used to switch models."""
    diagnostics = {}
    probabilities = np.asarray(fitted.predict(), dtype=float)

    print_title("Model Diagnostics")

    # --- Separation ---
    extreme = int(np.sum((probabilities > 0.999) | (probabilities < 0.001)))
    standard_errors = np.asarray(fitted.bse, dtype=float)
    huge_se = int(np.sum(standard_errors > 25))
    separated = extreme > 0 and huge_se > 0
    diagnostics["separation"] = {
        "extreme_fitted_probabilities": extreme,
        "very_large_standard_errors": huge_se,
        "suspected": bool(separated),
    }
    print("Separation check:")
    print(f"  Fitted probabilities within 0.001 of 0 or 1: {extreme} of {n}")
    print(f"  Coefficients with a standard error above 25: {huge_se}")
    if separated:
        printColor(
            "  Complete or quasi-complete separation is likely: some combination of predictors "
            "perfectly predicts the outcome. The affected coefficients and their confidence "
            "intervals are not trustworthy at any size. Consider merging sparse categories, "
            "dropping the offending predictor, or a penalised (Firth) logistic model.",
            "yellow",
        )
    else:
        print("  No sign of separation.")
    print()

    # --- Discrimination ---
    auc = _regression_auc(y_true, probabilities)
    diagnostics["auc"] = auc
    print(f"Discrimination (c-statistic / AUC): {auc:.3f}")
    if np.isfinite(auc):
        if auc < 0.6:
            quality = "poor - barely better than chance"
        elif auc < 0.7:
            quality = "weak"
        elif auc < 0.8:
            quality = "acceptable"
        elif auc < 0.9:
            quality = "good"
        else:
            quality = "excellent, but check for overfitting or a predictor that encodes the outcome"
        print(f"  Interpretation: {quality}.")
    print(
        "  This is apparent discrimination, measured on the same data the model was fitted to, "
        "so it is optimistic. Honest estimates need cross-validation or an external sample."
    )
    print()

    # --- Calibration ---
    hosmer = _regression_hosmer_lemeshow(y_true, probabilities)
    if hosmer is None:
        diagnostics["calibration"] = {"test": "skipped", "reason": "too few observations"}
        print("Calibration (Hosmer-Lemeshow): skipped (too few observations to form groups).")
    else:
        diagnostics["calibration"] = hosmer
        print(f"Calibration (Hosmer-Lemeshow, {hosmer['groups']} groups):")
        print(f"  Statistic: {round(hosmer['statistic'], 3)} on {hosmer['degrees_of_freedom']} df")
        print(f"  p-value: {_display_p_value(hosmer['p_value'])}")
        if hosmer["p_value"] > alpha:
            print(
                "  No evidence of poor calibration. Note that a non-significant result is the "
                "reassuring one for this test, which is the opposite of the usual reading."
            )
        else:
            printColor(
                "  Evidence of poor calibration: predicted probabilities do not match observed "
                "event rates across the risk range. The model may still rank patients well "
                "(see the c-statistic), but the probabilities it produces should not be quoted "
                "as absolute risks.",
                "yellow",
            )
    print()

    return diagnostics


def _regression_pretty_term(term):
    """
    Turn a patsy design-matrix name into something a reader can follow.

    C(Q('smoking'), Treatment(reference='never'))[T.current]  ->  smoking [current vs never]
    """

    categorical = re.match(
        r"^C\(\s*(?:Q\(['\"](?P<quoted>.+?)['\"]\)|(?P<plain>[^,\)]+?))\s*,\s*"
        r"Treatment\(reference=['\"](?P<reference>.*?)['\"]\)\s*\)\[T\.(?P<level>.+)\]$",
        term,
    )
    if categorical:
        name = categorical.group("quoted") or categorical.group("plain")
        return f"{name.strip()} [{categorical.group('level')} vs {categorical.group('reference')}]"

    quoted = re.match(r"^Q\(['\"](.+?)['\"]\)$", term)
    if quoted:
        return quoted.group(1)

    # Fall back to stripping Q('...') wrappers inside a larger expression (interactions, etc.)
    return re.sub(r"Q\(['\"](.+?)['\"]\)", r"\1", term)


def _regression_tidy(fitted, model, alpha):
    """Build the per-term results table, on the coefficient scale and the reported scale."""
    # A results object from get_robustcov_results() returns plain arrays rather than the indexed
    # Series a normal fit gives back, so read the term names off the design matrix and coerce
    # everything to arrays. That keeps this function correct for both the plain and robust fits.
    names = list(fitted.model.exog_names)
    confidence = np.asarray(fitted.conf_int(alpha=alpha), dtype=float)

    table = pd.DataFrame(
        {
            "term": [_regression_pretty_term(name) for name in names],
            "design_term": names,
            "coefficient": np.asarray(fitted.params, dtype=float),
            "std_error": np.asarray(fitted.bse, dtype=float),
            "statistic": np.asarray(fitted.tvalues, dtype=float),
            "p_value": np.asarray(fitted.pvalues, dtype=float),
            "ci_lower": confidence[:, 0],
            "ci_upper": confidence[:, 1],
        }
    )

    if model == "logistic":
        table["odds_ratio"] = np.exp(table["coefficient"])
        table["or_ci_lower"] = np.exp(table["ci_lower"])
        table["or_ci_upper"] = np.exp(table["ci_upper"])

    return table.reset_index(drop=True)


def _regression_print_coefficients(table, model, alpha, robust):
    """Print the coefficient table in the reported scale, with the null value made explicit."""
    confidence_level = round(100 - (alpha * 100), 10)
    if confidence_level == int(confidence_level):
        confidence_level = int(confidence_level)

    print_title("Model Coefficients")

    if model == "logistic":
        print(f"Effects are odds ratios with {confidence_level}% confidence intervals (null value = 1).")
        print()
        header = f"{'Term':<38}{'OR':>10}{'  ' + str(confidence_level) + '% CI':>22}{'p':>12}"
    else:
        scale = "robust (HC3)" if robust else "model-based"
        print(
            f"Effects are unstandardised coefficients with {confidence_level}% confidence "
            f"intervals (null value = 0), using {scale} standard errors."
        )
        print()
        header = f"{'Term':<38}{'Coef':>10}{'  ' + str(confidence_level) + '% CI':>22}{'p':>12}"

    print(header)
    print("-" * len(header))

    for _, row in table.iterrows():
        if model == "logistic":
            estimate = row["odds_ratio"]
            lower, upper = row["or_ci_lower"], row["or_ci_upper"]
        else:
            estimate = row["coefficient"]
            lower, upper = row["ci_lower"], row["ci_upper"]

        label = str(row["term"])
        if len(label) > 36:
            label = label[:35] + "…"
        interval = f"[{lower:.3f}, {upper:.3f}]"
        print(f"{label:<38}{estimate:>10.3f}{interval:>24}{str(_display_p_value(row['p_value'])):>12}")

    print()

    predictors = table[table["design_term"] != "Intercept"]
    significant = predictors[predictors["p_value"] < alpha]
    if significant.empty:
        printColor("No predictor is significantly associated with the outcome.", "red")
    else:
        printColor(
            f"Significantly associated with the outcome: {', '.join(significant['term'].tolist())}.",
            "green",
        )
    print()


def _regression_print_fit(fitted, model, y_true=None):
    """Print overall model fit, kept separate from the per-term effects."""
    print_title("Model Fit")
    summary = {"aic": float(fitted.aic), "bic": float(fitted.bic), "n": int(fitted.nobs)}

    if model == "linear":
        summary["r_squared"] = float(fitted.rsquared)
        summary["adj_r_squared"] = float(fitted.rsquared_adj)
        summary["f_p_value"] = float(fitted.f_pvalue)
        print(f"R-squared: {fitted.rsquared:.3f}")
        print(f"Adjusted R-squared: {fitted.rsquared_adj:.3f}")
        print(f"Overall model F-test p-value: {_display_p_value(fitted.f_pvalue)}")
        print(
            f"  The model explains {fitted.rsquared * 100:.1f}% of the variance in the outcome "
            f"within this sample."
        )
    else:
        summary["pseudo_r_squared"] = float(fitted.prsquared)
        summary["lr_p_value"] = float(fitted.llr_pvalue)
        print(f"McFadden pseudo R-squared: {fitted.prsquared:.3f}")
        print(f"Likelihood-ratio test p-value: {_display_p_value(fitted.llr_pvalue)}")
        print(
            "  Pseudo R-squared values are not comparable to a linear model's R-squared and are "
            "typically much lower; use it to compare models on the same data, not as a share of "
            "variance explained."
        )

    print(f"AIC: {fitted.aic:.2f}    BIC: {fitted.bic:.2f}")
    print()
    return summary


def _plot_regression_diagnostics(fitted, model, y_true, cooks_distance, do_graphs):
    """Diagnostic plots: residual behaviour for a linear model, discrimination and calibration for logistic."""
    if not do_graphs:
        return

    if model == "linear":
        fitted_values = np.asarray(fitted.fittedvalues, dtype=float)
        residuals = np.asarray(fitted.resid, dtype=float)

        figure, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].scatter(fitted_values, residuals, alpha=0.6, edgecolor="none")
        axes[0].axhline(0, linestyle="--", linewidth=1, color="grey")
        axes[0].set_xlabel("Fitted values")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs fitted\n(look for curvature or a funnel shape)")

        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title("Normal Q-Q of residuals\n(points should follow the line)")

        if cooks_distance is not None and len(cooks_distance):
            threshold = 4.0 / len(cooks_distance)
            axes[2].vlines(np.arange(len(cooks_distance)), 0, cooks_distance, linewidth=1)
            axes[2].axhline(threshold, linestyle="--", linewidth=1, color="red")
            axes[2].set_xlabel("Observation")
            axes[2].set_ylabel("Cook's distance")
            axes[2].set_title("Influence\n(red line = 4/n threshold)")
        else:
            axes[2].axis("off")

        plt.tight_layout()
        plt.show()
        print()
        return

    # --- Logistic: ROC and calibration ---
    probabilities = np.asarray(fitted.predict(), dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    order = np.argsort(-probabilities)
    labels = y_true[order]
    positives = labels.sum()
    negatives = labels.size - positives
    if positives > 0 and negatives > 0:
        true_positive_rate = np.concatenate([[0.0], np.cumsum(labels) / positives])
        false_positive_rate = np.concatenate([[0.0], np.cumsum(1 - labels) / negatives])
        auc = _regression_auc(y_true, probabilities)
        axes[0].plot(false_positive_rate, true_positive_rate, linewidth=2)
        axes[0].plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey")
        axes[0].set_xlabel("False positive rate (1 - specificity)")
        axes[0].set_ylabel("True positive rate (sensitivity)")
        axes[0].set_title(f"ROC curve (AUC = {auc:.3f})")
    else:
        axes[0].axis("off")

    hosmer = _regression_hosmer_lemeshow(y_true, probabilities)
    if hosmer is not None:
        groups = hosmer["table"]
        axes[1].plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey")
        axes[1].plot(groups["mean_predicted"], groups["observed_rate"], marker="o", linewidth=1.5)
        axes[1].set_xlabel("Mean predicted probability")
        axes[1].set_ylabel("Observed event rate")
        axes[1].set_title("Calibration by risk group\n(points should follow the line)")
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()
    print()


def _regression_detect_separation(data, outcome, predictors):
    """
    Look for separation before fitting, rather than inferring it from how the fit failed.

    statsmodels only sometimes warns about separation - in other cases it fails with a bare
    "Singular matrix" that is indistinguishable from ordinary collinearity. Since separation is
    the most common reason a logistic model fails on a small clinical dataset, and since the
    remedy is completely different, it is worth checking directly.

    Returns a list of human-readable descriptions, empty when nothing is found.
    """
    findings = []
    y = np.asarray(data[outcome], dtype=float)
    if np.unique(y).size < 2:
        return findings

    for name in predictors:
        series = data[name]
        is_factor = (
            isinstance(series.dtype, pd.CategoricalDtype)
            or series.dtype == object
            or series.dtype == bool
            or series.nunique() <= 10
        )
        if is_factor:
            for level, group in data.groupby(series, observed=True)[outcome]:
                rate = float(np.mean(np.asarray(group, dtype=float)))
                if rate in (0.0, 1.0):
                    outcome_word = "never" if rate == 0.0 else "always"
                    findings.append(
                        f'"{name}" = {level} ({len(group)} observation(s)): the event {outcome_word} '
                        f"occurs in this group"
                    )
        else:
            values = np.asarray(series, dtype=float)
            zeros, ones = values[y == 0], values[y == 1]
            if zeros.size and ones.size:
                if zeros.max() < ones.min() or ones.max() < zeros.min():
                    findings.append(
                        f'"{name}": a single cut-point on this variable separates the two outcome '
                        f"groups perfectly"
                    )
    return findings


_REGRESSION_SEPARATION_MESSAGE = (
    "The logistic model could not be fitted because of complete or quasi-complete separation: "
    "some combination of predictors perfectly predicts the outcome, so the maximum-likelihood "
    "estimate does not exist. Merge sparse categories, remove the offending predictor, or use a "
    "penalised (Firth) logistic model."
)


def _regression_fit(formula, data, model, separation_hint=False):
    """
    Fit the chosen model, translating the common failure modes into readable messages.

    Warnings are captured rather than silenced: statsmodels signals separation with a warning and
    only then fails with a bare "Singular matrix", so discarding the warnings would leave us
    unable to tell separation (a data problem with a specific remedy) apart from exact
    collinearity (a different problem with a different remedy).

    Returns:
        tuple: (fitted results, list of note strings worth reporting to the user)
    """
    notes = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if model == "linear":
                fitted = smf.ols(formula, data=data).fit()
            else:
                fitted = smf.logit(formula, data=data).fit(disp=0)

        categories = {type(item.message).__name__ for item in caught}
        messages = " ".join(str(item.message) for item in caught)
        if any("PerfectSeparation" in name for name in categories):
            notes.append(
                "statsmodels reported perfect separation while fitting: at least one coefficient "
                "is not identifiable, and its estimate and confidence interval are meaningless "
                "however large they look."
            )
        if any("Convergence" in name for name in categories) or "did not converge" in messages:
            notes.append(
                "The fitting algorithm did not converge. The reported estimates are wherever the "
                "optimiser stopped, not the maximum-likelihood solution, and should not be used."
            )
        return fitted, notes

    except PerfectSeparationError:
        raise Exception(_REGRESSION_SEPARATION_MESSAGE)
    except np.linalg.LinAlgError:
        # A singular matrix here has two very different causes. Separation announces itself with
        # a warning first, so use that to decide which remedy to recommend.
        separated = separation_hint or any(
            "PerfectSeparation" in type(item.message).__name__
            or "eparation" in str(item.message)
            for item in caught
        )
        if model == "logistic" and separated:
            raise Exception(_REGRESSION_SEPARATION_MESSAGE)
        raise Exception(
            "The model could not be fitted because the design matrix is singular. This usually "
            "means two predictors are perfectly collinear (one is an exact combination of the "
            "others, such as a total and all of its parts). Remove the redundant predictor and "
            "refit."
            + (
                " In a logistic model it can also mean separation, so check whether any predictor "
                "perfectly predicts the outcome."
                if model == "logistic"
                else ""
            )
        )


def _regression_univariable(data, outcome, predictors, model, alpha, reference, categorical_limit):
    """
    Fit each predictor on its own, for the crude-vs-adjusted table used in medical papers.

    This is descriptive only. It must not be used to decide which predictors enter the
    multivariable model: choosing predictors by their univariable p-value is a form of stepwise
    selection, and it invalidates the p-values and confidence intervals that get reported.
    """
    rows = []
    for name in predictors:
        formula, _ = _regression_build_formula(
            outcome, [name], data, reference, categorical_limit
        )
        try:
            fitted, _ = _regression_fit(formula, data, model)
        except Exception as error:
            printColor(f'Univariable model for "{name}" could not be fitted: {error}', "yellow")
            continue
        table = _regression_tidy(fitted, model, alpha)
        rows.append(table[table["design_term"] != "Intercept"])

    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def _regression_print_univariable(crude, adjusted, model, alpha):
    """Print crude and adjusted effects side by side."""
    confidence_level = round(100 - (alpha * 100), 10)
    if confidence_level == int(confidence_level):
        confidence_level = int(confidence_level)

    print_title("Univariable (Crude) vs Multivariable (Adjusted)")
    print(
        "Crude effects come from one model per predictor; adjusted effects come from the single "
        "model containing all of them. A large gap between the two means the other predictors "
        "are carrying part of the association."
    )
    printColor(
        "This table is descriptive. Do not use the crude p-values to decide which predictors to "
        "keep - selecting predictors from the data invalidates the confidence intervals and "
        "p-values of the final model.",
        "yellow",
    )
    print()

    scale = "OR" if model == "logistic" else "Coef"
    header = f"{'Term':<34}{'Crude ' + scale:>12}{'  crude CI':>22}{'Adj. ' + scale:>12}{'  adjusted CI':>22}"
    print(header)
    print("-" * len(header))

    adjusted_by_term = {row["design_term"]: row for _, row in adjusted.iterrows()}
    for _, row in crude.iterrows():
        label = str(row["term"])
        if len(label) > 32:
            label = label[:31] + "…"

        if model == "logistic":
            crude_estimate = row["odds_ratio"]
            crude_interval = f"[{row['or_ci_lower']:.3f}, {row['or_ci_upper']:.3f}]"
        else:
            crude_estimate = row["coefficient"]
            crude_interval = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"

        match = adjusted_by_term.get(row["design_term"])
        if match is None:
            adjusted_estimate, adjusted_interval = float("nan"), "-"
        elif model == "logistic":
            adjusted_estimate = match["odds_ratio"]
            adjusted_interval = f"[{match['or_ci_lower']:.3f}, {match['or_ci_upper']:.3f}]"
        else:
            adjusted_estimate = match["coefficient"]
            adjusted_interval = f"[{match['ci_lower']:.3f}, {match['ci_upper']:.3f}]"

        print(
            f"{label:<34}{crude_estimate:>12.3f}{crude_interval:>24}"
            f"{adjusted_estimate:>12.3f}{adjusted_interval:>24}"
        )
    print()


def regress(
    data,
    outcome=None,
    predictors=None,
    formula=None,
    alpha=0.05,
    outcome_type=None,
    force_model=None,
    reference=None,
    outcome_positive=None,
    univariable=False,
    do_graphs=True,
    categorical_limit=20,
):
    """
    Automatically selects and fits the appropriate regression model, then reports diagnostics.

    The model family is chosen from the outcome variable before fitting. Assumption checks run
    after the fit and are reported, not used to silently substitute a different model - there is
    no single agreed-upon fallback when a regression assumption fails, and the right response
    usually depends on why it failed. The one automatic change is a switch to HC3 robust standard
    errors under heteroscedasticity, which leaves the coefficients and the model untouched.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        outcome (str): Name of the outcome (dependent) column. Required unless formula is given.
        predictors (list of str): Names of the predictor (independent) columns. Required unless
            formula is given. A single string is accepted for a one-predictor model.
        formula (str, optional): A patsy formula such as "bp ~ age * sex + np.log(bmi)". Use this
            for interactions, transformations and polynomial terms. Overrides predictors.
        alpha (float): Significance level; confidence intervals are reported at 100(1-alpha)%.
            Default is 0.05.
        outcome_type (str, optional): 'cont' or 'cat'. If not provided, the function infers a type
            from the outcome and prints a yellow warning.
        force_model (str, optional): 'linear' or 'logistic'. Skips outcome inspection entirely.
        reference (dict, optional): Baseline level per categorical predictor, e.g.
            {'smoking': 'never'}. Defaults to the first level in sort order, which is printed.
        outcome_positive (optional): Which outcome level is modelled as the event in a logistic
            model. Defaults to the higher of the two levels, which is printed.
        univariable (bool): If True, also fit one model per predictor and print a crude-vs-adjusted
            table. Descriptive only - it is not a predictor selection procedure. Default is False.
        do_graphs (bool): Whether to draw diagnostic plots. Default is True.
        categorical_limit (int): Unique-value threshold used only when outcome_type is not set.
            Default is 20.

    Returns:
        dict: model kind, formula, sample size, the coefficient table (a tidy DataFrame ready for
        a manuscript table), overall fit, diagnostics, the reference levels used, and the fitted
        statsmodels object under "fitted_model" for any follow-up work.
    """
    if not isinstance(data, pd.DataFrame):
        raise Exception("data must be a pandas DataFrame.")

    # --- Resolve the two input styles into an outcome plus a set of columns to use ---
    if formula is not None:
        outcome = _regression_formula_outcome(formula, data)
        if outcome is None:
            raise Exception(
                "The left-hand side of the formula must be a plain column name (optionally "
                "quoted as Q('name')). Transformed outcomes such as np.log(y) are not supported "
                "yet - create the transformed column first, then use it as the outcome."
            )
        used_columns = _regression_formula_columns(formula, data)
        if outcome not in used_columns:
            used_columns.append(outcome)
        predictors = [column for column in used_columns if column != outcome]
    else:
        if outcome is None or predictors is None:
            raise Exception(
                "Provide either outcome and predictors, or a formula. For example: "
                "regress(df, outcome='bp', predictors=['age', 'sex'])."
            )
        if isinstance(predictors, str):
            predictors = [predictors]
        predictors = list(predictors)
        if not predictors:
            raise Exception("predictors is empty - a regression model needs at least one predictor.")
        used_columns = [outcome] + predictors

    missing_columns = [column for column in used_columns if column not in data.columns]
    if missing_columns:
        raise Exception(
            f"Column(s) not found in the data: {', '.join(map(str, missing_columns))}."
        )
    if outcome in predictors:
        raise Exception(
            f'"{outcome}" is both the outcome and a predictor. A model cannot explain a variable '
            f"with itself."
        )

    print_title("Regression")
    print("alpha is set to", alpha)
    print()

    # --- Complete-case analysis, reported in the same style as the comparison functions ---
    total_rows = len(data)
    data_clean = data.dropna(subset=used_columns).copy()
    dropped = total_rows - len(data_clean)
    if dropped > 0:
        print(
            f"Dropping {dropped} row(s) with missing values in the model variables "
            f"({len(data_clean)} of {total_rows} remain)."
        )
        printColor(
            "Complete-case analysis assumes the missing values are missing at random. If they "
            "are not, the estimates are biased and no diagnostic below will reveal it.",
            "yellow",
        )
    else:
        print("No missing values in the model variables to drop.")
    print()

    if len(data_clean) < 10:
        printColor(
            f"Cannot fit a regression model: only {len(data_clean)} complete observation(s) "
            f"remain.",
            "yellow",
        )
        return None

    # --- Guards on the variables themselves ---
    if data_clean[outcome].nunique() < 2:
        printColor(
            f'Cannot fit a model: the outcome "{outcome}" is constant after dropping missing '
            f"values, so there is nothing to explain.",
            "yellow",
        )
        return None

    constant_predictors = [name for name in predictors if data_clean[name].nunique() < 2]
    if constant_predictors:
        printColor(
            f"Dropping constant predictor(s) with no variance to explain anything: "
            f"{', '.join(map(str, constant_predictors))}.",
            "yellow",
        )
        if formula is not None:
            raise Exception(
                "A predictor in the formula is constant after dropping missing values. Remove it "
                "from the formula and refit."
            )
        predictors = [name for name in predictors if name not in constant_predictors]
        if not predictors:
            printColor("No predictors with variance remain; cannot fit a model.", "yellow")
            return None

    # --- Choose the model family from the outcome ---
    model = _regression_select_model(
        data_clean[outcome], outcome, outcome_type, force_model, categorical_limit, alpha
    )

    event_level = None
    baseline_level = None
    if model == "logistic":
        encoded, event_level, baseline_level = _regression_encode_binary_outcome(
            data_clean, outcome, outcome_positive
        )
        data_clean[outcome] = encoded
        printColor(
            f'Modelling P("{outcome}" = {event_level}). Odds ratios above 1 mean higher odds of '
            f'"{event_level}" rather than "{baseline_level}".',
            "cyan",
        )
    print()

    # --- Build the design and fit ---
    if formula is not None:
        model_formula = formula
        reference_levels = {}
        print(f"Using the supplied formula: {model_formula}")
    else:
        model_formula, reference_levels = _regression_build_formula(
            outcome, predictors, data_clean, reference, categorical_limit
        )
        if reference_levels:
            print("Categorical predictors and their reference (baseline) levels:")
            for name, detail in reference_levels.items():
                others = [level for level in detail["levels"] if level != detail["reference"]]
                print(
                    f'  {name}: reference = "{detail["reference"]}"; '
                    f"compared against {', '.join(map(str, others))}"
                )
            print("  Every effect for these predictors is relative to its reference level.")
            print()

    separation_findings = []
    if model == "logistic":
        separation_findings = _regression_detect_separation(data_clean, outcome, predictors)
        if separation_findings:
            printColor(
                "Warning: separation detected before fitting. A predictor (or a level of one) "
                "predicts the outcome perfectly, so the affected coefficient has no finite "
                "maximum-likelihood estimate:",
                "yellow",
            )
            for finding in separation_findings:
                printColor(f"  - {finding}", "yellow")
            printColor(
                "  Any odds ratio and confidence interval for the affected term is meaningless, "
                "however large it looks. Merge sparse categories, drop the predictor, or use a "
                "penalised (Firth) logistic model.",
                "yellow",
            )
            print()

    fitted, fit_notes = _regression_fit(
        model_formula, data_clean, model, separation_hint=bool(separation_findings)
    )
    n = int(fitted.nobs)
    for note in fit_notes:
        printColor(f"Warning: {note}", "yellow")
    if fit_notes:
        print()

    print_title("Model Adequacy")
    events = int(np.asarray(data_clean[outcome], dtype=float).sum()) if model == "logistic" else None
    size_check = _regression_sample_size_check(n, len(fitted.params), model, events=events)
    vif_table = _regression_vif(fitted)
    _regression_print_vif(vif_table)

    # --- Post-fit diagnostics (reported, not re-dispatched) ---
    robust = False
    cooks_distance = None
    if model == "linear":
        diagnostics, needs_robust = _regression_linear_diagnostics(fitted, alpha, n)
        cooks_distance = diagnostics.get("influence", {}).get("cooks_distance")
        if needs_robust:
            fitted = fitted.get_robustcov_results(cov_type="HC3")
            robust = True
    else:
        y_true = np.asarray(data_clean[outcome], dtype=float)
        diagnostics = _regression_logistic_diagnostics(fitted, y_true, alpha, n)

    diagnostics["multicollinearity"] = vif_table
    diagnostics["sample_size"] = size_check
    if model == "logistic":
        diagnostics["separation"]["pre_fit_findings"] = separation_findings

    # --- Report ---
    fit_summary = _regression_print_fit(fitted, model)
    coefficients = _regression_tidy(fitted, model, alpha)
    _regression_print_coefficients(coefficients, model, alpha, robust)

    crude = None
    if univariable:
        if formula is not None:
            printColor(
                "univariable=True is ignored when a formula is supplied, because an arbitrary "
                "formula cannot be reliably split into single-predictor models. Use the "
                "predictors argument instead.",
                "yellow",
            )
        else:
            crude = _regression_univariable(
                data_clean, outcome, predictors, model, alpha, reference, categorical_limit
            )
            if crude is not None:
                _regression_print_univariable(crude, coefficients, model, alpha)

    _plot_regression_diagnostics(
        fitted,
        model,
        np.asarray(data_clean[outcome], dtype=float) if model == "logistic" else None,
        cooks_distance,
        do_graphs,
    )

    return {
        "model": model,
        "formula": model_formula,
        "n": n,
        "n_dropped": int(dropped),
        "alpha": alpha,
        "coefficients": coefficients,
        "univariable": crude,
        "fit": fit_summary,
        "diagnostics": diagnostics,
        "robust_standard_errors": robust,
        "reference_levels": reference_levels,
        "outcome": outcome,
        "outcome_positive": event_level,
        "outcome_baseline": baseline_level,
        "fitted_model": fitted,
    }
