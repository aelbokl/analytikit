# Stat Kit
# Helper and wrapper functions for data cleaning, display and statistical analysis.
# Ahmed Elbokl (ahmed.elbokl@med.asu.edu.eg)
# Maha
# 2023-2026

# Imports
import pandas as pd
import numpy as np
import scipy.stats as stats
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scikit_posthocs import posthoc_dunn as dunn
import scikit_posthocs as sp
import copy
import math 
import warnings
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.stats.contingency_tables import SquareTable 
from scipy.stats import friedmanchisquare
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
    "method": "auto",
}


def _display_p_value(p_value):
    """Round a p-value for display without changing the value used for inference."""
    return round(float(p_value), 3)


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


def _wilcoxon_test(group1, group2=None):
    """Run the package's explicit two-sided Wilcoxon signed-rank policy."""
    return stats.wilcoxon(group1, group2, **_WILCOXON_OPTIONS)


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

# Function to calculate confidence interval for a proportion ci
def proportion_ci(count, nobs, alpha=0.05):
    """
    Calculate the confidence interval for a proportion.

    Parameters:
    count (int): The number of successes or events of interest.
    nobs (int): The total number of observations or trials.
    alpha (float, optional): The significance level. Defaults to 0.05.

    Returns:
    tuple: A tuple containing the lower and upper bounds of the confidence interval for the proportion.
    """
    # Type and value checks
    if not isinstance(count, int) or not isinstance(nobs, int):
        raise ValueError("count and nobs must be integers")
    if not isinstance(alpha, float) or not (0 < alpha < 1):
        raise ValueError("alpha must be a float between 0 and 1")
    if count > nobs:
        raise ValueError("count cannot be greater than nobs")
    if nobs == 0:
        raise ValueError("nobs must be greater than 0")

    # Calculate confidence interval
    ci_low, ci_upp = stats.binom.interval(alpha=1-alpha, n=nobs, p=count/nobs)
    
    # Normalize the interval
    return ci_low/nobs, ci_upp/nobs


#function to calculate the confidence interval for the difference between two independent means 
def calculate_ci_diff(ci_list, mode):
    """
    Calculate the difference between confidence intervals for multiple groups.
    
    Parameters:
    - ci_list: List of tuples [(lower1, upper1), (lower2, upper2), ...] for each group.
    
    Returns:
    - Tuple containing (min_diff, max_diff) representing the range of CI differences.
    """
    if len(ci_list) < 2:
        raise ValueError("At least two confidence intervals are required.")
    
 
    ci_diffs = []
    for i in range(len(ci_list)):
        for j in range(i + 1, len(ci_list)):
            lower_diff = ci_list[j][0] - ci_list[i][1]  # Lower bound of difference
            upper_diff = ci_list[j][1] - ci_list[i][0]  # Upper bound of difference
            ci_diffs.append((lower_diff, upper_diff))
    
    if mode=="mean":
        return np.min([d[0] for d in ci_diffs]), np.max([d[1] for d in ci_diffs])
    elif mode=="median":
        #calculate 2.5 and 97.5 percentiles
        return np.percentile([d[0] for d in ci_diffs], 2.5), np.percentile([d[1] for d in ci_diffs], 97.5)
    # elif mode=="proportion":
    #    
    else:
        raise ValueError("Invalid mode. Choose 'mean', 'median', or 'proportion'.")

#function to combine mean_ci/ median_ci/ proportion_ci with calculate_ci_diff to calculate ci diff from series 
def calculate_ci_diff_from_series(series_list, mode, num_samples=1000, alpha=0.05):
    """
    Calculate the difference between confidence intervals for multiple groups.
    
    Parameters:
    - series_list: List of series for each group.
    - mode: Type of CI to calculate. can be 'mean', 'median', or 'proportion'.
    - alpha: Confidence level (default is 0.05).

    Returns:
    - Tuple containing (min_diff, max_diff) representing the range of CI differences.
    """
    if mode=="mean":
        ci_list = [mean_ci(series) for series in series_list]
    elif mode=="median":
        ci_list = [median_ci(series, num_samples, alpha) for series in series_list]
    # elif mode=="proportion":
    #     ci_list = [proportion_ci(series.sum(), len(series)) for series in series_list]
    else:
        raise ValueError("Invalid mode. Choose 'mean', 'median', or 'proportion'.")
    return calculate_ci_diff(ci_list, mode)

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
):  # Depends on print_mean_std()
    # Note: this function does not handle the independent variable (grouping variable). It assumes that the series are already grouped.
    # groups: a list of pandas series/columns to be compared
    # group_labels: a list of labels for the groups. If not provided, the group names + group number will be used as labels. The list of labels has to correspond to the list of groups.
    # force_test: if provided, the function will use the test provided instead of determining it automatically. The test has to be a string and has to be one of the following: "ttest", "one-way ANOVA", "mannwhitney", "kruskalwallis", "chisquare", 'fisherexact'
    # alpha: the alpha value to be used for the test. Default is 0.05
    # data_type: 'cont' or 'cat'. If not provided, the function will attempt to infer it from the number of unique values (unreliable for low-resolution continuous variables).
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

            #ci difference
            ci_result = compute_confidence_interval_difference(groups[0], groups[1], method="median", paired=False, alpha=alpha)
            ci_lower, ci_upper = ci_result["CI"]
            ci_inclusive = ci_result["CI_inclusive"]

            

# # Compute the effect size (Cohen's U3) #N.B. #Cohen’s U₃ gives a percentile-based interpretation of effect size based on cohen's d#
# n1 = len(groups[0])
# n2 = len(groups[1])

# R1 = (n1 * (n1 + n2 + 1)) / 2
# R2 = n1 * n2 - R1

# u3 = (R1 - R2) / n1
# u3 = (statistic - (n1 * (n1 + 1)) / 2) / (n1 * n2)

# effect_size["label"] = "Cohen's U3"
# effect_size["value"] = u3

            #MAHA# #Compute the effect size for Mann-Whitney U test (Cliff’s Delta (δ))
            # delta = (2 * U) / (n1 * n2) - 1 
            # where U is the Mann-Whitney U statistic, n1 is the number of observations in the first group, and n2 is the number of observations in the second group.
            n1 = len(groups[0])
            n2 = len(groups[1])
            U = statistic  # Save unrounded for effect size calculation
            delta = (2 * U) / (n1 * n2) - 1
            effect_size["label"] = "Cliff's Delta (δ)"
            effect_size["value"] = round(delta, 3)
            
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
            #ci
            # ci_lower, ci_upper=calculate_ci_diff_from_series(groups, 'median')

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

    # #ci
    
    # if not continuous and 'contingency_table' in locals():
    #     for col in contingency_table.columns:
    #         count = contingency_table[col].sum()
    #         ci_low, ci_upp = proportion_ci(count, n)
    #         print(f"{col} proportion 95% CI: [{ci_low:.2f}, {ci_upp:.2f}]")

    # if not continuous:
    #     for col in contingency_table.columns:
    #         count = contingency_table[col].sum()
    #         ci_low, ci_upp = proportion_ci(count, n)
    #         print(f"{col} proportion 95% CI: [{ci_low:.2f}, {ci_upp:.2f}]")

    print("p-value:", _display_p_value(p_value))
    
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
    if len(effect_size) > 0:
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
            statistic_raw, p_value = _wilcoxon_test(groups[0], groups[1])
            statistic = round(statistic_raw, 3)

            #ci difference
            ci_result = compute_confidence_interval_difference(groups[0], groups[1], method="median", paired=True, alpha=alpha)
            ci_lower, ci_upper = ci_result["CI"]
            ci_inclusive = ci_result["CI_inclusive"]

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


    # #print ci
    # if not continuous and 'contingency_table' in locals():
    #     for col in contingency_table.columns:
    #         count = contingency_table[col].sum()
    #         ci_low, ci_upp = proportion_ci(count, n)
    #         print(f"{col} proportion 95% CI: [{ci_low:.2f}, {ci_upp:.2f}]")

    # if not continuous:
    #     for col in contingency_table.columns:
    #         count = contingency_table[col].sum()
    #         ci_low, ci_upp = proportion_ci(count, n)
    #         print(f"{col} proportion 95% CI: [{ci_low:.2f}, {ci_upp:.2f}]")


    print("p-value:", _display_p_value(p_value))
    
    if ci_lower!=None and ci_upper!=None:
        confidence_level= 100-(alpha*100)
        left_bracket = "[" if ci_inclusive[0] else "("
        right_bracket = "]" if ci_inclusive[1] else ")"
        print(f"{confidence_level}% CI: {left_bracket}{ci_lower:.2f}, {ci_upper:.2f}{right_bracket}")
        if not all(ci_inclusive):
            print("* Parentheses indicate an excluded endpoint.")
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
    result={
        "method": method,
        "correlation_coefficient": round(statistic, 3),
        "p_value": round(p_value, 3),
        "direction": direction,
        "significance": significance
    } 

    # # Iterate over the dictionary items
    for key, value in result.items():
        if key == "significance":
            color = "green" if str(value).lower().startswith("sign") else "red"
            printColor(f"{key}: {value}", color)
        else:
            print(f"{key}:", value)
    
    return result


# A function to calculate confidence interval difference between two groups
# works both for parametric (method="mean") and non-parametric (method="median") tests
# works for both paired (paired=True) and independent (paired=False) samples.
def compute_confidence_interval_difference(group1, group2, method="mean", paired=False, alpha=0.05, n_bootstrap=10000):
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
            diffs = np.asarray(group1, dtype=float) - np.asarray(group2, dtype=float)
            n = len(diffs)
            if n == 0:
                raise ValueError("Paired samples must contain at least one observation")

            test_result = _wilcoxon_test(group1, group2)
            test_stat, p_value = test_result.statistic, test_result.pvalue

            if np.all(diffs == 0):
                observed_diff = 0.0
                ci_lower, ci_upper = 0.0, 0.0
                ci_inclusive = (True, True)
            else:
                # Walsh averages: (d_i + d_j) / 2 for all i <= j
                walsh = np.array([(diffs[i] + diffs[j]) / 2
                                  for i in range(n) for j in range(i, n)])
                walsh.sort()
                observed_diff = np.median(walsh)

                def wilcoxon_p_value_at(shift):
                    if shift == 0:
                        return p_value
                    return _wilcoxon_test(diffs - shift).pvalue

                (ci_lower, ci_upper), ci_inclusive = _test_inverted_interval(
                    walsh,
                    wilcoxon_p_value_at,
                    observed_diff,
                    alpha,
                )
            test_method = _WILCOXON_OPTIONS["method"]

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
    }
