# Stat Kit
# Helper and wrapper functions for data cleaning, display and statistical analysis.
# Ahmed Elbokl (ahmed.elbokl@med.asu.edu.eg), 2023

# Imports
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from .helpers import percent, plus_minus, print_title, print_mean_std, print_median_iqr

sns.set()

# test successful import
def version():
    """
    Returns the version of the package
    """
    return "1.1.2"


# Drop columns more conveniently
def drop_columns(
    df, cols_to_drop
):  # Ignores missing columns and reports them. cols_to_drop is a python list of column names that need to be dropped.
    """
    Drops columns from a dataframe smoothly as it ignores missing columns (but reports them).
    """
    print("Number of columns to be dropped: {}".format(len(cols_to_drop)))
    missing_cols = [col for col in cols_to_drop if col not in df.columns]
    if len(missing_cols) > 0:
        print(
            "There are {} missing columns that cannot be dropped: {}".format(
                len(missing_cols), missing_cols
            )
        )
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        print("Dropping {} columns instead ..".format(len(cols_to_drop)))

    df.drop(cols_to_drop, axis=1, inplace=True)
    print("Done.")
    return cols_to_drop


# Lookup column names in a dataframe with partial match
def lookup_columns(search_string, df):
    """
    Looks up column names in a dataframe with partial match.
    """
    return [col for col in df.columns if search_string in col]


# compare two column lists to find common and different columns
def compare_columns(list1, list2):
    """
    Compares two column lists to find common and different columns.
    Returns three lists: common columns, only in list1, only in list2.
    """
    set1 = set(list1)
    set2 = set(list2)
    common = list(set1.intersection(set2))
    only_in_list1 = list(set1 - set2)
    only_in_list2 = list(set2 - set1)
    return common, only_in_list1, only_in_list2


# percent, plus_minus, and print_title are imported from helpers


## print_mean_std and print_median_iqr are imported from helpers


# Print Missing Value count and percentage in a pandas series/column
def print_missing(series):  ## Dependent on percent()
    """
    Returns missing value count and percentage of a series printed
    """
    # Count the missing values
    missing_count = series.isna().sum()
    # Calculate the percentage of missing values
    missing_percent = percent(missing_count, len(series))
    # Print the results
    print(
        "Missing values in '" + series.name + "':",
        missing_count,
        "(" + str(missing_percent) + "%)",
    )

# Suggest de-identification of columns in a dataframe
def suggest_deidentification(df: pd.DataFrame):
    """
    Suggests columns that may need de-identification based on common identifiers.
    """
    identifiers = [
        "name",
        "address",
        "phone",
        "email",
        "ssn",
        "dob",
        "date of birth",
        "patient id",
        "mrn",
        "medical record number",
        "zip code",
        "zip",
        "city",
        "state",
        "country",
    ]

    cols_to_deidentify = []
    for col in df.columns:
        for identifier in identifiers:
            if identifier in col.lower():
                cols_to_deidentify.append(col)
                break
    if len(cols_to_deidentify) > 0:
        print("Columns that may need de-identification:")
        for col in cols_to_deidentify:
            print("- " + col)
    else:
        print("No columns found that may need de-identification.")


# ---------------------------------------------------------------------------
# Missing data reporting
# ---------------------------------------------------------------------------

# Values that are frequently used to encode "missing" but that pandas reads as
# ordinary data. Matched case-insensitively after stripping whitespace.
_SENTINEL_STRINGS = {
    "",
    "na",
    "n/a",
    "n.a",
    "n.a.",
    "nan",
    "null",
    "none",
    "missing",
    "unknown",
    "unk",
    "not recorded",
    "not applicable",
    "not done",
    "nd",
    "nk",
    "n/k",
    "?",
    "-",
    "--",
    ".",
    "nil",
}

# Numeric placeholders. These are only reported when they sit far away from the
# rest of the distribution (see _numeric_sentinels), because 9, 99 and friends
# are perfectly ordinary values in most clinical variables.
_SENTINEL_NUMBERS = (-1, -9, -99, -999, -9999, 9, 77, 88, 99, 999, 9999)

_PRESENT_MARK = "●"  # filled circle
_MISSING_MARK = "○"  # hollow circle


def _numeric_sentinels(series):
    """
    Returns [(value, count)] for numeric placeholders that look like hidden
    missing codes: repeated, extreme, and separated from the rest of the data.
    """
    clean = series.dropna()
    if clean.empty:
        return []

    distinct = np.sort(clean.unique())
    if distinct.size < 3:  # too little spread to judge separation
        return []

    found = []
    for candidate in _SENTINEL_NUMBERS:
        if candidate not in distinct:
            continue
        count = int((clean == candidate).sum())
        if count < 2:  # a single occurrence is not a coding convention
            continue

        is_max = candidate == distinct[-1]
        is_min = candidate == distinct[0]
        if not (is_max or is_min):
            continue

        rest = distinct[:-1] if is_max else distinct[1:]
        neighbour = rest[-1] if is_max else rest[0]
        gap = abs(candidate - neighbour)
        spread = rest[-1] - rest[0]
        others = clean[clean != candidate]
        iqr = np.subtract(*np.percentile(others, [75, 25]))

        # A negative code in an otherwise non-negative variable is suspicious on
        # its own; otherwise the gap must dominate both the IQR and the spread.
        if candidate < 0 and rest[0] >= 0:
            found.append((candidate, count))
        elif gap > spread and gap >= 1.5 * iqr:
            found.append((candidate, count))

    return found


def _string_sentinels(series):
    """
    Returns [(value, count)] for string placeholders that look like hidden
    missing codes.
    """
    clean = series.dropna().astype(str)
    if clean.empty:
        return []

    normalised = clean.str.strip().str.lower()
    hits = normalised[normalised.isin(_SENTINEL_STRINGS)]
    if hits.empty:
        return []

    found = []
    for value, count in hits.value_counts().items():
        # Report the value as it appears in the data, not the normalised form.
        as_written = clean[normalised == value].iloc[0]
        label = "'" + as_written + "'" if as_written.strip() != "" else "'' (blank)"
        found.append((label, int(count)))
    return found


def find_sentinel_values(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Scans for values that probably encode missingness but are not NaN
    (e.g. '', 'NA', 'unknown', 999, -1).

    Reports only; nothing is coerced. Returns a dataframe with one row per
    (column, suspected value).
    """
    columns = list(df.columns) if columns is None else list(columns)
    rows = []
    for col in columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            candidates = _numeric_sentinels(series)
        elif pd.api.types.is_bool_dtype(series):
            candidates = []
        else:
            candidates = _string_sentinels(series)

        for value, count in candidates:
            rows.append(
                {
                    "column": col,
                    "suspected_value": value,
                    "count": count,
                    "%_of_rows": percent(count, len(series)) if len(series) else 0.0,
                }
            )

    return pd.DataFrame(rows, columns=["column", "suspected_value", "count", "%_of_rows"])


def missing_summary(df: pd.DataFrame, columns=None, hide_complete: bool = True) -> pd.DataFrame:
    """
    Per-column missing data counts and percentages, sorted by % descending.
    """
    columns = list(df.columns) if columns is None else list(columns)
    n_rows = len(df)

    rows = []
    for col in columns:
        n_missing = int(df[col].isna().sum())
        pct = percent(n_missing, n_rows) if n_rows else 0.0
        if pct == 0:
            flag = "complete"
        elif pct < 5:
            flag = ""
        elif pct < 20:
            flag = "notable"
        elif pct <= 50:
            flag = "high"
        else:
            flag = ">50% - consider dropping"
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "n_missing": n_missing,
                "%_missing": pct,
                "n_present": n_rows - n_missing,
                "flag": flag,
            }
        )

    summary = pd.DataFrame(
        rows, columns=["column", "dtype", "n_missing", "%_missing", "n_present", "flag"]
    )
    if hide_complete:
        summary = summary[summary["n_missing"] > 0]
    return summary.sort_values("%_missing", ascending=False).reset_index(drop=True)


def missing_patterns(df: pd.DataFrame, columns=None, max_patterns: int = 15) -> pd.DataFrame:
    """
    Collapses rows into unique missing/observed signatures across `columns`
    (the equivalent of R's mice::md.pattern).

    Each row is one pattern: filled circle = observed, hollow circle = missing.
    Sorted by frequency; truncated to `max_patterns`.
    """
    columns = list(df.columns) if columns is None else list(columns)
    if not columns or df.empty:
        return pd.DataFrame()

    mask = df[columns].isna()
    signatures = mask.astype(int).astype(str).agg("".join, axis=1)
    counts = signatures.value_counts()

    rows = []
    for pattern_no, (signature, count) in enumerate(counts.items(), start=1):
        row = {"pattern": pattern_no}
        for col, bit in zip(columns, signature):
            row[col] = _MISSING_MARK if bit == "1" else _PRESENT_MARK
        row["n_missing_here"] = signature.count("1")
        row["n_cases"] = int(count)
        row["%_of_cases"] = percent(int(count), len(df))
        rows.append(row)

    patterns = pd.DataFrame(rows)
    if max_patterns is not None and len(patterns) > max_patterns:
        patterns = patterns.head(max_patterns)
    return patterns.reset_index(drop=True)


def _classify_order_patterns(df: pd.DataFrame, order):
    """
    For an ordered list of columns (e.g. follow-up visits), classifies each row
    as complete, monotone (dropout) or intermittent (missing then present).

    Returns (attrition_table, counts_dict).
    """
    mask = df[order].isna().to_numpy()

    complete = (~mask).all(axis=1)
    all_missing = mask.all(axis=1)
    # A row is monotone if no observed value ever follows a missing one.
    intermittent = np.zeros(len(df), dtype=bool)
    for i in range(len(order) - 1):
        seen_missing = mask[:, : i + 1].any(axis=1)
        intermittent |= seen_missing & (~mask[:, i + 1])

    monotone_dropout = (~complete) & (~intermittent)

    counts = {
        "complete": int(complete.sum()),
        "monotone_dropout": int(monotone_dropout.sum()),
        "intermittent": int(intermittent.sum()),
        "all_missing": int(all_missing.sum()),
    }

    rows = []
    previous = None
    for index, col in enumerate(order):
        current = mask[:, index]
        if previous is None:
            new_missing = int(current.sum())
            returned = 0
        else:
            new_missing = int((current & ~previous).sum())
            returned = int((~current & previous).sum())
        rows.append(
            {
                "timepoint": col,
                "n_present": int((~current).sum()),
                "n_missing": int(current.sum()),
                "%_missing": percent(int(current.sum()), len(df)) if len(df) else 0.0,
                "newly_missing": new_missing,
                "returned": returned,
            }
        )
        previous = current

    attrition = pd.DataFrame(
        rows,
        columns=[
            "timepoint",
            "n_present",
            "n_missing",
            "%_missing",
            "newly_missing",
            "returned",
        ],
    )
    return attrition, counts


def plot_missing_matrix(df: pd.DataFrame, columns=None, sort_by=None, figsize=None):
    """
    Nullity matrix: one row per case, one column per variable.
    Dark = observed, light = missing.

    `sort_by` accepts a column name, a list of column names, or "n_missing"
    (sorts cases by how many values they are missing). Sorting is what makes
    staircase/dropout patterns visible.
    """
    columns = list(df.columns) if columns is None else list(columns)
    data = df[columns]

    if sort_by == "n_missing":
        n_missing_per_row = data.isna().sum(axis=1)
        data = data.loc[n_missing_per_row.sort_values(kind="mergesort").index]
    elif sort_by is not None:
        sort_cols = [sort_by] if isinstance(sort_by, str) else list(sort_by)
        data = df.sort_values(sort_cols, kind="mergesort", na_position="last")[columns]

    mask = data.isna().to_numpy().astype(int)

    if figsize is None:
        figsize = (max(6.0, 0.45 * len(columns) + 2.0), 5.0)

    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.grid(False)
    ax.imshow(
        mask,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(["#31456b", "#f2f2f2"]),
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=90, fontsize=8)
    ax.set_ylabel("cases (n = {})".format(len(data)))
    for edge in range(1, len(columns)):
        ax.axvline(edge - 0.5, color="white", linewidth=0.8)

    title = "Missing data matrix"
    if sort_by is not None:
        title += " (sorted by {})".format(
            sort_by if isinstance(sort_by, str) else ", ".join(sort_by)
        )
    ax.set_title(title)
    ax.legend(
        handles=[
            Patch(facecolor="#31456b", label="observed"),
            Patch(facecolor="#f2f2f2", edgecolor="#cccccc", label="missing"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=8,
    )
    plt.tight_layout()
    plt.show()


def plot_missing_bars(summary: pd.DataFrame, figsize=None):
    """
    Horizontal bar chart of % missing per column, from a missing_summary table.
    """
    if summary.empty:
        return

    if figsize is None:
        figsize = (6.5, max(2.5, 0.32 * len(summary) + 1.0))

    plt.figure(figsize=figsize)
    ax = plt.gca()
    order = summary.sort_values("%_missing")
    ax.barh(order["column"], order["%_missing"], color="#31456b")
    for y, (pct, n) in enumerate(zip(order["%_missing"], order["n_missing"])):
        ax.text(pct + 0.6, y, "{}%  (n={})".format(pct, n), va="center", fontsize=8)
    ax.set_xlim(0, min(100, max(order["%_missing"]) * 1.35 + 6))
    ax.set_xlabel("% missing")
    ax.set_title("Missing data per column")
    plt.tight_layout()
    plt.show()


def plot_missing_correlation(df: pd.DataFrame, columns=None, figsize=None):
    """
    Correlation between missingness indicators: which columns disappear
    together. Columns that are wholly present or wholly missing are dropped
    because their indicator has no variance.
    """
    columns = list(df.columns) if columns is None else list(columns)
    mask = df[columns].isna().astype(int)
    varying = [col for col in columns if 0 < mask[col].sum() < len(df)]

    if len(varying) < 2:
        print(
            "Nullity correlation skipped: needs at least 2 columns with partial "
            "missingness (found {}).".format(len(varying))
        )
        return None

    corr = mask[varying].corr()

    if figsize is None:
        side = max(4.5, 0.5 * len(varying) + 2.0)
        figsize = (side, side * 0.85)

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="vlag",
        annot=len(varying) <= 12,
        fmt=".2f",
        annot_kws={"fontsize": 8},
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "correlation of missingness"},
    )
    plt.title("Do these columns go missing together?")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()
    return corr


def missing_report(
    df: pd.DataFrame,
    columns=None,
    order=None,
    sort_by=None,
    max_patterns: int = 15,
    hide_complete: bool = True,
    detect_sentinels: bool = True,
    mar_tests: bool = False,
    test_against=None,
    mar_force: bool = False,
    plot: bool = True,
):
    """
    Reports the extent and shape of missing data, numerically first and then
    visually.

    Numeric output
        1. Per-column counts and percentages.
        2. Suspected hidden missing codes ('NA', 999, -1 ...) if
           detect_sentinels is True. Reported only, never coerced. This comes
           before the analyses below because it can invalidate them.
        3. Missing/observed patterns across columns (mice::md.pattern style),
           plus an attrition table and a monotone-vs-intermittent verdict when
           `order` is given.
        4. MAR testing, only when mar_tests=True (see missing_mar_tests).

    Visual output
        Nullity matrix, % missing bar chart, and a nullity correlation heatmap.

    Parameters
        columns : subset to analyse. Default: every column.
        order : ordered longitudinal columns, e.g. ["Bl", "T1", "T2", "T3", "T4"].
            Enables the dropout/attrition analysis and is prepended to the
            analysed columns.
        sort_by : row ordering for the matrix plot; a column name, a list of
            names, or "n_missing".
        max_patterns : truncate the pattern table.
        hide_complete : exclude columns with no missing values.
        detect_sentinels : scan for hidden missing codes.
        mar_tests : also run missing_mar_tests(). Off by default: it is an
            inferential step, it is quadratic in the number of columns, and it
            should be run only after any hidden missing codes found in section 2
            have been converted to NaN.
        test_against : predictors passed through to missing_mar_tests().
        mar_force : pass force=True to missing_mar_tests(), overriding its
            test-count guard.
        plot : draw the figures.

    Returns a dict of the underlying dataframes: summary, patterns, attrition,
    order_counts, sentinels, mar_tests, mar_flagged, and nullity_correlation.
    The mar_* keys are always present, and empty when mar_tests is False.

    Note on interpretation: counts and patterns can suggest that missingness is
    structured, but they cannot establish MCAR vs MAR vs MNAR on their own.
    A monotone pattern points to dropout rather than sporadic non-response;
    whether missingness is predicted by observed variables is what mar_tests
    tests.
    """
    n_rows = len(df)
    if n_rows == 0:
        print("The dataframe is empty; nothing to report.")
        return {}

    if columns is None:
        analysed = list(df.columns)
    else:
        analysed = list(columns)
    if order is not None:
        analysed = list(order) + [col for col in analysed if col not in set(order)]

    unknown = [col for col in analysed if col not in df.columns]
    if unknown:
        print("Ignoring {} column(s) not in the dataframe: {}".format(len(unknown), unknown))
        analysed = [col for col in analysed if col in df.columns]
    if order is not None:
        order = [col for col in order if col in df.columns]

    # Sections are numbered as they are printed, so switching one off does not
    # leave a gap in the numbering.
    section_number = {"n": 0}

    def section(label, sub=False):
        if not sub:
            section_number["n"] += 1
            print_title("{}. {}".format(section_number["n"], label))
        else:
            print_title("{}b. {}".format(section_number["n"], label))

    # ---- Layer 1: per column -------------------------------------------------
    print_title("Missing Data Report")
    full_summary = missing_summary(df, analysed, hide_complete=False)
    summary = full_summary[full_summary["n_missing"] > 0] if hide_complete else full_summary
    summary = summary.reset_index(drop=True)

    n_complete_cols = int((full_summary["n_missing"] == 0).sum())
    complete_rows = int(df[analysed].notna().all(axis=1).sum())
    total_cells = n_rows * len(analysed)
    missing_cells = int(df[analysed].isna().sum().sum())

    print("Rows: {}   Columns analysed: {}".format(n_rows, len(analysed)))
    print(
        "Missing cells: {} of {} ({}%)".format(
            missing_cells, total_cells, percent(missing_cells, total_cells) if total_cells else 0.0
        )
    )
    print(
        "Complete cases: {} ({}%)".format(complete_rows, percent(complete_rows, n_rows))
    )
    print(
        "Columns with no missing values: {} of {}".format(n_complete_cols, len(analysed))
    )

    section("Missing Values per Column")
    if summary.empty:
        print("No missing values in the analysed columns.")
    else:
        print(summary.to_string(index=False))
        if hide_complete and n_complete_cols:
            print(
                "\n({} complete column(s) hidden; pass hide_complete=False to list them.)".format(
                    n_complete_cols
                )
            )

    # ---- Hidden missing codes ------------------------------------------------
    # Reported before the pattern analysis: if these really encode missingness,
    # every table below is computed from the wrong mask.
    sentinels = pd.DataFrame()
    if detect_sentinels:
        sentinels = find_sentinel_values(df, analysed)
        section("Suspected Hidden Missing Codes")
        if sentinels.empty:
            print("None detected.")
        else:
            print(sentinels.to_string(index=False))
            print(
                "\nThese values are NOT counted as missing anywhere in this report. "
                "If they really encode missingness, replace them with NaN and re-run; "
                "the patterns below are computed as though they were real data."
            )

    # ---- Layer 2: patterns ---------------------------------------------------
    pattern_cols = list(summary["column"]) if hide_complete else analysed
    if order is not None:
        pattern_cols = order + [col for col in pattern_cols if col not in set(order)]

    patterns = pd.DataFrame()
    attrition = pd.DataFrame()
    order_counts = {}

    section("Missing Data Patterns")
    if not pattern_cols:
        print("No columns with missing values, so there are no patterns to show.")
    else:
        patterns = missing_patterns(df, pattern_cols, max_patterns=max_patterns)
        n_all_patterns = df[pattern_cols].isna().astype(int).astype(str).agg("".join, axis=1).nunique()
        print(
            "{} = observed, {} = missing. {} distinct pattern(s) across {} column(s).".format(
                _PRESENT_MARK, _MISSING_MARK, n_all_patterns, len(pattern_cols)
            )
        )
        print(patterns.to_string(index=False))
        if n_all_patterns > len(patterns):
            print(
                "\n(Showing the {} most frequent of {} patterns; raise max_patterns to see more.)".format(
                    len(patterns), n_all_patterns
                )
            )

    if order is not None and len(order) >= 2:
        attrition, order_counts = _classify_order_patterns(df, order)
        section("Follow-up Attrition", sub=True)
        print(attrition.to_string(index=False))
        print(
            "\nComplete across all {} timepoints: {} ({}%)".format(
                len(order),
                order_counts["complete"],
                percent(order_counts["complete"], n_rows),
            )
        )
        print(
            "Monotone (once missing, always missing): {} ({}%)".format(
                order_counts["monotone_dropout"],
                percent(order_counts["monotone_dropout"], n_rows),
            )
        )
        print(
            "Intermittent (observed again after a gap): {} ({}%)".format(
                order_counts["intermittent"],
                percent(order_counts["intermittent"], n_rows),
            )
        )
        incomplete = n_rows - order_counts["complete"]
        if incomplete == 0:
            print("\nNo missingness across the ordered columns.")
        elif order_counts["intermittent"] == 0:
            print(
                "\nVerdict: the pattern is MONOTONE - consistent with dropout/attrition "
                "rather than sporadic item non-response."
            )
        else:
            share = percent(order_counts["intermittent"], incomplete)
            print(
                "\nVerdict: the pattern is NON-MONOTONE - {}% of incomplete cases are "
                "observed again after a gap, so this is not pure dropout.".format(share)
            )
        print(
            "Either way, this describes the shape of the missingness, not its cause. "
            "Whether missingness is predicted by observed variables (MAR) is a separate test."
        )

    # ---- Layer 3: MAR testing (opt-in) ---------------------------------------
    mar_results = pd.DataFrame()
    mar_flagged = {}
    if mar_tests:
        mar = missing_mar_tests(
            df,
            columns=list(summary["column"]) if not summary.empty else None,
            test_against=test_against,
            force=mar_force,
            verbose=True,
        )
        mar_results = mar["results"]
        mar_flagged = mar["flagged"]
    elif not summary.empty:
        candidates = list(summary["column"])
        print(
            "\nNot tested: whether missingness is predicted by the observed data "
            "(MCAR vs MAR). To check, run:"
        )
        print(
            "    cleaning_kit.missing_mar_tests(df, columns={})".format(
                candidates if len(candidates) <= 6 else candidates[:6] + ["..."]
            )
        )
        print("    (or pass mar_tests=True to this function)")

    # ---- Layer 4: visuals ----------------------------------------------------
    nullity_corr = None
    if plot:
        matrix_cols = pattern_cols if pattern_cols else analysed
        if matrix_cols:
            plot_missing_matrix(df, matrix_cols, sort_by=sort_by)
        if not summary.empty:
            plot_missing_bars(summary)
            nullity_corr = plot_missing_correlation(df, matrix_cols)

    return {
        "summary": full_summary if not hide_complete else summary,
        "patterns": patterns,
        "attrition": attrition,
        "order_counts": order_counts,
        "sentinels": sentinels,
        "mar_tests": mar_results,
        "mar_flagged": mar_flagged,
        "nullity_correlation": nullity_corr,
    }


# ---------------------------------------------------------------------------
# Missing data: is missingness predicted by the observed data? (MAR check)
# ---------------------------------------------------------------------------


def _mar_describe_numeric(series) -> str:
    """
    Compact 'median (Q1, Q3)' description of a numeric group. A comma rather
    than a dash, so negative quartiles stay readable.
    """
    return "{:.2f} ({:.2f}, {:.2f})".format(
        series.median(), series.quantile(0.25), series.quantile(0.75)
    )


def _mar_describe_categorical(series, reference_level) -> str:
    """
    Compact '<level> n (%)' description of a categorical group, always reported
    against the same reference level so the two groups are comparable.
    """
    count = int((series == reference_level).sum())
    return "{} {} ({}%)".format(reference_level, count, percent(count, len(series)))


def _mar_numeric_test(values_missing, values_present, numeric_test):
    """
    Compares a numeric predictor between the missing and present groups.
    Returns (test_name, effect_label, effect_value, p_value).
    """
    first = np.asarray(values_missing, dtype=float)
    second = np.asarray(values_present, dtype=float)

    if numeric_test == "t":
        result = stats.ttest_ind(first, second, equal_var=False)
        # Hedges-corrected Cohen's d on the pooled SD.
        n1, n2 = first.size, second.size
        pooled_sd = np.sqrt(
            ((n1 - 1) * np.var(first, ddof=1) + (n2 - 1) * np.var(second, ddof=1))
            / (n1 + n2 - 2)
        )
        d = (first.mean() - second.mean()) / pooled_sd if pooled_sd > 0 else np.nan
        return "Welch t", "Cohen's d", float(d), float(result.pvalue)

    result = stats.mannwhitneyu(first, second, alternative="two-sided")
    # P(missing > present) + 0.5*P(equal); 0.5 is the null.
    probability = float(result.statistic) / (first.size * second.size)
    return "Mann-Whitney", "P(superiority)", probability, float(result.pvalue)


def _mar_categorical_test(labels_missing, labels_present):
    """
    Compares a categorical predictor between the missing and present groups.
    Returns (test_name, effect_label, effect_value, p_value).
    """
    groups = pd.Series(
        ["missing"] * len(labels_missing) + ["present"] * len(labels_present)
    )
    values = pd.Series(
        np.concatenate([np.asarray(labels_missing, dtype=object), np.asarray(labels_present, dtype=object)])
    )
    table = pd.crosstab(groups, values)
    if table.shape[1] < 2:
        return None

    chi2, p_value, _, expected = stats.chi2_contingency(table)
    total = table.to_numpy().sum()
    cramers_v = np.sqrt(chi2 / (total * (min(table.shape) - 1)))

    if table.shape == (2, 2) and expected.min() < 5:
        _, p_value = stats.fisher_exact(table.to_numpy())
        test_name = "Fisher exact"
    else:
        test_name = "Chi-square"

    return test_name, "Cramer's V", float(cramers_v), float(p_value)


def missing_mar_tests(
    df: pd.DataFrame,
    columns=None,
    test_against=None,
    numeric_test: str = "mannwhitney",
    correction: str = "holm",
    alpha: float = 0.05,
    min_group: int = 5,
    max_levels: int = 10,
    max_tests: int = 200,
    force: bool = False,
    verbose: bool = True,
):
    """
    Tests whether missingness is predicted by the observed data.

    For every column with missing values, the sample is split into cases
    *missing* on that column and cases *observed* on it, and every other column
    is compared between those two groups. A difference means missingness is not
    random with respect to that variable, which is evidence for MAR and names
    the variables that belong in an imputation model.

    Parameters
        columns : columns whose missingness is the outcome. Default: every
            column with at least one missing and one observed value.
        test_against : predictors to compare between the two groups.
            Default: every other column.
        numeric_test : "mannwhitney" (default, distribution-free) or "t"
            (Welch). The default is deliberate: these are screening tests over
            many variables of unknown shape.
        correction : "holm" (default), "fdr_bh", "bonferroni" or None.
            Applied across all tests performed in this call.
        min_group : minimum n per group for a comparison to be attempted.
        max_levels : categorical predictors with more levels are skipped.
        max_tests : refuse to run more than this many tests unless force=True.

    Returns a dict with:
        results : one row per comparison, sorted by adjusted p-value.
        skipped : comparisons that could not be run, with the reason.
        flagged : {column: [predictors that significantly predict its
            missingness]}.

    Interpretation. A significant row means missingness on that column depends
    on an observed variable: consistent with MAR, and that variable should be
    carried into any imputation model. A null result is NOT evidence of MCAR -
    it may simply be underpowered, and these are unadjusted screening
    comparisons. MNAR (missingness depending on the unrecorded value itself)
    cannot be detected from observed data by any test, so it can never be ruled
    out here.
    """
    if numeric_test not in ("mannwhitney", "t"):
        raise ValueError("numeric_test must be 'mannwhitney' or 't'")
    if correction not in ("holm", "fdr_bh", "bonferroni", None):
        raise ValueError("correction must be 'holm', 'fdr_bh', 'bonferroni' or None")

    empty = {
        "results": pd.DataFrame(),
        "skipped": pd.DataFrame(),
        "flagged": {},
    }

    n_rows = len(df)
    if n_rows == 0:
        if verbose:
            print("The dataframe is empty; nothing to test.")
        return empty

    # Outcomes: columns whose missingness varies.
    if columns is None:
        targets = [col for col in df.columns if 0 < df[col].isna().sum() < n_rows]
    else:
        targets = [col for col in columns if col in df.columns]
        constant = [col for col in targets if not 0 < df[col].isna().sum() < n_rows]
        if constant and verbose:
            print(
                "Skipping {} column(s) with no missing values or no observed values: "
                "{}".format(len(constant), constant)
            )
        targets = [col for col in targets if col not in set(constant)]

    if not targets:
        if verbose:
            print("No columns have a mix of missing and observed values; nothing to test.")
        return empty

    predictors = list(df.columns) if test_against is None else list(test_against)
    predictors = [col for col in predictors if col in df.columns]

    planned = sum(1 for target in targets for pred in predictors if pred != target)
    if planned == 0:
        if verbose:
            print("No predictor columns left to test against.")
        return empty
    if planned > max_tests and not force:
        print(
            "This would run {} tests ({} column(s) x {} predictor(s)), above "
            "max_tests={}.".format(planned, len(targets), len(predictors), max_tests)
        )
        print(
            "Narrow it with test_against=[...] or columns=[...], raise max_tests, "
            "or pass force=True. Testing everything against everything makes the "
            "multiplicity problem worse, not the analysis better."
        )
        return empty

    rows = []
    skipped = []
    for target in targets:
        is_missing = df[target].isna()
        n_missing = int(is_missing.sum())

        for pred in predictors:
            if pred == target:
                continue

            predictor = df[pred]
            observed = predictor.notna()
            group_missing = predictor[is_missing & observed]
            group_present = predictor[~is_missing & observed]

            if len(group_missing) < min_group or len(group_present) < min_group:
                skipped.append(
                    {
                        "missing_in": target,
                        "compared_on": pred,
                        "reason": "fewer than {} observed values in one group "
                        "(missing={}, present={})".format(
                            min_group, len(group_missing), len(group_present)
                        ),
                    }
                )
                continue

            if predictor[observed].nunique() < 2:
                skipped.append(
                    {"missing_in": target, "compared_on": pred, "reason": "constant column"}
                )
                continue

            is_numeric = pd.api.types.is_numeric_dtype(predictor) and not pd.api.types.is_bool_dtype(
                predictor
            )

            if is_numeric:
                kind = "numeric"
                test_name, effect_label, effect, p_value = _mar_numeric_test(
                    group_missing, group_present, numeric_test
                )
                described_missing = _mar_describe_numeric(group_missing)
                described_present = _mar_describe_numeric(group_present)
            else:
                n_levels = predictor[observed].nunique()
                if n_levels > max_levels:
                    skipped.append(
                        {
                            "missing_in": target,
                            "compared_on": pred,
                            "reason": "{} categorical levels (max_levels={})".format(
                                n_levels, max_levels
                            ),
                        }
                    )
                    continue

                kind = "categorical"
                outcome = _mar_categorical_test(group_missing, group_present)
                if outcome is None:
                    skipped.append(
                        {
                            "missing_in": target,
                            "compared_on": pred,
                            "reason": "only one level present across both groups",
                        }
                    )
                    continue
                test_name, effect_label, effect, p_value = outcome
                reference = predictor[observed].value_counts().index[0]
                described_missing = _mar_describe_categorical(group_missing, reference)
                described_present = _mar_describe_categorical(group_present, reference)

            rows.append(
                {
                    "missing_in": target,
                    "n_missing": n_missing,
                    "compared_on": pred,
                    "type": kind,
                    "group_missing": described_missing,
                    "group_present": described_present,
                    "test": test_name,
                    "effect": effect_label,
                    "effect_size": round(effect, 3),
                    "p": p_value,
                }
            )

    results = pd.DataFrame(rows)
    skipped_table = pd.DataFrame(skipped, columns=["missing_in", "compared_on", "reason"])

    if results.empty:
        if verbose:
            print("No comparison could be run.")
            if not skipped_table.empty:
                print(
                    "\n{} comparison(s) skipped. Most common reason: {}".format(
                        len(skipped_table), skipped_table["reason"].mode().iloc[0]
                    )
                )
        return {"results": results, "skipped": skipped_table, "flagged": {}}

    # Multiplicity correction across every test in this call.
    if correction is None:
        results["p_adj"] = results["p"]
        results["significant"] = results["p"] < alpha
        correction_label = "none (unadjusted)"
    else:
        from statsmodels.stats.multitest import multipletests

        rejected, adjusted, _, _ = multipletests(
            results["p"].to_numpy(), alpha=alpha, method=correction
        )
        results["p_adj"] = adjusted
        results["significant"] = rejected
        correction_label = correction

    results = results.sort_values(["p_adj", "p"], kind="mergesort").reset_index(drop=True)
    results["p"] = results["p"].map(lambda value: round(value, 4))
    results["p_adj"] = results["p_adj"].map(lambda value: round(value, 4))

    flagged = {
        target: group.loc[group["significant"], "compared_on"].tolist()
        for target, group in results.groupby("missing_in")
    }
    flagged = {target: preds for target, preds in flagged.items() if preds}

    if verbose:
        print_title("Is Missingness Predicted by the Observed Data? (MAR check)")
        print(
            "{} test(s) across {} column(s) with missing values. "
            "Multiplicity correction: {}.".format(
                len(results), results["missing_in"].nunique(), correction_label
            )
        )
        print(
            "Groups are cases MISSING vs OBSERVED on each column; numeric shown as "
            "median (Q1, Q3).\n"
        )
        display = results.copy()
        for column in ("p", "p_adj"):
            display[column] = display[column].map(
                lambda value: "<0.0001" if value < 0.0001 else "{:.4f}".format(value)
            )
        print(display.to_string(index=False))

        if not skipped_table.empty:
            print(
                "\n{} comparison(s) could not be run (see the 'skipped' table). "
                "Most common reason: {}".format(
                    len(skipped_table), skipped_table["reason"].mode().iloc[0]
                )
            )

        print("")
        if flagged:
            print(
                "Missingness IS predicted by observed variables in {} column(s):".format(
                    len(flagged)
                )
            )
            for target, preds in flagged.items():
                print("  {} <- {}".format(target, ", ".join(preds)))
            print(
                "\nThis is consistent with MAR. These predictors should be carried into "
                "any imputation model, and complete-case analysis is likely biased."
            )
        else:
            print(
                "No comparison reached significance after correction. This is NOT "
                "evidence of MCAR - it may simply be underpowered."
            )
        print(
            "These are screening comparisons, not confirmatory tests. And MNAR "
            "(missingness driven by the unrecorded value itself, e.g. a deficit not "
            "recorded because it had resolved) cannot be detected from observed data "
            "by any test, so it is never ruled out here."
        )

    return {"results": results, "skipped": skipped_table, "flagged": flagged}
