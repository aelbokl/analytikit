# analytikit

analytikit is a small Python package for practical data cleaning, descriptive summaries, and common statistical comparisons.

It currently includes the full package surface in this repository:

- `stat_kit` for statistical testing, confidence intervals, and correlation helpers.
- `cleaning_kit` for cleaning utilities and missing-data reporting.
- `helpers` for shared descriptive-statistics and formatting helpers.

## Package layout

### `stat_kit`

Main statistical utilities, including:

- `compare_ind` for comparing independent groups.
- `compare_dep` for comparing dependent or repeated-measures groups.
- `correlate` for Pearson or Spearman correlation.
- `correlate_biserial` for correlating a continuous variable with a two-level (binary) variable, auto-selecting between the point-biserial and rank-biserial correlation based on a per-group Shapiro-Wilk normality check.
- `correlate_p_biserial` for the point-biserial correlation (Pearson correlation with a 0/1-coded variable), with a Fisher-z confidence interval.
- `correlate_rank_biserial` for the rank-biserial correlation (Cliff's delta from the Mann-Whitney U test), for skewed, ordinal, or outlier-prone continuous variables.
- `mann_whitney_effect_ci` for the Mann-Whitney probability-of-superiority and Cliff's delta confidence interval, usable on its own.
- `regress` for regression modelling, auto-selecting a linear or logistic model from the outcome variable and reporting post-fit assumption diagnostics.
- `compute_confidence_interval_difference` for mean or median difference confidence intervals.
- plotting helpers used during comparisons.

`stat_kit.version()` currently returns `1.1.2`.

### `cleaning_kit`

Data cleaning and reporting helpers, including:

- `drop_columns` to safely drop columns while reporting missing column names.
- `lookup_columns` to find columns by partial match.
- `compare_columns` to compare two column lists.
- `print_missing` to report missing counts and percentages for one series.
- `suggest_deidentification` to flag columns that may contain identifiers.
- `missing_report` for a full missing-data report: per-column counts, hidden
  missing codes, missing/observed patterns, follow-up attrition, and three plots.
- `missing_mar_tests` to test whether missingness is predicted by the observed
  data (the MCAR vs MAR question).

The pieces `missing_report` assembles are also callable on their own:
`missing_summary`, `find_sentinel_values`, `missing_patterns`,
`plot_missing_matrix`, `plot_missing_bars` and `plot_missing_correlation`.

`cleaning_kit.version()` currently returns `1.1.2`.

### `helpers`

Shared helper functions used across the package, including:

- `percent`
- `plus_minus`
- `print_title`
- `mean_ci`
- `median_ci`
- `print_mean_std`
- `print_median_iqr`

## Installation

From the package root:

```bash
pip install -e .
```

Or install dependencies first:

```bash
pip install -r requirements.txt
```

To build a source distribution or wheel with modern tooling:

```bash
python -m build
```

## Dependencies

Runtime dependencies are declared in `pyproject.toml`:

- pandas
- numpy
- scipy
- pingouin
- statsmodels
- scikit-posthocs
- seaborn
- matplotlib

## Quick start

You can import from the package root:

```python
import analytikit

print(analytikit.version())
print(analytikit.percent(25, 40))
```

Or import individual modules explicitly:

```python
from analytikit import stat_kit, cleaning_kit
```

## Usage examples

### 1. Cleaning utilities

```python
import pandas as pd
from analytikit import cleaning_kit

df = pd.DataFrame({
    'patient_id': [1, 2, 3],
    'age': [30, 42, None],
    'group': ['A', 'A', 'B']
})

cleaning_kit.print_missing(df['age'])
print(cleaning_kit.lookup_columns('age', df))
cleaning_kit.suggest_deidentification(df)
```

### 2. Missing data reporting

Two functions, deliberately separate. `missing_report()` describes how much data
is missing and what shape the missingness has; it is cheap and safe to run on
anything. `missing_mar_tests()` tests whether missingness is *predicted* by the
observed data, which is an inferential step you should choose to take.

#### `missing_report()`

```python
import pandas as pd
from analytikit import cleaning_kit as ck

report = ck.missing_report(
    df,
    order=['baseline_NIHSS', 'T1_deficit', 'T2_deficit', 'T3_deficit', 'T4_deficit'],
    sort_by='baseline_NIHSS',
)
```

Numerics are printed first, then the figures are drawn. Sections are numbered as
they print, so switching one off does not leave a gap.

1. **Missing values per column** — `column | dtype | n_missing | %_missing | n_present | flag`,
   sorted by percentage descending, under a header giving total missing cells,
   complete cases, and how many columns are fully complete. `flag` escalates
   `notable` → `high` → `>50% - consider dropping`.
2. **Suspected hidden missing codes** — values that probably encode missingness
   but that pandas reads as data. This comes *before* the pattern analysis
   because, if these are really missing, every table below it is computed from
   the wrong mask. Reported only; nothing is coerced.
3. **Missing data patterns** — the equivalent of R's `mice::md.pattern()`. Each
   row is one unique missing/observed signature (● observed, ○ missing) with the
   number and percentage of cases that share it. This is what distinguishes
   dropout from sporadic item non-response.
4. **Follow-up attrition** (only when `order` is given) — per timepoint
   `n_present`, `n_missing`, `%_missing`, `newly_missing` and `returned`, then a
   count of cases that are complete, monotone (once missing, always missing) or
   intermittent (observed again after a gap), and a monotone / non-monotone
   verdict.
Finally, when `mar_tests=True`, the **MAR check** is delegated to
`missing_mar_tests()` — scoped to the columns that actually have missingness —
and prints under its own title after the numbered sections.

Then three figures: the **nullity matrix** (one row per case, dark = observed),
a **bar chart** of % missing, and a **nullity correlation heatmap** showing which
columns go missing together — a pair at 1.00 means one missed visit rather than
several independent lapses.

`missing_report()` parameters:

- `columns=None`: subset to analyse. Default: every column.
- `order=None`: ordered longitudinal columns, e.g. `['Bl', 'T1', 'T2', 'T3', 'T4']`.
  Enables the attrition analysis and is placed first among the analysed columns.
- `sort_by=None`: row ordering for the matrix plot — a column name, a list of
  names, or `'n_missing'`. Worth setting: unsorted, the matrix is noise; sorted
  by a baseline severity or a follow-up variable, staircase patterns become
  obvious.
- `max_patterns=15`: truncate the pattern table (the total number of distinct
  patterns is still reported).
- `hide_complete=True`: exclude columns with no missing values.
- `detect_sentinels=True`: scan for hidden missing codes.
- `mar_tests=False`: also run `missing_mar_tests()`. Off by default — it is
  inferential, it is quadratic in the number of columns, and it should be run
  only after any hidden codes found in section 2 have been converted to `NaN`.
  When it is off, the numeric output ends with a ready-to-paste call.
- `test_against=None`, `mar_force=False`: passed through to `missing_mar_tests()`.
- `plot=True`: draw the figures.

It returns a dict of the underlying dataframes — `summary`, `sentinels`,
`patterns`, `attrition`, `order_counts`, `mar_tests`, `mar_flagged` and
`nullity_correlation` — so the tables are usable programmatically and not only
as printed output. The `mar_*` keys are always present, and empty when
`mar_tests` is `False`.

Every piece is also callable on its own, which is useful when you want one table
or one figure rather than the whole report: `missing_summary(df, columns)`,
`find_sentinel_values(df, columns)`, `missing_patterns(df, columns)`,
`plot_missing_matrix(df, columns, sort_by=...)`, `plot_missing_correlation(df, columns)`,
and `plot_missing_bars(summary)` — which takes a `missing_summary()` table rather
than the dataframe.

#### Hidden missing codes

`find_sentinel_values()` is what section 2 calls, and it is useful on its own:

```python
ck.find_sentinel_values(df)
#   column suspected_value  count  %_of_rows
#      age              -1      3       2.50
#      sex            'NA'      5       4.17
#      ldl             999      8       6.67
```

Strings are matched case-insensitively after stripping (`''`, `NA`, `N/A`,
`unknown`, `missing`, `?`, `-`, `.`, and similar). Numeric codes are held to a
stricter test — the value must be repeated, extreme, and separated from the rest
of the distribution (a gap wider than the whole remaining range *and* at least
1.5 × IQR), or a negative code in an otherwise non-negative column. That guard is
the point: it catches `999` in an LDL column while leaving a real age of 99
alone. Nothing is coerced; if these values do encode missingness, replace them
with `NaN` and re-run the report.

#### `missing_mar_tests()`

```python
mar = ck.missing_mar_tests(df)                       # every column with missingness
mar = ck.missing_mar_tests(
    df,
    columns=['T4_deficit'],
    test_against=['baseline_NIHSS', 'centre', 'sex'],
)
```

For each column with missing values, the sample is split into cases **missing**
on that column and cases **observed** on it, and every other column is compared
between those two groups:

- **Numeric predictors** → Mann–Whitney, reported as median (Q1, Q3) per group
  with P(superiority) as the effect size (null 0.5, the same convention as
  `compare_ind()`). `numeric_test='t'` switches to Welch's t with Cohen's d. The
  distribution-free default is deliberate: these are screening comparisons over
  many variables of unknown shape.
- **Categorical predictors** → chi-square, falling back to Fisher's exact on a
  2×2 with any expected count below 5, with Cramér's V as the effect size. Both
  groups are described against the same reference level so the two cells are
  comparable.

The results table is one row per comparison —
`missing_in | n_missing | compared_on | type | group_missing | group_present | test | effect | effect_size | p | p_adj | significant`
— sorted by adjusted p-value:

```
missing_in  n_missing    compared_on        type       group_missing        group_present         test         effect  effect_size       p   p_adj  significant
T4_deficit         67 baseline_NIHSS     numeric 10.20 (8.75, 11.20) 13.60 (12.50, 14.70) Mann-Whitney P(superiority)        0.000 <0.0001 <0.0001         True
       ldl         34         centre categorical    Alex 34 (100.0%)      Alex 9 (10.47%)   Chi-square     Cramer's V        0.841 <0.0001 <0.0001         True
```

followed by a plain-language verdict:

```
Missingness IS predicted by observed variables in 2 column(s):
  T4_deficit <- baseline_NIHSS
  ldl <- centre
```

`missing_mar_tests()` parameters:

- `columns=None`: columns whose missingness is the outcome. Default: every column
  with at least one missing *and* one observed value.
- `test_against=None`: predictors to compare between the two groups. Default:
  every other column.
- `numeric_test='mannwhitney'`: or `'t'` for Welch's t with Cohen's d.
- `correction='holm'`: multiplicity correction applied across every test in the
  call. Also accepts `'fdr_bh'`, `'bonferroni'` and `None`.
- `alpha=0.05`: significance level.
- `min_group=5`: minimum observed values per group for a comparison to run.
- `max_levels=10`: categorical predictors with more levels are skipped.
- `max_tests=200`, `force=False`: the function refuses to run more than
  `max_tests` comparisons and tells you the count, because the test count is
  `n_columns_with_missing × n_predictors` and grows fast. Narrow `test_against`,
  raise the limit, or pass `force=True`.
- `verbose=True`: set to `False` to get the tables back without printing.

It returns a dict with `results` (the table above), `skipped` (comparisons that
could not run, with the reason: a group under `min_group`, a constant column, too
many categorical levels) and `flagged` — a machine-readable
`{column: [predictors]}` map for feeding an imputation model.

#### What these functions can and cannot tell you

A monotone pattern points to dropout rather than sporadic non-response, and a
significant MAR row means missingness depends on an observed variable: that
variable belongs in any imputation model, and complete-case analysis is likely
biased. But counts and patterns alone cannot establish MCAR vs MAR vs MNAR.

Two limits are printed on every run, and are worth repeating here. A null result
is **not** evidence of MCAR — it may simply be underpowered, and these are
unadjusted screening comparisons rather than confirmatory tests. And **MNAR
cannot be detected from observed data by any test**: if a deficit went unrecorded
precisely because it had resolved, the value that would explain the missingness
is the one you do not have. These functions can rule MAR *in*; they can never
rule MNAR out.

### 3. Independent-group comparison

```python
import pandas as pd
from analytikit import stat_kit

df = pd.DataFrame({
    'group': ['A'] * 5 + ['B'] * 5,
    'score': [1, 2, 3, 4, 5, 2, 3, 2, 4, 5]
})

group_a = df[df['group'] == 'A']['score']
group_b = df[df['group'] == 'B']['score']

stat_kit.compare_ind([group_a, group_b], group_labels=['A', 'B'])
```

`compare_ind()` parameters:

- `groups`: list of pre-split pandas Series, where each Series is one independent group to compare. Missing values are dropped within each group before testing.
- `group_labels=None`: optional list of display names matching the order of `groups`. If omitted, labels are generated from each Series name.
- `alpha=0.05`: significance threshold used for normality checks, the main hypothesis test, and any post-hoc testing.
- `categorical_limit=20`: unique-value cutoff used only when `data_type` is not supplied. Values above this threshold are treated as continuous; lower counts are treated as categorical.
- `force_test=None`: **accepted but currently ignored.** The parameter is reserved in the signature, but the present implementation always selects the test automatically. Passing `force_test='mannwhitney'` produces output identical to omitting it, with no warning — so do not rely on it to pin a test. Use `force_normality`, `force_non_normality`, or `data_type` to steer the choice instead. (`correlate()` and `correlate_biserial()` do honour their own `force_test` arguments.)
- `force_normality=False`: skips Shapiro-Wilk checking and forces the continuous-data path to use parametric tests.
- `force_non_normality=False`: skips Shapiro-Wilk checking and forces the continuous-data path to use non-parametric tests.
- `data_type=None`: set to `"cont"` to force continuous handling, or `"cat"`, `"ordinal"`, or `"nominal"` to force categorical handling. If omitted, the function infers the type from `categorical_limit` and prints a warning.
- `do_graphs=True`: enables the summary plots produced after the test.
- `graphs_for_non_significance=False`: if `False`, plots are skipped for non-significant results; if `True`, plots are shown even when `p >= alpha`.
- `subject_palette=None`: optional colour sequence for the comparison plots. See `make_subject_palette()`.
- `mann_whitney_location_shift_ci=False`: when a Mann-Whitney U test is selected, additionally report the Hodges-Lehmann location-shift estimate and its confidence interval. Off by default, because a location shift is only interpretable when a common-shape model is defensible; the primary effect reported is the probability of superiority alongside Cliff's delta.

For continuous data, `compare_ind()` auto-selects between an independent t-test, one-way ANOVA, Mann-Whitney U, or Kruskal-Wallis depending on group count and normality. For categorical data, it uses a contingency-table approach, choosing between chi-squared and Fisher's exact test. Post-hoc testing is added automatically when more than two groups are compared: Tukey HSD after a one-way ANOVA, Dunn's test after a Kruskal-Wallis.

`compare_ind()` prints its report and returns `None` — the results are the printed output, not a return value.

### 4. Dependent-group comparison

```python
import pandas as pd
from analytikit import stat_kit

baseline = pd.Series([10, 12, 9, 11, 10], name='baseline')
followup = pd.Series([12, 13, 11, 14, 12], name='followup')

stat_kit.compare_dep([baseline, followup], group_labels=['Baseline', 'Follow-up'])
```

`compare_dep()` parameters:

- `groups`: list of matched pandas Series representing repeated measurements on the same subjects. The function concatenates them and drops any row with a missing value in any time point, so only complete cases are analyzed.
- `group_labels=None`: optional labels for each repeated-measures condition, in the same order as `groups`.
- `alpha=0.05`: significance threshold used throughout normality testing, main testing, and post-hoc comparisons.
- `categorical_limit=20`: fallback unique-value threshold for automatic type inference when `data_type` is not provided.
- `force_test=None`: **accepted but currently ignored**, exactly as in `compare_ind()`. The test is always chosen automatically from data type, normality, and the number of repeated conditions; passing a value changes nothing and prints no warning. Use `force_normality`, `force_non_normality`, or `data_type` instead.
- `force_normality=False`: forces parametric continuous tests and bypasses automatic normality-based switching.
- `force_non_normality=False`: forces non-parametric continuous tests and bypasses automatic normality-based switching.
- `data_type=None`: use `"cont"` for continuous repeated measures, or `"cat"`, `"ordinal"`, or `"nominal"` for categorical repeated measures. If omitted, the function infers the type and warns.
- `do_graphs=True`: enables the generated plots for the selected analysis.
- `graphs_for_non_significance=False`: controls whether plots are still generated when the result is not statistically significant.
- `difference_decimals=None`: for the non-parametric paired path, round the paired differences to this many decimals before ranking. Set it to your actual measurement precision when floating-point subtraction would otherwise turn exact ties into tiny non-zero differences and distort the ranks.

For continuous repeated measures, `compare_dep()` auto-selects between a paired t-test, repeated-measures ANOVA, Wilcoxon signed-rank test, and Friedman test. For categorical repeated measures, it branches to McNemar's test, Cochran's Q, the Stuart-Maxwell test, or Friedman, depending on the number of categories and time points.

Like `compare_ind()`, `compare_dep()` prints its report and returns `None`.

### 5. Correlation

```python
import pandas as pd
from analytikit import stat_kit

df = pd.DataFrame({
    'age': [21, 22, 23, 24, 25],
    'score': [60, 65, 68, 74, 80]
})

result = stat_kit.correlate(df, 'age', 'score')
print(result)
```

`correlate()` parameters:

- `data`: pandas DataFrame containing both variables.
- `x`: name of the first column to correlate.
- `y`: name of the second column to correlate.
- `force_test=None`: set to `"pearson"` to force Pearson correlation or `"spearman"` to force Spearman correlation. If omitted, the function runs Shapiro-Wilk on both variables when possible and chooses Pearson for apparently normal data and Spearman otherwise.
- `alpha=0.05`: significance threshold used both for the normality decision and for labeling the final correlation as significant or not significant.

`correlate()` drops rows with missing values in either column before testing. It returns `None` instead of a result if fewer than 3 complete observations remain or if either variable has zero variance.

### 6. Biserial correlation (continuous vs. two-level variable)

```python
import pandas as pd
from analytikit import stat_kit

df = pd.DataFrame({
    't0_berg': [45, 48, 50, 52, 40, 38, 30, 44, 55, 41,
                49, 53, 47, 39, 36, 51, 42, 46, 33, 54],
    'improved': [1, 1, 1, 1, 0, 0, 0, 1, 1, 0,
                 1, 1, 1, 0, 0, 1, 0, 1, 0, 1],
})

result = stat_kit.correlate_biserial(df, 't0_berg', 'improved', alpha=0.05)
print(result)
```

`stat_kit.correlate_biserial()` runs Shapiro-Wilk on the continuous variable within
each of the two groups (the same normality check used by `compare_ind()`), prints the
verdict, and then delegates to one of the two functions below:

- Both groups look normal → `correlate_p_biserial()` (point-biserial correlation).
- Either group does not → `correlate_rank_biserial()` (rank-biserial correlation).

Parameters common to all three:

- `data`: pandas DataFrame containing both variables.
- `continuous_var`: name of the continuous numeric column.
- `binary_var`: name of the column with exactly two observed levels.
- `alpha=0.05`: significance threshold, used for the normality screen (on `correlate_biserial`) and for the correlation's own confidence interval and significance label.
- `plot=True`: draws a box plot with an overlaid strip plot, one box per level of `binary_var`.
- `minimum_group_n=2`: smallest group size the correlation will analyze; smaller groups are skipped rather than analyzed, since the effect size and its interval are not reliably estimable below this. Pass `1` to analyze anyway.

Additional `correlate_biserial()` parameters:

- `force_test=None`: set to `"point-biserial"` or `"rank-biserial"` to skip the normality screen and use that test directly.
- `force_normality=False` / `force_non_normality=False`: skip the screen and force the parametric or rank-based branch, matching the same-named parameters on `compare_ind()`.
- `data_type=None`: set to `"ordinal"` to route straight to the rank-biserial correlation without screening.
- `method=None`: forwarded to `correlate_rank_biserial()`'s underlying Mann-Whitney test (`"exact"` or `"asymptotic"`); ignored when the parametric branch is selected.

Additional `correlate_rank_biserial()` parameter:

- `method=None`: `"exact"` or `"asymptotic"` Mann-Whitney method. Defaults to the package's resolved policy (see `compare_ind()`).

The three functions share the same input validation and level-coding: the two observed
levels of `binary_var` are sorted, coded `0` and `1`, and that coding — not row order —
determines the sign of the reported coefficient. All three drop rows with missing values
in either column before analysis and return `None`, with a yellow explanation printed,
instead of raising when the pair cannot support a biserial correlation: too few complete
observations, `binary_var` collapsed to fewer than two observed levels after dropping
missing values, a group smaller than `minimum_group_n`, or a constant or non-numeric
continuous variable. This makes it safe to call them in a loop over many variable pairs.

`correlate_p_biserial()` returns the point-biserial correlation coefficient with a
Fisher-z confidence interval; its p-value is algebraically identical to the
equal-variance two-sample t-test, so it assumes roughly normal, equal-variance groups.
`correlate_rank_biserial()` returns Cliff's delta (equivalently, `2 * P(group1 > group0)
- 1`, with ties counted as half a win) with a test-compatible confidence interval, plus
the equivalent probability-of-superiority effect size; it makes no normality assumption
and is unaffected by monotone transformations of the continuous variable.
`correlate_biserial()` returns whichever delegate's result dictionary it selected, with
two entries added: `selected_method` and `normality` (the per-group Shapiro-Wilk
statistics, p-values, and the resulting verdict).

### 7. Regression

`regress()` follows the same automation idea as `compare_ind()` and `compare_dep()`: it inspects the data and selects the model for you. The difference is *what* it inspects. A comparison test is chosen partly from an assumption check (normality), but a regression model is chosen from the **outcome variable**, which can be settled before anything is fitted.

```python
import numpy as np
import pandas as pd
from analytikit import regress

rng = np.random.default_rng(0)
n = 150

df = pd.DataFrame({
    'age': rng.normal(55, 12, n).round(),
    'sex': rng.choice(['female', 'male'], n),
    'smoking': rng.choice(['never', 'former', 'current'], n),
})
df['systolic_bp'] = (
    95
    + 0.6 * df['age']
    + 5 * (df['sex'] == 'male')
    + 8 * (df['smoking'] == 'current')
    + rng.normal(0, 8, n)
).round()

result = regress(
    df,
    outcome='systolic_bp',
    predictors=['age', 'sex', 'smoking'],
    outcome_type='cont',
    reference={'smoking': 'never'},
)
```

For a binary outcome the model switches to logistic and effects are reported as odds ratios:

```python
risk = -6 + 0.07 * df['age'] + 0.9 * (df['sex'] == 'male')
df['died'] = np.where(rng.binomial(1, 1 / (1 + np.exp(-risk))) == 1, 'yes', 'no')

result = regress(
    df,
    outcome='died',
    predictors=['age', 'sex'],
    outcome_type='cat',
    outcome_positive='yes',   # model P(died = 'yes')
    univariable=True,         # also print the crude-vs-adjusted table
)
```

Use `formula` for interactions, transformations, or polynomial terms:

```python
result = regress(df, formula='systolic_bp ~ age * sex', outcome_type='cont')
```

Note that a model needs enough complete rows to be estimable: `regress()` returns `None` with a
warning if fewer than 10 remain after dropping missing values.

#### How the model is chosen

| Outcome | Model fitted | Effects reported as |
|---|---|---|
| Continuous (`outcome_type='cont'`) | Ordinary least squares | Unstandardised coefficients, null value 0 |
| Binary (`outcome_type='cat'`) | Logistic | Odds ratios, null value 1 |

If `outcome_type` is not supplied, the type is inferred from the outcome and a yellow warning is printed, exactly as in `compare_ind()`. A numeric outcome with more distinct values than `categorical_limit` is treated as continuous; an outcome with exactly two levels is treated as binary. Set `outcome_type` explicitly to suppress the warning.

#### `regress()` parameters

- `data`: a pandas DataFrame containing the outcome and all predictors.
- `outcome=None`: name of the outcome (dependent) column. Required unless `formula` is given.
- `predictors=None`: list of predictor (independent) column names. A single string is accepted for a one-predictor model. Required unless `formula` is given.
- `formula=None`: a patsy formula such as `'bp ~ age * sex + np.log(bmi)'`, for interactions, transformations, and polynomial terms. Overrides `predictors`. The left-hand side must be a plain column name.
- `alpha=0.05`: significance threshold. Confidence intervals are reported at 100(1 − alpha)%, and every diagnostic test is judged against this same value.
- `outcome_type=None`: `'cont'` forces a linear model; `'cat'` (also `'nominal'`, `'binary'`, `'ordinal'`) forces a binary logistic model. If omitted, the type is inferred and a warning is printed.
- `force_model=None`: `'linear'` or `'logistic'`. Skips outcome inspection entirely, in the spirit of `force_test` elsewhere in the package.
- `reference=None`: dict giving the baseline level per categorical predictor, e.g. `{'smoking': 'never'}`. Defaults to the first level in sort order, which is always printed.
- `outcome_positive=None`: which outcome level is modelled as the event in a logistic model. Defaults to the higher of the two levels, which is always printed.
- `univariable=False`: if `True`, also fits one model per predictor and prints a crude-vs-adjusted table. Descriptive only — see the note on predictor selection below.
- `do_graphs=True`: enables the diagnostic plots.
- `categorical_limit=20`: unique-value cutoff used only when `outcome_type` is not supplied.

#### Return value

`regress()` prints a narrated report and returns a dict:

- `model`: `'linear'` or `'logistic'`.
- `formula`: the patsy formula actually fitted.
- `n`, `n_dropped`: observations used, and rows dropped for missing values.
- `coefficients`: a tidy DataFrame with one row per model term — `term` (a readable label such as `smoking [current vs never]`), `design_term`, `coefficient`, `std_error`, `statistic`, `p_value`, `ci_lower`, `ci_upper`, plus `odds_ratio`, `or_ci_lower`, `or_ci_upper` for logistic models. This is the table to paste into a manuscript.
- `univariable`: the crude-effects DataFrame when `univariable=True`, otherwise `None`.
- `fit`: overall model fit (R², adjusted R² and the F-test p-value for linear; McFadden pseudo R² and the likelihood-ratio p-value for logistic; AIC and BIC for both).
- `diagnostics`: every check described below, including the VIF table.
- `robust_standard_errors`: whether HC3 errors were substituted.
- `reference_levels`, `outcome_positive`, `outcome_baseline`: the coding decisions made, so they can be reported in a methods section.
- `fitted_model`: the underlying `statsmodels` results object, for any follow-up work.

```python
result['coefficients'].to_csv('table2.csv', index=False)
result['fitted_model'].summary()          # the full statsmodels output
result['diagnostics']['multicollinearity']  # the VIF table
```

#### Reading the diagnostics

Regression assumptions can only be checked *after* a model is fitted, and there is no single agreed-upon fallback when one fails — the right response usually depends on why it failed. So `regress()` reports what it found and leaves the decision to you, rather than silently substituting a different model.

Reported for both model types:

- **Multicollinearity (VIF)**, per design-matrix column. High values between separate predictors mean they carry overlapping information and their individual coefficients are unstable. High values among the dummies of one categorical predictor, or between an interaction term and its components, are expected and are not a problem.
- **Sample size**: observations per model term for linear models; events per variable (EPV) for logistic. An EPV below 10 is flagged, as reviewers of medical work routinely ask for it.

Linear models:

- **Residual normality** (Shapiro-Wilk on the residuals; skipped when n > 5000). Failing this does not bias the coefficients — it affects small-sample p-values and intervals.
- **Homoscedasticity** (Breusch-Pagan). This is the one check that triggers an automatic response: the model is refitted with HC3 robust standard errors. The coefficients and the model are unchanged; only the standard errors, p-values and confidence intervals are corrected. `result['robust_standard_errors']` records whether this happened.
- **Linearity** (Ramsey RESET). A failure suggests a missing non-linear term — add a quadratic or spline term through `formula`.
- **Influential observations** (Cook's distance, flagged above 4/n). Check these for data-entry errors, but do not delete a point merely because it is influential; a genuine extreme value is information.
- **Independence** (Durbin-Watson). Only meaningful when the rows have a real order, such as a time series or repeated measures.

Logistic models:

- **Separation**, checked *before* fitting. If a predictor or one of its levels perfectly predicts the outcome, the affected coefficient has no finite estimate, and its odds ratio and interval are meaningless however large they look. The offending variable and level are named.
- **Discrimination** (c-statistic / AUC). Note this is *apparent* discrimination, measured on the same data the model was fitted to, so it is optimistic; honest estimates need cross-validation or an external sample.
- **Calibration** (Hosmer-Lemeshow). Unusually, a **non-significant** result is the reassuring one here.

With `do_graphs=True`, linear models also plot residuals-vs-fitted, a normal Q-Q of the residuals, and Cook's distance; logistic models plot the ROC curve and a calibration curve by risk group.

#### Categorical predictors and reference levels

Categorical predictors are dummy-coded automatically, and the baseline level is always printed:

```
Categorical predictors and their reference (baseline) levels:
  smoking: reference = "never"; compared against current, former
```

Every effect for that predictor is relative to its reference, and the coefficient table labels it explicitly (`smoking [current vs never]`). Set `reference={'smoking': 'never'}` to choose the baseline deliberately rather than accepting sort order. For logistic models the outcome level being modelled as the event is reported the same way, since the direction of every odds ratio depends on it.

You do **not** need to build dummy variables yourself, and you should not: hand-made dummies lose the reference level from the printed output and from `result['reference_levels']`.

##### Which predictors count as categorical

Detection is by **dtype**, not by how few distinct values a column happens to have:

| Column dtype | Treated as | Enters the model as |
|---|---|---|
| `object` (strings) | Categorical | One dummy term per non-reference level |
| `category` with string levels | Categorical | One dummy term per non-reference level |
| Any numeric dtype | Continuous | A single linear term |

A numeric column is treated as continuous **however few values it takes**. This is deliberate — silently dummy-coding a numeric dose, lab value, or score would be the worse mistake — but it means a category you encoded as numbers is not detected:

```python
df['stage'] = [1, 2, 3, ...]   # I, II, III encoded as integers
regress(df, outcome='survival_days', predictors=['stage'], outcome_type='cont')
# -> a single 'stage' coefficient, NOT two dummy terms
```

That fit assumes the step from stage II to III is exactly the same size as the step from I to II, and it reports one coefficient where you expected two. Nothing warns you about it. A `0`/`1` coded binary variable is the harmless case — a binary numeric term is algebraically identical to a dummy — but anything with three or more coded levels needs to be declared:

```python
df['stage'] = df['stage'].astype(str)
regress(df, outcome='survival_days', predictors=['stage'], outcome_type='cont')

# or leave the column alone and declare it in the formula:
regress(df, formula='survival_days ~ C(stage)', outcome_type='cont')
```

Use `.astype(str)` rather than `.astype('category')` here — see the known issue below.

Note that `categorical_limit` does **not** affect this. It applies only to inferring the type of the *outcome* when `outcome_type` is not set; it plays no part in classifying predictors.

If a numeric coding really is ordinal and you want to model it as a single linear trend, leaving it numeric is a legitimate choice — just make it a deliberate one, and say so in your methods, since it constrains the effect to be equally spaced across levels.

##### Known issue: non-string categorical levels

Two predictor types are currently rejected with a `PatsyError` reading `specified level '...' not found`, because the reference level is quoted as a string when the formula is built while the underlying levels are not strings:

- a `category` dtype column whose levels are numbers (`df['stage'].astype('category')` where stage holds `1`, `2`, `3`)
- any `bool` dtype column (`df['flag'] = df['age'] > 50`)

Until this is fixed, convert those columns to strings first, which works in both cases:

```python
df['stage'] = df['stage'].astype(str)
df['flag'] = df['flag'].astype(str)     # 'True' / 'False'
```

A `category` dtype whose levels are already strings is unaffected and works normally.

#### Missing data

`regress()` performs **complete-case (listwise) analysis**: any row with a missing value in any model variable is dropped, and the count is reported. This assumes the values are missing at random; if they are not, the estimates are biased and no diagnostic will reveal it.

There is no imputation, and this is deliberate. Mean or median imputation — the easiest thing to automate — shrinks the variable's variance and makes the standard errors too small, producing narrower confidence intervals from *less* information. Multiple imputation (MICE) is the defensible alternative, but it is not a preprocessing step: it fits several models and pools them with Rubin's rules, which changes what the reported coefficient and interval mean. If you need it, handle it explicitly outside `regress()`.

Beyond dropping incomplete rows and zero-variance predictors, `regress()` performs no cleaning: no scaling, no outlier removal, no transformation of skewed variables, and no collapsing of sparse categories. Prepare the data first — `cleaning_kit.missing_report()` and `cleaning_kit.missing_mar_tests()` are there to help you decide what happens to the missing values before you get here.

##### What to prepare before calling `regress()`

| Step | Handled by `regress()`? |
|---|---|
| Dummy-coding categorical predictors | **Yes** — automatic, with the reference level reported |
| Coding a binary outcome as 0/1 | **Yes** — automatic, with the event level reported |
| Declaring numeric-coded categories | **No** — cast to `str`/`category`, or wrap in `C()` (see above) |
| Imputing missing values | **No** — incomplete rows are dropped listwise |
| Scaling, outlier handling, transformations, collapsing sparse categories | **No** |

In short: you do not need to encode, but you do need to decide what happens to missing values, and to make sure a category encoded as numbers is not mistaken for a measurement.

#### Predictor selection

`regress()` fits exactly the predictors you name. It offers no stepwise, forward, or backward selection.

This is a deliberate omission. Choosing predictors by their p-values biases the surviving coefficients away from the null and understates their standard errors, so the confidence intervals and p-values the final model reports are not valid as printed. Setting `univariable=True` prints crude effects beside adjusted ones for the usual medical-paper table, but that table is descriptive — it is not a way to decide what enters the model.

#### Not yet implemented

Linear and binary logistic models are available. Poisson, negative binomial, ordinal, multinomial and Cox models are recognised by `force_model` and raise a clear "not implemented yet" message rather than silently substituting something else. A multi-level outcome declared with `outcome_type='cat'` is refused for the same reason.

### 8. Confidence interval for group differences

```python
from analytikit import stat_kit

group1 = [1, 2, 3, 4, 5]
group2 = [2, 3, 4, 5, 6]

ci = stat_kit.compute_confidence_interval_difference(
    group1,
    group2,
    method='mean',
    paired=False,
)
print(ci)
```

`compute_confidence_interval_difference()` parameters:

- `group1`, `group2`: the two samples, as any array-like (lists, NumPy arrays, or pandas Series). For `paired=True` they must be the same length and aligned element by element.
- `method='mean'`: `'mean'` gives a t-based confidence interval for the difference in means. `'median'` gives a test-inverted Hodges-Lehmann location shift — the name `'median'` is kept for backward compatibility, but the estimate is a location shift, not a difference of medians.
- `paired=False`: set to `True` for dependent or repeated measurements on the same subjects.
- `alpha=0.05`: significance level; the interval is reported at 100(1 − alpha)%.
- `n_bootstrap=10000`: **retained for backward compatibility only and no longer used.** The non-parametric interval is obtained by test inversion rather than bootstrap resampling, so changing this value has no effect.
- `difference_decimals=None`: for the paired non-parametric path, round the paired differences to this many decimals before ranking. Use your measurement precision when floating-point subtraction would otherwise create artificial rank differences.

It returns a dict with:

- `difference`: the point estimate (mean difference, or the Hodges-Lehmann location shift).
- `CI`: the `(lower, upper)` interval.
- `CI_inclusive`: a `(bool, bool)` pair saying whether each endpoint is included. Discrete non-parametric intervals can have an excluded endpoint, which the comparison functions render as a parenthesis rather than a bracket.
- `test_stat`, `p_value`: the statistic and p-value of the corresponding test.
- `test_method`: which test produced the interval.
- `null_value_included`: whether zero falls inside the interval.

Note that the non-parametric paired path currently raises a `ValueError` from SciPy when every paired difference is identical (a degenerate input with no rank information), rather than returning a result or a handled message.

## Import behavior

The package root exports exactly these names:

- from `stat_kit`: `version`, `compare_ind`, `compare_dep`, `correlate`, `regress`, `compute_confidence_interval_difference`, `plot_group_comparison`, `plot_dependent_group_comparison`, `make_subject_palette`
- from `helpers`: `percent`, `plus_minus`, `print_title`, `mean_ci`, `median_ci`, `print_mean_std`, `print_median_iqr`
- the `cleaning_kit` module itself, as `analytikit.cleaning_kit`

Anything not on that list — including `correlate_biserial`, `correlate_p_biserial`, `correlate_rank_biserial`, and `mann_whitney_effect_ci` — must be reached through the module rather than the package root:

```python
from analytikit import regress    # exported at the root
from analytikit import stat_kit   # module access for everything else

# Reached through the module, not the package root:
#   stat_kit.correlate_biserial(...)
#   stat_kit.correlate_p_biserial(...)
#   stat_kit.correlate_rank_biserial(...)
#   stat_kit.mann_whitney_effect_ci(...)
```

Everything in `cleaning_kit` works the same way — the module is exported, the
functions inside it are not:

```python
from analytikit import cleaning_kit as ck

ck.missing_report(df)
ck.missing_mar_tests(df)
```

All of the following are valid:

```python
import analytikit
from analytikit import stat_kit
from analytikit import cleaning_kit
```

## Version note

The package metadata and module version helpers are aligned at `1.1.2`:

- `stat_kit.version()` returns `1.1.2`
- `cleaning_kit.version()` returns `1.1.2`
- `pyproject.toml` declares the package version as `1.1.2`

## Smoke test

After installation, run a quick import check:

```bash
python smoke_test.py
```

## Contributing

If you add or change functionality:

- update this README
- keep examples aligned with the public API
- add tests where practical

## Authors

Ahmed Elbokl (ahmed.elbokl@med.asu.edu.eg), Maha Sabry and contributors.

## License

See `pyproject.toml` for package metadata including license information.
