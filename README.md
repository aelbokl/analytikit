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
- `compute_confidence_interval_difference` for mean or median difference confidence intervals.
- plotting helpers used during comparisons.

`stat_kit.version()` currently returns `1.1.2`.

### `cleaning_kit`

Data cleaning and reporting helpers, including:

- `drop_columns` to safely drop columns while reporting missing column names.
- `lookup_columns` to find columns by partial match.
- `compare_columns` to compare two column lists.
- `print_missing` to report missing counts and percentages.
- `suggest_deidentification` to flag columns that may contain identifiers.

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

### 2. Independent-group comparison

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
- `force_test=None`: reserved override parameter in the current signature. The present implementation still selects the test automatically from the data rather than switching on this argument.
- `force_normality=False`: skips Shapiro-Wilk checking and forces the continuous-data path to use parametric tests.
- `force_non_normality=False`: skips Shapiro-Wilk checking and forces the continuous-data path to use non-parametric tests.
- `data_type=None`: set to `"cont"` to force continuous handling, or `"cat"`, `"ordinal"`, or `"nominal"` to force categorical handling. If omitted, the function infers the type from `categorical_limit` and prints a warning.
- `do_graphs=True`: enables the summary plots produced after the test.
- `graphs_for_non_significance=False`: if `False`, plots are skipped for non-significant results; if `True`, plots are shown even when `p >= alpha`.

For continuous data, `compare_ind()` auto-selects between an independent t-test, one-way ANOVA, Mann-Whitney U, or Kruskal-Wallis depending on group count and normality. For categorical data, it uses a contingency-table approach such as chi-squared or Fisher's exact test.

### 3. Dependent-group comparison

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
- `force_test=None`: reserved in the public signature, but the current implementation still chooses the statistical test automatically from data type, normality, and number of repeated conditions.
- `force_normality=False`: forces parametric continuous tests and bypasses automatic normality-based switching.
- `force_non_normality=False`: forces non-parametric continuous tests and bypasses automatic normality-based switching.
- `data_type=None`: use `"cont"` for continuous repeated measures, or `"cat"`, `"ordinal"`, or `"nominal"` for categorical repeated measures. If omitted, the function infers the type and warns.
- `do_graphs=True`: enables the generated plots for the selected analysis.
- `graphs_for_non_significance=False`: controls whether plots are still generated when the result is not statistically significant.

For continuous repeated measures, `compare_dep()` auto-selects between a paired t-test, repeated-measures ANOVA, Wilcoxon signed-rank test, and Friedman test. For categorical repeated measures, it branches to tests such as McNemar's, Cochran's Q, Stuart-Maxwell, or Friedman depending on the number of categories and time points.

### 4. Correlation

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

### 5. Confidence interval for group differences

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

## Import behavior

The package root currently exports:

- selected `stat_kit` functions such as `version`, `compare_ind`, `compare_dep`, `correlate`, and `compute_confidence_interval_difference`
- selected `helpers` functions such as `percent`, `mean_ci`, and descriptive print helpers
- the `cleaning_kit` module as `analytikit.cleaning_kit`

This means all of the following are valid:

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
