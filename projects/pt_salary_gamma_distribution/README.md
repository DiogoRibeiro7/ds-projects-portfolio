# Portugal salary distribution with grouped public tables

This project studies the distribution of monthly earnings in Portugal using the public **GEP/MTSSS Quadros de Pessoal** chronological-series Excel workbooks.

The repository does **not** assume that Portuguese salaries follow a Gamma distribution. It tests that claim against alternative positive-support models using the public grouped tables that the workbooks actually publish.

## Main notebook

Run:

```text
notebooks/01_salary_gamma_distribution_portugal.ipynb
```

The paired source file is:

```text
notebooks/01_salary_gamma_distribution_portugal.py
```

## Data coverage

The public grouped-distribution analysis in this repository is based on the salary-bracket and decile-summary tables that are reproducibly available in the official workbooks. In practice, that grouped analysis window is:

```text
2007-2024
```

The older 1999-2009 workbook still provides remuneration context, but it does not contain the same public grouped salary-bracket and decile tables needed for grouped likelihood fitting.

## Methodology

The notebook does four things:

1. downloads the official Excel files listed in `data/source_manifest.csv`;
2. extracts:
   - salary-bracket counts;
   - salary-bracket percentages;
   - total worker counts;
   - median monthly gain;
   - decile cutpoints;
   - mean monthly gain by decile;
3. fits grouped distributions year by year:
   - Gamma;
   - Lognormal;
   - Weibull;
   - Generalized Gamma;
4. validates fitted models with:
   - AIC and BIC;
   - grouped chi-square residuals;
   - observed-versus-fitted bracket shares;
   - observed-versus-fitted decile means;
   - Gamma parameter trends;
   - model-winner tracking over time;
   - an optional Pareto tail diagnostic.

The grouped likelihood excludes the exact-minimum-wage equality bin from the continuous fit because a point mass is not represented cleanly by the smooth families used here.

## Reproducibility

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then execute the notebook from the project root:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_salary_gamma_distribution_portugal.ipynb
```

## Outputs

The notebook writes processed tables to:

```text
data/processed/
```

and figures to:

```text
reports/figures/
```

Key outputs include:

- `download_status.csv`
- `salary_brackets_raw_long.csv`
- `salary_summaries_raw_long.csv`
- `salary_bins_long.csv`
- `salary_bin_percentage_checks.csv`
- `manual_validation_checks.csv`
- `totals_comparison.csv`
- `distribution_fit_results.csv`
- `model_winners_by_year.csv`
- `grouped_residuals_by_bin.csv`
- `decile_fit_validation.csv`
- `decile_fit_validation_summary.csv`
- `gamma_parameter_trend.csv`
- `pareto_tail_diagnostics.csv`

## Interpretation

The notebook is designed to support a careful conclusion:

- Gamma can be a useful approximation to the body of the earnings distribution.
- That claim should be tested against alternatives rather than assumed.
- Minimum-wage concentration and the upper tail are the most likely reasons for systematic misspecification.

## Optional microdata path

If a local anonymized CSV with a `monthly_earnings` column is placed under `data/private/`, the package can run an optional microdata comparison branch. This is not required for the public notebook and is intentionally excluded from the core reproducible pipeline.
