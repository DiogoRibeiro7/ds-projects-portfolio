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

## Main findings

At the current state of the notebook:

- the reproducible grouped public-data window is `2007-2024`;
- the grouped extraction checks reconcile exactly with the sampled source cells and published totals;
- `Lognormal` is the consistent full grouped-data winner in the current AIC/BIC comparison;
- `Gamma` remains useful as a benchmark for the body of the distribution, but not as the best-supported full-distribution claim;
- the main stress points for smooth models are the minimum-wage mass and the upper tail.

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
   - sensitivity analysis for minimum-wage and top-bracket handling;
   - grouped chi-square residuals;
   - observed-versus-fitted bracket shares;
   - observed-versus-fitted decile means;
   - bootstrap parameter ranges for selected years;
   - Gamma parameter trends;
   - model-winner tracking over time;
   - a Pareto tail diagnostic;
   - tail-only Lognormal-versus-Pareto comparison;
   - real-wage context and wage-compression context.

The grouped likelihood excludes the exact-minimum-wage equality bin from the continuous fit because a point mass is not represented cleanly by the smooth families used here.

## Reproducibility

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run the full analysis pipeline from the project root:

```bash
python scripts/run_full_analysis.py
```

This command does three things in order:

1. syncs `notebooks/01_salary_gamma_distribution_portugal.py` and `notebooks/01_salary_gamma_distribution_portugal.ipynb`;
2. executes the notebook headlessly with `nbconvert`;
3. regenerates the compact markdown and JSON summary files.

To inspect the planned steps without running them:

```bash
python scripts/run_full_analysis.py --dry-run
```

If you only need the summary refresh after a successful run:

```bash
python scripts/summarize_notebook_outputs.py
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

The summary script writes:

```text
reports/notebook_summary.md
reports/notebook_summary.json
```

Key outputs include:

- `download_status.csv`
- `bracket_metadata_by_year.csv`
- `main_findings_first.csv`
- `minimum_wage_context.csv`
- `wage_compression_metrics.csv`
- `salary_brackets_raw_long.csv`
- `salary_summaries_raw_long.csv`
- `salary_bins_long.csv`
- `salary_bin_percentage_checks.csv`
- `manual_validation_checks.csv`
- `harmonized_broad_band_shares.csv`
- `harmonized_share_change_first_last.csv`
- `totals_comparison.csv`
- `distribution_fit_results.csv`
- `model_winners_by_year.csv`
- `sensitivity_fit_results.csv`
- `sensitivity_summary.csv`
- `bic_gap_gamma_vs_lognormal.csv`
- `bootstrap_parameter_ranges.csv`
- `grouped_residuals_by_bin.csv`
- `decile_fit_validation.csv`
- `decile_fit_validation_summary.csv`
- `gamma_parameter_trend.csv`
- `pareto_tail_diagnostics.csv`
- `tail_model_comparison.csv`
- `tail_model_winners.csv`
- `top_bracket_shares_over_time.csv`
- `top_share_fit_comparison.csv`
- `optional_microdata_schema.csv`

## Figure index

The main notebook figures currently include:

- `context_totals_and_median.png`: worker totals and published median gain over time.
- `context_decile_means.png`: published mean earnings by selected deciles.
- `real_wage_context.png`: nominal versus real wage context.
- `minimum_wage_regime.png`: exact-minimum-wage mass, below-minimum-wage mass, and RMMG level.
- `wage_compression_metrics.png`: decile-ratio measures of compression and spread.
- `distribution_change_harmonized_bands.png`: broad-band share shifts between the first and last grouped years.
- `aic_bic_model_comparison.png`: full model competition by AIC and BIC.
- `bic_gap_gamma_vs_lognormal.png`: direct BIC gap between Gamma and Lognormal.
- `sensitivity_gamma_vs_lognormal_bic_gap.png`: robustness of the Gamma-versus-Lognormal gap under alternative grouped-fit rules.
- `gamma_parameter_trends.png`: Gamma shape, scale, and implied mean over time.
- `gamma_bootstrap_ranges.png`: grouped-bootstrap parameter ranges for selected years.
- `grouped_histogram_with_fits_*.png`: grouped histogram views with fitted density overlays.
- `grouped_residuals_*.png`: grouped Pearson residual heatmaps for representative years.
- `decile_validation_*.png`: observed versus fitted mean earnings by decile.
- `decile_error_by_model_over_time.png`: model comparison on decile-fit error.
- `pareto_tail_alpha.png`: descriptive Pareto tail diagnostic over time.
- `tail_model_comparison.png`: tail-only Lognormal versus Pareto comparison.
- `top_bracket_shares_over_time.png`: observed upper-tail bracket concentration over time.
- `observed_vs_fitted_top_shares.png`: observed versus model-implied upper-tail shares.

## Interpretation

The notebook is designed to support a careful conclusion:

- Gamma can be a useful approximation to the body of the earnings distribution.
- That claim should be tested against alternatives rather than assumed.
- Minimum-wage concentration and the upper tail are the most likely reasons for systematic misspecification.

## Glossary

- `Grouped likelihood`: fitting a distribution to interval counts instead of individual observations.
- `Open-top bin`: the highest published salary bracket, with no finite upper bound.
- `Decile mean`: the published average wage inside one tenth of the distribution.
- `RMMG`: the Portuguese monthly minimum wage used in the public tables.
- `Body of the distribution`: the broad middle mass of wages, excluding the most extreme upper tail.

## Optional microdata path

If a local anonymized CSV with a `monthly_earnings` column is placed under `data/private/`, the package can run an optional microdata comparison branch. This is not required for the public notebook and is intentionally excluded from the core reproducible pipeline.

The expected optional schema is documented in:

```text
data/private/MICRODATA_SCHEMA.md
```
