# Portugal Housing, Tourism and Short-Term Accommodation

A reproducible empirical case study of housing affordability, tourism intensity and short-term accommodation in Portugal, with Lisbon as the main longitudinal case study and a municipality-level comparison across Portugal.

The project is deliberately conservative about identification. It does **not** interpret tourism, Airbnb or Alojamento Local as proven causes of rent growth. Its v1 contribution is measurement, descriptive decomposition, national comparison and a short current-vintage panel association model.

## v1 empirical result

The final municipality panel uses the common current NUTS-2024 window 2022–2024. The primary two-way fixed-effects model is

```text
log(rent_it / income_it)
    = municipality FE
    + year FE
    + beta * log(tourism_intensity_it)
    + error_it
```

with municipality-clustered standard errors.

The primary unbalanced sample contains 614 municipality-year observations from 208 municipalities with at least two complete years. The estimated tourism-intensity coefficient on the affordability proxy is

```text
beta = 0.03562
95% CI = [-0.01643, 0.08767]
p = 0.180
```

The balanced sensitivity contains 594 observations from 198 municipalities and gives

```text
beta = 0.03927
95% CI = [-0.02577, 0.10430]
p = 0.237
```

The direction and magnitude are similar across the two samples, but both intervals include zero. The evidence therefore supports a **modest positive estimated within-municipality association that remains statistically uncertain**, not a claim that tourism caused higher housing costs.

Using the exact same rows, the primary rent coefficient is 0.03443 while the income coefficient is -0.00119. The affordability coefficient therefore comes almost entirely from the rent side. The exact identity

```text
beta_affordability = beta_rent - beta_income
```

holds numerically to floating-point precision.

The frozen result is stored in `results/processed/municipality_twfe_results.json`, with workflow provenance in `results/processed/municipality_twfe_provenance.json`.

## Core quantities

### Local Housing Decoupling Index

```text
LHDI_it = 100 * (rent_it / income_it) / (rent_i0 / income_i0)
```

Values above 100 mean the rent-price-to-income proxy has increased relative to that municipality's baseline. The numerator is median EUR/m² for new leases and the denominator is median gross declared income per taxpayer, so LHDI is a normalised affordability proxy rather than a literal household rent share.

### Tourism intensity

```text
TI_it = overnight_stays_it / resident_population_it
```

### Tourist Housing Conversion Rate

```text
THCR_it = 1000 * AL_units_it / housing_stock_it
```

THCR remains a useful conceptual measure, but it is **not part of the v1 municipality TWFE model**. Housing-stock data do not extend far enough into the current panel, and the public RNAL snapshot does not reconstruct historical active AL stock because it lacks cancellation/end dates.

### External Housing Pressure Index

EHPI remains a proposed model-based counterfactual quantity. It is not reported in v1 because a defensible annual historical STR exposure series is not yet available.

## Lisbon longitudinal evidence

The NUTS-2013 and NUTS-2024 Lisbon municipality series were joined only after an empirical overlap audit. All available overlap observations were numerically identical for the published Lisbon municipality values, allowing a guarded direct splice for the case-study series.

The post-2022 affordability deterioration is concentrated in 2022–2023. Lisbon rent rose 18.17% while income rose 3.10%, increasing the rent-to-income proxy by 14.61%. From 2023 to 2024, rent rose 4.66% and income 4.37%, leaving the ratio almost unchanged at +0.29%.

Across 2022–2024, rent increased 23.68% and income 7.60%, producing a 14.94% increase in the rent-to-income proxy. LHDI rose from 106.20 in 2022 to 121.72 in 2023 and 122.07 in 2024.

Tourism intensity did not exceed its 2019 benchmark over the same recovery period: 2024 was 95.03% of the 2019 level and 2025 was 97.39%.

## National municipality comparison

For the 2022–2023 endpoint comparison, complete rent-and-income observations were available for 200 of 308 municipalities. Lisbon ranked 22nd of those 200 municipalities for deterioration in the rent-to-income proxy, corresponding to the 89.5th percentile. Porto ranked 26th, at the 87.5th percentile.

The observed-sample median deterioration was 5.84%, compared with 14.61% for Lisbon. These rankings apply only to municipalities with observed rent and income endpoints; missing rent observations must not be interpreted as zeros or as a random national sample.

## Short-term-accommodation evidence

The project keeps three distinct concepts separate:

```text
RNAL registration != active tourist dwelling != platform listing
```

The current RNAL register is suitable for a current snapshot and a surviving-registration proxy, but not for reconstructing historical active stock.

For Lisbon, independently verified Inside Airbnb snapshots provide point-in-time platform counts:

| Snapshot | Listed units | Entire home/apt | Private room | Shared room | Hotel room |
|---|---:|---:|---:|---:|---:|
| 2024-12-14 | 24,181 | 17,867 | 5,806 | 287 | 221 |
| 2026-06-23 | 24,876 | 18,444 | 6,131 | 148 | 153 |

Missing platform years remain missing. The project does not interpolate a 2025 platform count or use these two snapshots as an annual municipality exposure series.

## Data sources

| Measure | Legacy INE indicator | Current INE indicator | Source |
|---|---:|---:|---|
| Median rent of new leases, EUR/m² | `0009631` | `0012600` | INE local housing rent statistics |
| Median gross declared income per taxpayer | `0009934` | `0012749` | INE local income statistics |
| Classical family dwelling stock | `0008329` | `0014137` | INE housing stock statistics |
| Resident population | `0008272` | `0012917` | INE annual population estimates |
| Tourist overnight stays by accommodation type | `0009877` | `0013214` | INE tourist accommodation survey |
| Alojamento Local | — | — | Turismo de Portugal / RNAL |
| Point-in-time platform listings, Lisbon | — | — | Inside Airbnb dated snapshots |

The execution layer uses INE as the statistical source. Where direct INE transport is unavailable from GitHub-hosted runners, the public Pipeworx endpoint is used only as a transport proxy for the native INE indicator payload.

## Reproducibility

The empirical pipeline is script- and CI-driven. Important entry points are:

- `scripts/build_lisbon_longitudinal.py` for the Lisbon annual series;
- `scripts/build_lisbon_descriptive.py` for the rent/income decomposition;
- `scripts/build_municipality_comparison.py` for the 2022–2023 cross-municipality comparison;
- `scripts/build_municipality_panel_support.py` for the 2022–2024 repeated-support audit;
- `scripts/run_municipality_twfe.py` for the final v1 TWFE association model.

The notebooks are narrative companions to those executable scripts rather than independent hidden analysis pipelines.

## Notebooks

1. `notebooks/01_build_housing_pressure_indices.ipynb` summarises the frozen measurement layer and committed Lisbon evidence.
2. `notebooks/02_model_housing_decoupling.ipynb` reads the frozen TWFE result, checks the coefficient decomposition and documents the v1 interpretation.

## Interpretation boundary

The v1 finding is descriptive:

> Municipalities experiencing larger within-municipality increases in tourism intensity over 2022–2024 also have a modestly larger estimated increase in the rent-to-income proxy, but the association is imprecisely estimated and is not statistically distinguishable from zero at conventional levels.

This result does not identify a tourism treatment effect. Credible causal work would require a separate design such as a regulation-based event study or another defensible source of exogenous variation.

## Status

**Portfolio v1 · Frozen empirical case study**

The measurement layer, Lisbon longitudinal decomposition, national municipality comparison, panel-support audit and current-vintage TWFE association model are reproducibly executed. Historical STR exposure remains too sparse for the originally proposed THCR/EHPI panel model, and that limitation is preserved rather than filled with interpolation or a current-register reconstruction.
