# U.S. County Excessive Drinking: Map, Explain, and Predict

This project starts by reproducing a county-level U.S. choropleth of **excessive drinking** (County Health Rankings / CDC BRFSS), then grows into a full analytics notebook: it validates the target, cross-checks it against independent data sources, engineers a verified ~30-feature predictor set, and **models** the question *"what county characteristics are associated with higher excessive drinking, and how well can we predict it?"*

## What the notebook does

1. **Loads and caches** the County Health Rankings analytic CSV locally, recording provenance (source URL, fetch time, SHA-256, row count) for every remote source so reruns are offline-reproducible and auditable.
2. **Extracts** the excessive drinking field, detecting the proportion-vs-percentage scale once and applying it consistently to the estimate and both confidence-interval bounds.
3. **Validates** structural and value invariants (FIPS well-formed/unique, estimates in `[0, 100]`, estimate within its CI, no state-summary rows).
4. **Reproduces the map** with the screenshot-style palette, plus an "honest" Viridis variant, a companion uncertainty (CI-width) map, and a geographic coverage check that surfaces counties dropped due to FIPS changes (reorganized Alaska census areas; Oglala Lakota County, SD).
5. **Cross-validates the target** against CDC PLACES county binge drinking and NIAAA state per-capita sales, reporting both Pearson and Spearman correlations and population-weighted state means.
6. **Engineers a verified feature set** — ~30 County Health Rankings predictors whose meaning was confirmed against known value ranges and anchor counties (the raw file has opaque coded column names and no data dictionary row).
7. **Explores** the relationships (target distribution, per-feature correlations, correlation structure, strongest bivariate fits).
8. **Models** the rate with a ladder from a naive baseline up to gradient boosting (`LinearRegression`, `Ridge`, `ElasticNet`, `RandomForest`, `HistGradientBoosting`, `LightGBM`), using **state-grouped** cross-validation and a **held-out-states** test set to avoid spatial leakage.
9. **Interprets** the model with permutation importance, SHAP values, partial-dependence plots, and a **spatial residual map**.

### Headline result

County drinking is partly predictable (R² ≈ 0.4–0.5 on held-out states) from demographics, socioeconomics, and co-occurring health behaviors. All models cluster tightly and a regularized **linear** model is as accurate as gradient boosting — the structure is largely additive. Strongest associated features: physical inactivity, age structure, demographic composition, and social connectedness.

## What the measure means

The relevant County Health Rankings field is:

```text
v049_rawvalue = Excessive Drinking raw value
```

This is the age-adjusted percentage of adults reporting binge or heavy drinking. It is not a direct measurement of alcohol sales, alcohol-use disorder, or alcohol-related deaths.

## Files

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── reproduce_excessive_drinking_map.ipynb
├── data/
│   ├── raw/            # cached source files + provenance.json (gitignored)
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
└── outputs/            # tidy CSVs tracked; PNG/HTML regenerated (gitignored)
    └── .gitkeep
```

Cached source data and large generated artifacts (PNG/HTML) are gitignored; the notebook re-downloads and regenerates them on first run.

## How to run

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Then start Jupyter:

```bash
jupyter lab
```

Open:

```text
notebooks/reproduce_excessive_drinking_map.ipynb
```

Run all cells.

The notebook will try to download the 2024 County Health Rankings analytic CSV directly from the official source.

If the official URL changes, download the analytic data manually from the County Health Rankings data documentation page and place it here:

```text
data/raw/analytic_data2024.csv
```

Then rerun the notebook.

## Outputs

The notebook writes:

```text
outputs/excessive_drinking_county_2024.csv           # tidy county estimates
outputs/excessive_drinking_county_2024.html          # interactive choropleth
outputs/excessive_drinking_county_2024.png           # static choropleth
outputs/excessive_drinking_vs_places_county_2023.csv # CHR vs CDC PLACES (county)
outputs/excessive_drinking_vs_niaaa_state_2023.csv   # CHR vs NIAAA (state)
data/processed/modeling_table.csv                    # target + engineered features
```

The HTML file is an interactive county-level choropleth. The statistical plots
(distributions, correlation heatmap, model comparison, SHAP, partial dependence) are
embedded inline in the notebook.

## Main data sources

- County Health Rankings & Roadmaps data documentation
- County Health Rankings 2024 Data Dictionary
- CDC excessive drinking definition and BRFSS-related documentation
- CDC PLACES county-level binge drinking (2023)
- NIAAA apparent per-capita alcohol consumption (2023)
- Plotly public county FIPS GeoJSON

## Modeling notes

- **Leakage-aware evaluation:** entire states are held out for testing and cross-validation uses `GroupKFold` by state, since counties within a state are correlated. A random split would overstate accuracy.
- **No target leakage in features:** the target's own CI bounds, alcohol-impaired driving deaths (a consequence), and CDC PLACES binge drinking (the same construct) are excluded from predictors.
- **Associational, not causal:** features are mutually correlated and some may be partly downstream of drinking; the model describes association, not causation.
- All randomness is seeded (`RANDOM_STATE = 42`).

## Reproducibility note

This reproduces the same type of map from the official county-level data source. It may not match the screenshot pixel-for-pixel because the screenshot likely uses a specific styling, binning scheme, and possibly a specific release year. Every remote source is cached under `data/raw/` with a provenance record (URL, fetch time, SHA-256, row count), and dependencies are pinned in `requirements.txt`.
