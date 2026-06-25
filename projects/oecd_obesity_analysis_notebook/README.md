# Is OECD adult obesity still rising, plateauing, or reversing?

A rigorous panel analysis of adult obesity across OECD countries. The notebook tests a
widely shared claim — that obesity might be *reversing* — and shows that the apparent
reversal is largely a **data-composition artifact**, not a real downturn.

## Main notebook

- `oecd_obesity_analysis.ipynb`

## The core problem

OECD obesity data is **survey-wave based and unbalanced**: the latest year differs by
country, and the set of reporting countries changes every year. **Self-reported** BMI also
systematically understates **measured** BMI. A naive yearly cross-country average mixes a
real trend with composition and methodology effects — so we avoid it and model trends
*within* countries instead.

## What the notebook does

1. **Acquires and caches** OECD Body-weight (SDMX) and World Bank population data, with a
   provenance record (URL, fetch time, SHA-256, row count) for each source.
2. **Cleans** the data using robust **code-based** filters (not fragile text matching) and
   validates structural/value invariants.
3. **Audits coverage** with a country×year heatmap that motivates the whole approach.
4. **Quantifies the measurement gap** between measured and self-reported obesity and fits a
   bias-correction model (mean gap ≈ +3.8 pp; measured = 3.11 + 1.03·self-reported, R²=0.71).
5. **Estimates within-country trends** with OLS slopes, 95% confidence intervals, and
   significance, for every country with ≥5 observations.
6. **Pools** the trend with country **fixed effects** (cluster-robust SE) and a
   **random-intercept mixed model**.
7. **Tests for a post-2013 slowdown** with a segmented (piecewise) model.
8. **Exposes and fixes the population-weighting artifact** with a balanced-panel estimate.
9. **Visualises** everything, including interactive Plotly choropleths and a multi-country
   line chart (saved as standalone HTML).
10. **Synthesises** the evidence into a single table and a data-driven verdict.

## Key findings

- **Within countries, obesity is still rising**: ≈ **+0.34 pp/year (3.4 pp/decade)**,
  95% CI [2.5, 4.2], p ≈ 3×10⁻¹⁵.
- **No statistically significant slowdown after 2013** (slope change ≈ −0.04 pp/yr, p ≈ 0.5).
- The apparent recent reversal is a **composition artifact**: in 2024 only five small
  countries report and the US drops out, so the weighted average falls even though no
  country's obesity fell.
- **Verdict:** the data does **not** support a broad reversal.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux  (.venv\Scripts\activate on Windows)
pip install -r requirements.txt
jupyter lab
```

Open `oecd_obesity_analysis.ipynb` and run all cells. The notebook downloads from live
OECD and World Bank APIs on first run and caches everything under `data/raw/`; reruns are
offline-reproducible. If an API is temporarily unavailable or rate-limited, rerun later.

## Outputs

```text
data/processed/*.csv            # cleaned panel, bias table, per-country & pooled trends, weighted series
outputs/evidence_summary.csv    # the synthesis table
outputs/figures/*.png           # coverage heatmap, bias scatter, trend forest plot, weighting fix
outputs/interactive/*.html      # interactive choropleths + multi-country line chart
```

Cached data and regenerated artifacts are gitignored; the notebook recreates them on run.

## Data sources

- OECD Body weight (Health: Non-medical determinants of health), OECD SDMX API
- World Bank total population (`SP.POP.TOTL`)

## Notes

All estimates are **descriptive, not causal**. Self-reported data dominates the panel and
understates obesity; measured coverage is too sparse for measured-only trends in most
countries. The balanced-panel weighting trades coverage for year-to-year comparability.
