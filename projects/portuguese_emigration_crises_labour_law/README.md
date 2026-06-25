# Portuguese Emigration, Crises, and Labour-Law Changes

This README explains the notebook:

`portuguese_emigration_crises_labour_law.ipynb`

The notebook analyses the annual number of Portuguese people who left Portugal, using public emigration-flow data and marking major economic crises and labour-law changes directly on the plots.

## Purpose

The goal is to create a reproducible exploratory analysis of Portuguese emigration over the latest available long-run annual window.

The notebook is descriptive. It does not claim that labour-law changes caused emigration increases or decreases. The crisis periods and legal changes are added as visual markers to help interpret the timeline.

## Main data source

The notebook uses the Portuguese Emigration Observatory table:

**E.2 Estimates of total outflows of Portuguese emigrants, 2000–2024**

The main plotted series is:

- `oem_estimate`: Portuguese Emigration Observatory estimate of total Portuguese emigrant outflows.

The notebook also includes INE / Statistics Portugal fields where available:

- `ine_total`
- `ine_permanent`
- `ine_temporary`

## Time range

The requested window was the last 26 years.

The latest complete annual flow table used by the notebook covers:

**2000–2024**

This gives the latest available observed annual window in the source table. When 2025 data becomes available, the notebook can be updated by changing:

```python
START_YEAR = 2000
END_YEAR = 2025
```

and adding or scraping the new row.

## What the notebook does

The notebook:

1. Loads annual Portuguese emigration outflow data.
2. Tries to scrape the live Observatory table.
3. Falls back to an embedded copy of the table if the website structure changes.
4. Creates crisis-event annotations.
5. Creates labour-law event annotations.
6. Produces several plots:
   - Observatory-estimated total Portuguese emigrant outflows.
   - Observatory estimate versus INE total outflows.
   - INE permanent versus temporary outflows.
   - Year-over-year change in the Observatory estimate.
7. Exports cleaned CSV files when run.

## Crisis markers included

The notebook marks the following periods:

| Period | Label | Notes |
|---|---|---|
| 2001–2002 | Dot-com / 9-11 shock | Global slowdown context. |
| 2008–2009 | Global financial crisis | International financial crisis. |
| 2011–2014 | Portugal bailout / Troika programme | Sovereign-debt crisis and adjustment programme. |
| 2020–2021 | COVID-19 shock | Pandemic and international mobility restrictions. |
| 2022–2023 | Inflation / energy shock | Post-pandemic inflation and energy-price shock. |

## Labour-law markers included

The notebook marks these Portuguese labour-law changes:

| Year | Law | Label |
|---|---|---|
| 2003 | Law no. 99/2003 | 2003 Labour Code |
| 2009 | Law no. 7/2009 | 2009 Labour Code revision |
| 2012 | Law no. 23/2012 | 2012 labour reform |
| 2019 | Law no. 93/2019 | 2019 Labour Code amendment |
| 2023 | Law no. 13/2023 | 2023 Decent Work Agenda |

## How to run

Install the required packages (pinned in `requirements.txt`):

```bash
pip install -r requirements.txt
```

Then launch Jupyter:

```bash
jupyter notebook portuguese_emigration_crises_labour_law.ipynb
```

or:

```bash
jupyter lab portuguese_emigration_crises_labour_law.ipynb
```

## Output files

Running the notebook creates `data/raw/` (cached World Bank downloads + `provenance.json`)
and an `outputs/` folder with:

```text
outputs/
├── emigration_outflows_portugal_2000_2024.csv
├── emigration_context_crisis_events.csv
├── emigration_context_labour_law_events.csv
├── emigration_macro_context.csv            # emigration + macro/destination covariates
├── emigration_economic_context.csv         # wages, median/mean income, cost of living, GDP/cap
├── emigration_inbound_flow.csv             # emigration, net migration, implied immigration
├── emigration_by_destination.csv           # Portuguese inflows by destination (OECD)
├── emigration_female_share.csv             # female share over time
└── figures/
    ├── emigration_vs_unemployment.png
    ├── macro_correlation_matrix.png
    ├── wages_income_cost_of_living.png
    ├── inbound_vs_outbound_flow.png
    ├── emigration_by_destination.png
    ├── emigration_female_share.png
    ├── interrupted_time_series.png
    ├── change_point_detection.png
    ├── structural_counterfactual.png
    ├── event_study.png
    └── synthetic_control.png
```

## Interpretation notes

The main pattern in the data is the sharp increase in Portuguese emigration during the sovereign-debt crisis and adjustment period.

In the Observatory estimate, the peak year is **2013**, with about **120,000** Portuguese emigrant outflows.

In the INE total-outflow series, the highest value is **2014**, with **134,624** outflows.

The labour-law markers should be read as historical context, not as causal evidence. A causal analysis would require additional variables and a proper identification strategy.

## Implemented extensions

The second half of the notebook ("Implemented extensions") moves from description toward
cautious inference. **Data extensions** add macroeconomic and destination-country context
from the World Bank API (cached locally with a provenance record):

- Portuguese unemployment, youth unemployment, GDP growth, inflation, GDP per capita, net migration;
- destination-country (ES, FR, CH, LU, GB, DE) unemployment and a GDP-weighted demand index;
- **wages and living standards** — mean gross wage (OECD, USD PPP), **median and mean
  equivalised disposable income** (OECD Income Distribution Database), cost of living
  (consumer price index), and GDP per capita in PPP terms;
- an **implied inbound flow** (immigration ≈ net migration + emigration) showing Portugal's
  swing from net immigration (2000s) to net emigration (2011–2014 crisis) and back to
  strong net immigration since ~2016;
- **emigration by destination country and gender** — Portuguese inflows into each OECD
  country (OECD International Migration Database mirror statistics), the shifting top
  destinations (Switzerland, Spain, Germany, France, UK, Luxembourg), and the female share
  over time.

**Modelling extensions** then analyse the series:

- **interrupted time-series** (segmented regression at the 2011 bailout, HAC standard errors);
- **change-point detection** (`ruptures`, breaks chosen by the data, not imposed);
- a **structural time-series counterfactual** (state-space local-linear-trend projection of a
  no-crisis path, with an excess-emigration estimate);
- an **event-study** view (emigration re-centred on event time, indexed to the 2011 event);
- a **synthetic-control-style** comparison (synthetic Portugal built from a donor pool of
  non-programme EU countries on the net-migration rate).

Headline: emigration tracks unemployment (r ≈ 0.86); a significant 2011 break; data-chosen
change-points bracket the surge and its unwind; the counterfactual estimates large
crisis-period excess outflow. All estimates are **illustrative and associational** — the
short, overlapping-event series cannot identify the independent effect of any single
labour-law change.

### Disaggregation status

- **Destination country** and **gender** — implemented from the OECD International Migration
  Database (Portuguese inflows by destination; female share over time).
- **Age** and **education** flows remain **data-limited**: OECD migration flows carry no age
  dimension and only employment-rate education, and Eurostat's emigration-by-age/education
  series (`migr_emi2` etc.) were unreachable at run time. They are documented rather than
  approximated; with Eurostat reachable they would slot into the same destination-style cells.

## Caveats

The notebook compares different statistical series. The Observatory estimate and the INE series are not identical measures.

The Observatory estimate is based on arrivals in destination countries. INE estimates are based on Portuguese statistical sources. Differences between the two series should therefore be expected.

The plots are useful for exploration, but they should not be interpreted as proof that any specific law or crisis caused a specific change in emigration.
