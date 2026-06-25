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

Install the required packages:

```bash
pip install pandas matplotlib requests beautifulsoup4 notebook
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

When the final export cell is run, the notebook creates an `outputs/` folder with:

```text
outputs/
├── portuguese_emigration_outflows_2000_2024.csv
├── crisis_events.csv
└── labour_law_events.csv
```

## Interpretation notes

The main pattern in the data is the sharp increase in Portuguese emigration during the sovereign-debt crisis and adjustment period.

In the Observatory estimate, the peak year is **2013**, with about **120,000** Portuguese emigrant outflows.

In the INE total-outflow series, the highest value is **2014**, with **134,624** outflows.

The labour-law markers should be read as historical context, not as causal evidence. A causal analysis would require additional variables and a proper identification strategy.

## Suggested extensions

A stronger version of this project could add:

- unemployment rate;
- youth unemployment;
- real wages;
- GDP growth;
- inflation;
- housing-cost indicators;
- destination-country labour-market demand;
- destination-country wages;
- destination-country unemployment;
- flows by age group;
- flows by education level;
- flows by destination country.

Possible modelling extensions:

- interrupted time-series analysis;
- event-study models;
- Bayesian structural time series;
- change-point detection;
- synthetic-control comparisons with similar countries.

## Caveats

The notebook compares different statistical series. The Observatory estimate and the INE series are not identical measures.

The Observatory estimate is based on arrivals in destination countries. INE estimates are based on Portuguese statistical sources. Differences between the two series should therefore be expected.

The plots are useful for exploration, but they should not be interpreted as proof that any specific law or crisis caused a specific change in emigration.
