# Portugal GDP Bayesian Revision

This project implements a Bayesian workflow to estimate Portugal's 2025 GDP and GDP per capita after the revised resident-population information became available.

The purpose is methodological. A higher resident-population estimate mechanically lowers GDP per capita if GDP is held fixed, but that is only a denominator-only correction. A more complete approach treats population, GDP, GDP per capita and the PPS index as uncertain quantities that should be updated when new information arrives.

## Core idea

The model estimates:

```text
p(GDP_2025, Population_2025, GDPpc_2025 | historical data, revised population, GDP growth signal)
```

It starts from the longest historical population and macroeconomic series that can be fetched, then patches the latest revised INE population observation for 2025.

The current implementation uses:

- a Bayesian AR(1) model for log population growth;
- a noisy observation model for the revised 2025 population estimate;
- a Bayesian regression for nominal GDP growth;
- a direct signal from real GDP growth plus inferred GDP-deflator growth;
- posterior simulation for GDP, population and GDP per capita;
- a separate sensitivity layer for the Eurostat/AMECO GDP-per-capita-in-PPS index.

## Official anchors included

The repository includes a small `data/known_observations.csv` file with manually anchored observations:

- revised 2025 resident population: 11,424,031;
- foreign residents in 2025: 1,597,539;
- 2025 real GDP volume growth: 1.9%;
- preliminary GDP-per-capita-in-PPS index used as a sensitivity input;
- population correction factor used in the article being examined.

These anchors should be treated as input observations, not as final conclusions.

## Data sources

The data-loading layer is designed to use:

- INE for the revised 2025 population estimate and 2025 GDP growth;
- World Bank API for long historical series from 1960 onward;
- AMECO for European Commission macroeconomic series and forecasts;
- Eurostat for GDP per capita in PPS and PPP-related methodology.

The runtime used to generate this repository had no outbound internet access, so the notebook and scripts are written to fetch data when run locally.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

jupyter notebook notebooks/01_bayesian_gdp_population_revision.ipynb
```

Or run the command-line version:

```bash
python scripts/run_population_gdp_model.py
```

## Interpretation

The model should not be read as producing one definitive corrected number. It produces posterior distributions. That is the point.

A good output is not simply:

```text
Portugal is at 77.0% of the EU average.
```

A better output is:

```text
Given the revised population observation, historical GDP dynamics and the observed 2025 real GDP growth signal, Portugal's GDP per capita is likely lower than the preliminary estimate, with uncertainty depending on assumptions about GDP revisions, population measurement error, deflator growth, PPPs and the EU reference average.
```

## Why this matters

The denominator-only correction is a useful first approximation. It is not enough to support a full causal story about convergence, productivity, immigration, tourism or public-policy failure.

This Bayesian version makes the assumptions explicit and separates:

1. the mechanical population effect;
2. the possible GDP revision effect;
3. uncertainty in the data;
4. uncertainty in the GDP-per-capita-in-PPS ranking.
