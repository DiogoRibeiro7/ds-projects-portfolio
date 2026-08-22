# City Wage-Cost Analysis: US, UK, and EU

Reproducible Python notebooks that test a single question across three regions:

> Does the higher *nominal* income in big, expensive cities survive after you
> adjust for what it actually costs to live there?

The thesis is that a city is not automatically an opportunity machine. It may
raise nominal income, but that gain can be absorbed by housing, local prices,
taxes, or household-specific costs. The notebooks make that wage-versus-cost
trade-off explicit using only public data.

All four notebooks are **executed end to end** against live public sources; the
findings below are read off the committed outputs, not assumed.

## Headline findings

**United States** (BLS OEWS May 2024 + BEA Regional Price Parities 2024, 145 metros)

- The nominal city premium is almost universal but the *real* premium is not.
  **San Jose** leads on purchasing power; **San Francisco's** wages survive its
  high prices, but in **Los Angeles only ~47%** of workers come out ahead of
  Tulsa after cost adjustment (vs 91% nominally), and **Miami** pairs near-top
  prices with bottom-tier real wages.
- The split is occupational: high-skill jobs keep a large real premium (NYC vs
  Tulsa: data scientists +\$28.6k, teachers +\$32.8k), while low-wage service
  jobs can go **negative** (NYC retail: −\$489 despite a \$7.4k nominal raise).

**United Kingdom** (ONS Explore Local Statistics, 337 local authorities)

- London boroughs pay the highest wages *and* have the worst housing
  affordability. The general-price "city tax" is small (~7% in London), but
  house-price-to-earnings ratios of **12–22×** (vs ~7× nationally) swamp the wage
  premium for anyone buying in. In the UK, the city tax is **housing**.

**European Union** (Eurostat NUTS2 disposable income per inhabitant + national price levels)

- Nominal income and real living standards rank capital-regions very
  differently. **Munich** leads on price-adjusted income; high-cost **Dublin**
  and **Copenhagen** fall ~85+ rank places; low-price **Eastern capitals**
  (Bucharest, Budapest, Warsaw, Prague) are the big real gainers, while
  **Berlin, Vienna and Brussels** sit *below* their national median region.

**Comparative** — the same mechanism runs through a different channel in each
region: survive the local **rent** (US), the national **price level** (EU), or
**house prices** (UK). A higher nominal income is necessary but never sufficient.

## Notebooks

```text
notebooks/01_us_city_wage_cost_analysis.ipynb
notebooks/02_uk_city_wage_cost_analysis.ipynb
notebooks/03_eu_region_city_proxy_income_cost_analysis.ipynb
notebooks/04_comparative_framework_us_uk_eu.ipynb
```

Run them in order. Notebooks 01–03 each save a CSV under `outputs/`; notebook 04
reads all three and builds the cross-region comparison.

## Data design

### United States

BLS OEWS metropolitan occupation-level wages (the **MSA** file, 5-digit CBSA
codes) and BEA/FRED Regional Price Parities (all-items `RPPALL*` and rent
`RPPSERVERENT*`). Real wage = nominal wage ÷ (RPP / 100).

> **BLS access note.** BLS serves its files behind a bot filter. The loader sends
> a browser User-Agent, and if BLS still blocks the request (common on data-centre
> and cloud IPs, which get an "Access Denied" page) it falls back to the Internet
> Archive mirror and caches the ZIP under `data/raw/`. If both fail, download the
> metro ZIP manually from <https://www.bls.gov/oes/tables.htm> into `data/raw/`
> and re-run.

### United Kingdom

ONS Explore Local Statistics gross median weekly pay and housing affordability
ratios (filtered to genuine local authorities). A coarse ONS 2016 Relative
Regional Consumer Price Level provides a non-housing price adjustment; housing is
analysed separately because it dominates UK local cost differences. An optional
Nomis ASHE section shows how to pull occupation-level wages.

### European Union

Eurostat NUTS2 net disposable household income **per inhabitant** (`nama_10r_2hhinc`,
`B6N`, `EUR_HAB`) deflated by national price-level indices (`tec00120`,
EU27 = 100). Major cities are mapped to their NUTS2 region — a city-region proxy,
not a true city-level cost model.

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

## Important limitations

The three regions do not have identical data availability, so the cost
adjustment is not identical:

- **US** — closest to the original question: metro occupation wages and all-items
  RPP (which include housing).
- **UK** — strong on local wages and housing affordability, weak on current local
  consumer-price parities (the general-price adjustment is a flat 2016 benchmark).
- **EU** — strong on harmonised regional income and cross-country PPP, weak on
  *within*-country city housing costs (national price levels miss the fact that
  Munich or Paris are far dearer than their national average).

Do not read the outputs as a definitive "should I move" answer. The correct
conclusion is conditional on occupation, household composition, housing tenure,
career stage, and remote-work options. See `METHODOLOGY.md`.

## Thesis (now supported by the data)

Cities reliably raise *nominal* income, but a real city premium exists only when
the income premium exceeds the local cost premium. That holds for high-skill
workers in genuinely high-wage cities (San Francisco, San Jose, Munich) and
fails for many median and low-wage workers — most clearly through **housing**,
the channel that most often turns a nominal wage gain into a real affordability
loss.
