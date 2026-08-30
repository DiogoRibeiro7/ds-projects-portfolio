# Portugal Housing, Tourism and Short-Term Accommodation

A deliberately small data-science project studying whether housing costs in Portugal have decoupled from local purchasing power, and how that change is associated with tourism intensity and short-term accommodation.

The project starts at municipality-year level from 2017 onward, with Lisbon as the main case study. It does **not** assume that Airbnb or Alojamento Local caused rent growth. The first objective is measurement; the second is conditional association and counterfactual description under an explicit statistical model.

## Core quantities

### Tourist Housing Conversion Rate (THCR)

```text
THCR_it = 1000 * AL_units_it / housing_stock_it
```

Interpretation: short-term accommodation units per 1,000 conventional dwellings.

### Local Housing Decoupling Index (LHDI)

```text
LHDI_it = 100 * (rent_it / income_it) / (rent_i0 / income_i0)
```

Values above 100 mean the rent-price-to-income proxy has increased relative to that municipality's baseline. Because the numerator is EUR/m² for new leases and the denominator is income per taxpayer, LHDI is a normalised decoupling index, not a literal household rent share.

### Tourism Intensity (TI)

```text
TI_it = overnight_stays_it / resident_population_it
```

### External Housing Pressure Index (EHPI)

```text
EHPI_it = 100 * predicted_rent_observed_it / predicted_rent_counterfactual_it
```

EHPI is generated only after an explicit statistical model and counterfactual have been defined. It is a model-based descriptive counterfactual, not automatically a causal effect.

## Notebooks

1. `notebooks/01_build_housing_pressure_indices.ipynb`
   - retrieves and caches INE data;
   - ingests dated RNAL snapshots;
   - constructs a municipality-year panel;
   - computes THCR, LHDI and tourism intensity;
   - produces descriptive checks.

2. `notebooks/02_model_housing_decoupling.ipynb`
   - fits two-way fixed-effects models;
   - uses municipality-clustered standard errors;
   - examines lagged exposure;
   - constructs EHPI only after the model is defined;
   - keeps association distinct from causal interpretation.

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

The NUTS-2013 and NUTS-2024 Lisbon municipality series are bridged only after an empirical overlap audit. Every available overlap observation is numerically identical; the bridge therefore preserves the legacy series through its final year and appends only later observations from the current indicator. If a future source revision breaks that equality, the guarded bridge fails.

## Historical short-term-rental exposure

The project separates two estimands rather than pretending that one source measures both.

**National municipality panel.** The current RNAL register is used only to construct a surviving-registration proxy. A registration date tells us when a currently surviving registration began; it does not tell us whether registrations absent from today's register were active in earlier years. This series is therefore a sensitivity measure, not reconstructed historical active stock.

**Lisbon case study.** Historical platform exposure is built from dated Inside Airbnb listing snapshots. Each verified snapshot is a point-in-time census of listings visible to that data collection, not a count of occupied nights and not necessarily a one-to-one count of licensed RNAL establishments. The ingestion layer stores the retrieval bytes, SHA-256 digest, source URL, snapshot date and row count.

Verified snapshot URLs belong in `data/manifests/inside_airbnb_lisbon.csv`. Missing years remain missing: `annualise_listing_snapshots()` chooses the latest observed snapshot within each year and performs no interpolation or backfilling.

This means the preferred hierarchy is:

```text
Lisbon dated platform snapshots
    -> primary historical exposure series for the Lisbon case study

RNAL current-register survivor reconstruction
    -> nationwide sensitivity proxy only
```

Neither measure is automatically interpreted causally.

## First executed Lisbon observations

The first empirical workflow execution successfully fetched and summarised two independently verified snapshots:

| Snapshot | Listed units | Entire home/apt | Private room | Shared room | Hotel room |
|---|---:|---:|---:|---:|---:|
| 2024-12-14 | 24,181 | 17,867 | 5,806 | 287 | 221 |
| 2026-06-23 | 24,876 | 18,444 | 6,131 | 148 | 153 |

Between these two observed dates, total listed units are 2.9% higher and entire-home listings are 3.2% higher. The entire-home share is nearly unchanged, at about 73.9% and 74.1%, respectively. These are comparisons between two point-in-time platform censuses, not estimates of housing units converted to tourism use and not causal effects on rents.

The compact outputs and source hashes are stored in `results/processed/inside_airbnb_lisbon_annual.csv` and `results/processed/inside_airbnb_lisbon_provenance.csv`. Raw third-party listing files are not committed.

## Lisbon descriptive decomposition

The first decomposition uses the exact identity

```text
Delta log(rent / income) = Delta log(rent) - Delta log(income)
```

to separate movement in the affordability proxy into rent and income components. It is descriptive accounting, not a causal model.

The post-2022 deterioration is concentrated almost entirely in 2022–2023. Rent rose 18.17% while median declared income rose 3.10%, increasing the rent-to-income proxy by 14.61%. In log terms, the affordability gap widened by 13.64 percentage points. From 2023 to 2024, rent rose 4.66% and income 4.37%, so the rent-to-income proxy changed by only 0.29%.

Across 2022–2024 as a whole, rent increased 23.68% and income 7.60%, producing a 14.94% deterioration in the rent-to-income proxy. LHDI consequently rose from 106.20 in 2022 to 121.72 in 2023 and 122.07 in 2024.

Tourism followed a different timing pattern. Tourism intensity collapsed by 65.92% between 2019 and 2021, then recovered strongly. In 2024 it was 95.03% of its 2019 level and in 2025 it reached 97.39% of the 2019 benchmark. The observed December 2024 platform snapshot contains 36.35 listings and 26.86 entire-home listings per 1,000 residents. These parallel movements are not interpreted as evidence that tourism or platform listings caused the rent acceleration.

The reproducible annual indices and episode decompositions are stored in `results/processed/lisbon_descriptive_annual.csv` and `results/processed/lisbon_descriptive_episodes.csv`.

## Important measurement rule

`RNAL registration != active tourist dwelling != platform listing`.

The public RNAL layer exposes registration dates but no cancellation/end-date field. A present-day registry therefore cannot be treated as a complete historical active-stock series. Likewise, a platform listing is an observable market listing rather than proof of occupancy, legal status or conversion from a conventional long-term dwelling.

## Status

**Research case study · Active**

The analytical framework, data clients, index definitions, NUTS-vintage bridge, Lisbon descriptive decomposition, tests and first observed Lisbon platform snapshots are now executed reproducibly. The platform-exposure series remains sparse because 2025 and earlier archive dates have not yet been independently verified, so the project remains non-featured and should not yet be presented as a completed causal or policy analysis.
