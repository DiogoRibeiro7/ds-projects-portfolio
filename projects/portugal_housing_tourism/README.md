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

| Measure | INE indicator | Source |
|---|---:|---|
| Median rent of new leases, EUR/m² | `0009631` | INE local housing rent statistics |
| Median gross declared income per taxpayer | `0009934` | INE local income statistics |
| Classical family dwelling stock | `0008329` | INE housing stock statistics |
| Resident population | `0008272` | INE annual population estimates |
| Tourist overnight stays | `0009182` | INE tourist accommodation survey |
| Alojamento Local | — | Turismo de Portugal / RNAL |
| Active/listed short-term rentals, optional | — | Inside Airbnb Lisbon snapshots |

## Important measurement rule

`RNAL registration != active tourist dwelling`.

The public RNAL layer exposes registration dates but no cancellation/end-date field. A present-day registry therefore cannot be treated as a complete historical active-stock series. The project keeps a surviving-registration proxy separate from a defensible dated active-stock measure.

## Status

**Research case study · Active**

The analytical framework, data clients, index definitions and tests are implemented. Historical short-term-rental exposure remains the main empirical data constraint before the project should be presented as a completed causal or policy analysis.
