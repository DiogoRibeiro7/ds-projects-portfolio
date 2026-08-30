# v1 empirical freeze

## Scope

Portfolio v1 freezes the evidence currently supported by reproducible public-source workflows:

- Lisbon longitudinal rent, income, population and tourism series with an audited NUTS-vintage bridge;
- verified Lisbon Inside Airbnb snapshots for 2024-12-14 and 2026-06-23;
- Lisbon rent-versus-income descriptive decomposition;
- 2022-2023 municipality endpoint comparison;
- 2022-2024 municipality panel-support audit;
- 2022-2024 tourism-intensity TWFE association model.

The freeze does not include a historical annual THCR series, EHPI counterfactual or causal tourism/STR effect.

## Municipality panel support

The current-vintage panel contains 308 municipalities in the metadata universe. Rent coverage is the limiting measure:

| Measure | 2022 | 2023 | 2024 | At least 2 years | All 3 years |
|---|---:|---:|---:|---:|---:|
| Rent | 203 | 215 | 212 | 208 | 199 |
| Income | 298 | 298 | 299 | 298 | 298 |
| Population | 308 | 308 | 308 | 308 | 308 |
| Overnight stays | 307 | 308 | 308 | 308 | 307 |

After requiring positive complete values for rent, income, population and overnight stays, the TWFE primary sample contains 614 observations from 208 municipalities. The balanced sensitivity contains 594 observations from 198 municipalities.

## Frozen TWFE result

Primary model:

```text
log(rent / income) ~ log(tourism intensity) + municipality FE + year FE
```

with standard errors clustered by municipality.

| Sample | Coefficient | Clustered SE | 95% CI | p-value |
|---|---:|---:|---:|---:|
| Primary unbalanced | 0.035621 | 0.026557 | [-0.016430, 0.087673] | 0.179820 |
| Balanced sensitivity | 0.039267 | 0.033182 | [-0.025769, 0.104303] | 0.236657 |

On the primary sample, the rent coefficient is 0.034434 and the income coefficient is -0.001187. The affordability coefficient identity holds to floating-point precision.

## Interpretation

The point estimate is positive and similar in the primary and balanced samples, but both confidence intervals include zero. The frozen v1 interpretation is therefore:

> A modest positive within-municipality tourism-affordability association is estimated over 2022-2024, driven almost entirely by rents rather than incomes, but the association is statistically uncertain and is not a causal estimate.

No specification is substituted post hoc to obtain statistical significance.

## Provenance

The first successful TWFE execution was GitHub Actions run `33319206560`, workflow run number 3, on PR head `70b372d0569fd8ff0a1a472505665cf68a88c425`.

The uploaded evidence artifact was `portugal-housing-tourism-twfe`, artifact ID `9734394970`, SHA-256:

```text
dffd6102d64ac528f1dc81bddae26d97caaab5a834cd2e588b99d880537d423a
```

The compact result and provenance are committed as:

- `results/processed/municipality_twfe_results.json`;
- `results/processed/municipality_twfe_provenance.json`.

## Future research boundary

A later version may investigate historical STR exposure or causal policy variation, but it must not backfill missing platform years, interpret the current RNAL register as historical active stock, or silently use housing-stock values beyond their observed availability.
