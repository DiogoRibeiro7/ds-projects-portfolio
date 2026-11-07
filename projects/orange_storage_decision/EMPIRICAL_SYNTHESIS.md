# Empirical quotation synthesis and research closure

**Status:** completed, reproducible **wholesale-quotation** analysis. This is
not a postharvest field trial, an identified storage treatment effect, a
real-time selling strategy or evidence of positive producer returns. The
separate [scientific synthesis](SCIENTIFIC_SYNTHESIS.md) covers the completed
**synthetic** selling-decision methods and is not retroactively validated by
the observations here.

## 1. What was observed

The frozen GPP SIMA orange extract contains **3,412 official wholesale
quotation rows** dated 2025-01-02 to 2026-10-30. It is stored as
`data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz` with an
uncompressed CSV SHA-256 of
`b039f254b4fb18a41b7997208a43ceae46ee9f48db978934ab9fad6de9e24da0`.
The analysis retains the original product descriptor, region, wholesale
market, price unit and dated *frequent quotation*. There are **485 observed
2026 starting panel-weeks**; a panel-week takes its last recorded quotation
for that exact product/region/market and calendar week.

These are *nominal quotes*, without traded volumes, first-publication
timestamps or confirmed availability to a grower on the quoted date. They
are not farm-gate prices, sale executions or observations on the same physical
orange lot over time. See [source provenance](MARKET_PRICES.md) and the
[exact-week pairing specification](FORWARD_PRICE_WINDOWS.md).

## 2. Horizon selection and the observation gap

For an observed 2026 starting quote in week `t`, a future quotation is
paired **only** for the identical product, region and market in week `t+h`,
for `h` = 1, 2, 3 or 4 calendar weeks. The week beginning 2026-10-26, which
contains the final source quote, is conservatively treated as incomplete:
future target weeks must begin no later than **2026-10-19** to be eligible.
An ineligible target is censored, while a matured but unquoted target is
*missing*, never assigned a zero, carried-forward or imputed price.

| Horizon (weeks) | Time-eligible anchors | Exact endpoints | Missing eligible | Fixed common-maturity anchors | Common-maturity endpoints | Fully matched four-horizon anchors |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 453 | 433 | 20 | 420 | 401 | 343 |
| 2 | 442 | 403 | 39 | 420 | 382 | 343 |
| 3 | 431 | 373 | 58 | 420 | 363 | 343 |
| 4 | 420 | 343 | 77 | 420 | 343 | 343 |

There are **three distinct populations**:

1. **Horizon-eligible:** all starting quotes whose particular future week
   elapsed. Its population changes with `h`.
2. **Common-maturity:** the fixed **420** starting quotes for which *all*
   four future weeks elapsed, whether or not the future quote was recorded.
3. **Fully matched:** **343** starting quotes with observed exact endpoints
   at every horizon, selected **using future quote availability**. It is
   appropriate for a like-for-like retrospective description, not a
   prospective policy or an unbiased representation of all 2026 starts.

The 4-week unidentified fraction among eligible starts is **77/420 =
18.3%**. At one week, **20/453 = 4.4%** are missing after eligibility.
Neither fraction quantifies unrecorded fruit sales or physical spoilage.

## 3. What the matched wholesale quotation movements show

On the fixed 343-anchor complete-case subset, the empirical *future quote*
changes are:

| Weeks ahead | Mean signed change (EUR/kg) | Median change (EUR/kg) | Higher quote | Lower quote |
| ---: | ---: | ---: | ---: | ---: |
| 1 | +0.0036 | 0.0000 | 11.4% | 7.0% |
| 2 | +0.0073 | 0.0000 | 22.7% | 13.4% |
| 3 | +0.0101 | 0.0000 | 30.9% | 17.8% |
| 4 | +0.0117 | 0.0000 | 37.3% | 22.2% |

The median quote is unchanged at each horizon; the positive mean does not
mean most prices increased. The frequencies describe *quoted movements
observed after the start*, with overlapping weeks and product/market
dependence. They are not forecast accuracy, independent Bernoulli trials,
realised holding returns or evidence that a grower could sell on either
date at these wholesale prices. The broader all-available samples have
different composition. See [the forward-window evidence](FORWARD_PRICE_WINDOWS.md).

## 4. What hypothetical storage frictions do to the arithmetic

For an observed quote pair, the strictly **illustrative** difference per
initial kilogram is

\[
A_{t,h}(\rho,c)=\rho^h P_{t+h}-hc-P_t,
\]

where weekly retained-saleable fraction `rho` and weekly storage cost `c`
(EUR per *original* kilogram) are assumptions, **not fitted or observed
data**. The experiment fixes `rho` to 1, 0.99, 0.98 or 0.95 and `c` to 0,
0.005, 0.010 or 0.020. Break-even includes an unchanged value within the
declared numerical tolerance; strict improvement requires a positive
margin. For the 343 fully matched starts:

| Assumed weekly retention | Assumed cost/week (EUR/initial kg) | Week 1 break-even | Week 1 strict improvement | Week 4 break-even | Week 4 strict improvement |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100% | 0.000 | 93.0% | 11.4% | 77.8% | 37.3% |
| 99% | 0.005 | 11.4% | 11.4% | 29.7% | 29.7% |
| 98% | 0.010 | 11.4% | 11.4% | 12.5% | 12.5% |
| 95% | 0.020 | 4.7% | 4.7% | 0.0% | 0.0% |

At zero friction, **break-even is not the same as an improvement**. Most
one-week quotes are unchanged, which passes a nonnegative test but cannot
pay even a small unmodelled cost. Under 99% weekly retention and EUR 0.005
weekly assumed cost, only **29.7%** of the four-week matched quote pairs
give a positive hypothetical arithmetic difference. These are conditional
illustrations, not estimates of the frequency of *profitable storage*.

Inverting the same inequality produces raw, **unclipped** hypothetical
limits: at 99% weekly retention, the *median* one-week maximum weekly cost
is **−EUR 0.007 per original kg**; at EUR 0.005 assumed weekly cost, the
*median* required one-week retained-saleable share is **100.67%**. Negative
cost ceilings and retention above 100% mean those median pairs cannot
break even under the remaining assumptions. They do not estimate actual
cost or biological loss. See [scenario calculations](QUOTE_BREAK_EVEN_SCENARIOS.md)
and [continuous feasibility frontiers](QUOTE_FEASIBILITY_FRONTIER.md).

## 5. Missing endpoints: identification, not prediction

Retrospective complete cases can be misleading when the future quote is
missing for eligible starts. Without any assumption about the missing
binary outcome, let `N` be the number eligible, `S` the number of observed
hypothetical successes and `M` the missing endpoints. The sharp bounds are

\[
\frac{S}{N}\leq\text{eligible break-even fraction}\leq\frac{S+M}{N}.
\]

For the **horizon-eligible** population, examples from the fixed grid are:

| Scenario | Horizon | Observed-only fraction | Worst-case eligible bound | Best-case eligible bound |
| --- | ---: | ---: | ---: | ---: |
| No assumed cost/loss; weak break-even | 1 | 92.8% | 88.7% | 93.2% |
| No assumed cost/loss; strict improvement | 1 | 10.6% | 10.2% | 14.6% |
| No assumed cost/loss; weak break-even | 4 | 77.8% | 63.6% | 81.9% |
| No assumed cost/loss; strict improvement | 4 | 37.3% | 30.5% | 48.8% |
| 99% retention, EUR 0.005/week; weak break-even | 4 | 29.7% | 24.3% | 42.6% |

The width of the week-four bounds is the missing share, **77/420 =
18.3 percentage points**. These are *logical identification bounds*, not
confidence intervals or evidence for a missing-at-random assumption. The
common-maturity cohort fixes denominator 420 at every horizon, but does
not resolve unavailable later quotes. The detailed
[missing-endpoint audit](QUOTE_MISSING_ENDPOINT_BOUNDS.md) keeps all absent
future quotation values unknown and reports market partitions.

## 6. Decision and transferability boundaries

The evidence now supports an *auditable description of dated wholesale
quotes*, short-horizon changes, hypothetical loss/cost thresholds, and
the consequences of missing later quotations. It does **not** support:

- a recommended sale day, a fitted orange storage-loss curve or measured
  real-world fruit survival;
- an economically profitable grower policy, attributable uplift, causal
  treatment effect or realised farmer selling price;
- an unbiased national orange-price index, independent-sample confidence
  intervals, or validation of the synthetic study's assumed prices;
- a real-time strategy, since first-publication and revision timestamps
  are unavailable and future matched-panel membership is unknown at `t`.

To move beyond this research/portfolio phase, collect independent lot-level
maturity, defects and mass losses; storage environment and actual costs;
transport and commission charges; sales/unsold quantity and prices received;
buyer eligibility and capacity; and time-stamped information availability.
Evaluate prespecified future harvest lots against a real operational
comparator. The [prospective protocol](PROTOCOL.md) defines that separate
field-validation gate. Until it is satisfied, the project remains
**unfeatured** and explicitly non-operational.

## 7. Reproduction and completion gate

All analysis is offline and uses base R. Run from
`projects/orange_storage_decision`:

```sh
# Integrated empirical headline and cross-stage consistency audit.
Rscript tests/test_sima_empirical_closure.R

# Frozen-source module tests and local-link/archived-synthetic audit.
Rscript tests/test_sima_forward_quote_windows.R
Rscript tests/test_sima_quote_break_even.R
Rscript tests/test_sima_quote_feasibility_frontier.R
Rscript tests/test_sima_missing_quote_bounds.R
Rscript tests/test_release_audit.R
```

The CI workflow also regenerates machine-readable CSVs, PNG figures and
`RESULTS.md` reports for each empirical stage as named 30-day artifacts.
**Those transient artifacts are not a permanent public dataset**: the
committed source archive, code, tests and the exact reproduction commands in
each linked method document are the durable record. Synthetic studies retain
their separately committed original results; no synthetic baseline was
overwritten to make this synthesis.

**Research closure criterion:** all source integrity, cohort, arithmetic,
missingness and release-link audits pass; the distinction between observed
quotations and assumed storage economics remains explicit. On that basis,
the reproducible **research phase** is complete. Field deployment is a
distinct, currently unmet milestone.
