# What wholesale quote rise would cover hypothetical storage losses?

This R-only extension uses the frozen [official GPP SIMA wholesale orange
archive](MARKET_PRICES.md) and the [observed 1–4 week quote pairs](FORWARD_PRICE_WINDOWS.md).
It adds a *transparent break-even threshold* calculation—not measured
storage economics, an intervention, or evidence that holding a fruit lot
would be profitable.

## The narrow hypothetical question

Suppose a grower could sell one original kilogram at the observed wholesale
quotation in calendar week \(t\), and could instead sell the surviving,
still-saleable mass at the same product/market's quote in week \(t+h\).

If we **assume**, without estimating either parameter:

- \(\rho\): weekly fraction of mass remaining saleable, \(0 < \rho \leq 1\);
- \(c\): storage cost in EUR per **original** kilogram per calendar week,
  \(c\geq 0\);
- \(h\): number of quoted calendar weeks ahead, \(h\in\{1,2,3,4\}\);

then the *illustrative wholesale-quotation advantage* per initial kg is

\[
A_{t,h}(\rho,c) = \rho^h P_{t+h} - h c - P_t.
\]

The quote-based break-even criterion is

\[
P_{t+h} \geq \frac{P_t + h c}{\rho^h},
\quad\text{or}\quad
P_{t+h}-P_t \geq
P_t\left(\rho^{-h}-1\right)+\frac{h c}{\rho^h}.
\]

Strict improvement requires a **positive** difference, not mere equality.
The evaluation reports both `break_even_or_better` and `strictly_better`
using a predeclared numerical equality tolerance of \(10^{-10}\) EUR/kg.

These formulas assume a constant weekly saleable fraction, a linear cost
per original kg, stable grading and the ability to realise quoted wholesale
prices for the same physical lot. **None of those conditions is observed in
the GPP SIMA data.** This is a sensitivity bound, not a farm return model.

## Observed and hypothetical inputs are kept separate

The *only observed inputs* come from the independent archive of **3,412
GPP SIMA wholesale quotations** and the conservative right-censored pairing
described in [FORWARD_PRICE_WINDOWS.md](FORWARD_PRICE_WINDOWS.md):

| Horizon | Eligible 2026 anchors | Exact observed endpoints |
| ---: | ---: | ---: |
| 1 week | 453 | 433 |
| 2 weeks | 442 | 403 |
| 3 weeks | 431 | 373 |
| 4 weeks | 420 | 343 |

All scenario comparisons use the **same 343 observed anchor panel-weeks**
with a quotation at each of the four horizons. The all-available population
is also reported separately; it varies by horizon and is not silently
substituted for the balanced population.

Only the **parameters** are hypothetical. The grid is fixed in code and
the report before evaluating any result:

| Hypothetical parameter | Prespecified values |
| --- | --- |
| Weekly saleable fraction \(\rho\) | 100%, 99%, 98%, 95% |
| Weekly cost \(c\), EUR per original kg | 0, 0.005, 0.010, 0.020 |
| Calendar-week horizon \(h\) | 1, 2, 3, 4 |

The 16 cost/retention combinations are evaluated at each horizon using
observed same-panel endpoint prices. **The chosen grid values do not come
from quality inspection, storage-cost surveys or GPP estimates**. They
illustrate sensitivity only; no scenario is asserted to be realistic.

The frictionless case \(\rho=1,c=0\) is a regression control: its
`strictly_better` frequency must exactly reproduce the
[forward-window audit](FORWARD_PRICE_WINDOWS.md)'s proportion of higher
wholesale quotations. Unchanged quotations count as *break-even* only in
this zero-friction case, not as strict improvement.

## Descriptive output

For each horizon/scenario, and separately by wholesale market on the
balanced cohort, the report gives:

- Number of exact dated pairs, break-even frequency, strict-improvement
  frequency and mean/median illustrative advantage per original kg.
- Empirical lower-decile illustrative advantage and mean required forward
  quote increase, both explicitly conditional on recorded endpoints.
- The all-available-pairs counterpart, to expose horizon-selection effects.
- A row-level audit of observed dates/prices, assumed retained fraction,
  assumed cost, required price and resulting algebraic advantage.

No Monte Carlo estimation, fitted retention distribution, selected
"optimal holding period", p-value or IID confidence interval is presented.
Overlapping market weeks are dependent, and the balanced cohort is
*selected using future availability*, so it is not a prospective policy.

## Reproduction

From `projects/orange_storage_decision`, with base R:

```sh
Rscript tests/test_sima_quote_break_even.R
Rscript scripts/run_sima_quote_break_even.R \
  data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz \
  results/replay_sima_quote_break_even
```

The destination must not exist. The report writes `scenario_detail.csv`,
`scenario_grid.csv`, `coverage.csv`, `all_available.csv`, `balanced.csv`,
`market_balanced.csv`, `settings.R`, `RESULTS.md`, and `quote_break_even.png`.
The existing GitHub Actions R job tests the arithmetic, source denominators,
missing-endpoint handling, frictionless identity, monotonicity in cost and
saleable fraction, and invalid parameters; it uploads a separate
`orange-sima-quote-break-even` artifact.

The subsequent [continuous feasibility frontier](QUOTE_FEASIBILITY_FRONTIER.md)
solves the same inequality for maximum hypothetical weekly storage cost
or minimum retained-saleable fraction **per observed quote pair**. The
frontier checks all sixteen grid scenarios against this direct calculation;
negative raw cost ceilings and loss allowances are never clipped.
The [missing-future-quotation audit](QUOTE_MISSING_ENDPOINT_BOUNDS.md)
separately bounds what these scenario frequencies could have been among
eligible anchors whose later quote was absent, without assuming that
missing prices follow the observed distribution.

## Evidence boundaries

Wholesale quoted prices are neither what a grower receives nor proof that
the same fruit lot survives, retains grade, can be sold, or commands that
price weeks later. No original lot-level maturity, postharvest loss, quality
inspection, transport, commissions, energy consumption, holding cost or
sale execution is in the archive. The source also lacks first-publication
timestamps, so quotation dates are not certified real-time data access.

Consequently, **a positive illustrative quote-based advantage is not a
positive net return**. A real decision model requires the independent
[prospective lot and price observation protocol](PROTOCOL.md). The
completed synthetic orange-storage study and its release conclusions are
untouched.
