# When Should a Heterogeneous Batch of Oranges Be Sold?

An exploratory, reproducible R case study in postharvest survival analysis and
optimal stopping. The same harvest date does **not** imply the same maturity at
harvest or the same remaining commercial life for every orange.

## Research question

> Can repeated, representative inspections of a small sample of fruit support
> a better sell-or-hold decision for the whole stored batch, without inspecting
> every orange?

The economic problem is to maximise **expected net selling value at delivery**,
not to wait until the last fruit becomes inedible. Fresh-market acceptance is
not identical to food safety, and the model does not certify that fruit is safe
to eat.

## Critical biological boundary

Oranges are non-climacteric: an orange harvested before adequate internal
maturity does not become properly sweet simply through storage. Skin colour
alone is not a reliable maturity label. Initial maturity should be based on
relevant observations such as soluble-solids-to-acidity ratio and quality
inspection, with thresholds appropriate to cultivar, market and regulation.

- UC ANR, [Use taste, rather than rind color, to determine when to harvest
  citrus](https://ucanr.edu/node/111326/printable/print).
- UC Davis Postharvest Center, [Orange: Maturity, Quality and Postharvest
  Handling](https://postharvest.ucdavis.edu/produce-facts-sheets/orange).

An *initially immature* stratum is kept in the inventory example but excluded
from fresh-market revenue. It is not modelled as ripening into the eligible
population. Alternative processing outlets require their own separate quality
thresholds, costs and prices; this version does not optimise those routes.

## Statistical model

At harvest, partition the lot into independently defined maturity/condition
strata, indexed by \(g\), with initial saleable mass \(M_g\). Each initially
eligible stratum has a conditional commercial survival curve

\[
S_g(t)=\Pr(T_g>t\mid g,\text{commercially acceptable at harvest})
     =\exp\left[-\left(t/\lambda_g\right)^{k_g}\right].
\]

Independent, representative cross-sectional inspections observe
\(Y_{g,t}\sim\operatorname{Binomial}(n_{g,t},S_g(t))\). Weibull parameters are
estimated separately by maximising the aggregated binomial likelihood.
Inspections are never treated as individual time-to-event observations.
This is a parametric assumption to validate empirically, not a claim that
real fruit deterioration must be Weibull-distributed.
Multiplying a fruit-count survival fraction by cohort mass additionally assumes
that, within a stratum, the probability of deterioration is not systematically
related to fruit mass. A real application must test this or use mass-weighted
loss measurements.

For a common sale day \(d\), delivery lag \(L\), deterministic expected price
\(p_g(d)\), decision day \(d_0\), and per-initial-kilogram holding cost \(c\),
the illustrative expected value is

\[
V(d)=\sum_{g\in\mathcal E}M_g S_g(d+L)p_g(d)
 - c\left(\sum_{g\in\mathcal E}M_g\right)(d-d_0+L).
\]

Only initially fresh-market-eligible groups \(\mathcal E\) appear. Prices are
locked on the sale day, but the fruit must pass quality inspection at delivery.
Choose the earliest day attaining the highest expected net value. If physical
sorting and separate sales are possible at no extra cost, optimise each stratum
independently as a contrasting upper-bound policy.

### What sampling uncertainty means here

The parametric bootstrap generates inspection counts from the fitted survival
model, refits the curves and recomputes the optimal sale day. It describes
uncertainty due to finite inspection samples *conditional on* the assumed
Weibull specification and supplied price path. Bootstrap failed refits are
counted and reported. Price forecast uncertainty, common storage shocks,
cross-fruit dependence, grading error and model misspecification are not yet
included in that uncertainty interval.

## Run the demonstration

Requires R; no external packages, internet, private data or downloaded datasets.

```bash
cd projects/orange_storage_decision
Rscript tests/test_model.R
Rscript scripts/demo.R
```

The seeded demo creates a fictitious 1,000 kg lot with two initially eligible
strata and a 150 kg initially immature stratum. It simulates cross-sectional
inspections at days 4, 8, 12 and 16, fits separate Weibull curves, and makes a
day-16 decision using only observations from days 16 or earlier. A two-day
delivery lag is included. The two cohort-specific price paths and their future
increases are constructed assumptions—not actual orange price forecasts.

The demo prints:

1. Fitted survival shape and scale by eligible stratum.
2. Expected-value-maximising whole-batch sale day.
3. Optimal separately graded sale days and their hypothetical value difference.
4. Bootstrap distribution of the whole-batch sale day and failed refit count.

## Out-of-time policy evaluation

The follow-up experiment compares maturity-stratified Weibull curves with a
mass-standardised, pooled Weibull baseline. It makes decisions using only
day-12 inspections, then scores independent days 16/20/24 and economic
returns on identical simulated fruit outcomes and price paths. Paired
whole-lot intervals measure variation across the simulated harvest lots.

Held-out scoring also supports a single future inspection day and all-pass or
all-fail samples. These do not need to identify a new survival curve. Every
scored day must cover all eligible strata for inventory standardisation;
unequal sample sizes are allowed, but missing strata are rejected explicitly.

```bash
Rscript tests/test_evaluation.R
Rscript scripts/run_policy_evaluation.R
```

Read [EVALUATION.md](EVALUATION.md) for the estimands, leakage controls,
reproducibility settings and interpretation limits. No positive effect is
assumed or claimed in advance.

## Scientific status and limitations

Status: **methodological prototype, synthetic evidence only**. No empirical
profit improvement or predictive accuracy is claimed. This is deliberately
`featured: false` in the portfolio catalogue.

Before application, the study needs representative observations across
multiple harvest lots and storage environments, traceable maturity grading,
predefined market acceptance standards, genuine historical price scenarios,
loss accounting, out-of-lot validation, uncertainty calibration and comparison
against operational baselines. The synthetic prospective simulation does not
replace an empirical out-of-lot test. See [PROTOCOL.md](PROTOCOL.md) for the
proposed empirical design and anti-leakage rules.
