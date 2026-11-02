# Adaptive inspection when the shelf-life prior is wrong

## Research question

The [one-step Bayesian inspection rule](ADAPTIVE_POLICY.md) calculates the
value of another inspection under three possible commercial-survival curves.
Its first evaluation draws the true curve from precisely those hypotheses.
That is an informative implementation check, but an optimistic world: real
storage lifetimes may be shorter or longer than the assumed library predicts.

This extension asks whether **paying for contingent information still helps**
when true deterioration differs systematically from the decision model.
It is separate from the earlier [survival extrapolation stress](SURVIVAL_STRESS.md),
which keeps a fitted fixed-date decision after the day-12 cutoff; here both
the *inspection decision* and any subsequent day-16 sale-date update are
evaluated under misspecification.

## Frozen decision model, perturbed truth

Before target outcomes are generated, fix the same three survival hypotheses,
prior probabilities and known hypothetical price paths as in
`R/adaptive_evaluation.R`. The sale/inspect policy is unchanged. Within each
lot, draw one latent library scenario \(h\) once, then create three *true*
storage worlds by multiplying each eligible cohort's scale by
\(\delta\in\{0.75,1.00,1.25\}\):

\[
S^{\mathrm{true}}_{hg,\delta}(t)
=\exp\!\left[-\left(\frac{t}{\delta\lambda_{hg}}\right)^{k_g}\right].
\]

The policy **never receives** \(\delta\), the generating scenario ID, true
Weibull parameters or future failure times. Its posterior still updates over
the *unchanged* three-hypothesis library using the representative day-6 and
day-12 counts available before its day-12 action. The action is to commit to
the best fixed sale date, or pay for a nondestructive day-16 inspection and
then reoptimise the sale date from that new inspection alone.

The scale shifts measure systematic shelf-life misspecification, not a change
to the assumed prior weights. Some shifted distributions can fall outside the
predeclared hypothesis range. All scenarios keep the same initially eligible
fruit mass, price assumptions and two-day delivery lag.

## Paired prospective construction

For each independent simulated harvest lot:

1. Draw the observed maturity-stratum mass split, one latent library scenario,
   one exogenous price path and common **predrawn uniforms** for day-6/day-12
   inspections. Each true scale regime thresholds these uniforms by its own
   survival probabilities, producing its own valid binomial counts.
2. Fit the scenario posterior and freeze the day-12 commit-or-inspect plan
   *separately in each world*, using **only** its already-observed counts.
3. After **all** plans are frozen, draw common day-16 inspection uniforms and
   common unit-scale Weibull fruit failure times. Scale those same unit draws
   by each regime's true lifetimes; this couples potential outcomes across
   regimes without leaking them to the policies.
4. The day-16 counts affect only the contingent-inspection and always-inspect
   policies. The fixed policy keeps its day-12 commitment. Score all sales on
   the identical fruit, price and inventory *within* each world, subtracting
   inspection labour only when actually ordered.
5. Select a true-parameter optimal date **after the policy decisions**, for
   scoring context only. Its realised finite-lot return is not necessarily a
   pathwise upper bound; it is unavailable to the actual policies.

The resulting per-lot estimand is the realised adaptive-minus-fixed gain
\(G_{i,\delta}\). We also compare the *change* in that gain relative to the
unchanged world on the **same** lot:

\[
D_{i,\delta}=G_{i,\delta}-G_{i,1}.
\]

The fixed decision is the best *predecision* commitment within the Bayesian
model, not an oracle with the true deterioration curve. The planned gain is
\(\max(0,I-F)\), while the realised gain includes the observed fruit outcomes
and any labour cost. The difference between them is a descriptive calibration
diagnostic, not an unbiased estimate of information value on one lot.

## Summaries and uncertainty

Report each regime's inspection frequency, mean planned gain, mean realised
gain, fraction below its fixed-sale baseline, mean gain over immediate sale,
and descriptive gap to the truth-curve oracle. Report contrasts of shorter and
longer worlds against the unchanged world. Both marginal and contrast 95%
percentile intervals resample **whole independent lots**, using the *same*
bootstrap lot indices across regimes. Do not resample individual fruit,
inspection rows or correlated regime records as independent observations.

Negative gains are retained. All seeds and scenarios are fixed by the script;
neither the inspection policy nor a winner is selected after observing target
returns. Comparisons are not corrected for multiple testing.

## Reproduce

From `projects/orange_storage_decision/`, using base R only:

```sh
Rscript tests/test_adaptive_stress.R
Rscript scripts/run_adaptive_stress.R 200
```

This writes `records.csv`, `by_shift.csv`, `contrasts.csv`, `settings.R` and
`RESULTS.md` beneath `results/adaptive_stress/` by default. The CI smoke run
uses a smaller number of lots in a temporary directory.

## Interpretation limits

The model still assumes known deterministic future prices, fixed maturity
strata, perfect nondestructive inspections, independent binomial sample labels
and Weibull deterioration with correctly specified shapes. A ±25% scale
perturbation is an illustrative stress, not an empirically calibrated range.
Correlated storage shocks, inspection misclassification, destructive sampling,
mass-dependent deterioration and price forecast revisions remain outside this
experiment. Real archived and prospective harvest lots would be needed to
calibrate priors, validate the decision rule, and estimate commercial value.
