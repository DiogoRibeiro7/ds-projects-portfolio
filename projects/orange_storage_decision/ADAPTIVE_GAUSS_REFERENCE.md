# Higher-order quadrature reference for adaptive orange inspection

## Purpose

The [grid-resolution study](ADAPTIVE_GRID_RESOLUTION.md) compares finite
midpoint-quantile grids of size 3×3 to 9×9. Its largest grid is still an
approximation, so agreement between those grids does not establish the
accuracy of the *continuous* grader-rate posterior or the decisions it
implies. This follow-up constructs a different numerical reference and
evaluates the actual finite policies against it.

The target is **decision error from numerical integration**, conditional on
the same independent reference audit, observed quality labels, survival
scenarios and synthetic price forecasts. It is not empirical selling profit.

## Conditional grader-rate integration

The reference audit has TP, FN, TN and FP counts from independently verified
commercial quality labels. Before that audit, Se and Sp have independent
Beta(1,1) priors. The audit supplies Beta(1+TP, 1+FN) for sensitivity and
Beta(1+TN, 1+FP) for specificity, before orientation conditioning.

As in the preceding models, the admissible grader has `Se + Sp > 1`.
Instead of discarding tensor-grid nodes along the diagonal, integrate over
the full informative triangle with a smooth change of variables:

\[
 x=\mathrm{Se},\qquad
 \mathrm{Sp}=1-x+xu,\qquad
 (x,u)\in(0,1)^2,\qquad
 \left|\frac{\partial(\mathrm{Se},\mathrm{Sp})}
 {\partial(x,u)}\right|=x.
\]

The normalisation is the untruncated Beta posterior probability of the
informative triangle:

\[
 Z=\int_0^1\!\int_0^1 x\,
 f_{\mathrm{Se}}(x)f_{\mathrm{Sp}}(1-x+xu)\,du\,dx.
\]

Apply an order-64 product Gauss–Legendre rule to this smooth integrand, using
nodes and weights computed by the symmetric Jacobi eigenproblem in base R.
Normalised integration weights represent the conditional grader-rate
posterior. The order-32 rule supplies an independent numerical stability
comparison, including the posterior on survival scenarios and the net value
of the inspection option.

There is no need to create thousands of explicit duplicate survival
hypotheses. For each survival scenario \(h\), integrate the historical
binomial-label likelihood over the grader rates to obtain the joint state
weights. The day-16 inspection outcome calculation integrates the *product*
of all cohort likelihoods over those same shared grader rates. It then
aggregates to three survival hypotheses before optimising the sale date.
Independent integration for each cohort would be incorrect because grader
accuracy is shared across the cohort observations.

Expected economic values always use true commercial survival at delivery,
not reported-label acceptance. The direct inspection labour cost is charged
only if inspection is selected.

## Numerical policy regret

For each independent synthetic snapshot, construct all policies with only
day-10 reference-audit and day-6/12 target inspection counts. The finite
grids `3×3`, `5×5`, `7×7` and `9×9` use the existing implementation and the
same information. No day-16 observations or fruit outcomes are generated.

Under the order-64 reference posterior, evaluate the complete finite-grid
policy as a frozen mapping:

- If it commits on day 12, value its chosen fixed sale day under the
  reference survival posterior.
- If it elects to inspect on day 16, enumerate every possible pair of
  reported cohort counts, use the **finite policy's** prescribed contingent
  selling day for each pair, then value this mapping using the **reference**
  joint predictive probabilities. Subtract inspection labour once.

The reference policy is optimised over the same two actions and sale dates.
Its conditional expected value minus the evaluated finite policy's value
is the **expected numerical regret in EUR**. Apart from floating-point
roundoff, it is nonnegative by construction. The comparison is not made
against a true-parameter oracle or realised individual fruit losses.

Additional diagnostics are:

- Total variation distance between the three-hypothesis survival posteriors.
- Difference in the day-12 `commit`/`inspect` action and fixed sale day.
- Reference-predictive probability mass for day-16 outcomes where the
  finite and reference contingent sale dates disagree.
- Absolute difference in the estimated net inspection-option value.
- The order-32 versus order-64 posterior and value differences, to expose
  possible instability of the numerical reference itself.

## Experimental design and limitations

Each synthetic snapshot draws cohort masses, a latent `short/typical/long`
survival scenario, a known hypothetical price path and hidden grading rates
uniformly between 0.75 and 0.95. Two independent reference-sample budgets
are paired: 8 and 80 verified fruit of each true quality class. Their
confusion-count streams share common random numbers, as do all compared
policies' predecision target-lot labels. The policies never see hidden truth.

The standard report uses 20 independent snapshots and two Gauss orders,
32 and 64. These are **finite quadrature orders**, not a rigorous certified
error bound. Very concentrated posteriors, other audit sizes, price paths and
near-tie decisions could require further refinement; agreement between
orders is useful evidence only in the tested snapshots. The report is
descriptive: it does not attach a bootstrap interval to numerical errors or
claim that they predict field losses.

Actual fruit deterioration, grader calibration, reference-standard error,
cross-fruit dependence, price forecast uncertainty and inspection
destructiveness remain separate validation problems. This remains a
methodological R prototype, not a profit forecast.

## Reproduce

From `projects/orange_storage_decision` (base R only):

```bash
Rscript tests/test_adaptive_gauss_reference.R
Rscript scripts/run_adaptive_gauss_reference.R 20
```

The command writes `records.csv`, `by_grid.csv`, `settings.R` and `RESULTS.md`
under `results/adaptive_gauss_reference/` or a supplied directory. GitHub
Actions runs the analytical tests and a small report smoke test. Generated
reports are not committed.
