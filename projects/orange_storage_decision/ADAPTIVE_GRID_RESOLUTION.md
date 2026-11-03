# Resolution sensitivity of adaptive grader-calibration uncertainty

## Motivation

The [adaptive calibration-uncertainty model](ADAPTIVE_CALIBRATION_UNCERTAINTY.md)
approximates a continuous posterior over grader sensitivity and specificity
with a small, truncated grid of midpoint Beta quantiles. Its default 5×5 grid
is a numerical choice, not a demonstrated convergence threshold. A different
grid can change the posterior on survival hypotheses, the value assigned to a
day-16 inspection and therefore the sale date.

This follow-up measures how sensitive those **operational decisions** are to
the available quadrature resolution. It does not try to prove exact numerical
convergence or superior commercial performance.

## Controlled comparison

Use the same finite `short/typical/long` survival library, day-12 decision,
optional nondestructive day-16 inspection, price path and two-day delivery
lag as the earlier adaptive experiments. For each latent synthetic lot:

1. Draw one hidden survival hypothesis, cohort masses, prices and true grader
   sensitivity/specificity in `[0.75, 0.95]`.
2. Independently draw true-positive and true-negative reference-audit streams.
   The *scarce* audit uses the first 8 fruit of each true reference class;
   the *ample* audit uses the first 80 of each. Both audits are available by
   day 10. No target-lot inspection supplies reference truth.
3. Draw representative day-6/12 target-lot counts from common quality-state
   and grader-error uniforms. All policies see the **same aggregate reported
   labels**, without access to the hidden true states.
4. For each audit budget, construct the conditional posterior over grader
   accuracy separately at grid sizes `3×3`, `5×5`, `7×7`, `9×9`. Each grid uses
   the same audit confusion counts and target inspection counts. Independently
   freeze every day-12 commit-or-inspect plan.
5. Only afterward draw the potential day-16 inspection counts and physical
   fruit lifetimes, shared across all eight decisions. A committed policy
   cannot use the day-16 sample to revise its sale day. Score every policy on
   the same true fruit outcomes and price path, charging inspection labour
   only when ordered.

The grid approximation starts from independent Beta(1,1) priors updated by
reference TP/FN and TN/FP counts and conditioned on `Se + Sp > 1`. Grids are
Cartesian products of midpoint marginal quantiles. Nodes outside the
informative-grader region are **explicitly counted and excluded**; remaining
nodes have equal weights. Each retained rate node is paired with every
survival hypothesis and updated using the noisy-label binomial likelihood.
The expected sales value always uses commercial survival, not reported-label
probability.

## What is reported

Within each audit budget and grid size, compare with the **9×9 policy
computed on the same latent lot**:

- Fraction of lots choosing a different `inspect` or `commit` action.
- Fraction selecting a different eventual sale day.
- Mean absolute difference in the reference-only posterior mean of Se and Sp.
- Mean absolute difference in the *planned* net inspection-option advantage.
- Realised net EUR difference, its strict-loss fraction and a 95% percentile
  interval from paired whole-lot bootstrap resampling.
- Mean retained and excluded rate-node counts, making the numerical state
  budget visible.

The bootstrap draws whole latent lots, reusing the same resampled IDs for
every budget and grid. It preserves dependence among policies evaluated on
the same fruit. No bootstrap interval covers a hypothetical continuous-grid
answer: it describes variation across synthetic lots **relative to the finite
9×9 policy**. The 9×9 self-comparison is zero by construction.

## Interpretation boundaries

The 9×9 grid is only the largest rate grid supported by the prototype.
With three survival hypotheses it produces up to 243 joint states, near the
explicit 250-state limit. It is not a gold standard for integration error.
Equal-weight grid nodes and truncation at `Se + Sp = 1` can cause
non-monotone changes as the number of nodes grows. Agreement at the examined
resolutions offers a **limited robustness diagnostic**, not proof of
convergence, an error bound or a deployment-ready rule.

The generating grading rates are uniform on `[0.75, 0.95]`, while the
decision's weak Beta priors have a different distribution. Reference labels
are assumed accurate and stable across cohorts and days. The Weibull
survival library, prices, independence of inspections and nondestructive
checks remain assumptions. Real operations need independently audited data,
grader-shift checks, measured losses and prospective out-of-lot validation.

## Reproduce

From `projects/orange_storage_decision` (base R only):

```bash
Rscript tests/test_adaptive_grid_resolution.R
Rscript scripts/run_adaptive_grid_resolution.R 60
```

The report script writes `records.csv`, `by_grid.csv`, `settings.R` and
`RESULTS.md` under `results/adaptive_grid_resolution/`, or a supplied
output directory. Its default sample is illustrative, not an automated
acceptance threshold. GitHub Actions runs the R test and a small report
smoke test; generated results are not committed.
