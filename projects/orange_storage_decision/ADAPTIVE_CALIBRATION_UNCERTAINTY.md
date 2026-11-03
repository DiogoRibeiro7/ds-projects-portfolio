# Adaptive inspection under finite grading-calibration uncertainty

## Question

The [adaptive label-error model](ADAPTIVE_LABEL_ERROR.md) treats grading
sensitivity and specificity as known constants. In practice, these rates are
estimated from a finite, independently audited reference sample. Does the
inspect-or-commit policy change when uncertainty in those estimates is
integrated into its decisions instead of replacing them with single values?

The existing [calibration bootstrap](CALIBRATION.md) studies how finite
reference samples affect fitted Weibull decisions. This study instead follows
the **joint Bayesian uncertainty in commercial survival hypotheses and
grading rates** through the one-step adaptive decision. The two methods answer
different questions and should not be interpreted as equivalent intervals.

## Reference likelihood and finite rate grid

An independent, reference-standard audit records true positives (TP), false
negatives (FN), true negatives (TN) and false positives (FP). The reference
truth is assumed accurate. The grader sensitivity is the probability of a
reported accept among truly acceptable reference fruit; specificity is the
probability of a reported reject among truly unacceptable reference fruit.

Begin with independent \(\mathrm{Beta}(1,1)\) priors over sensitivity
\(\mathrm{Se}\) and specificity \(\mathrm{Sp}\). After auditing, the
untruncated posteriors factor as

\[
\begin{aligned}
\mathrm{Se}\mid D &\sim \mathrm{Beta}(1+\mathrm{TP},1+\mathrm{FN}),\\
\mathrm{Sp}\mid D &\sim \mathrm{Beta}(1+\mathrm{TN},1+\mathrm{FP}).
\end{aligned}
\]

As in the label-error model, restrict the grader orientation to
\(\mathrm{Se}+\mathrm{Sp}>1\). The model is a joint prior truncated to this
informative region, updated by the audit; a finite audit with poor empirical
rates is **not discarded**. Instead, each marginal Beta posterior is sampled
at deterministic midpoint quantiles \((i-\tfrac12)/m\), \(i=1,\ldots,m\),
and the Cartesian grid is conditioned on the informative region. The retained
nodes receive equal normalised weights. This is a small, transparent
**quadrature approximation**, not an exact continuous posterior. The workflow
records how many nodes were excluded. Default \(m=5\) gives at most 25 grader
states and, with three survival hypotheses, at most 75 joint latent states.

## Joint posterior and decision

Pair each grader-rate node \(r\) with each prespecified survival hypothesis
\(h\). Its prior weight is \(P(h)P(r\mid D)\), where \(D\) includes only
reference audits available by the day-12 decision. For each target-lot
inspection count \(Y_{g,t}\) observed at or before day 12, update the *joint*
state likelihood using

\[
Y_{g,t}\mid h,r\sim\mathrm{Binomial}\left(n_{g,t},
  \mathrm{Se}_r S_{g,h}(t) +
  (1-\mathrm{Sp}_r)[1-S_{g,h}(t)]\right).
\]

This update creates posterior dependence between grader rates and survival
hypotheses. On day 12, evaluate a fixed sale date against paying for one
optional nondestructive day-16 inspection. The latter enumerates future
reported counts under **every joint state**, updates the same posterior once
and selects a day-16-or-later sale. Both alternatives value saleable mass using
the *true* Weibull survival curve of \(h\), never the label probability.
Labour is charged only when inspection occurs.

The comparison uses the **same audit** and target observations but plugs in
the rate grid's weighted mean Se and Sp before updating the survival prior.
This is a posterior-mean plug-in, not a maximum-likelihood estimator. A single
known survival curve cannot acquire economic information value merely because
the grader's accuracy remains uncertain.

## Prospective paired experiment

For each synthetic lot:

1. Draw a latent `short/typical/long` survival scenario, grade masses and
   price path. Draw hidden grader sensitivity and specificity independently
   and uniformly in \([0.75,0.95]\) for simulation; neither rate is passed to
   the policies. This truth-generating distribution is **not** the weak
   Beta(1,1) policy prior.
2. Independently draw reference-grade outcomes for 80 known-acceptable and
   80 known-unacceptable fruit; the *scarce* audit uses the first 8 of each
   and the *ample* audit uses all 80. Audits are independent of the target lot
   and recorded on day 10.
3. Draw common day-6/12 latent-quality and grading uniforms for independent,
   nondestructive target-lot inspections. Both policies and audit budgets see
   the same aggregate target-lot labels, without knowing the hidden truth.
4. Freeze the integrated and plug-in day-12 plans for **both** audit budgets.
   Only afterward generate potential day-16 labels and fruit lifetimes,
   shared across all alternatives. Committed policies cannot revise dates.
5. Score both adaptive policies, the integrated policy's best fixed day-12
   commitment, and an always-inspect benchmark under the same fruit outcomes
   and prices. Include inspection labour exactly once where applicable.

The primary contrast is integrated-minus-plug-in realised net EUR within
each reference budget. A second contrast compares integrated adaptive to
its own fixed commitment. A third contrasts the integrated-minus-plug-in
premium under ample versus scarce audit samples on **the same latent lot**.
Use the same whole-lot bootstrap indices for every contrast. Report losses
and negative intervals as observed, without claiming that marginalisation
must improve every finite-sample realised result.

## Boundaries

- The finite midpoint grid conditions on the informative-grader orientation.
  Its resolution can change results; convergence to a continuous posterior is
  not demonstrated by the default smoke test.
- True grading rates vary by lot in the simulation, but within a lot are
  constant across cohorts and days. Reference labels are assumed correct;
  calibration bias and reference-test error are outside scope.
- Inspections are independent, representative and nondestructive; fruit
  quality outcomes used for scoring are held out. Future price forecasts and
  the survival scenario library are synthetic and treated as known.
- The experiment is **not** empirical proof of benefit or calibrated interval
  coverage. Real deployment requires external reference-grade audits,
  grader-shift validation and out-of-lot economic testing.

## Reproduce

From `projects/orange_storage_decision` (base R only):

```bash
Rscript tests/test_adaptive_calibration_uncertainty.R
Rscript scripts/run_adaptive_calibration_uncertainty.R 200
```

The script writes `records.csv`, `by_budget.csv`, `contrast.csv`,
`settings.R` and `RESULTS.md` under `results/adaptive_calibration_uncertainty/`
unless another output directory is supplied. Generated reports are not
committed; GitHub Actions runs the test and a small report smoke test.
