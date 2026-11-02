# Adaptive inspection when quality labels are imperfect

## Question

If a grader sometimes accepts spoiled fruit or rejects sound fruit, should
the day-12 **inspect on day 16 or commit now** decision change? Does using
externally calibrated grading accuracy outperform treating the same reported
inspection labels as ground truth?

This is an extension of [ADAPTIVE_POLICY.md](ADAPTIVE_POLICY.md), which assumes
perfect labels, and [INSPECTION_ERROR.md](INSPECTION_ERROR.md), which corrects a
*fitted* Weibull likelihood rather than a sequential finite-scenario policy.
The survival hypotheses here remain fixed and prespecified.

## Observation model

For survival hypothesis \(h\), true cohort survival at day \(t\) is
\(S_{g,h}(t)\). A reported *acceptable* label has probability

\[
q_{g,h}(t) = \mathrm{Se}\,S_{g,h}(t)
             + (1-\mathrm{Sp})[1-S_{g,h}(t)],
\]

where \(\mathrm{Se}=P(\text{accept label}\mid\text{truly acceptable})\) and
\(\mathrm{Sp}=P(\text{reject label}\mid\text{truly unacceptable})\). For
independent representative checks, the observed acceptable count has the
distribution \(Y_{g,t}\mid h\sim\mathrm{Binomial}(n_{g,t},q_{g,h}(t))\).

The posterior over hypotheses multiplies these noisy-label likelihoods for
**observations available by the decision cutoff**. The day-16 value of
information enumerates possible reported counts using the same \(q\), not
\(S\), then optimises future sale dates under each resulting posterior.
The expected sale revenue still uses **true commercial survival** \(S\) at
delivery, never noisy reported acceptance \(q\). Inspection labour is charged
once when inspection is ordered. For \(\mathrm{Se}=\mathrm{Sp}=1\), the
original perfect-label policy is recovered exactly.

The policy requires \(0\leq\mathrm{Se},\mathrm{Sp}\leq1\) and
\(\mathrm{Se}+\mathrm{Sp}>1\). This last constraint fixes the known label
orientation and excludes a useless or systematically inverted grader.
Sensitivity/specificity are assumed externally calibrated and fixed across
cohorts and days. Fitting both rates from these same target-lot observations
would generally confound grading error with deterioration.

## Paired synthetic evaluation

For each independent latent lot:

1. Draw initial grade masses, a true survival hypothesis from the fixed
   `short/typical/long` prior, and a common synthetic price path. The true
   hypothesis stays hidden from both policies.
2. Draw independent uniforms for **true quality** and **grading error** at
   days 6 and 12. Convert those to aggregate reported labels under each
   grading regime. The policies see counts, not individual truth labels.
3. Independently freeze two day-12 policies for each regime:
   - **Calibrated adaptive:** use the specified Se/Sp in both Bayesian
     updating and exact day-16 inspection-value enumeration.
   - **Naive adaptive:** observe exactly the same reported counts but assume
     Se=Sp=1 in both places.
4. Only after all policies have committed, draw day-16 potential inspection
   labels and hidden fruit lifetimes. Use the same uniforms and fruit
   outcomes for both policies and across all grading regimes. Inspecting
   policies may use the day-16 aggregate count to choose a subsequent date;
   a committed policy cannot. Both score the same actual fruit and prices.
5. Retain inspection labour, negative differences and zero differences.
   Compare calibrated adaptive against naive adaptive, its own optimal
   day-12 fixed commitment, and a hypothetical always-inspect rule.

Prespecified accuracy regimes:

| Regime | Sensitivity | Specificity |
|---|---:|---:|
| Perfect control | 1.00 | 1.00 |
| Moderate error | 0.90 | 0.90 |
| Noisy grader | 0.75 | 0.80 |

The price/survival assumptions, two saleable grades, day-12 decision, day-16
optional inspection and day-28 horizon match the adaptive-policy prototype.
Checks are nondestructive and independently sampled. The latent truth is
*inside* the fixed survival library; unlike
[ADAPTIVE_STRESS.md](ADAPTIVE_STRESS.md), this experiment changes grader
accuracy, not shelf life.

## Estimands and uncertainty

The primary paired contrast is
\(\text{calibrated adaptive EUR}-\text{naive adaptive EUR}\), per latent lot
under each grading regime. A second contrast compares calibrated adaptive to
its own day-12 fixed alternative; this isolates the extra inspection option
under each accuracy level. The across-regime contrast subtracts that second
premium under perfect labels on the **same latent lot**. Whole-lot percentile
bootstrap intervals resample the same lot IDs jointly across regimes and
policies. They quantify Monte Carlo variation, not a field-performance claim.

This setup cannot establish whether the supplied confusion rates are accurate,
whether errors are correlated across graders or fruits, or whether collecting
an inspection changes the commercial batch. Price forecasts, shelf-life model
misspecification, destructive checks and uncertain reference-label calibration
remain out of scope. A real trial needs independent grading audits, observed
commercial outcomes and predeclared out-of-lot testing.

## Reproduce

From `projects/orange_storage_decision` (base R only):

```bash
Rscript tests/test_adaptive_label_error.R
Rscript scripts/run_adaptive_label_error.R 200
```

The script writes `records.csv`, `by_regime.csv`, `contrasts.csv`,
`settings.R` and `RESULTS.md` to `results/adaptive_label_error/` (or a supplied
output directory). Generated results are not checked in; the test suite and
report command run in GitHub Actions with a small smoke-test sample.
