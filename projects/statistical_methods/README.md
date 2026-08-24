# Statistical Methods Examples

This folder collects statistical-method examples used in the portfolio. The
modules cover Bayesian A/B testing, non-parametric tests, multiple-testing
corrections, bootstrap methods, power analysis, causal inference, bandits, and
validation helpers. The code is reference material for review and local
experimentation rather than a maintained standalone package.

## Contents

- `bayesian_ab_testing.py`: conversion and continuous-metric Bayesian tests,
  multivariate testing helpers, hierarchical examples, and simulation utilities.
- `advanced_statistical_tests.py`: non-parametric tests, multiple-testing
  corrections, bootstrap intervals/tests, power calculations, and effect-size
  helpers.
- `causal_inference.py`: instrumental variables, difference-in-differences,
  regression discontinuity, propensity score matching, and synthetic control
  examples.
- `multi_armed_bandits.py`: epsilon-greedy, Thompson sampling, UCB, LinUCB,
  simulation, and dynamic-bandit examples.
- `enhanced_bayesian_testing.py`: network-effect, time-dependent, and
  meta-analysis examples.
- `power_analysis_simulations.py`: simulation-based power and sensitivity
  analysis helpers.
- `statistical_validation_suite.py` and
  `enhanced_statistical_validation.py`: validation checks and comparison
  helpers.
- `ENHANCED_STATISTICAL_METHODS.md` and
  `STATISTICAL_METHODS_DOCUMENTATION.md`: longer notes retained as reference
  documentation.

## Quick Start

Install the core dependencies used across the examples:

```bash
pip install numpy scipy pandas statsmodels scikit-learn matplotlib seaborn
```

Run examples from this folder, or add `projects/statistical_methods` to
`PYTHONPATH` before importing the modules.

```python
from bayesian_ab_testing import BayesianABTesting

tester = BayesianABTesting()
result = tester.test_conversion(
    conversions_a=120,
    visitors_a=1000,
    conversions_b=150,
    visitors_b=1000,
)

print(result.probability_b_better)
```

## Review Notes

- Use these modules as examples of statistical reasoning and implementation
  patterns, not as a drop-in decision system.
- Several files include demonstrations under `if __name__ == "__main__"` for
  quick local inspection.
- Before adapting any method, check its assumptions, sample-size requirements,
  and validation path for the specific dataset or experiment design.
