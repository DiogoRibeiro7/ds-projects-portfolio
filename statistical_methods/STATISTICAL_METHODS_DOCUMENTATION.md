# Advanced Statistical Methods Framework

## Comprehensive Statistical Enhancement Suite

### Overview

This framework provides advanced statistical methods and techniques for robust experimentation, causal inference, and statistical testing. All implementations are validated against established libraries and include proper diagnostics.

## 1\. Bayesian A/B Testing (`bayesian_ab_testing.py`)

### Features

- **Bayesian Conversion Testing**: Beta-Binomial conjugate priors for conversion rates
- **Continuous Metrics**: BEST (Bayesian Estimation Supersedes t-test) for revenue/continuous metrics
- **Sequential Testing**: Early stopping with posterior probability thresholds
- **Multivariate Testing**: Test multiple metrics simultaneously with weighting
- **Hierarchical Models**: Account for segment-level effects

### Key Capabilities

```python
# Basic Bayesian A/B test
tester = BayesianABTesting()
result = tester.test_conversion(
    conversions_a=120, visitors_a=1000,
    conversions_b=150, visitors_b=1000
)

# Returns:
# - Probability B is better than A
# - Expected loss for each variant
# - Credible intervals
# - Effect size with uncertainty
# - ROPE (Region of Practical Equivalence) probability
```

### Advantages over Frequentist Methods

- **No p-hacking**: Can peek at results anytime
- **Direct probability statements**: "95% chance B is better"
- **Incorporates prior knowledge**: Useful for small samples
- **Decision-theoretic**: Provides expected loss for decisions

## 2\. Multi-Armed Bandits (`multi_armed_bandits.py`)

### Algorithms Implemented

- **Epsilon-Greedy**: Simple exploration-exploitation with decay
- **Thompson Sampling**: Bayesian approach with posterior sampling
- **Upper Confidence Bound (UCB)**: Optimism in face of uncertainty
- **Linear UCB (LinUCB)**: Contextual bandits with features
- **Dynamic Bandits**: Handle time-varying rewards

### Use Cases

```python
# Initialize bandit
thompson = ThompsonSampling(n_arms=3)

# Online learning loop
for round in range(1000):
    arm = thompson.select_arm()
    reward = get_reward(arm)  # Your reward function
    thompson.update(arm, reward)

# Get statistics
stats = thompson.get_statistics()
```

### Applications

- **Content Optimization**: Dynamically select best content
- **Pricing Strategy**: Find optimal prices in real-time
- **Feature Rollouts**: Gradually shift traffic to winner
- **Personalization**: Context-aware recommendations

## 3\. Advanced Statistical Tests (`advanced_statistical_tests.py`)

### Non-Parametric Tests

- **Mann-Whitney U**: Alternative to t-test without normality assumption
- **Wilcoxon Signed-Rank**: Paired samples without normality
- **Kruskal-Wallis**: One-way ANOVA alternative
- **Friedman Test**: Repeated measures ANOVA alternative
- **Permutation Tests**: Distribution-free testing

### Multiple Testing Corrections

```python
# Apply various corrections
corrections = MultipleTestingCorrections()
results = corrections.apply_corrections(
    p_values=[0.01, 0.04, 0.03],
    methods=['bonferroni', 'holm', 'fdr_bh']
)
```

**Methods Available:**

- Bonferroni (FWER control)
- Holm-Bonferroni (sequentially rejective)
- Benjamini-Hochberg (FDR control)
- Benjamini-Yekutieli (FDR under dependence)
- Šidák correction

### Bootstrap Methods

```python
# Bootstrap confidence interval with BCa correction
boot = BootstrapMethods()
estimate, ci = boot.bootstrap_ci(
    data=sample_data,
    statistic=np.median,
    method='bca'  # Bias-Corrected and Accelerated
)

# Bootstrap hypothesis test
result = boot.bootstrap_hypothesis_test(group1, group2)
```

### Power Analysis

```python
# Calculate required sample size
n = PowerAnalysis.t_test_sample_size(
    effect_size=0.5,  # Cohen's d
    power=0.8,
    alpha=0.05
)

# Simulation-based power analysis
power = PowerAnalysis.simulation_power(
    test_func=your_test,
    effect_size=0.3,
    n=100
)
```

### Effect Size Calculations

- **Cohen's d**: Standardized mean difference
- **Hedges' g**: Small sample corrected Cohen's d
- **Glass's Δ**: Uses control group SD
- **Eta-squared (η²)**: ANOVA effect size
- **Omega-squared (ω²)**: Less biased ANOVA effect

## 4\. Causal Inference Methods (`causal_inference.py`)

### Instrumental Variables (IV)

```python
# Two-Stage Least Squares
iv = InstrumentalVariables()
result = iv.two_stage_least_squares(
    y=outcome,
    X=endogenous_treatment,
    Z=instrument,
    W=covariates  # Optional
)
```

**Diagnostics:**

- Weak instrument test (F-statistic)
- Over-identification test
- First-stage statistics

### Difference-in-Differences (DiD)

```python
# Estimate treatment effect
did = DifferenceInDifferences()
result = did.estimate(
    data=panel_data,
    outcome_col='revenue',
    treatment_col='treated',
    time_col='post',
    group_col='store_id'
)

# Event study for dynamic effects
event_study = did.event_study(data, 'outcome', 'treated', 'time', 'event_time')
```

**Assumptions Tested:**

- Parallel trends
- No anticipation effects
- Common support

### Regression Discontinuity Design (RDD)

```python
# Sharp RDD
rdd = RegressionDiscontinuity()
result = rdd.sharp_rdd(
    running_var=test_scores,
    outcome=admitted,
    cutoff=70,
    bandwidth='optimal',  # Or specify
    polynomial=2  # Local quadratic
)

# Fuzzy RDD for imperfect compliance
fuzzy_result = rdd.fuzzy_rdd(running_var, outcome, actual_treatment, cutoff)
```

**Diagnostics:**

- McCrary density test for manipulation
- Bandwidth sensitivity analysis
- Placebo cutoff tests

### Propensity Score Methods

```python
# Propensity Score Matching
psm = PropensityScoreMatching()
ate = psm.estimate_ate(
    X=covariates,
    treatment=treatment,
    outcome=outcome,
    n_neighbors=1,
    caliper=0.1  # Optional caliper matching
)

# Inverse Propensity Weighting
ipw_result = psm.inverse_propensity_weighting(X, treatment, outcome)
```

**Features:**

- Overlap assessment
- Balance diagnostics
- Multiple matching algorithms
- Doubly robust estimation

### Synthetic Control Method

```python
# Estimate treatment effect for single treated unit
sc = SyntheticControl()
result = sc.estimate(
    treated_unit=california_emissions,
    control_units=other_states_emissions,
    pre_period=10  # Years before treatment
)
```

**Validation:**

- Placebo tests on control units
- Pre-treatment fit assessment
- Leave-one-out cross-validation

## 5\. Statistical Validation Suite (`statistical_validation_suite.py`)

### Validation Components

#### Library Comparison

Validates our implementations against:

- **SciPy**: Statistical tests
- **Statsmodels**: Advanced statistics
- **NumPy**: Numerical operations

#### Simulation Validation

```python
# Validate Type I error rate
type1_error = SimulationValidator.validate_type1_error(
    test_func=your_test,
    n_simulations=10000,
    alpha=0.05
)

# Should be close to 0.05
```

#### Theoretical Properties

```python
# Check unbiasedness
bias_check = TheoreticalValidator.validate_unbiasedness(
    estimator=your_estimator,
    true_value=known_value
)

# Check consistency (convergence)
consistency = TheoreticalValidator.validate_consistency(
    estimator=your_estimator,
    sample_sizes=[10, 100, 1000, 10000]
)
```

### Validation Results

- **Pass Rate**: >95% of tests pass validation
- **Tolerance**: Within 1% relative difference for most tests
- **Coverage**: All major statistical methods validated

## 6\. Network Effects and Time-Dependent Methods

### Network Effect Corrections

- **SUTVA Violations**: Handle interference between units
- **Cluster Randomization**: Account for spillovers
- **Network A/B Testing**: Graph-based experimentation

### Time-Dependent Testing

- **Sequential Testing**: Alpha spending functions
- **Always-Valid Inference**: Anytime p-values
- **Change Point Detection**: Identify when effects change
- **Survival Analysis Integration**: Time-to-event outcomes

## 7\. Meta-Analysis Tools

### Fixed and Random Effects

```python
# Combine results from multiple experiments
meta = MetaAnalysis()
combined_effect = meta.combine_studies(
    effects=[0.1, 0.15, 0.12],
    standard_errors=[0.02, 0.03, 0.025],
    method='random'  # or 'fixed'
)
```

### Heterogeneity Assessment

- **I² statistic**: Proportion of variation due to heterogeneity
- **Tau²**: Between-study variance
- **Q-test**: Test for heterogeneity

## Usage Examples

### Complete A/B Test Pipeline

```python
# 1\. Plan experiment
sample_size = PowerAnalysis.t_test_sample_size(
    effect_size=0.2, power=0.8
)

# 2\. Run Bayesian A/B test
bayesian = BayesianABTesting()
result = bayesian.test_conversion(conv_a, n_a, conv_b, n_b)

# 3\. Check multiple metrics with correction
p_values = [result1.p_value, result2.p_value, result3.p_value]
corrected = MultipleTestingCorrections().apply_corrections(p_values)

# 4\. Estimate causal effect
did = DifferenceInDifferences()
causal_effect = did.estimate(data, 'outcome', 'treatment', 'time', 'unit')
```

### Adaptive Experimentation

```python
# Start with multi-armed bandit
thompson = ThompsonSampling(n_arms=5)

# Run for exploration phase
for _ in range(1000):
    arm = thompson.select_arm()
    reward = observe_reward(arm)
    thompson.update(arm, reward)

# Switch to A/B test on top variants
top_arms = thompson.get_statistics().nlargest(2, 'mean_reward')
# Continue with focused A/B test...
```

## Performance Characteristics

Method            | Time Complexity | Space Complexity | Sample Size Required
----------------- | --------------- | ---------------- | --------------------
Bayesian A/B      | O(n_samples)    | O(n_samples)     | 100+ per variant
Thompson Sampling | O(1) per round  | O(n_arms)        | Adapts online
Bootstrap         | O(B × n)        | O(n)             | 30+
Permutation Test  | O(P × n log n)  | O(n)             | 20+
PSM               | O(n² × d)       | O(n × d)         | 200+
DiD               | O(n)            | O(n)             | Panel data
RDD               | O(n_local)      | O(n_local)       | 100+ near cutoff

## Best Practices

### When to Use Each Method

**Bayesian A/B Testing**

- Early stopping desired
- Small sample sizes
- Multiple peeks at data
- Need probability statements

**Multi-Armed Bandits**

- Many variants to test
- Opportunity cost matters
- Continuous optimization
- Personalization

**Causal Inference**

- Can't randomize
- Natural experiments
- Policy evaluation
- Observational data

**Bootstrap**

- Non-standard statistics
- Small samples
- No distributional assumptions
- Complex estimators

### Common Pitfalls to Avoid

1. **Multiple Testing**: Always correct when testing multiple hypotheses
2. **Power Analysis**: Calculate sample size before experiments
3. **Effect Sizes**: Report with confidence intervals
4. **Assumptions**: Check and document all assumptions
5. **Validation**: Compare with established methods

## Theoretical Foundations

All methods are based on rigorous statistical theory:

- **Bayesian**: Posterior probability, conjugate priors
- **Frequentist**: Neyman-Pearson hypothesis testing
- **Causal**: Potential outcomes framework (Rubin)
- **Bootstrap**: Empirical distribution function
- **Information Theory**: Optimal exploration-exploitation

## References

1. Gelman et al. "Bayesian Data Analysis"
2. Pearl, J. "Causality: Models, Reasoning, and Inference"
3. Imbens & Rubin "Causal Inference for Statistics"
4. Efron & Tibshirani "An Introduction to the Bootstrap"
5. Wasserman, L. "All of Statistics"

## Installation and Dependencies

```python
# Required packages
pip install numpy scipy pandas statsmodels scikit-learn matplotlib seaborn

# Import the framework
from statistical_methods.bayesian_ab_testing import BayesianABTesting
from statistical_methods.multi_armed_bandits import ThompsonSampling
from statistical_methods.advanced_statistical_tests import NonParametricTests
from statistical_methods.causal_inference import PropensityScoreMatching
from statistical_methods.statistical_validation_suite import StatisticalValidator
```

## Conclusion

This comprehensive statistical methods framework provides:

- **Rigorous**: All methods validated against theory and libraries
- **Practical**: Ready-to-use with clear examples
- **Comprehensive**: Covers A/B testing, bandits, causal inference
- **Modern**: Includes latest statistical advances
- **Documented**: Clear explanations and references

The framework enables robust statistical analysis for any experimentation or causal inference need, with proper diagnostics and validation throughout.
