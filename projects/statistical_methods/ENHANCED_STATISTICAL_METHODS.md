# Statistical Methods Reference Notes

## Overview

This document summarizes selected statistical-method examples in this portfolio,
including Bayesian experimentation, network effects, time-dependent testing,
meta-analysis, power simulations, causal inference, bandits, and validation
checks.

## Table of Contents

1. [Bayesian A/B Testing](#bayesian-ab-testing)
2. [Network Effect Corrections](#network-effect-corrections)
3. [Time-Dependent Testing](#time-dependent-testing)
4. [Meta-Analysis Tools](#meta-analysis-tools)
5. [Power Analysis Simulations](#power-analysis-simulations)
6. [Causal Inference Methods](#causal-inference-methods)
7. [Multi-Armed Bandits](#multi-armed-bandits)
8. [Statistical Validation Suite](#statistical-validation-suite)

## Bayesian A/B Testing

### Theoretical Foundation

Bayesian A/B testing uses Bayes' theorem to update prior beliefs with observed data:

```
P(θ|data) ∝ P(data|θ) × P(θ)
```

Where:
- `P(θ|data)` is the posterior distribution
- `P(data|θ)` is the likelihood
- `P(θ)` is the prior distribution

### Key Advantages

1. **Probabilistic Interpretation**: Direct probability that variant B is better than A
2. **Early Stopping**: Can stop as soon as sufficient evidence is gathered
3. **Uncertainty Quantification**: Full posterior distributions provide uncertainty estimates
4. **Small Sample Performance**: Better performance with small samples than frequentist methods

### Implementation Features

#### 1. Basic Bayesian Testing
- Beta-Binomial conjugate prior for conversion rates
- BEST (Bayesian Estimation Supersedes t-test) for continuous metrics
- Region of Practical Equivalence (ROPE) for effect size assessment

#### 2. Multivariate Testing
- Joint posterior distributions for multiple metrics
- Weighted composite scores
- Correlation handling between metrics

#### 3. Hierarchical Models
- Segment-level effects with population-level shrinkage
- Partial pooling for improved estimates
- Heterogeneity assessment

## Network Effect Corrections

### Theoretical Foundation

Network effects violate the Stable Unit Treatment Value Assumption (SUTVA):

```
Y_i(treatment) ≠ f(Z_i) but Y_i(treatment) = f(Z_i, Z_neighbors)
```

### Methodology

#### 1. Direct and Indirect Effects
- **Direct Effect**: Treatment effect on treated units
- **Indirect Effect**: Spillover effect through network connections
- **Total Effect**: Combined direct and indirect effects

#### 2. Estimation Approach
```python
Total Effect = Direct Effect + Spillover × Network Exposure
```

Where Network Exposure = proportion of treated neighbors

#### 3. Key Metrics
- **Spillover Ratio**: Indirect Effect / Total Effect
- **Clustering Coefficient**: Network connectivity measure
- **Degree Distribution**: Network structure characterization

### Applications
- Social network experiments
- Marketplace platforms
- Viral features
- Referral programs

## Time-Dependent Testing

### Theoretical Foundation

Time-dependent effects model treatment effects that vary over time:

```
τ(t) = τ_0 × f(t)
```

Where `f(t)` captures temporal patterns:
- Decay: `f(t) = exp(-λt)`
- Growth: `f(t) = 1 - exp(-λt)`
- Seasonal: `f(t) = sin(ωt + φ)`

### Implementation Features

#### 1. Temporal Pattern Detection
- Exponential decay estimation
- Trend component extraction
- Seasonality detection via autocorrelation
- Peak effect identification

#### 2. Gaussian Process Smoothing
- Non-parametric effect estimation
- Confidence bands over time
- Automatic hyperparameter optimization

#### 3. Event Study Analysis
- Pre/post treatment effects
- Dynamic treatment effects
- Parallel trends testing

## Meta-Analysis Tools

### Theoretical Foundation

Meta-analysis combines results from multiple studies:

#### Fixed Effects Model
```
θ̂ = Σ(w_i × θ_i) / Σw_i
```
Where `w_i = 1/σ_i²` (inverse variance weighting)

#### Random Effects Model
```
θ̂ = Σ(w_i* × θ_i) / Σw_i*
```
Where `w_i* = 1/(σ_i² + τ²)` and `τ²` is between-study variance

### Key Features

#### 1. Heterogeneity Assessment
- **Q-statistic**: Tests for heterogeneity
- **I² statistic**: Proportion of variance due to heterogeneity
- **τ²**: Between-study variance estimate

#### 2. Forest Plots
- Individual study effects with confidence intervals
- Pooled effect estimate
- Visual heterogeneity assessment

#### 3. Publication Bias Detection
- Funnel plots
- Egger's test
- Trim and fill method

## Power Analysis Simulations

### Theoretical Foundation

Statistical power is the probability of detecting a true effect:

```
Power = P(Reject H₀ | H₁ is true) = 1 - β
```

### Implementation Features

#### 1. Analytical Power Calculations
- Closed-form solutions for standard tests
- Effect size calculations (Cohen's d, h, f²)
- Sample size determination

#### 2. Simulation-Based Power
- Complex experimental designs
- Non-standard test statistics
- Multiple testing scenarios

#### 3. Sequential Testing
- Alpha spending functions (O'Brien-Fleming, Pocock)
- Conditional power for futility assessment
- Expected sample size reduction

#### 4. Adaptive Designs
- Thompson sampling for arm selection
- Response-adaptive randomization
- Convergence monitoring

### Key Formulas

#### Sample Size for Proportions
```
n = [(z_α + z_β)² × (p₁(1-p₁) + p₂(1-p₂))] / (p₁ - p₂)²
```

#### Minimum Detectable Effect
```
MDE = (z_α + z_β) × √[2p̄(1-p̄)/n]
```

## Causal Inference Methods

### 1. Instrumental Variables (IV)

#### Theory
Identifies causal effects when treatment is endogenous:
```
Y = βX + ε, where Cov(X,ε) ≠ 0
```

Use instrument Z where:
- **Relevance**: Cov(Z,X) ≠ 0
- **Exclusion**: Cov(Z,ε) = 0

#### Two-Stage Least Squares (2SLS)
1. First stage: X̂ = γZ + η
2. Second stage: Y = βX̂ + ν

### 2. Difference-in-Differences (DiD)

#### Estimator
```
τ_DiD = (Ȳ_treat,post - Ȳ_treat,pre) - (Ȳ_control,post - Ȳ_control,pre)
```

#### Key Assumption
Parallel trends: Treatment and control would have evolved similarly without treatment

### 3. Regression Discontinuity (RDD)

#### Sharp RDD
```
τ_RD = lim[E[Y|X=c⁺] - E[Y|X=c⁻]]
```

#### Fuzzy RDD
Uses discontinuity as instrument for treatment uptake

### 4. Propensity Score Methods

#### Matching
Find control units with similar propensity scores:
```
e(X) = P(Treatment = 1|X)
```

#### Inverse Propensity Weighting (IPW)
```
τ_IPW = E[Y×T/e(X)] - E[Y×(1-T)/(1-e(X))]
```

### 5. Synthetic Control

Creates weighted combination of control units to match treated unit:
```
W* = argmin ||X₁ - X₀W||
```

## Multi-Armed Bandits

### Theoretical Foundation

Balance exploration vs exploitation to maximize cumulative reward:

```
Regret = T×μ* - Σ μ_a(t)
```

### Algorithms Implemented

#### 1. ε-Greedy
- Explore with probability ε
- Exploit best arm with probability 1-ε
- Decaying ε over time

#### 2. Thompson Sampling
- Sample from posterior distributions
- Select arm with highest sample
- Naturally balances exploration/exploitation

#### 3. Upper Confidence Bound (UCB)
```
UCB_i = μ̂_i + c×√(2ln(t)/n_i)
```

#### 4. LinUCB (Contextual)
```
a_t = argmax[θ̂ᵀx + α√(xᵀA⁻¹x)]
```

### Applications
- Content recommendation
- Ad placement optimization
- Clinical trials
- Feature rollouts

## Statistical Validation Suite

### Validation Approaches

#### 1. Comparison with Established Libraries
- statsmodels
- scipy.stats
- R packages via rpy2

#### 2. Known Test Cases
- Standard normal quantiles
- Chi-square critical values
- t-distribution properties

#### 3. Monte Carlo Validation
- Central Limit Theorem convergence
- Bootstrap consistency
- Permutation test calibration

#### 4. Theoretical Property Validation
- **Unbiasedness**: E[θ̂] = θ
- **Consistency**: θ̂ →ᵖ θ as n→∞
- **Efficiency**: Minimum variance among unbiased estimators

### Validation Metrics

1. **Absolute Error**: |θ̂ - θ_ref|
2. **Relative Error**: |θ̂ - θ_ref|/|θ_ref|
3. **Coverage**: P(θ ∈ CI) = 1-α
4. **Type I Error**: P(Reject H₀|H₀ true) = α
5. **Power**: P(Reject H₀|H₁ true) = 1-β

## Sensitivity Analysis

### Framework

Assess robustness to assumption violations:

```
θ̂_adjusted = θ̂_base + Σ δ_i × violation_i
```

### Key Analyses

#### 1. Missing Data Bounds
- **Lee Bounds**: Trimming for differential attrition
- **Manski Bounds**: Worst-case bounds
- **Horowitz-Manski**: Tighter worst-case bounds

#### 2. Unmeasured Confounding
- Rosenbaum bounds for hidden bias
- E-value for minimum confounding strength
- Sensitivity to unobserved covariates

#### 3. Model Specification
- Functional form sensitivity
- Distributional assumptions
- Outlier influence

## Best Practices

### 1. Experiment Design
- [ ] Calculate required sample size upfront
- [ ] Plan for multiple testing if needed
- [ ] Consider sequential/adaptive designs
- [ ] Account for network effects if applicable
- [ ] Document assumptions clearly

### 2. Analysis
- [ ] Check data quality and SRM
- [ ] Validate statistical assumptions
- [ ] Use appropriate test for data type
- [ ] Report uncertainty (CI, credible intervals)
- [ ] Conduct sensitivity analysis

### 3. Interpretation
- [ ] Distinguish statistical from practical significance
- [ ] Consider multiple comparisons
- [ ] Report effect sizes, not just p-values
- [ ] Acknowledge limitations
- [ ] Provide actionable recommendations

## Implementation Examples

### Example 1: Network Effect Corrected A/B Test

```python
from enhanced_bayesian_testing import NetworkEffectBayesianTest

# Initialize tester
network_test = NetworkEffectBayesianTest()

# Run test with network data
result = network_test.test_with_network_effects(
    data=experiment_df,
    outcome_col='converted',
    treatment_col='treatment',
    user_col='user_id',
    network_edges=edge_list
)

print(f"Direct effect: {result.direct_effect:.3f}")
print(f"Spillover effect: {result.indirect_effect:.3f}")
print(f"Total effect: {result.total_effect:.3f}")
```

### Example 2: Time-Dependent Analysis

```python
from enhanced_bayesian_testing import TimeDependentBayesianTest

# Initialize tester
time_test = TimeDependentBayesianTest()

# Analyze temporal patterns
result = time_test.test_with_temporal_effects(
    data=time_series_df,
    outcome_col='revenue',
    treatment_col='treatment',
    time_col='date',
    granularity='daily'
)

# Visualize results
time_test.plot_temporal_results(result)
```

### Example 3: Power Analysis for Sequential Testing

```python
from power_analysis_simulations import PowerAnalysisSimulator

# Initialize simulator
simulator = PowerAnalysisSimulator(n_simulations=10000)

# Simulate sequential testing
result = simulator.simulate_sequential_testing(
    true_effect=0.02,
    max_sample_size=5000,
    check_points=[1000, 2000, 3000, 4000, 5000],
    spending_function='obrien_fleming'
)

print(f"Power: {result.observed_power:.3f}")
print(f"Early stopping rate: {result.early_stopping_rate:.3f}")
```

## References

1. **Bayesian Methods**
   - Gelman, A., et al. (2013). Bayesian Data Analysis, 3rd Edition.
   - Kruschke, J. (2014). Doing Bayesian Data Analysis, 2nd Edition.

2. **Causal Inference**
   - Pearl, J. (2009). Causality: Models, Reasoning, and Inference.
   - Imbens, G. & Rubin, D. (2015). Causal Inference for Statistics, Social, and Biomedical Sciences.

3. **Network Effects**
   - Aronow, P. & Samii, C. (2017). "Estimating Average Causal Effects Under General Interference."
   - Ugander, J., et al. (2013). "Graph Cluster Randomization."

4. **Sequential Testing**
   - Jennison, C. & Turnbull, B. (1999). Group Sequential Methods.
   - Proschan, M., et al. (2006). Statistical Monitoring of Clinical Trials.

5. **Multi-Armed Bandits**
   - Lattimore, T. & Szepesvári, C. (2020). Bandit Algorithms.
   - Slivkins, A. (2019). "Introduction to Multi-Armed Bandits."

## Closing Notes

These examples show several approaches to experimentation and causal analysis:

1. Bayesian A/B testing and posterior decision summaries.
2. Network and temporal treatment-effect examples.
3. Meta-analysis, power simulation, and adaptive-design utilities.
4. Causal inference implementations for common observational-study designs.
5. Bandit algorithms and statistical validation helpers.

Before reusing the code, verify assumptions, sample-size requirements, and
diagnostics against the specific experiment or dataset.
