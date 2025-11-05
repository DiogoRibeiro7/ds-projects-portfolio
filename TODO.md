---
- Demonstrate strong **EDA and storytelling** with data.
- Show **solid modeling practice**: 'baselines, evaluation, diagnostics.'
- >-
  Add **at least one advanced angle** per project (causal, uplift, optimization,
  deployment, uncertainty, etc.).
- Keep everything **reproducible**: 'clean repo structure, tests for key utilities, and clear documentation.'
---

# ROADMAP

This roadmap lists project topics to showcase end-to-end data science skills, from exploratory data analysis to advanced modeling, causal inference, optimization, and MLOps.

You do **not** need to do everything. Think of this as a menu. Pick a few projects and push them through all phases:

- **Tier 1 – Data Analysis / EDA**
- **Tier 2 – Predictive / Statistical Modeling**
- **Tier 3 – Advanced Layer (Causal / MLOps / Optimization / Etc.)**

## 1\. Customer Churn & Retention

**Goal:** Show customer analytics + classification + uplift / decision layer.

### Tier 1 – Data Analysis

- Explore churn vs:

  - Tenure
  - Price
  - Engagement metrics
  - Support interactions

- Build simple customer segments:

  - RFM-style or k-means segmentation
  - Describe each segment (size, churn rate, value)

- Plot survival-style curves (Kaplan–Meier-like):

  - Time-to-churn distributions by segment

### Tier 2 – Predictive Modeling

- Build a churn classifier:

  - Logistic regression and at least one tree-based model
  - Calibrated probabilities (Platt, isotonic, etc.)

- Compare:

  - Simple interpretable models vs. more complex models

- Model diagnostics:

  - ROC/PR curves
  - Reliability diagrams
  - SHAP values to explain main churn drivers

### Tier 3 – Advanced / Decision Layer

- Implement **uplift modeling** for retention campaigns:

  - Two-model or meta-learner approach

- Optimize **budget-constrained targeting**:

  - Decide which customers to contact under a fixed budget

- Policy simulation:

  - Estimate expected ROI when targeting top X% by uplift
  - Compare to naive strategies (target by risk only, or random)

--------------------------------------------------------------------------------

## 2\. Dynamic Pricing / Discount Optimization

**Goal:** Show pricing analytics + demand modeling + simple optimization.

### Tier 1 – Analysis

- Analyze relationship between price and demand:

  - Price elasticity per segment or product category

- Study seasonality and time effects:

  - Day-of-week, hour-of-day, holidays

- Basket and cross-selling analysis:

  - Which products are bought together
  - Support and confidence metrics

### Tier 2 – Modeling

- Build demand forecasting models as a function of:

  - Price
  - Seasonality
  - Promotions / campaigns

- Train and compare:

  - Baseline models vs. regularized regressors vs. tree-based models

- Run scenario simulations:

  - Effect on revenue when changing price for key products

### Tier 3 – Optimization / RL

- Formulate a **pricing policy**:

  - Objective: maximize revenue or profit under constraints

- Implement a simple **multi-armed bandit / contextual bandit**:

  - Learn which prices perform better over time

- Evaluate policies offline:

  - Inverse propensity scoring or doubly robust estimators

--------------------------------------------------------------------------------

## 3\. Time Series Forecasting (Energy / Traffic / Finance / Etc.)

**Goal:** Show time-series EDA, forecasting, and uncertainty.

### Tier 1 – Analysis

- Decompose series into:

  - Trend, seasonality, residuals

- Detect anomalies:

  - Rule-based and simple statistical methods

- Cross-correlation analysis:

  - Relationship with external regressors (e.g., temperature vs. energy load)

### Tier 2 – Modeling

- Build standard models:

  - ARIMA/SARIMA/Prophet or similar
  - Regressors with lagged features and exogenous variables

- Use robust evaluation:

  - Rolling-origin cross-validation
  - Error distribution analysis (MAPE, RMSE, etc.)

### Tier 3 – Advanced / Deployment

- Implement **probabilistic forecasts**:

  - Quantile models or conformal prediction intervals

- Champion–challenger setup:

  - Compare current "champion" model vs. new challengers

- Deployment simulation:

  - REST API for real-time forecasts
  - Batch pipeline for daily predictions
  - Monitoring dashboards for drift and error over time

--------------------------------------------------------------------------------

## 4\. Recommender Systems

**Goal:** Show user–item modeling, ranking metrics, and business constraints.

### Tier 1 – Exploration

- Analyze interaction matrix:

  - Sparsity
  - Long-tail popularity

- Cohort analysis:

  - User engagement over time
  - Content type preferences

### Tier 2 – Modeling

- Implement baseline recommenders:

  - Popularity-based
  - Item-based similarity

- Introduce matrix factorization or implicit feedback models:

  - ALS / BPR or similar

- Evaluate ranking quality:

  - HR@k, NDCG@k, coverage

### Tier 3 – Advanced

- Context-aware signals:

  - Time, device, location as context

- Re-ranking to include:

  - Diversity
  - Novelty
  - Simple business rules (e.g. not repeating the same item)

- Simple offline simulation:

  - Compare different recommendation policies

--------------------------------------------------------------------------------

## 5\. Experimentation & Causal Inference

**Goal:** Show you understand A/B tests and causal estimators.

### Tier 1 – A/B Test Analysis

- Use a real or simulated A/B dataset:

  - Compute lift and confidence intervals
  - Assess power and minimal detectable effect

- Pre-experiment checks:

  - Metric variance and seasonality
  - Covariate balance

### Tier 2 – Observational Causal Inference

- Take a panel or observational dataset with interventions:

  - Marketing campaigns, price changes, policy changes

- Apply:

  - Propensity score matching / weighting
  - Difference-in-differences or synthetic control

- Validate assumptions and interpret causal effects

### Tier 3 – Advanced

- Design and analyze **geo-experiments**:

  - Pre-period calibration + post-period effect estimation

- Hierarchical Bayesian models:

  - Pool treatment effects across geos or segments

- Portfolio / meta-analysis:

  - Combine multiple experiments to plan future tests

--------------------------------------------------------------------------------

## 6\. NLP & Text Projects (From Analytics to LLMs)

**Goal:** Show ability to go from classic NLP to modern embedding/LLM usage.

### Tier 1 – Text Analysis

- Preprocess text:

  - Tokenization, cleaning, vocabulary analysis

- Compute:

  - TF–IDF features
  - Basic sentiment scores

- Explore:

  - Frequent n-grams
  - Changes before/after certain events

### Tier 2 – Modeling

- Build text classifiers:

  - E.g. review → rating, ticket → category

- Implement topic models:

  - LDA or NMF to identify latent themes

- Evaluate:

  - Standard metrics and qualitative inspection of outputs

### Tier 3 – Advanced

- Use embeddings (e.g. sentence embeddings) for:

  - Semantic search
  - Clustering of documents or users

- Build a simple RAG-style system:

  - Retrieve relevant documents and summarize them

- Human-in-the-loop evaluation:

  - Inspect failure cases and refine prompts or retrieval strategy

--------------------------------------------------------------------------------

## 7\. Anomaly Detection in Time Series / Sensors

**Goal:** Show ability to detect anomalies and think about streaming.

### Tier 1 – Analysis

- Perform EDA on time-dependent data:

  - Identify extreme values, trends, seasonality

- Implement simple anomaly rules:

  - Thresholds, 3σ rules

- Visualize anomalies:

  - Overlay on time series plots

### Tier 2 – Modeling

- Build statistical models:

  - EWMA, ARIMA with residual analysis

- Train multivariate anomaly detectors:

  - Isolation Forest
  - Local Outlier Factor
  - Robust covariance

- Evaluate using:

  - Labeled anomalies or synthetic anomaly injection

### Tier 3 – Advanced

- Implement online / streaming detection:

  - Sliding windows
  - Incremental statistics

- Distinguish:

  - Point, contextual, and collective anomalies

- Integrate into a simple alerting pipeline:

  - Alerts, false positive analysis, alert fatigue metrics

--------------------------------------------------------------------------------

## 8\. Graph & Network Analysis

**Goal:** Show skills on relational data and graph algorithms.

### Tier 1 – Exploration

- Build a graph from available data:

  - Social network, transactions, or web links

- Compute:

  - Degree distribution
  - Centrality measures
  - Communities and connected components

### Tier 2 – Modeling

- Community detection:

  - Louvain or spectral methods

- Link prediction:

  - Which users are likely to connect
  - Which products co-occur

- Node classification:

  - Fraud vs. normal, or power users vs. casual

### Tier 3 – Advanced

- Graph embeddings:

  - Node2Vec, DeepWalk, or similar

- Temporal graphs:

  - Evolution of networks over time

- Downstream application:

  - Graph-based recommender or fraud detection demo

--------------------------------------------------------------------------------

## 9\. End-to-End MLOps Project (Domain-Agnostic)

**Goal:** Demonstrate you can go from notebook to production-style pipeline.

### Tier 1 – Offline Work

- Clean dataset, EDA, and baselines.
- Create a clear project structure:

  - `src/`, `notebooks/`, `tests/`, `configs/`

- Document feature engineering and data validation steps.

### Tier 2 – Reproducible Training

- Package code as a Python module (`pyproject.toml`).
- Use configuration management:

  - YAML/JSON configs for experiments

- Add:

  - Logging of metrics
  - Experiment tracking (MLflow, wandb, or light custom solution)

- Write unit tests for:

  - Data loaders
  - Core feature engineering
  - Model utilities

### Tier 3 – Deployment & Monitoring

- Serve the model with:

  - FastAPI/Flask + Docker container

- Implement batch scoring:

  - Daily or hourly prediction jobs
  - Log inputs, outputs, and prediction metadata

- Build simple monitoring:

  - Data drift checks
  - Performance decay tracking
  - Basic alerting rules

--------------------------------------------------------------------------------

## 10\. Fairness, Robustness & Interpretability

**Goal:** Show you understand ethical and practical concerns in modeling.

### Tier 1 – Baseline Fairness & Performance

- Choose a supervised task (e.g. credit, churn, hiring).
- Measure performance by subgroup:

  - Gender, age bands, region, etc.

- Analyze:

  - Label imbalance
  - Feature distribution differences across groups

### Tier 2 – Methods

- Compare:

  - Black-box model vs interpretable model (logistic, GA2M, etc.)

- Use:

  - SHAP and partial dependence for explanations

- Compute basic fairness metrics:

  - Demographic parity, equalized odds, etc.

### Tier 3 – Advanced

- Explore **fairness-aware training**:

  - Reweighing
  - Constraints
  - Adversarial debiasing (if feasible)

- Robustness checks:

  - Add noise or shift distributions
  - Analyze model performance under perturbations

- Build:

  - Explanation reports for individual predictions
  - Short "model card" summarizing risks, limitations, and intended use

--------------------------------------------------------------------------------

## How to Use This Roadmap

1. **Pick 3–4 flagship projects**

  - Cover different domains (e.g. churn, forecasting, experimentation, MLOps).
  - For each, commit to going at least to **Tier 2**, preferably touching **Tier 3**.

2. **Define deliverables per project**

  - Clean Jupyter notebooks or scripts.
  - A `README.md` explaining problem, methods, and how to run the code.
  - At least a short written "decision memo" or analysis for one project.

3. **Track progress**

  - For each project, maintain a simple checklist:

    - [ ] Data & EDA
    - [ ] Baseline model
    - [ ] Advanced method (causal / uplift / optimization / deployment / etc.)
    - [ ] Documentation & visuals
    - [ ] Tests and basic CI (for the more engineering-heavy projects)

4. **Iterate**

  - Start simple, then add complexity where it shows the most value.
  - Use insights from one project (e.g., uplift modeling) in another when it makes sense.

This roadmap is intentionally rich. Use it as a long-term guide and select the pieces that best align with the profile you want to present.
