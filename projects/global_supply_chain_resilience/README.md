# Global Supply Chain Resilience Under Disruption

## Status

`active-portfolio` — research design and reproducible data pipeline are being built. No empirical results are claimed yet.

The ingestion layer is implemented and the first real-data probe is wired against the official OECD 2025 regular ICIO archive for 2016–2022. The probe targets the 2022 table, records exact-source provenance, and emits schema evidence before any vintage-specific economic mapping or graph metric is computed.

## Research question

Which country-industry nodes create the greatest systemic supply-chain vulnerability, how do disruptions propagate through production dependencies, and which diversification strategies reduce exposure at the lowest cost?

The project treats the supply chain as a directed weighted production network rather than as a collection of independent bilateral trade flows.

## Primary data

- **OECD Inter-Country Input-Output (ICIO) tables**: primary source for country-industry production dependencies. The current release covers 1995–2022, 80 economies plus a rest-of-world aggregate, and 50 economic activities.
- **UN Comtrade**: product-level bilateral trade data for a semiconductor case study. This is kept analytically distinct from input-output dependence.
- **World Bank Logistics Performance Index**: contextual logistics information only. It is not treated as an annual panel because survey waves are sparse.

## Data-ingestion contract

Downloaded OECD archives are immutable raw artifacts. The ingestion layer records a SHA-256 digest of the exact source bytes and preserves source row/column identifiers. Before analysis it rejects empty tables, duplicate labels, missing numeric accounting content, non-positive gross output, negative values inside the economically identified intermediate-use block, and technical-coefficient columns whose intermediate-input shares exceed one.

Raw ICIO tables are **not** globally constrained to non-negative values because components such as changes in inventories may legitimately be negative. Sign constraints are applied only after the economic meaning of a block has been established.

The raw-table parser does **not** guess which columns constitute the square intermediate-use block. That extraction mapping must be explicit for each ICIO vintage, because silently inferring final-demand or value-added columns would contaminate the production network.

For the intermediate-use matrix `Z` and gross-output vector `x`, the project defines

\[
A_{ij}=\frac{Z_{ij}}{x_j}.
\]

This orientation is frozen: row `i` is the supplying country-industry node and column `j` is the using country-industry node.

## First empirical gate

The reproducible probe downloads the official 2016–2022 regular ICIO archive and selects the unique 2022 CSV member. It records the archive SHA-256, table shape, row/column labels, overlap between row and column identifiers, and the count of negative numeric cells. These outputs are evidence for the subsequent vintage-specific mapping of intermediate use, final demand, value added, and gross output; they are not substantive supply-chain results.

## Analytical structure

### 1. Build the production network

For year `t`, define a directed weighted graph

\[
G_t=(V,E,W_t),
\]

where each node is a country-industry pair and each directed edge represents intermediate-input dependence.

### 2. Structural dependency

Measure supplier concentration, import dependence, upstream/downstream exposure, centrality, community structure, and concentration indices. All measures will be defined prospectively before interpretation.

### 3. Input-output stress testing

Using

\[
x=Ax+f,
\qquad
x=(I-A)^{-1}f,
\]

construct explicit scenario-based stress tests. These are model-based counterfactual exercises under fixed technical-coefficient assumptions, **not causal estimates of real disruptions**.

### 4. Shock propagation

Simulate disruptions to selected country-industry nodes and quantify direct and indirect exposure. Sensitivity analysis will vary shock magnitude, affected nodes, and modelling assumptions rather than presenting a single deterministic scenario as truth.

### 5. Diversification optimization

Choose alternative sourcing weights `w` by balancing sourcing cost and resilience risk:

\[
\min_{\mathbf w}\; \lambda C(\mathbf w)+(1-\lambda)R(\mathbf w)
\]

subject to allocation, capacity, concentration, and feasibility constraints. Results will be reported as a cost-resilience frontier rather than a single supposedly optimal policy.

### 6. Semiconductor case study

Use product-level trade flows to examine semiconductor concentration and alternative sourcing patterns. Product trade flows and ICIO production dependencies will remain separate estimands throughout the analysis.

## Planned outputs

1. `01_build_network.ipynb` — ingest, validate, and construct country-industry networks.
2. `02_structural_dependency.ipynb` — concentration, centrality, communities, and dependence.
3. `03_stress_tests.ipynb` — Leontief-based disruption scenarios and sensitivity analysis.
4. `04_shock_propagation.ipynb` — systemic exposure and resilience metrics.
5. `05_diversification_optimization.ipynb` — constrained sourcing optimization and Pareto frontier.
6. `06_semiconductor_case_study.ipynb` — detailed product-level application.

Reusable code lives in `src/`; notebooks are analysis/reporting surfaces rather than the sole implementation.

## Scientific guardrails

- No random mixing of years when temporal validation is relevant.
- No causal language for static input-output stress tests.
- No assumption that trade value equals technological dependence.
- No use of LPI as annual data between survey waves.
- Network rankings will include stability/sensitivity checks.
- Optimization assumptions, penalties, capacities, and feasibility constraints will be explicit.
- Missingness, aggregation changes, concordances, and rest-of-world treatment will be documented.
- Results will distinguish exposure, vulnerability, systemic importance, and resilience rather than using those terms interchangeably.

## Portfolio objective

Demonstrate network science, linear algebra, economic modelling, stress testing, uncertainty analysis, constrained optimization, and decision support in one coherent applied project.
