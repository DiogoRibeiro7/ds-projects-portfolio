# Global Supply Chain Resilience Under Disruption

## Status

`active-portfolio` — the 2022 OECD ICIO production system, structural-dependency baseline, supplier-side importance audit, shock-candidate freeze, and preregistered downstream propagation experiment are now implemented and empirically validated. Diversification optimization is the next prospective decision layer.

The validated 2022 active production system contains 3,999 country-industry nodes after excluding 51 zero-output labels. The preregistered propagation model passed its admissibility gate with spectral radius approximately 0.5964. These are model-based exposure results under fixed technical coefficients, not causal forecasts of real disruptions.

## Research question

Which country-industry nodes create the greatest systemic supply-chain vulnerability, how do disruptions propagate through production dependencies, and which diversification strategies reduce exposure with the least sourcing reallocation?

The project treats the supply chain as a directed weighted production network rather than as a collection of independent bilateral trade flows.

## Primary data

- **OECD Inter-Country Input-Output (ICIO) tables**: primary source for country-industry production dependencies. The current release covers 1995–2022, 80 economies plus a rest-of-world aggregate, and 50 economic activities.
- **UN Comtrade**: product-level bilateral trade data for a semiconductor case study. This is kept analytically distinct from input-output dependence.
- **World Bank Logistics Performance Index**: contextual logistics information only. It is not treated as an annual panel because survey waves are sparse.

## Data-ingestion contract

Downloaded OECD archives are immutable raw artifacts. The ingestion layer records a SHA-256 digest of the exact source bytes and preserves source row/column identifiers. Before analysis it rejects empty tables, duplicate labels, missing numeric accounting content, negative values inside the economically identified intermediate-use block, and non-positive gross output for active production nodes.

Raw ICIO tables are **not** globally constrained to non-negative values because components such as changes in inventories may legitimately be negative. Sign constraints are applied only after the economic meaning of a block has been established.

The raw-table parser does **not** guess which columns constitute the square intermediate-use block. That extraction mapping is explicit for the 2025 ICIO vintage.

For the intermediate-use matrix `Z` and gross-output vector `x`, the project defines

\[
A_{ij}=\frac{Z_{ij}}{x_j}.
\]

This orientation is frozen: row `i` is the supplying country-industry node and column `j` is the using country-industry node.

A technical-coefficient column is **not** rejected merely because its entries sum above one. Coefficient construction is kept separate from productivity and invertibility. Any inverse-based propagation analysis must pass its own spectral-radius and numerical admissibility gate.

## Empirical validation

The production pipeline downloads the official 2016–2022 regular ICIO archive, verifies exact-source provenance, calibrates the release-balance envelope on 2016–2021, and applies the same frozen criterion to the independent 2022 holdout before constructing the active production system.

The 2022 downstream exposure model then uses

\[
q=s+A^\top q,
\]

and only solves

\[
q=(I-A^\top)^{-1}s
\]

if the empirical technical-coefficient matrix satisfies the preregistered admissibility gate. The observed 2022 matrix passed with

\[
\rho(A)\approx0.5964.
\]

The seven frozen supplier shocks are `CHN_C26`, `CHN_C20`, `USA_G`, `CHN_C27`, `RUS_B06`, `NOR_B06`, and `USA_B06`. They were selected mechanically from threshold-persistent supplier rankings before propagation outcomes were inspected.

## Analytical structure

### 1. Build the production network

For year `t`, define a directed weighted graph

\[
G_t=(V,E,W_t),
\]

where each node is a country-industry pair and each directed edge represents intermediate-input dependence.

### 2. Structural dependency

Measure direct foreign-input dependence, domestic input share, supplier-country concentration, effective supplier counts, and supplier-side downstream importance. Ratio rankings are accompanied by material-scale filters and prospective threshold sensitivity checks.

### 3. Input-output stress testing

Use explicitly assumption-bound fixed-coefficient experiments. Inverse-based calculations are allowed only after a separate productivity/invertibility gate. They are **not causal estimates of real disruptions**.

### 4. Shock propagation

Simulate the frozen single-supplier disruptions and quantify direct and higher-order exposure, foreign spillovers, and output-equivalent exposure. The primary 10% shock is accompanied by 5% and 20% linearity checks and input-share threshold sensitivity diagnostics.

### 5. Diversification optimization

The optimization layer does **not** invent monetary procurement costs. It minimizes observed sourcing reallocation burden subject to prespecified reductions in worst-case direct exposure to the frozen shocks.

For a selected buyer `j`, counterfactual sourcing preserves the observed supplying-activity composition exactly while allowing geographic reallocation within each activity. Supplier headroom, concentration safeguards, self-supply restrictions, and alternative-supplier support are frozen prospectively. The main frontier asks how much sourcing turnover is required to reduce direct worst-case risk by 25%, 50%, and 75%.

The primary system evaluation uses the 50% risk-reduction target with 5% supplier headroom, then re-runs the full admissibility gate and all seven propagation shocks for each buyer-specific counterfactual. Buyer-specific policies are evaluated separately rather than summed into an infeasible globally coordinated reallocation.

### 6. Semiconductor case study

Use product-level trade flows to examine semiconductor concentration and alternative sourcing patterns. Product trade flows and ICIO production dependencies remain separate estimands throughout the analysis.

## Planned outputs

1. `01_build_network.ipynb` — ingest, validate, and construct country-industry networks.
2. `02_structural_dependency.ipynb` — concentration, centrality, communities, and dependence.
3. `03_stress_tests.ipynb` — fixed-coefficient disruption scenarios and sensitivity analysis.
4. `04_shock_propagation.ipynb` — systemic exposure and resilience metrics.
5. `05_diversification_optimization.ipynb` — constrained sourcing optimization and reallocation-resilience frontier.
6. `06_semiconductor_case_study.ipynb` — detailed product-level application.

Reusable code lives in `src/`; notebooks are analysis/reporting surfaces rather than the sole implementation.

## Scientific guardrails

- No random mixing of years when temporal validation is relevant.
- No causal language for static input-output stress tests.
- No assumption that trade value equals technological dependence.
- No use of LPI as annual data between survey waves.
- Network rankings include stability/sensitivity checks.
- Optimization assumptions, headroom constraints, concentration safeguards, and feasibility constraints are explicit.
- Reallocation burden is not called monetary cost without an external cost source.
- Missingness, aggregation changes, concordances, and rest-of-world treatment are documented.
- Exposure, vulnerability, systemic importance, resilience, and optimization burden are kept conceptually distinct.

## Portfolio objective

Demonstrate network science, linear algebra, economic modelling, stress testing, uncertainty analysis, constrained optimization, and decision support in one coherent applied project.
