# Global Supply Chain Resilience Under Disruption

## Status

`validated-portfolio` — the 2022 OECD ICIO production-network analysis, structural-dependency baseline, supplier-side importance audit, preregistered propagation experiment, diversification optimization, and UN Comtrade semiconductor case study are implemented and empirically validated.

The project studies two deliberately separate systems:

1. country-industry production dependence from OECD ICIO;
2. product-level semiconductor trade concentration from UN Comtrade.

They are compared descriptively only. Trade value is not treated as technological dependence, fabrication origin, or a substitute for input-output structure.

## Research question

Which country-industry nodes create the greatest systemic supply-chain vulnerability, how do disruptions propagate through production dependencies, and which diversification strategies reduce exposure with the least sourcing reallocation?

## Core 2022 production-network evidence

The validated active ICIO production system contains **3,999 country-industry nodes**, after excluding 51 zero-output labels. The downstream fixed-coefficient model

\[
q=s+A^\top q
\]

is solved only after its numerical admissibility gate. For 2022,

\[
\rho(A)=0.5963952234<1.
\]

The seven supplier shocks were frozen before propagation outcomes were inspected:

- `CHN_C26`
- `CHN_C20`
- `USA_G`
- `CHN_C27`
- `RUS_B06`
- `NOR_B06`
- `USA_B06`

At the primary 10% shock level, per-unit amplification ranks highest for `CHN_C20`, `RUS_B06`, and `NOR_B06`; absolute exposure rankings differ because scale and network position are distinct quantities.

## Diversification result

The optimization layer minimizes observed sourcing reallocation burden rather than inventing monetary procurement costs. Ten buyers were frozen prospectively for the primary 50% direct-risk-reduction target with 5% supplier headroom.

Six buyer-specific problems were feasible:

- `NOR_C19`
- `BLR_C19`
- `SVK_C19`
- `SWE_C19`
- `HUN_C19`
- `ARE_C26`

Four were infeasible under the frozen constraints:

- `USA_C19`
- `CHN_C26`
- `CHN_C20`
- `CHN_C22`

The central decision result is:

\[
\boxed{
\text{local diversification can reduce buyer exposure without reducing global systemic exposure}
}
\]

For all six feasible buyer-specific policies, worst-case direct exposure fell by 50%, yet global propagated exposure increased slightly after the counterfactual network was re-evaluated. This is a model-based result under fixed technical coefficients, not a causal forecast of real-world disruption.

## Semiconductor case study — HS 8542

The confirmatory trade study uses 2022 annual UN Comtrade data for **HS 8542 — electronic integrated circuits**. The primary reporter universe contains **167 reporters** after prospectively excluding the overlapping EU and ASEAN aggregate reporters.

### Import concentration

The median positive-importer threshold defines **84 material importers**. Across them:

- median all-reported partner HHI: **0.1844**;
- median largest-partner share: **31.96%**;
- median top-three share: **64.31%**.

The largest HS 8542 import markets are China, Hong Kong SAR, Singapore, Other Asia, nes, and Korea.

Among named suppliers, China is especially pervasive across the 84 material importers:

- largest named supplier for **35/84**;
- at least 10% share for **60/84**;
- at least 25% for **28/84**;
- at least 50% for **6/84**.

`Other Asia, nes` is kept separate from named-country statistics. It represents about **30.16%** of material-importer bilateral value and about **99.3%** of all residual/special-partner value, so silently assigning it to Taiwan would materially distort the evidence.

### Export scale

Among the frozen primary reporters, **149** report positive HS 8542 exports to World. Their reported total is approximately **1.077 trillion** in current trade-value units.

The largest exporter-reported sources are:

1. Hong Kong SAR — about 213.8 bn;
2. Other Asia, nes — about 183.7 bn;
3. China — about 154.5 bn;
4. Singapore — about 122.0 bn;
5. Korea — about 112.8 bn.

These are commercial trade positions, not fabrication-capacity estimates.

### Mirror-data audit

The top 50 named bilateral import links were frozen by importer-reported value and then checked against reverse exporter-reported observations.

Of 50 links:

- 49 have an observed mirror;
- median relative difference is **36.96%**;
- the 90th percentile is **71.86%**;
- **18/49** differ by at least 50%;
- exporter-reported value is below importer-reported value in **37/49** observed links.

The largest observed relative discrepancy is Thailand → China, with about a **90.13%** max-denominator difference. This audit supports the methodological rule that importer and exporter reports must remain separate observations rather than being averaged into a synthetic flow.

## HS 8542 versus ICIO C26

The trade evidence was compared descriptively with OECD ICIO activity **C26 — computer, electronic and optical products** using exact ISO3-to-ICIO country matches only.

For export scale:

\[
\rho_S=0.9144
\]

across 74 matched countries.

For downstream importance:

\[
\rho_S=0.8814
\]

across 79 matched countries.

Both top-10 lists overlap in 8 of 10 countries. The agreement is therefore strong, but the rank discrepancies are informative. Hong Kong SAR is the clearest example: it ranks **#1** in HS 8542 export trade but only **#43** in C26 foreign intermediate sales, consistent with the importance of trade routing and re-export structures.

The comparison is descriptive only: HS 8542 is narrower than C26, and agreement does not validate either dataset or identify technological dependence.

## Optional HS6 decomposition — closed by preregistered gate

The secondary decomposition was allowed only if the five frozen HS6 codes were comparable across every classification version represented in the 167-reporter universe.

The frozen classification counts are:

- H2: 1 reporter
- H3: 6
- H4: 6
- H5: 41
- H6: 113

H3–H6 contain all five frozen codes, but H2 contains `854290` and lacks `854231`, `854232`, `854233`, and `854239`.

Therefore:

\[
\boxed{\text{the global 167-reporter HS6 decomposition is not permitted}}
\]

No H2 reporter is dropped post hoc and no six-digit trade-value analysis is run.

## Data and reproducibility

Primary sources:

- **OECD Inter-Country Input-Output tables** for production dependencies;
- **UN Comtrade** for the semiconductor trade case study;
- **World Bank Logistics Performance Index** as contextual logistics information only.

Every substantive stage is separated from its design gate. Source downloads and analytical artifacts record exact provenance, including workflow run IDs, artifact IDs, digests, retrieval metadata, and source hashes where applicable.

The compact final evidence ledger is:

`protocol/final_evidence_2022.json`

Raw and derived empirical artifacts remain GitHub Actions evidence rather than being copied into the repository as stale CSV snapshots.

## Scientific guardrails

- No causal language for static input-output stress tests.
- No assumption that trade value equals technological dependence or fabrication origin.
- No random mixing of years when temporal validation matters.
- No silent reconciliation of importer and exporter mirror data.
- No fuzzy or post-result country remapping in the HS8542↔C26 comparison.
- `Other Asia, nes` is not mapped to Taiwan.
- No forced HS6 decomposition across incompatible HS revisions.
- Reallocation burden is not called monetary cost without an external cost source.
- Exposure, vulnerability, systemic importance, resilience, and optimization burden remain distinct concepts.

## Portfolio objective

Demonstrate network science, linear algebra, economic modelling, stress testing, constrained optimization, reproducible evidence gates, and decision support in one coherent applied project.
