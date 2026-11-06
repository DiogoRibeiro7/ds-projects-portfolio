# Real orange-price ingestion: Portugal's GPP SIMA

This is a **separate empirical data-ingestion layer** for the completed
[synthetic orange-storage study](README.md). It does not alter the fourteen
archived simulations, their price paths or the scientific synthesis. No
empirical sale-policy evaluation has been performed with these quotations.

## Official sources and definitions

- [Portuguese Ministry of Agriculture: SIMA market quotations](https://agricultura.gov.pt/portal/sima)
  describes GPP's market information system. Its minimum, maximum and
  most-frequent prices are **market quotations**, assessed from information
  gathered from economic operators. They are not verified executed sales.
- [GPP SIMA market consultation, wholesale markets](https://regsima.gpp.pt/regsima/consulta/mercados?tm=11)
  is the actual source of the archived CSV. The interface labels this section
  *Mercados Abastecedores*, distinct from *Mercados de Produção*.
- [GPP's SIMA export guide](https://www.gpp.pt/images/Agricultura/SIMA/SIMA_Manual.pdf)
  confirms that searches can be exported as CSV and that exports cover all
  matching rows rather than only the visible page. The older guide mentions
  comma delimiters, but **the file downloaded on 5 November 2026 actually uses
  semicolons and Latin-1 encoding**. The R parser follows the observed file.
- [GPP SIMA methodological document](https://www.gpp.pt/images/Agricultura/Documento_Metodologico.pdf)
  details collection, analysis and the market-quotation process. Consult this
  before assigning an economic interpretation to the observations.

The EU's [Agri-food Data Portal](https://agridata.ec.europa.eu/extensions/dataportal/fruit-and-vegetables.html)
is a **different** source with different reporting stages and coverage. Its
[fruit and vegetable API documentation](https://agridata.ec.europa.eu/extensions/API_Documentation/fruitandvegetables.html)
states that the old API was decommissioned in October 2025. This PR does not
guess a replacement endpoint or mix European prices into the Portuguese series.

## Frozen observed data (not a mock dataset)

| Field | Value |
| --- | --- |
| Source | GPP SIMA, *Mercados Abastecedores* |
| Retrieval | 5 November 2026, via the official website export control |
| UI query | Sector `Frutos Frescos`; species `Laranja`; all markets |
| Requested period | 1 January 2025 through 31 October 2026 (inclusive) |
| Actual observed dates | 2 January 2025 through 30 October 2026 |
| Exported observations | **3,412** quotation rows |
| Markets | `MAC` (1,620), `MAP` (1,059), `MARL` (733) |
| Export columns | `Produto`, `Espécie`, `Região`, `Mercado`, `Semana`, `Data`, `Mínima`, `Máxima`, `Freq` |
| Original format | Semicolon-delimited CSV; Latin-1; unquoted numeric decimal points |
| Raw file size | 461,934 bytes before lossless gzip compression |
| Original CSV SHA-256 | `b039f254b4fb18a41b7997208a43ceae46ee9f48db978934ab9fad6de9e24da0` |
| Snapshot | [`data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz`](data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz) |

The product descriptor includes grade, calibre, packaging, geographic origin
and an explicit `EUR/kg` unit (for example, an `Algarve` orange in a crate).
These dimensions can vary within a market/date and **must not be pooled
without a documented comparable-product definition**. The `Data` field is the
quoted observation date; retain the accompanying SIMA week number rather than
pretending each CSV row is an independent weekly aggregate.

The exported file has no `Provisórias` column, even though the website has a
provisional-status field. Normalized `is_provisional` therefore remains `NA`
(unknown), not `FALSE` (final). Future exports with explicit indicators can be
filtered, but unknown rows are not silently classified as final.

This archive records the source, query and hash for reproducibility. It is
**not** a declaration that the upstream source has adopted a particular open
data licence; verify reuse terms before any separate redistribution.

## Reproducing and updating with base R

From `projects/orange_storage_decision`:

```sh
# Validate the frozen real-data snapshot and corrupt-input guardrails.
Rscript tests/test_sima_market_prices.R

# Normalize the archived quotations to a new, untracked CSV file.
Rscript scripts/ingest_sima_orange_prices.R \
  data/raw/sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz \
  /tmp/sima-orange-normalized.csv wholesale
```

The output path must not exist, to avoid overwriting either the raw source or
previous experiments. Use your preferred existing directory in place of
`/tmp` if required. No R packages, API keys or network access are needed to
replay the checked-in dataset. The GitHub Actions workflow checks the raw
SHA-256 and runs the tests and an independent command-line export smoke test.

To refresh the series, open the GPP wholesale-market consultation URL above,
select `Frutos Frescos` → `Laranja`, choose all markets and a declared date
range, search and click `EXPORTAR`. Archive the resulting file and a new
retrieval manifest; do not silently replace this frozen snapshot. Inspect its
header, price units, origin descriptors, encoding and provisional field before
using it. The importer also accepts an uncompressed CSV from the same portal.

The parsed output preserves the complete `product_descriptor`, `quote_date`,
`sima_week`, `region`, `market`, `market_stage`, `price_unit`, the three quoted
prices, source URL/file/checksum and possible provisional status.

Only the verified wholesale export is supported in this first ingestion.
Production-market data require an independently validated source URL, format
and quotation definition before inclusion.

Data validation also rejects:

- Non-orange rows or any unit other than `EUR/kg`.
- Missing/invalid dates, weeks, market metadata and monetary values.
- Nonpositive prices, or a frequent quotation outside its min–max interval.
- Repeated keys for the same date, descriptor, region and market.
- Unexpected provisional labels and malformed/ambiguous column layouts.

## What this dataset does—and does not—measure

These data can support descriptive price dispersion, seasonality, consistent
product/market panels and future **out-of-time** forecasting studies.
They **cannot yet** validate the original sale-or-hold policy:

1. Wholesale quotations are neither farm-gate returns nor the amount a given
   grower can sell at, and they do not include transport, commissions or costs.
2. Origin and quality class vary; the observed price mix is not a controlled
   counterfactual price path for one fixed harvest batch.
3. SIMA gives no linked lot maturity, postharvest inspection, deterioration,
   deliveries, realised sales or unsold mass. Those data are needed to score
   the economic decision.
4. A future forecast test must train **only on prices available by each
   decision date**, and record publication/revision lags if known. This PR
   performs ingestion and validation only; it does not estimate an orange
   price-risk distribution from future data.

The [rolling-origin forecast follow-up](ROLLING_PRICE_BACKTEST.md) defines
like-for-like product/market panels and compares two fixed historical price
baselines, with honest missing-data coverage and transaction-stage limits.
The [release-delay sensitivity extension](AVAILABILITY_STRESS.md) further
tests hypothetical delays using the original dated rows, **not** observed GPP
publication timestamps; it cannot establish a real-time availability history.
The [product-mix audit](COMPOSITION_AUDIT.md) additionally reports comparable
2025/2026 month-by-month panels, unmatched entries/exits and unquoted-week
coverage so pooled product mixes are not mistaken for a national price index.
An [exact prior-year seasonal forecast audit](SEASONAL_52_BENCHMARK.md)
separately assesses how much of the 2026 target sample has a quoted
same-product, same-market week 52 weeks earlier, without filling gaps.
The follow-on [level-drift audit](LEVEL_DRIFT_AUDIT.md) separates the
pre-existing prior-year-to-recent quote gap from the subsequently observed
price innovation, reporting all availability denominators explicitly.
Biological shelf-life validation still requires a separate observational or
prospective dataset.
