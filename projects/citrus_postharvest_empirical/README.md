# What Citrus Coatings Actually Preserve

An independent, reproducible **empirical reanalysis** of measured postharvest
quality and experimentally induced fungal decay in **organic oranges**.

The dataset is from Martínez-Zamora and colleagues (2025), University of
Cartagena / University of Murcia, deposited under CC BY 4.0 at
[Zenodo 16423047](https://zenodo.org/records/16423047). The companion paper is
[Foods 14(15), 2616](https://doi.org/10.3390/foods14152616).

**Status:** real source files verified against the upstream MD5 checksums;
descriptive R pipeline and tests prepared. This is not a reproduction of the
authors' entire experimental analysis or their inferential significance tests.

## Fixed research question

In the published orange experiments, how do the recorded decay, sporulation,
firmness and weight-loss outcomes differ across an unwaxed control, wax-only
control, and waxes incorporating avocado, orange or grapefruit by-products?

There are exactly **three empirical endpoints / panels**:

| Panel | Original orange measurements | Observation unit | Days |
| --- | ---: | --- | --- |
| Firmness | 200 | 10 readings per treatment/day (unit N) | 0, 5, 10, 15 |
| Weight loss | 60 | 3 readings per treatment/day (unit %) | 5, 8, 10, 15 |
| Inoculated fruit decay and sporulation | 15 experimental boxes, 90 time/outcome rows | 3 boxes per treatment; 20 oranges/box | 5, 7, 10 |

The source CSV interleaves orange and lemon measurements. We read **only** the
orange columns. Within those columns, firmness and weight-loss observations
are side-by-side *independent tables*: equal spreadsheet row numbers do **not**
identify the same fruit or permit within-fruit pairing. The in-vivo study
involves deliberately inoculated fruit and is not equivalent to natural storage
decay in an ordinary commercial batch.

### Explicitly out of scope

- No Portuguese prices, transport, cold-chain economics, profitability or
  optimal selling-day recommendations.
- No synthetic survival extrapolation or transfer of these measurements to the
  existing `orange_storage_decision` project.
- No pooling with the lemon data, in-vitro screening, or studies whose raw
  measurements are available only on author request.
- No fruit-level trajectories, repeated-measures model, causal claim beyond
  the experiment, or population inference from unreported experimental IDs.

## Reproduce

Requires base R 4.5.1 or compatible and Cairo PNG graphics support. No R
packages, tokens or download access are required because the three small,
attributed original files are included byte-for-byte.

```bash
cd projects/citrus_postharvest_empirical
Rscript tests/test_empirical_sources.R
Rscript scripts/run.R data/raw results/generated
```

Outputs, generated on demand:

- `orange_quality_summary.csv`: means, SDs and ranges, *separately* by
  treatment/day/outcome; descriptive SDs are not confidence intervals.
- `orange_in_vivo_summary.csv`: exact counts and denominators, pooled within
  the reported day/treatment, with per-box count preserved upstream.
- `endpoint_contrasts.csv`: observed, unadjusted differences versus the
  unwaxed and wax-only controls, within the same endpoint and sampling day.
- `orange_firmness.png`, `orange_weight_loss.png`, `orange_decay.png`: three
  independent descriptive figures suitable for manuscript illustrations.

GitHub Actions runs the same test and generation commands and uploads the
generated evidence as an artifact. See [METHODS.md](METHODS.md) for the unit of
analysis, denominator checks and interpretation rules. See
[data/raw/PROVENANCE.md](data/raw/PROVENANCE.md) for attribution and hashes.

## A first descriptive read, not a ranking

At day 10 in the *inoculated* orange panel, observed decay is 55% for the
unwaxed control, 30% for wax alone, 25% for avocado-enriched wax, 30% for
orange-enriched wax, and 50% for grapefruit-enriched wax (three boxes of 20
oranges per group). Avocado's 5-percentage-point advantage over **wax alone**
must not be confused with its 30-point difference from the **unwaxed** group.

At day 15 in the *separate quality panel*, mean weight loss is about 6.99%
unwaxed versus 4.25% for grapefruit-enriched wax, while grapefruit-enriched
wax performs comparatively poorly on day-10 inoculated decay. Thus even the
direction of a naive 'best coating' claim depends on which distinct outcome is
selected. The two percentages come from different tests and times.

## Source attribution

Martínez-Zamora, L., et al. (2025). *Carnauba Wax Coatings Enriched with
Essential Oils or Fruit By-Products Reduce Decay and Preserve Postharvest
Quality in Organic Citrus*. **Foods, 14**(15), 2616.
https://doi.org/10.3390/foods14152616. Original dataset:
https://zenodo.org/records/16423047, CC BY 4.0. The original authors are
credited; errors in this independent reanalysis are ours.
