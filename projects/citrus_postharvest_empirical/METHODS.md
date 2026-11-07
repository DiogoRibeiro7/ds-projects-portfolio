# Protocol: fixed observational reanalysis

## 1. Evidence boundary

Primary evidence is the unchanged original `Quality Data.csv` and
`in vivo data.csv` from Zenodo record 16423047. `README.txt` states collection
occurred in Cartagena in 2022--2024 under laboratory conditions, and declares
CC BY 4.0. The linked primary article provides the experimental motivation.
We do not treat a laboratory study as representative of Portuguese warehouses,
production seasons or wholesale-market prices.

The code verifies **original-file MD5 checksums** before reading or plotting.
Any replaced, truncated or cleaned-up raw file fails closed. Decimal commas
in the quality CSV and its UTF-8 byte order mark are handled explicitly.

## 2. Three measurement panels

**Firmness:** leftmost three columns in `Quality Data.csv`. Exactly 200 orange
measurements: five conditions, four days (0, 5, 10, 15) and ten rows per cell.
Newton units. No fruit identifier; a source spreadsheet row is an audit trail,
not a longitudinal subject identifier.

**Weight loss:** columns 4--6 in the same file. Exactly 60 orange measurements:
five conditions, four days (5, 8, 10, 15) and three rows per cell. Values are
percentages as supplied. No measured day-zero weight-loss outcome is present:
we do not manufacture a zero or baseline. Weight loss and firmness are **not
paired** despite the rows being printed side-by-side.

**Inoculated decay:** orange section *only* of `in vivo data.csv`. Five
conditions, three source rows/boxes per condition, 20 fruit per box, with
recorded decay and sporulation counts on days 5, 7 and 10. The reported
percentages must equal the counts divided by the 20-fruit denominator. We
aggregate counts over the **three source boxes** only to report observed
percentages, e.g. 15/60 = 25%, and retain the number of boxes and denominator.
We do not treat the three days of a box as independent experiments.

Conditions: `CTRL` = unwaxed; `CTRL-W` = wax without by-product; `AVOCADO`,
`ORANGE` and `GRAPEFRUIT` = carnauba wax with the specified fruit by-product.
In the quality CSV, the treatment name `ORANGE` means orange by-product, not a
species field. The unused lemon table and in-vitro screening are **not**
analysed. The source is not a clean, pooled trial across all outcomes.

## 3. Estimands and comparisons

For each (treatment, day, outcome) quality cell, report sample size, arithmetic
mean, sample SD, minimum and maximum. An SD is descriptive dispersion; it is
not a standard error, a confidence interval or a model of biological
replication. For inoculated decay/sporulation, report the integer sum of
affected fruit over the sum of experimental-box denominators.

Endpoint-specific descriptive differences are defined as

`observed_treatment_mean - observed_control_mean`

at day 15 for the quality measures and day 10 for inoculated decay and
sporulation. The comparisons against `CTRL` (unwaxed) and `CTRL-W` (wax only)
answer different questions. We do not run t-tests, regression, mixed-effects
models or multiplicity adjustments because fruit-level/block-level allocation
and repeat-observation metadata are insufficiently specified in the two CSVs
to justify a new inferential analysis.

## 4. Interpretation safeguards

1. Same spreadsheet row does not imply same fruit across quality outcomes.
2. The end points have different sampling days and biological contexts.
3. The day-zero baseline firmness means differ among groups. Day-15 raw
   differences do not automatically measure preservation *caused* by a coating.
4. Inoculation-based decay is not the naturally arising probability of
   commercial spoilage or mass loss in storage.
5. A coating's smaller day-15 mean weight loss does not make it a globally
   superior coating if its other endpoints differ.
6. No costs, selling prices or market matching were measured by this dataset.
7. No individual-level fruit survival curve or shelf-life extension in days
   is identified by this reanalysis.

## 5. Exact workflow and output audit

Run `Rscript tests/test_empirical_sources.R` and
`Rscript scripts/run.R data/raw results/generated` from this project root.
The test enforces hashes, source-section extraction, all cell sizes, integer
denominators, reported percentage arithmetic and independently checked
endpoint values. The generated CSVs and figures are reproducible summaries;
source CSV files are not mutated or re-sorted. CI runs inside a pinned
`rocker/r-ver:4.5.1` image, with no analysis dependencies.

These are useful, bounded descriptive facts that can support a separately
versioned article. Future commercial storage modelling or independent
external datasets require a new stated question, not a silent scope change.
