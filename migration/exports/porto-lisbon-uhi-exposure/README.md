# Porto and Lisbon Urban Heat Island Exposure Analysis

This project rebuilds a Porto/Lisbon urban heat island exposure table from official European data sources and uses it to answer two questions: **what makes cells hot** (do parks cool the city?) and **who is exposed** to that heat.

The workflow combines:

- Eurostat GISCO Census 2021 1 km grid data;
- Eurostat GISCO Urban Audit city boundaries;
- the EEA public service exposing the Copernicus/UrbClim urban heat island model;
- OpenStreetMap park/vegetation polygons (Overpass API) for the green-cover analysis.

## Headline result: parks cool the city

Relating each cell's **green cover** (OSM parks, gardens, woodland, grass) to its modelled UHI intensity shows a clear negative relationship in both cities — greener cells are cooler:

- population-weighted correlation of green cover with UHI: **≈ −0.40 (Lisbon), −0.32 (Porto)**;
- regression slope: roughly **0.13 °C cooling per +10 percentage points of green cover** in Lisbon (≈0.09 °C in Porto);
- Lisbon cells with **>30% green run about half a degree cooler** than cells with <5% green.

Built-up centres without parks are the hotspots; well-vegetated cells are the coolest. (An association in modelled data, not a controlled experiment — green cover, building density and coastal position move together.)

## Output table

The preparation step writes `data/manual/porto_lisbon_cells.csv` with this schema:

```text
city,cell_id,population_total,age_0_14,age_65_plus,employed,born_outside_eu,uhi_intensity_celsius,is_uhi_exposed
```

Counts are area-weighted when a 1 km cell is split by a city boundary, so the prepared table contains fractional values.

## Boundary choice

The default real-data pipeline uses the **Eurostat Urban Audit `CITIES` polygons** for `PT001C` (Lisbon) and `PT002C` (Porto), not municipality boundaries. Those polygons are materially larger than the municipalities that most people informally mean by “Lisbon” or “Porto”, so city totals in this project are expected to be much larger than municipality-only census totals.

## UHI assumption

UHI intensity is sampled from the EEA ArcGIS service that exposes the UrbClim-based raster. Some coastal or no-data fragments do not return a direct raster value at the representative point. For those fragments, the preparation script assigns the nearest non-missing UHI value within the same city and reports how many cells required that fill.

## Install

```bash
poetry install
```

## Prepare real data

```bash
poetry run python scripts/prepare_real_data.py \
  --output data/manual/porto_lisbon_cells.csv \
  --threshold 2.0
```

This command downloads or reuses cached source files in `data/raw/`, intersects the Eurostat grid with the Urban Audit city polygons, samples UHI values, validates the final schema, and prints quality checks.

## Run the analysis

```bash
poetry run python scripts/run_analysis.py \
  --input data/manual/porto_lisbon_cells.csv \
  --output outputs/porto_lisbon_group_representation.csv \
  --summary-output outputs/porto_lisbon_city_summary.csv \
  --plot-dir plots
```

## Run the notebook

```bash
poetry run jupyter lab notebooks/porto_lisbon_uhi_exposure.ipynb
```

The notebook is configured for the real prepared table by default.

## Key findings

With the current real-data pipeline and a `2.0 °C` exposure threshold:

- Lisbon has a much lower exposed population share than Porto in the prepared table, about `1.05%` versus `6.42%`.
- In Lisbon, `born_outside_eu` is **significantly** over-represented in exposed cells (≈`1.60`, 90% bootstrap CI excludes 1) — the strongest signal in the study.
- In Porto, `older_65_plus` and `not_employed` are significantly over-represented; `born_outside_eu` is **not** statistically distinguishable from proportional at the threshold (CI includes 1), so it is not a both-cities result.
- The notebook now reports these ratios **with uncertainty** (bootstrap CIs), adds a **threshold-free dose-response** check (which reveals Lisbon's migrant signal is an extreme-tail effect, not a gradient), and a **double-burden** quadrant analysis (≈30% of Porto's population lives where high heat meets high age-vulnerability).

These findings should be read together with the boundary and raster assumptions above. They are valid for the **Urban Audit city polygons** and the **published UHI model surface** used here, not for municipality-only definitions of Lisbon and Porto.

## Interpretation scope

This is an exposure and spatial representation analysis. It does not estimate mortality, morbidity, or causal heat effects. It answers a narrower question: which groups are more concentrated in cells with higher modelled urban heat island intensity.
