# Porto and Lisbon Urban Heat Island Exposure Analysis

This project rebuilds a Porto/Lisbon urban heat island exposure table from official European data sources and uses that table to compare which population groups are more represented in hotter urban cells.

The workflow combines:

- Eurostat GISCO Census 2021 1 km grid data;
- Eurostat GISCO Urban Audit city boundaries;
- the EEA public service exposing the Copernicus/UrbClim urban heat island model.

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

## Interpretation scope

This is an exposure and spatial representation analysis. It does not estimate mortality, morbidity, or causal heat effects. It answers a narrower question: which groups are more concentrated in cells with higher modelled urban heat island intensity.
