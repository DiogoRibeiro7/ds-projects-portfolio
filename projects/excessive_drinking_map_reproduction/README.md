# U.S. County Excessive Drinking Map Reproduction

This project reproduces a county-level U.S. choropleth map of **excessive drinking** using County Health Rankings / CDC BRFSS-derived data.

The notebook extracts the relevant county-level field, cleans the FIPS codes, saves a tidy CSV, and builds an interactive Plotly map.

## What the measure means

The relevant County Health Rankings field is:

```text
v049_rawvalue = Excessive Drinking raw value
```

This is the age-adjusted percentage of adults reporting binge or heavy drinking. It is not a direct measurement of alcohol sales, alcohol-use disorder, or alcohol-related deaths.

## Files

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── reproduce_excessive_drinking_map.ipynb
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
└── outputs/
    └── .gitkeep
```

## How to run

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Then start Jupyter:

```bash
jupyter lab
```

Open:

```text
notebooks/reproduce_excessive_drinking_map.ipynb
```

Run all cells.

The notebook will try to download the 2024 County Health Rankings analytic CSV directly from the official source.

If the official URL changes, download the analytic data manually from the County Health Rankings data documentation page and place it here:

```text
data/raw/analytic_data2024.csv
```

Then rerun the notebook.

## Outputs

The notebook writes:

```text
outputs/excessive_drinking_county_2024.csv
outputs/excessive_drinking_county_2024.html
```

The HTML file is an interactive county-level choropleth.

## Main data sources

- County Health Rankings & Roadmaps data documentation
- County Health Rankings 2024 Data Dictionary
- CDC excessive drinking definition and BRFSS-related documentation
- Plotly public county FIPS GeoJSON

## Reproducibility note

This reproduces the same type of map from the official county-level data source. It may not match the screenshot pixel-for-pixel because the screenshot likely uses a specific styling, binning scheme, and possibly a specific release year.
