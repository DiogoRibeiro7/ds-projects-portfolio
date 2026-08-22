# Porto Lisbon UHI Exposure Export Report

## Decision

Create a standalone-ready copy of the flagship urban heat island exposure project.
No source files, notebooks, plots, scripts, or portfolio assets were moved or deleted from
`ds-projects-portfolio`.

## Export Location

`migration/exports/porto-lisbon-uhi-exposure`

## Copied From Portfolio

- `projects/porto_lisbon_uhi_exposure/src/uhi_exposure`
- `projects/porto_lisbon_uhi_exposure/tests`
- `projects/porto_lisbon_uhi_exposure/notebooks`
- `projects/porto_lisbon_uhi_exposure/scripts`
- `projects/porto_lisbon_uhi_exposure/plots/demo_*.png`
- Existing README, `pyproject.toml`, and `.gitignore`

## Export-Only Metadata Adjustment

The export copy widens the Python metadata from `<3.13` to `<3.14`, allows NumPy 2.x,
allows modern GeoPandas, and allows newer PyArrow wheels so it can validate in the current
Python 3.13 workspace without downgrading or source-building shared environment packages.
The original portfolio project metadata is unchanged.

## Kept In Portfolio

The original `projects/porto_lisbon_uhi_exposure` directory remains in place and unchanged.

## Validation Commands

Run from the export directory:

```bash
python -m pip install -e .
pytest tests/ -q
```
