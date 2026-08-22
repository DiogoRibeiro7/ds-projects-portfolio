# City Wage Cost Global Export Report

## Decision

Create a standalone-ready copy of the flagship notebook project.
No source files, notebooks, docs, or portfolio assets were moved or deleted from
`ds-projects-portfolio`.

## Export Location

`migration/exports/city-wage-cost-global`

## Copied From Portfolio

- `projects/city_wage_cost_global/notebooks`
- `projects/city_wage_cost_global/README.md`
- `projects/city_wage_cost_global/METHODOLOGY.md`
- `projects/city_wage_cost_global/requirements.txt`

## Kept In Portfolio

The original `projects/city_wage_cost_global` directory remains in place and unchanged.

## Validation Commands

Run from the export directory:

```bash
python -m pip install -r requirements.txt
python -c "import json, pathlib; [json.load(open(p, encoding='utf-8')) for p in pathlib.Path('notebooks').glob('*.ipynb')]"
python -c "import eurostat, matplotlib, nbformat, numpy, openpyxl, pandas, pyarrow, requests"
```
