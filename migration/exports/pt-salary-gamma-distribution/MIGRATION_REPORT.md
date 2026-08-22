# Portugal Salary Gamma Distribution Export Report

## Decision

Create a standalone-ready copy of the flagship salary-distribution project.
No source files, notebooks, scripts, reports, or portfolio assets were moved or deleted from
`ds-projects-portfolio`.

## Export Location

`migration/exports/pt-salary-gamma-distribution`

## Copied From Portfolio

- `projects/pt_salary_gamma_distribution/src/pt_salary_gamma_distribution`
- `projects/pt_salary_gamma_distribution/tests`
- `projects/pt_salary_gamma_distribution/notebooks`
- `projects/pt_salary_gamma_distribution/scripts`
- `projects/pt_salary_gamma_distribution/reports/notebook_summary.md`
- Existing README, ROADMAP, requirements, and `.gitignore`

## Export-Only Metadata Adjustment

The portfolio project did not include package metadata. The export adds a minimal
`pyproject.toml` so the package can be installed and tested as a standalone repository.
The original portfolio project is unchanged.

## Kept In Portfolio

The original `projects/pt_salary_gamma_distribution` directory remains in place and unchanged.

## Validation Commands

Run from the export directory:

```bash
python -m pip install -e ".[test]"
python -c "import json, pathlib; [json.load(open(p, encoding='utf-8')) for p in pathlib.Path('notebooks').glob('*.ipynb')]"
pytest tests/ -q
```
