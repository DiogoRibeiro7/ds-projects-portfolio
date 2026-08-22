# Final Migration Report

## Summary

The migration completed the non-destructive decomposition path agreed in this branch.

The repository remains a data science projects portfolio. Focused export snapshots were
created under `migration/exports/`, validated, documented, and pushed. No original
portfolio projects or source directories were moved or deleted.

## Produced Export Snapshots

| Export | Status |
|---|---|
| `migration/exports/experimentation-toolkit` | validated |
| `migration/exports/genai-rag-engineering` | validated |
| `migration/exports/portugal-gdp-bayesian-revision` | validated |
| `migration/exports/porto-lisbon-uhi-exposure` | validated |
| `migration/exports/city-wage-cost-global` | validated |
| `migration/exports/pt-salary-gamma-distribution` | validated |
| `migration/exports/customer-analytics` | validated |

## Reports Produced

- `docs/repository_decomposition_inventory.md`
- `MIGRATION_MATRIX.md`
- `migration/migration_matrix.csv`
- `migration/flagship_extraction_report.md`
- `migration/experimentation_toolkit_report.md`
- `migration/genai_extraction_report.md`
- `migration/final_disposition_report.md`

## Tooling Produced

- `migration/export_repo.py`
- `migration/validate_export.py`
- `migration/templates/`
- `migration/manifests/example.json`
- `migration/tests/`

Tooling validation passed:

- `ruff check migration`
- `ruff format --check migration`
- `mypy migration`
- `pytest migration`

## Portfolio Metadata Produced

- `projects.yml`
- `docs/architecture.md`
- `docs/history_preservation.md`

These files document the current portfolio architecture, machine-readable project catalogue,
and manual history-preservation steps.

## Archived Or Retained Material

Broad infrastructure, legacy archive material, performance tooling, and stale workflow
automation are classified in the migration matrix as `ARCHIVE` or `KEEP` depending on their
role. They were not promoted into standalone exports unless they directly supported a
coherent project/package.

## Generated Material

Generated artifacts are identified as `DELETE_GENERATED` candidates in the matrix. Nothing
was deleted in this migration. Cleanup remains a separate explicit task.

## Hub Conversion Decision

The original prompt sequence included a final phase to convert the repository into a tiny
portfolio hub. That phase was intentionally not executed because the user's later policy was
to keep data science projects and portfolio files in this repository.

Instead, this branch adds catalogue and architecture metadata while preserving active
portfolio content.

## Manual GitHub Operations Still Required

- Review and merge `migration/copy-first-exports` into `main`.
- Create standalone GitHub repositories from validated export snapshots if desired.
- Create the preservation tag `monolith-final-2026-08` before any future cleanup.
- Decide separately whether stale root workflows should be repaired or disabled.
- Decide separately whether generated artifacts should be removed or externalized.
