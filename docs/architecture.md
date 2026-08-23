# Portfolio Architecture

This repository remains a data science projects portfolio. It contains active projects,
notebooks, examples, documentation, tests, and selected reusable source modules.

The migration work added copy-first export snapshots under `migration/exports/` for
components that now also exist as independent repositories. Originals stay in this
repository unless removal is explicitly approved.

## Repository Model

The working model is:

- portfolio source remains here;
- reusable or flagship components may also have validated export snapshots;
- generated artifacts are identified for possible cleanup but are not removed by default;
- stale infrastructure remains archival/reference unless a focused destination is approved.

## Why Not Convert To A Tiny Hub

The original decomposition prompt proposed converting this repository into a lightweight
hub. That was superseded by the user's later policy: keep data science projects and
portfolio files in this repo, and move/copy unrelated material only when there is a real
destination.

This avoids destroying project context just to make the repository smaller.

## Validated Export Snapshots

- `migration/exports/experimentation-toolkit` -> `https://github.com/DiogoRibeiro7/experimentation-toolkit`
- `migration/exports/genai-rag-engineering` -> `https://github.com/DiogoRibeiro7/genai-rag-engineering`
- `migration/exports/portugal-gdp-bayesian-revision` -> `https://github.com/DiogoRibeiro7/portugal-gdp-bayesian-revision`
- `migration/exports/porto-lisbon-uhi-exposure` -> `https://github.com/DiogoRibeiro7/porto-lisbon-uhi-exposure`
- `migration/exports/city-wage-cost-global` -> `https://github.com/DiogoRibeiro7/city-wage-cost-global`
- `migration/exports/pt-salary-gamma-distribution` -> `https://github.com/DiogoRibeiro7/pt-salary-gamma-distribution`
- `migration/exports/customer-analytics` -> `https://github.com/DiogoRibeiro7/customer-analytics`

## Catalogue

`projects.yml` is the machine-readable catalogue for featured portfolio work. It includes
validated export snapshots and active portfolio notebooks/projects.

Required fields per project:

- `name`
- `repository`
- `category`
- `status`
- `summary`
- `methods`
- `technologies`
- `featured`

## Boundaries

Independent repositories have been created from the export snapshots. This repository
remains the canonical portfolio source for the original active project files.

Generated artifacts and local caches are not promoted as portfolio content. They are tracked
as cleanup candidates in the migration matrix, but cleanup requires explicit approval.
