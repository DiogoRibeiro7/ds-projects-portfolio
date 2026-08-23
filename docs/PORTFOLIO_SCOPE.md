# Portfolio Scope Contract

## What belongs in the public portfolio surface

- `src/`: reusable modules and production-style reference code.
- `notebooks/`: featured notebooks and documented notebook workflows.
- `projects/`: curated project writeups and demos.
- `examples/`: runnable portfolio demos and scripts.
- `tests/`: lightweight validation for active modules, where quality is enforced in CI.
- `docs/`: public documentation, guides, and module references.
- `docs/contributor/*`: contributor-facing process docs.
- `deployment/docker`: portfolio-facing container entrypoints.
- `deployment/model_server`: reference model serving package (when used by docs/tests).

## What is explicitly out-of-scope for the portfolio narrative

- `archive/legacy/`: legacy project trees and historical reference material.
- `archive/runtime/`: generated runtime outputs.
- `archive/quality-reports/`: historical quality snapshots.
- `archive/internal/`: internal platform ops scaffolding and experiment infrastructure.
- `artifacts/`: large generated artifacts and run logs.
- `archive/internal/deployment-platform/`: historical platform/ops deployment manifests, configs,
  Helm charts, and utility scripts.

## Disposition guidance for new content

1. **Keep** in top-level active surface if the file is useful for portfolio review,
   demos, or contributor onboarding.
2. **Move to `archive/`** for legacy experiments, operational tooling, or internal
   scaffolding that is not required for day-to-day portfolio consumption.
3. **Remove** only by explicit issue when data is stale, duplicated, or no longer relevant.

## 2-minute placement guidance

- `src/`, `notebooks/`, `projects/`, `docs/` are user-facing portfolio assets.
- `archive/` is for historical or internal context that should stay discoverable.
- Internal workflows and deployment-heavy stacks should live under `archive/internal/`.
