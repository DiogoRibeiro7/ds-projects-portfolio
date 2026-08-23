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

- `artifacts/`: large generated artifacts and run logs.
- Historical project trees and reference notebooks that are not part of the
  active portfolio.
- Generated runtime outputs and local run traces.
- Historical quality snapshots.
- Internal platform ops scaffolding, deployment manifests, configs, Helm charts,
  and utility scripts.

## Disposition guidance for new content

1. **Keep** in top-level active surface if the file is useful for portfolio review,
   demos, or contributor onboarding.
2. **Move out of this repository** for legacy experiments, operational tooling,
   or internal scaffolding that is not required for day-to-day portfolio
   consumption.
3. **Remove** only by explicit issue when data is stale, duplicated, or no longer relevant.

## 2-minute placement guidance

- `src/`, `notebooks/`, `projects/`, `docs/` are user-facing portfolio assets.
- Historical/internal context should live in separate archive repositories or
  Git history, not in the active portfolio tree.
