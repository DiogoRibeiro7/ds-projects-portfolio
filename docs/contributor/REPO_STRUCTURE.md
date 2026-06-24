# Repository Structure Contract

This repository is intentionally a monorepo, but not all directories have the same lifecycle.

## Active code (quality-gated)

- `src/`: reusable production-style Python modules.
- `tools/`: small developer utilities used by checks and workflows.
- `tests/`: automated unit/integration/regression tests for active code.
- `scripts/`: repo scripts that support active workflows.
- `docs/`: user and contributor documentation.

## Portfolio and reference content (not quality-gated by default)

- `projects/`: portfolio implementations and demos, including per-project experiments.
- `notebooks/`: exploratory and domain notebooks.
- `examples/`: runnable examples and notebook-facing demos.
- `deployment/`: portfolio-facing deployment entry points (primarily `docker/` and `model_server/`).
- `tutorials/`: narrative learning material.

This document intentionally keeps historical and internal scaffolding in `archive/` and
explicitly marks `archive/internal/deployment-platform/` as non-portfolio deployment ops.

## Archive and generated outputs

- `archive/quality-reports/`: historical quality snapshots (e.g., mypy report dumps).
- `archive/runtime/`: local/generated runtime outputs that should not be committed.
- `archive/runtime/notebooks/artifacts/`: notebook runtime outputs moved out of active paths when useful for reproduction evidence.
- `archive/legacy/projects/`: legacy project trees moved out of active project paths.
- `artifacts/`: model/data artifacts kept only when intentionally versioned for portfolio evidence.
- `notebooks/artifacts/**/production/`: notebook run traces and production-style run outputs; ignore by default.
- `archive/internal/deployment-platform/`: legacy/ops deployment scaffolding
  (configs, Helm charts, Kubernetes manifests, MLflow hooks, and utility scripts).

For the public-facing contract and clear "keep vs archive" rule, see
[`docs/PORTFOLIO_SCOPE.md`](../PORTFOLIO_SCOPE.md).

## Rules

1. New reusable logic belongs in `src/`, not inside notebooks or project folders.
2. CI quality gates (format/lint/typecheck/tests) should target active code paths first.
3. Generated outputs must stay out of top-level root; place them under `archive/runtime/` or ignored artifact paths.
4. Versioned generated artifacts are only allowed when they are explicit portfolio deliverables and documented in contributor/development policy.
5. Legacy project material should be moved into `archive/legacy/projects/`.
