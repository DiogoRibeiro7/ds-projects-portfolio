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

Historical and internal scaffolding should live outside this repository. The
active tree should contain only maintained portfolio assets and supporting
tooling.

## Generated Outputs

- `artifacts/`: model/data artifacts kept only when intentionally versioned for portfolio evidence.
- `notebooks/artifacts/**/production/`: notebook run traces and production-style run outputs; ignore by default.

For the public-facing contract and clear "keep vs move out" rule, see
[`docs/PORTFOLIO_SCOPE.md`](../PORTFOLIO_SCOPE.md).

## Rules

1. New reusable logic belongs in `src/`, not inside notebooks or project folders.
2. CI quality gates (format/lint/typecheck/tests) should target active code paths first.
3. Generated outputs must stay out of top-level root; place them in ignored artifact paths or separate archive repositories.
4. Versioned generated artifacts are only allowed when they are explicit portfolio deliverables and documented in contributor/development policy.
5. Legacy project material should be moved to a separate archive repository or
   left in Git history.
