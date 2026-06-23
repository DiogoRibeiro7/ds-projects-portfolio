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
- `deployment/`: deployment references and environment-specific manifests.
- `tutorials/`: narrative learning material.

## Archive and generated outputs

- `archive/quality-reports/`: historical quality snapshots (e.g., mypy report dumps).
- `archive/runtime/`: local/generated runtime outputs that should not be committed.
- `archive/legacy/projects/`: legacy project trees moved out of active project paths.
- `artifacts/`: model/data artifacts kept only when intentionally versioned for portfolio evidence.
- `notebooks/artifacts/**/production/`: notebook run traces and production-style run outputs; ignore by default.

## Rules

1. New reusable logic belongs in `src/`, not inside notebooks or project folders.
2. CI quality gates (format/lint/typecheck/tests) should target active code paths first.
3. Generated outputs must stay out of top-level root; place them under `archive/runtime/` or ignored artifact paths.
4. Legacy project material should be moved into `archive/legacy/projects/`.
