# History Preservation And Branch Operations

This document records the manual Git operations that should happen before any future
structural cleanup or repository conversion.

No destructive history rewriting is required for the current copy-first migration branch.

## Current Policy

- Keep project/source history in `ds-projects-portfolio`.
- Keep originals after copy-first exports.
- Do not delete project/source files unless a validated destination exists and removal is
  explicitly approved.
- Do not rewrite Git history to hide removed files.

## Branch State

The repository previously consolidated `develop` and `main` so `main` is the canonical
default branch. The migration work lives on:

`migration/copy-first-exports`

## Recommended Manual Steps Before Any Future Cleanup

1. Confirm `main` is the intended default branch on GitHub.
2. Confirm `migration/copy-first-exports` is pushed and reviewed.
3. Merge the migration branch into `main` with a normal merge or squash according to the
   repository's preferred review workflow.
4. Create a final monolith tag before any future removal-oriented cleanup:

   ```bash
   git switch main
   git pull --ff-only
   git tag -a monolith-final-2026-08 -m "Final full portfolio state before cleanup"
   git push origin monolith-final-2026-08
   ```

5. Only after that tag exists, consider narrow cleanup PRs for generated artifacts or stale
   infrastructure, and only with explicit approval.

## Operations Not Approved By This Migration

- `git reset --hard`
- history rewriting
- deleting `projects/`, `notebooks/`, `src/`, `tests/`, or `docs/`
- deleting generated artifacts without a separate cleanup approval
- replacing this repository with a tiny link-only hub
