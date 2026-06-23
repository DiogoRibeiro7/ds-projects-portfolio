# Portfolio Release Checklist

Use this checklist for **major portfolio updates**:

- README and public landing page changes
- Project showcase or featured notebook updates
- New/updated demos, screenshots, docs, or portfolio evidence

For smaller, low-visibility changes (single typo fixes, internal refactors, docs wording only in internal files), this checklist can be used as a light reference or linked selectively.

## Scope and definition of major update

- [ ] I reviewed what changed in this PR and marked it as major portfolio scope because it changes public-facing portfolio value.
- [ ] Checklist was copied into the PR description before merge and any intentionally skipped check is explicitly justified.

## Portfolio quality

- [ ] Featured portfolio items are accurate (`README.md`, `docs/README_ENHANCED.md`).
- [ ] Every changed featured notebook has: clean structure, clear objective, run steps, expected outputs, and next-step guidance.
- [ ] Featured notebook list and project/readme links were updated where needed.
- [ ] All visual portfolio artifacts (`plots`, `tables`, `screenshots`) were validated for readability and match the current results.
- [ ] No high-friction changes were made without updating the corresponding audience-facing context (project README, results summary, or portfolio card).

## Technical validation

- [ ] Featured demo/project commands were identified and smoke-tested locally (example:
  `python examples/run_demo.py` or project-specific run scripts).
- [ ] Static checks for changed code were executed (`make check` or equivalent target).
- [ ] Documentation build was run (`cd docs && make html`) if docs were changed in behavior or references.
- [ ] Link integrity was validated in changed docs (`git grep`/manual verification of all new/changed links).
- [ ] Notebook execution health was verified when notebooks were modified (`scripts/run_notebook_tests.py` or equivalent smoke execution).
- [ ] No generated runtime artifacts were introduced (e.g. `docs/_build`, `.ipynb_checkpoints`,
  `outputs`, `results`, `coverage`, `logs`).
- [ ] Artifact policy is updated if needed in
  [docs/development.md](docs/development.md), `.gitignore`, and/or workflow rules.

## Reviewer handoff

- [ ] PR description includes a short summary of what changed and why this update is portfolio-relevant.
- [ ] Known caveats/risks are explicitly called out and visible to reviewers.
- [ ] Any intentionally skipped checks are justified and have an owner.
- [ ] A follow-up maintainer review can validate the portfolio narrative in <5 minutes using the PR summary.

For major portfolio releases, include this checklist in the PR description and keep a completed copy with the merge commit.
