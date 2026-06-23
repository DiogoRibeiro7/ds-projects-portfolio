# Portfolio Release Checklist

Use this checklist for PRs that materially affect portfolio content (notebooks,
project READMEs, screenshots, demos, or public landing pages).

## Portfolio quality

- [ ] Core portfolio pages (`README.md`, `docs/README_ENHANCED.md`) were
  updated to keep featured content accurate.
- [ ] Every touched project README includes clear run instructions and expected
  outcomes.
- [ ] Notebook structure and narrative follow the portfolio template where
  applicable (goal, setup, results, next steps).
- [ ] Broken links in updated docs were checked with link-aware review.
- [ ] Visual artifacts (plots/tables/screenshots) remain readable and match current
  outputs where relevant.

## Technical validation

- [ ] Local smoke tests were run for modified demos/projects.
- [ ] Static checks for changed code were executed (`make check` or equivalent
  target).
- [ ] Documentation build was run (`cd docs && make html`) when doc behavior or
  references changed.
- [ ] No generated runtime artifacts were introduced (e.g. `docs/_build`, `.ipynb_checkpoints`,
  `test_results`, `outputs`, `results`, `logs`, `coverage` artifacts).
- [ ] Artifact policy is unchanged or updated in
  [docs/development.md](docs/development.md) and `.gitignore`.

## Reviewer handoff

- [ ] PR description includes a short summary of what changed and why.
- [ ] Known caveats/risks are explicitly called out.
- [ ] Any intentionally omitted checks are justified in the PR body.

For major portfolio releases, include this checklist in the PR description and
keep a completed copy with the merge commit.
