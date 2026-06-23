## Portfolio release checklist

For PRs that materially change portfolio-facing content, complete this section
and copy the result into the PR description before opening:

- [ ] This change is a major portfolio update (public README/docs/notebooks/projects/screenshots).
- [ ] Covered items from
  [docs/PORTFOLIO_RELEASE_CHECKLIST.md](/docs/PORTFOLIO_RELEASE_CHECKLIST.md)
  were run and passed.
- [ ] A final summary of portfolio impact is included for reviewer validation.
- [ ] If any checks were skipped, each exception is documented with owner + reason.

### Compact required checks

- [ ] Portfolio-facing links were verified (README/docs/project README/demos).
- [ ] Featured project and notebook guidance was reviewed for this PR.
- [ ] At least one representative demo run command was executed successfully.
- [ ] No generated runtime artifacts were introduced or left tracked.

### Optional checks (run when relevant)

- [ ] `make check` completed after the changes.
- [ ] `cd docs && make html` completed after doc changes.
- [ ] Smoke tests executed for project/demo paths changed in this PR.
