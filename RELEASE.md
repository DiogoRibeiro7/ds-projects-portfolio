# Release Process

## Versioning

We follow [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`):

- **MAJOR**: incompatible API changes or breaking behavior.
- **MINOR**: backwards-compatible feature additions.
- **PATCH**: backwards-compatible bug fixes / doc updates.

Every change merged to `main` should bump `CHANGELOG.md` under the `Unreleased` section. During release, move those notes to a new versioned section (e.g., `## [1.2.0] - 2026-01-07`).

## Changelog Strategy

- Use `CHANGELOG.md` to summarize highlights (features, fixes, deprecations).
- Keep entries terse (“Added X”, “Fixed Y bug”). Include PR/issue links when possible.

## Release Workflow

1. **Ensure main is clean**
   ```bash
   git checkout main
   git pull origin main
   ```
2. **Decide the next version** (SemVer).
3. **Update metadata**
   - Bump `project.version` in `pyproject.toml`.
   - Move `CHANGELOG.md` “Unreleased” entries under a new `## [x.y.z] - YYYY-MM-DD` header.
4. **Run the quality gate**
   ```bash
   make check
   make docs   # optional but recommended
   ```
5. **Commit the release prep**
   ```bash
   git commit -am "chore(release): vX.Y.Z"
   ```
6. **Tag the release**
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin main --tags
   ```
7. **Build artifacts**
   ```bash
   make build
   ```
   Upload the `dist/` artifacts to your package registry (PyPI, internal storage, etc.) as needed.
8. **Open a GitHub release** (optional but recommended)
   - Title: `vX.Y.Z`
   - Body: copy the changelog entry.
9. **Reset `CHANGELOG.md`**
   - Re-add the `## [Unreleased]` placeholder for future work.

### GitHub Actions alternative

Instead of doing this manually, run the `Release` workflow from the Actions tab:

1. Select `Release` → `Run workflow`.
2. Provide the target SemVer (e.g., `1.3.0`) that already exists in `pyproject.toml` and `CHANGELOG.md`.
3. The workflow will:
   - run `make check` + `make docs`
   - build and upload `dist/` artifacts
   - tag the repo (`vX.Y.Z`) and create a GitHub release

Use that automation once metadata is prepared (version/changelog updated).

Keep releases deterministic: do not merge additional commits between running `make check` and tagging. For emergencies, cut a hotfix branch off the last tag and follow the same steps.
