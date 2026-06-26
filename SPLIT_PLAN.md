# Repository Split Plan

Plan for breaking the `ds-portfolio` monorepo into focused repositories.
**This is a plan only — nothing here has been executed.**

## 1. Goal and principles

- Turn one 182 MB monorepo (594 files, 293 commits) into a set of focused repos.
- **Maximise portfolio value**: polished analyses become individually pinnable
  GitHub repos with their own README, data, and narrative.
- **Isolate the reusable library** (`src/`) from the one-off analyses.
- **Shed history bloat**: a single 43 MB `artifacts/medical_imaging/model.pt`
  plus committed notebook outputs dominate the 182 MB `.git`. New repos should
  not inherit it.
- Keep each repo independently lintable, testable, and CI-able.

## 2. Why this split is safe (coupling evidence)

- **No genuine cross-project imports.** The only hit
  (`projects.advanced_customer_segmentation`) is that project's own tests
  importing itself.
- **Only 2 of 22 projects import `src/`:**
  - `feature_engineering` → `src.feature_engineering.*` (it is effectively a
    **demo of the core library**, including `from src.feature_engineering.utils
    import *`).
  - `streamlit_apps/ab_test_calculator.py` → `src.statistics.core`
    (`calculate_power`, `calculate_sample_size`) only.
- `src/` is consumed by `tests/` and the 43 top-level `notebooks/`, not by the
  analyses.

The 20 remaining analyses detach with zero code changes.

## 3. Target topology

| New repo | Contents | Source today |
|---|---|---|
| **`ds-portfolio-core`** | Reusable library + its tests, demo notebooks, docs, infra. Published as installable package `ds-portfolio`. | `src/`, `tests/`, `notebooks/`, `docs/`, `pyproject.toml`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/`, `Makefile`, `.pre-commit-config.yaml`, `requirements*.txt`, `setup.py`, `setup.cfg`, `pytest.ini` + `projects/feature_engineering` (moved in as `examples/feature_engineering`) |
| **`ds-apps`** | Interactive apps. Depends on `ds-portfolio-core` (or vendors 2 functions). | `projects/dashboard_enhanced`, `projects/streamlit_apps` |
| **One repo per Tier-A analysis** (see 4) | Standalone analysis, self-contained. | one `projects/<name>` each |
| **`ds-experiments`** | Thin/unpolished analyses grouped together. | the Tier-B `projects/*` |
| **`ds-portfolio-archive`** (optional) | Legacy + internal platform. | `archive/` |

## 4. Per-project tiering

### Tier A — own standalone repo (polished, self-contained)

| Project | Notes |
|---|---|
| `city_wage_cost_global` | ✅ done; reference template for the others |
| `porto_lisbon_uhi_exposure` | 13 py + nb, README, requirements |
| `pt_salary_gamma_distribution` | 12 py, README, requirements |
| `portugal_gdp_bayesian_revision` | README, requirements |
| `advanced_customer_segmentation` | has its own test suite |
| `ab_testing` | 17 notebooks — **add a README on the way out** |
| `deep_learning` | 6 notebooks, README |
| `nlp` | README |
| `churn_prediction` | README |
| `excessive_drinking_map_reproduction` | README, requirements |
| `oecd_obesity_analysis_notebook` | README, requirements |
| `portuguese_emigration_crises_labour_law` | README, requirements |
| `time_series` | 7 notebooks — **add a README on the way out** |

### Tier B — group into `ds-experiments` (thin, no README/requirements)

`causal_inference` (1 file), `customer_segmentation` (basic vs the "advanced"
one), `machine_learning`, `portugal_gdp_income_distribution` (1 notebook).

### Decide case-by-case

- `statistical_methods` (11 py) and `performance_optimization` (12 py) are
  sizeable **code toolkits**, not analyses. Two options: fold into
  `ds-portfolio-core` (as `src/` modules or `examples/`), or promote to their own
  repos **after adding a README**. Recommendation: fold into core.
- `feature_engineering` → **into `ds-portfolio-core`** (it is a library demo).

## 5. Resolved decisions (recommendations)

1. **Granularity:** tiered, as above — not 22 repos, not one. Flagship analyses
   get individual repos; thin ones are grouped.
2. **History:** **fresh `git init` + single snapshot commit** for each Tier-A
   analysis and `ds-experiments`. This drops the 182 MB bloat and is far simpler.
   Preserve history only where it adds value — `ds-portfolio-core` and
   `advanced_customer_segmentation` (shows engineering depth) — via
   `git filter-repo` **with blob stripping** (section 7).
3. **Library dependency:** keep `ds-portfolio-core` as the published package.
   `ds-apps` either `pip install ds-portfolio-core` or vendors the two
   `src.statistics.core` functions. Recommendation: vendor the two functions so
   `ds-apps` has no external coupling.

### Still genuinely open (your call)

- Whether `statistical_methods` / `performance_optimization` fold into core or
  become repos.
- Whether to merge the Portugal econ analyses (`portugal_gdp_*`,
  `pt_salary_gamma_distribution`) into a single `portugal-economics` repo vs keep
  separate for individual pinning.
- Whether `archive/` is migrated or simply dropped.

## 6. Coupling to resolve before moving

| Item | Action |
|---|---|
| `feature_engineering` → `src.feature_engineering` | Moves *with* core; rewrite imports from `src.feature_engineering` to the installed package path. |
| `streamlit_apps` → `src.statistics.core` | Vendor `calculate_power` / `calculate_sample_size` into the app, or add `ds-portfolio-core` as a dependency. |
| `advanced_customer_segmentation` self-import via `projects.` namespace | Rewrite `from projects.advanced_customer_segmentation...` to a relative/package import once it is a repo root. |

## 7. History & bloat strategy

**Every** history-preserving split must strip the large blobs or each clone
carries 182 MB.

- **Fresh snapshot (default for analyses):**
  ```bash
  # reference only — run later, per project
  mkdir ../<repo> && cp -r projects/<name>/* ../<repo>/
  cd ../<repo> && git init && git add . && git commit -m "Initial import of <name>"
  ```
- **History-preserving (core + advanced_customer_segmentation):**
  ```bash
  # reference only
  git clone --no-local . ../<repo> && cd ../<repo>
  git filter-repo --path <subdir>/ --path-rename <subdir>/:        # keep only that path
  git filter-repo --strip-blobs-bigger-than 5M                     # drop model.pt etc.
  ```
- Remove `artifacts/medical_imaging/model.pt` (43 MB) from any preserved history;
  if the model is needed, store it via Git LFS or a release asset, not in git.

## 8. Shared tooling each new repo needs (templated)

The monorepo config is global; splits lose it unless copied:

- **`.gitignore`** — copy the relevant slice (`data/`, `outputs/`, `*.parquet`,
  `.ipynb_checkpoints`, caches). ⚠️ the current global `**/data/` rule silently
  excludes `data/manual/` template files — fix per repo.
- **Lint/type/test config** — extract each project's needs from the root
  `pyproject.toml` (ruff, isort, black, mypy, pytest) into a per-repo
  `pyproject.toml`.
- **CI** — a minimal `.github/workflows/ci.yml` per repo (lint + notebook smoke
  test). The current `notebook-tests.yml` is a good base.
- **`.pre-commit-config.yaml`**, **`LICENSE`**, **`CONTRIBUTING.md`** — copy.
- **`requirements.txt`** — each Tier-A project already has one or needs a small
  one pinned from `requirements-core.txt`.

## 9. Suggested sequencing (phased, low-risk)

1. **Pilot** — split `city_wage_cost_global` first (already clean). Establish the
   per-repo template: README, `.gitignore`, `requirements.txt`, `pyproject.toml`,
   `ci.yml`. Reuse for the rest.
2. **Core** — carve out `ds-portfolio-core` (history-preserving + blob strip),
   move `feature_engineering` in, publish the package, confirm `tests/` pass.
3. **Tier-A analyses** — one repo each (fresh snapshot), reusing the pilot template.
4. **`ds-apps`** — vendor the 2 functions, move the two app projects.
5. **`ds-experiments`** — group the Tier-B leftovers.
6. **Decide & migrate** `statistical_methods`, `performance_optimization`,
   `archive/`.
7. **Retire** the monorepo: leave a stub README pointing to the new repos (don't
   delete history).

## 10. Per-repo "definition of done" checklist

- [ ] Builds/imports with no reference to `src.` or `projects.` (unless it is core)
- [ ] Own `README.md` with headline + how-to-run
- [ ] Own `.gitignore` (data/outputs excluded; `data/manual/` templates kept)
- [ ] Own `requirements.txt` (pinned) and/or `pyproject.toml`
- [ ] Minimal CI green (lint + notebook smoke run)
- [ ] No blob > 5 MB in history
- [ ] `LICENSE` present

## 11. Risks

- **History bloat** propagating into every repo (mitigation: §7 blob strip / fresh snapshot).
- **Silent data loss** from the global `**/data/` ignore (mitigation: per-repo `.gitignore` review).
- **Lost CI/lint** if config isn't templated (mitigation: §8).
- **Broken imports** in the 2 coupled projects (mitigation: §6 done before moving).
- **Notebook reproducibility** — splits must keep each notebook's data fetch +
  cache logic intact (the analyses re-download public data on first run).
