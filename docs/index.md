# Data Science Projects Portfolio

This documentation is the reviewer-facing entry point for the portfolio. It
summarizes how the repository is organized, which projects to inspect first, and
which commands validate the maintained surface.

```{toctree}
:caption: Portfolio docs
:maxdepth: 2
:hidden:

portfolio_overview
architecture
PORTFOLIO_SCOPE
installation
quickstart
usage
DATA_QUALITY
ROBUSTNESS
methodology/index
tutorials/index
api/index
```

```{toctree}
:caption: Contributor docs
:maxdepth: 2
:hidden:

internal
```

## Review Path

Start with the shortest public path:

1. Read the repository
   [`README.md`](https://github.com/DiogoRibeiro7/ds-projects-portfolio/blob/main/README.md)
   for the project narrative.
2. Inspect
   [`projects.yml`](https://github.com/DiogoRibeiro7/ds-projects-portfolio/blob/main/projects.yml)
   for the featured project catalogue.
3. Open the featured notebooks listed in the README when evaluating storytelling,
   methods, and reproducibility.
4. Use [`docs/architecture.md`](architecture.md) and
   [`docs/PORTFOLIO_SCOPE.md`](PORTFOLIO_SCOPE.md) to understand what belongs in
   the active portfolio surface.

## Portfolio Signal

The repository is organized around practical data science work:

- Statistical modeling, experimentation, and uncertainty communication.
- Healthcare, finance, customer analytics, geospatial, time-series, and GenAI
  project examples.
- Reusable Python modules, tests, and lightweight validation workflows.
- Clear separation between active portfolio material and archived/reference
  material.

## Quick Validation

From the repository root:

```bash
python examples/run_demo.py
make check
```

For documentation-only changes:

```bash
cd docs
make html
```

## Maintained Boundaries

The active tree should stay easy to review. Runtime outputs, caches, generated
artifacts, local experiment dumps, and obsolete process work should not be part
of the public portfolio surface.
