# Documentation Style Guide

This repository standardizes on **Google-style docstrings** for every public
module, class, function, and method. The sections below define what qualifies as
public, how docstrings are structured, how inline comments are used, and how
automation enforces these rules.

## Public API Definition

Public symbols are anything meant to be imported or executed by consumers:

- All Python packages under `src/` (CLI entry points, shared libraries, FastAPI
  services, data utilities).
- Project packages under `projects/` that expose importable Python modules used
  by notebooks, examples, or tests.
- Standalone scripts in `scripts/` (for example
  `scripts/advanced_experimentation_platform.py`, `scripts/analyze_experiment.py`).
- Any module exported from `__init__.py`, referenced by documentation, invoked
  by CLI entry points (`pyproject.toml` / `src/api/ml_api.py`), or imported by
  automated jobs/tests/notebooks.

Inside a public module, a symbol counts as public when:

- Its name does not start with `_`.
- It is listed in `__all__`.
- It is referenced by another public symbol or by tooling (e.g., CLI, notebooks,
  dashboards).

All public symbols must have docstrings. Private helpers may omit docstrings
only when their behavior is already explained by the surrounding documentation.

## Docstring Fundamentals

1. **Summary line**: Imperative, ≤80 characters, describing what the symbol
   does rather than how it is implemented.
2. **Args / Parameters**: Every parameter appears in declaration order. Include
   types (matching type hints), accepted units/ranges (e.g., probabilities in
   `[0, 1]`, durations in seconds), default behavior, and required shapes.
3. **Returns / Yields**: Required even for `None` returns (state side effects).
   For generator functions use `Yields`. Include container shapes or schemas.
4. **Raises**: List every intentionally raised exception; omit generic runtime
   failures such as `TypeError` from Python internals.
5. **Side Effects**: Document file I/O, network calls, global cache mutations,
   or environment manipulation in a dedicated `Side Effects:` section or
   `Notes:` paragraph.
6. **Complexity**: When the runtime dominates feature usage (e.g., `O(n log n)`
   simulations), add a `Complexity:` sentence after `Side Effects`.
7. **Examples**: Provide runnable snippets for public APIs referenced by CLI,
   documentation, or notebooks. The “top 20” most-used public functions/classes
   identified via repository-wide `rg` usage counts must include an `Examples`
   section describing typical and edge-case behavior.

Classes and dataclasses include an `Attributes:` section documenting fields that
callers read. Module docstrings summarize responsibilities, note constants, and
cross-reference related modules or documentation.

### Canonical Template

````markdown
def run_power_simulation(
    baseline_rate: float,
    effect_size: float,
    sample_size: int,
    *,
    alpha: float = 0.05,
) -> SimulationResult:
    """Run a Monte Carlo power simulation for a single metric.

    Args:
        baseline_rate (float): Baseline conversion probability in [0, 1].
        effect_size (float): Absolute lift relative to ``baseline_rate``.
        sample_size (int): Users per variant in each simulation.
        alpha (float): Significance level used to classify the outcome.

    Returns:
        SimulationResult: Power, confidence interval bounds, diagnostics.

    Raises:
        ValueError: If ``baseline_rate`` or ``effect_size`` fall outside (0, 1).

    Side Effects:
        Uses NumPy's global RNG; set ``np.random.seed`` for reproducibility.

    Complexity:
        O(sample_size × simulations) time, O(1) memory.

    Examples:
        >>> result = run_power_simulation(0.12, 0.02, 5000, alpha=0.01)
        >>> result.power
        0.84
    ```
````

### Inline Comments

- **Explain intent**: Comment *why* the code is structured a certain way (e.g.,
  “Guardrail: stop Holm correction at first failure to preserve monotonicity”).
- **Required** for invariants, numerical tricks, concurrency/rate-limiting,
  caching behavior, and business rules that the code alone does not convey.
- **Forbidden** when narrating obvious operations (e.g., `# increment i`), when
  they duplicate the docstring, or when they describe how to read Python syntax.
- For longer explanations, use short paragraph-style comments (`# ...`) rather
  than block comments. Prefix actionable follow-ups with
  `TODO(username): <issue link>` if a tracker exists.

## Type Hints, Raises, Returns, and Examples

- All public callables must be fully type hinted, including `*args`, `**kwargs`,
  return values, and class attributes (dataclasses, TypedDicts).
- Keep docstring descriptions synchronized with type hints. If additional
  constraints exist (e.g., arrays must be shape `(N, D)`), document them in the
  docstring.
- Every raised exception listed in `Raises` must match the implementation; if
  behavior varies by parameter values, explain the guardrail.
- Functions invoked by tutorials, CLIs, or tests must provide at least one
  runnable example covering default behavior and a second note for edge cases
  (e.g., “zero-variance data returns `np.nan` effect sizes”).

## Documentation Workflow

1. Update or add docstrings when touching public code.
2. Add inline comments when you introduce domain-specific logic or guardrails.
3. Run `pre-commit run --all-files` for formatting, linting, type checking, and
   repository hygiene hooks.
4. Build docs with `(cd docs && make html)` to ensure docstrings render
   correctly inside the API reference.

## Automation

- `ruff` enforces formatting and the selected lint rules configured in
  `pyproject.toml`.
- Docstrings are maintained by review policy for public APIs, tutorials, and
  reusable project code.
- CI (`.github/workflows/ci.yml`) runs lint → type check → unit/integration/data
  tests → docs build to keep docstrings, coverage, and site generation aligned.
- Pre-commit hooks (`.pre-commit-config.yaml`) execute formatting, `ruff`,
  `mypy`, and repository hygiene checks locally. Fix issues before opening a
  pull request.

## Review Checklist

1. Does the module/class/function count as public per the definition above?
2. Does the docstring follow the Google template with Args/Returns/Raises,
   Side Effects (if applicable), Complexity (if non-trivial), and Examples for
   high-usage APIs?
3. Are inline comments limited to intent/invariant explanations?
4. Are type hints, exception handling, and docstring text kept in sync?
5. Were docs rebuilt to verify rendering?

Following these rules keeps documentation consistent, enforceable, and genuinely
useful for everyone consuming the data science toolkit.
