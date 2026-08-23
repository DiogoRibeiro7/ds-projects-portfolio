# Documentation Contribution Guide

This guide complements `CONTRIBUTING.md` by focusing exclusively on docstrings,
inline comments, and the documentation site. Follow it when adding features,
fixing bugs, or improving notebooks to keep documentation consistent.

## 1. Pick the Right Targets

1. Identify which modules/symbols are public using `DOCS_STYLE.md`.
2. Locate the most-consumed APIs via:
   - `rg -n "from <module>" -g"*.py"` to inspect imports.
   - CLI entry points defined in `pyproject.toml`.
   - Tests (`tests/`), tutorials (`tutorials/`), and docs (`docs/usage.md`,
     `docs/api/`) to see what gets exercised most frequently.
3. Update docstrings for any public symbol you touch, and add `Examples`
   sections when the API is used by notebooks, examples, docs, or tests.

## 2. Writing Docstrings

- Follow the Google-style template (summary → Args → Returns → Raises →
  Side Effects → Complexity → Examples). See `DOCS_STYLE.md` for details.
- Include parameter types, units, default behavior, and edge-case handling.
- Describe side effects (file I/O, network, cache) and computational complexity
  when they influence usage or cost.
- Examples **must** be runnable snippets importing the real function/class.
  Include one default path and one edge case (zero variance, empty input, etc.).
- Keep docstrings synchronized with type hints and tests. If behavior is
  ambiguous, inspect usages before documenting. Use `TODO(username): <issue>`
  when the code disagrees with documentation and needs work.

## 3. Inline Comments

- Comment *why* the code needs invariants, guardrails, or non-obvious math.
- Add “guardrail” comments before tricky loops, statistical adjustments,
  caching layers, and concurrency logic. Avoid repeating what the code already
  states.
- Keep comments short, using plain sentences prefixed with `# `. Multi-line
  explanations should read like small paragraphs rather than bullet lists.

## 4. Documentation Validation Workflow

1. Run `pre-commit run --all-files` to execute Ruff, formatting, mypy, and
   repository hygiene hooks.
2. Build the docs site: `cd docs && make html`. This confirms that docstrings
   render correctly in the API reference.

## 5. Pull Request Checklist

- [ ] Every public module/class/function touched has a Google-style docstring.
- [ ] Usage-heavy APIs touched by the change contain runnable `Examples`.
- [ ] Inline comments explain intent/guardrails for complex logic.
- [ ] Type hints, Raises, and docstrings stay in sync.
- [ ] `pre-commit` hooks and the docs build pass locally.

Following this workflow keeps the documentation site, code, and CI automation in
lockstep, preventing regressions while making the APIs approachable.
