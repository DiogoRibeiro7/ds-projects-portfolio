# Migration Tooling

This directory contains non-destructive tooling for creating and validating copy-first export snapshots from the portfolio repository.

The tools do not move or delete source material. They copy explicitly listed paths from a JSON manifest into a destination directory and validate the resulting candidate repository.

## Export

```bash
python migration/export_repo.py migration/manifests/example.json
```

Manifest shape:

```json
{
  "name": "example-export",
  "destination": "migration/exports/example-export",
  "overwrite": false,
  "entries": [
    {"source": "README.md", "target": "README.md"},
    {"source": "src/example", "target": "src/example", "exclude": ["__pycache__"]}
  ]
}
```

Only relative source paths are allowed. Paths that escape the repository root or destination root are rejected.

## Validate

```bash
python migration/validate_export.py migration/exports/example-export
```

The validator checks for common extraction mistakes:

- unresolved `src.*` and `projects.*` imports;
- files larger than 5 MB unless allowlisted;
- missing README, dependency metadata, CI, or tests when required;
- accidental `.git/`, generated caches, notebook checkpoints, and secret-like files;
- obsolete monorepo repository URLs.

## Quality Gates

```bash
ruff check migration
ruff format --check migration
mypy migration
pytest migration
```
