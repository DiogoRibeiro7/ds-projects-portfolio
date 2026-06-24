# Development Guide

This document was moved to [docs/contributor/development.md](./contributor/development.md).

Use the contributor guide for the authoritative workflow, tooling, and release notes:

- [docs/contributor/development.md](./contributor/development.md)

> NOTE: Full `mypy` coverage across production packages is still blocked by legacy type debt and missing third-party stubs. Use Python 3.11 for local type-checking:
>
> ```bash
> py -3.11 -m mypy src tools --python-version 3.11 --ignore-missing-imports
> ```

See the same note and the latest commands in
`docs/contributor/development.md`.
