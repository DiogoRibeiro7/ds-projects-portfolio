from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_BYTES = 5 * 1024 * 1024
OBSOLETE_URLS = (
    "github.com/diogoribeiro7/data-science-portfolio",
    "diogoribeiro7.github.io/data-science-portfolio",
)
SECRET_NAME_PARTS = (
    ".env",
    "id_rsa",
    "id_dsa",
    "id_ed25519",
    "secret",
    "secrets",
    "credential",
    "credentials",
)
GENERATED_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
    ".ipynb_checkpoints",
    "htmlcov",
}


@dataclass(frozen=True)
class ValidationOptions:
    max_bytes: int = DEFAULT_MAX_BYTES
    allow_large: tuple[str, ...] = ()
    allow_src_imports: bool = False
    allow_projects_imports: bool = False
    require_ci: bool = True
    require_tests: bool = False


@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    message: str


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _matches(path: str, patterns: tuple[str, ...]) -> bool:
    return path in patterns or any(Path(path).match(pattern) for pattern in patterns)


def _text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _import_roots(path: Path) -> set[str]:
    source = _text(path)
    if source is None:
        return set()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return set()
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", maxsplit=1)[0])
    return roots


def validate_export(root: Path, options: ValidationOptions) -> list[Finding]:
    export_root = root.resolve()
    findings: list[Finding] = []

    if not export_root.exists() or not export_root.is_dir():
        return [
            Finding(
                code="missing_root",
                path=str(root),
                message="Export root does not exist or is not a directory",
            )
        ]

    if not (export_root / "README.md").is_file():
        findings.append(Finding("missing_readme", "README.md", "Expected README.md"))

    dependency_files = ("pyproject.toml", "requirements.txt", "environment.yml")
    if not any((export_root / name).is_file() for name in dependency_files):
        findings.append(
            Finding(
                "missing_dependencies",
                ".",
                "Expected pyproject.toml, requirements.txt, or environment.yml",
            )
        )

    ci_dir = export_root / ".github" / "workflows"
    if (
        options.require_ci
        and not any(ci_dir.glob("*.yml"))
        and not any(ci_dir.glob("*.yaml"))
    ):
        findings.append(
            Finding("missing_ci", ".github/workflows", "Expected CI workflow")
        )

    if options.require_tests and not (export_root / "tests").is_dir():
        findings.append(Finding("missing_tests", "tests", "Expected tests directory"))

    if (export_root / ".git").exists():
        findings.append(
            Finding("accidental_git", ".git", "Export must not contain .git")
        )

    for path in _iter_files(export_root):
        relative = _relative(path, export_root)
        parts = set(path.relative_to(export_root).parts)

        generated = parts.intersection(GENERATED_DIRS)
        if generated:
            findings.append(
                Finding(
                    "generated_cache",
                    relative,
                    f"Generated/cache path present: {sorted(generated)[0]}",
                )
            )

        lowered_name = path.name.lower()
        if any(part in lowered_name for part in SECRET_NAME_PARTS):
            findings.append(
                Finding("secret_like_file", relative, "Secret-like filename")
            )

        if path.stat().st_size > options.max_bytes and not _matches(
            relative, options.allow_large
        ):
            findings.append(Finding("large_file", relative, "File exceeds size limit"))

        if path.suffix == ".py":
            roots = _import_roots(path)
            if "src" in roots and not options.allow_src_imports:
                findings.append(
                    Finding("unresolved_src_import", relative, "Found src.* import")
                )
            if "projects" in roots and not options.allow_projects_imports:
                findings.append(
                    Finding(
                        "unresolved_projects_import",
                        relative,
                        "Found projects.* import",
                    )
                )

        text = _text(path)
        if text is not None:
            for obsolete_url in OBSOLETE_URLS:
                if obsolete_url in text:
                    findings.append(
                        Finding(
                            "obsolete_repo_url",
                            relative,
                            f"Found obsolete monorepo URL: {obsolete_url}",
                        )
                    )

    return findings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an exported repository candidate."
    )
    parser.add_argument("export_root", type=Path)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--allow-large", action="append", default=[])
    parser.add_argument("--allow-src-imports", action="store_true")
    parser.add_argument("--allow-projects-imports", action="store_true")
    parser.add_argument("--no-ci-required", action="store_true")
    parser.add_argument("--require-tests", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit JSON findings")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = ValidationOptions(
        max_bytes=args.max_bytes,
        allow_large=tuple(args.allow_large),
        allow_src_imports=args.allow_src_imports,
        allow_projects_imports=args.allow_projects_imports,
        require_ci=not args.no_ci_required,
        require_tests=args.require_tests,
    )
    findings = validate_export(args.export_root, options)
    if args.json:
        print(json.dumps([finding.__dict__ for finding in findings], indent=2))
    else:
        for finding in findings:
            print(f"{finding.code}: {finding.path}: {finding.message}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
