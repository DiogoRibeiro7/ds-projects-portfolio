from __future__ import annotations

from pathlib import Path

from validate_export import ValidationOptions, validate_export


def _basic_export(tmp_path: Path) -> Path:
    root = tmp_path / "export"
    root.mkdir()
    (root / "README.md").write_text("# Demo\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    workflows = root / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text("name: CI\n", encoding="utf-8")
    return root


def _codes(root: Path, options: ValidationOptions | None = None) -> set[str]:
    return {
        finding.code
        for finding in validate_export(root, options or ValidationOptions())
    }


def test_validate_clean_minimal_export(tmp_path: Path) -> None:
    root = _basic_export(tmp_path)

    assert validate_export(root, ValidationOptions()) == []


def test_validate_unresolved_import_detection(tmp_path: Path) -> None:
    root = _basic_export(tmp_path)
    package = root / "src" / "demo"
    package.mkdir(parents=True)
    (package / "bad.py").write_text(
        "import src.statistics\nfrom projects.foo import bar\n",
        encoding="utf-8",
    )

    codes = _codes(root)

    assert "unresolved_src_import" in codes
    assert "unresolved_projects_import" in codes


def test_validate_large_file_detection_and_allowlist(tmp_path: Path) -> None:
    root = _basic_export(tmp_path)
    large = root / "large.bin"
    large.write_bytes(b"x" * 200)

    strict_codes = _codes(root, ValidationOptions(max_bytes=100))
    allowed_codes = _codes(
        root, ValidationOptions(max_bytes=100, allow_large=("large.bin",))
    )

    assert "large_file" in strict_codes
    assert "large_file" not in allowed_codes


def test_validate_generated_cache_and_secret_like_files(tmp_path: Path) -> None:
    root = _basic_export(tmp_path)
    cache = root / "__pycache__"
    cache.mkdir()
    (cache / "demo.pyc").write_bytes(b"cache")
    (root / ".env").write_text("TOKEN=abc\n", encoding="utf-8")

    codes = _codes(root)

    assert "generated_cache" in codes
    assert "secret_like_file" in codes


def test_validate_missing_required_files(tmp_path: Path) -> None:
    root = tmp_path / "export"
    root.mkdir()

    codes = _codes(root, ValidationOptions(require_tests=True))

    assert "missing_readme" in codes
    assert "missing_dependencies" in codes
    assert "missing_ci" in codes
    assert "missing_tests" in codes


def test_validate_obsolete_url_detection(tmp_path: Path) -> None:
    root = _basic_export(tmp_path)
    (root / "README.md").write_text(
        "https://github.com/diogoribeiro7/data-science-portfolio\n",
        encoding="utf-8",
    )

    assert "obsolete_repo_url" in _codes(root)
