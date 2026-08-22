from __future__ import annotations

import json
from pathlib import Path

import pytest

from export_repo import ExportError, export_repo, parse_manifest


def _write_manifest(tmp_path: Path, data: dict[str, object]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_export_copies_explicit_files_with_rename(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "source").mkdir()
    (repo / "source" / "README.md").write_text("hello", encoding="utf-8")

    manifest_path = _write_manifest(
        tmp_path,
        {
            "name": "demo",
            "destination": "out",
            "entries": [{"source": "source/README.md", "target": "docs/README.md"}],
        },
    )

    report = export_repo(parse_manifest(manifest_path), repo_root=repo)

    assert (repo / "out" / "docs" / "README.md").read_text(encoding="utf-8") == "hello"
    assert report["file_count"] == 1


def test_export_excludes_matching_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "keep.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "src" / "drop.pyc").write_bytes(b"cache")

    manifest_path = _write_manifest(
        tmp_path,
        {
            "name": "demo",
            "destination": "out",
            "entries": [{"source": "src", "target": "src", "exclude": ["*.pyc"]}],
        },
    )

    export_repo(parse_manifest(manifest_path), repo_root=repo)

    assert (repo / "out" / "src" / "keep.py").is_file()
    assert not (repo / "out" / "src" / "drop.pyc").exists()


def test_export_refuses_overwrite_without_flag(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("hello", encoding="utf-8")
    (repo / "out").mkdir()
    manifest_path = _write_manifest(
        tmp_path,
        {
            "name": "demo",
            "destination": "out",
            "entries": [{"source": "README.md"}],
        },
    )

    with pytest.raises(ExportError, match="Destination already exists"):
        export_repo(parse_manifest(manifest_path), repo_root=repo)


def test_manifest_rejects_absolute_and_parent_paths(tmp_path: Path) -> None:
    absolute_manifest = _write_manifest(
        tmp_path,
        {
            "name": "bad",
            "destination": "out",
            "entries": [{"source": str(tmp_path / "README.md")}],
        },
    )
    with pytest.raises(ExportError, match="must be relative"):
        parse_manifest(absolute_manifest)

    parent_manifest = _write_manifest(
        tmp_path,
        {
            "name": "bad",
            "destination": "out",
            "entries": [{"source": "../README.md"}],
        },
    )
    with pytest.raises(ExportError, match="must not contain"):
        parse_manifest(parent_manifest)


def test_manifest_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ExportError, match="Malformed JSON"):
        parse_manifest(path)
