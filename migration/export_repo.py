from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("migration.export_repo")


class ExportError(ValueError):
    """Raised when an export manifest or copy operation is unsafe."""


@dataclass(frozen=True)
class ManifestEntry:
    source: Path
    target: Path
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExportManifest:
    name: str
    destination: Path
    entries: tuple[ManifestEntry, ...]
    overwrite: bool = False
    report_path: Path | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CopiedFile:
    source: str
    target: str
    size_bytes: int
    executable: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExportError(f"Malformed JSON manifest: {path}") from exc
    if not isinstance(data, dict):
        raise ExportError("Manifest root must be a JSON object")
    return data


def _relative_path(value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ExportError(f"{field_name} must be a non-empty string")
    path = Path(value)
    if path.is_absolute():
        raise ExportError(f"{field_name} must be relative: {value}")
    if any(part == ".." for part in path.parts):
        raise ExportError(f"{field_name} must not contain '..': {value}")
    return path


def _safe_resolve(base: Path, relative: Path, field_name: str) -> Path:
    resolved = (base / relative).resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as exc:
        raise ExportError(f"{field_name} escapes base directory: {relative}") from exc
    return resolved


def parse_manifest(path: Path) -> ExportManifest:
    data = _load_json(path)
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ExportError("Manifest field 'name' must be a non-empty string")

    destination = _relative_path(data.get("destination"), "destination")
    entries_data = data.get("entries")
    if not isinstance(entries_data, list) or not entries_data:
        raise ExportError("Manifest field 'entries' must be a non-empty list")

    entries: list[ManifestEntry] = []
    for index, item in enumerate(entries_data):
        if not isinstance(item, dict):
            raise ExportError(f"entries[{index}] must be an object")
        source = _relative_path(item.get("source"), f"entries[{index}].source")
        target = _relative_path(
            item.get("target", item.get("source")),
            f"entries[{index}].target",
        )
        excludes_raw = item.get("exclude", [])
        if not isinstance(excludes_raw, list) or not all(
            isinstance(pattern, str) for pattern in excludes_raw
        ):
            raise ExportError(f"entries[{index}].exclude must be a list of strings")
        entries.append(
            ManifestEntry(
                source=source,
                target=target,
                exclude=tuple(excludes_raw),
            )
        )

    report_path = None
    if data.get("report_path") is not None:
        report_path = _relative_path(data["report_path"], "report_path")

    metadata_raw = data.get("metadata", {})
    if not isinstance(metadata_raw, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in metadata_raw.items()
    ):
        raise ExportError("metadata must be an object with string keys and values")

    return ExportManifest(
        name=name,
        destination=destination,
        entries=tuple(entries),
        overwrite=bool(data.get("overwrite", False)),
        report_path=report_path,
        metadata=dict(metadata_raw),
    )


def _is_excluded(relative: Path, patterns: tuple[str, ...]) -> bool:
    text = relative.as_posix()
    return any(
        fnmatch.fnmatch(text, pattern) or fnmatch.fnmatch(relative.name, pattern)
        for pattern in patterns
    )


def _copy_file(source: Path, target: Path) -> CopiedFile:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    mode = source.stat().st_mode
    if mode & stat.S_IXUSR:
        target.chmod(target.stat().st_mode | stat.S_IXUSR)
    return CopiedFile(
        source=str(source),
        target=str(target),
        size_bytes=target.stat().st_size,
        executable=bool(target.stat().st_mode & stat.S_IXUSR),
    )


def _copy_entry(
    repo_root: Path,
    destination_root: Path,
    entry: ManifestEntry,
) -> list[CopiedFile]:
    source = _safe_resolve(repo_root, entry.source, "source")
    target = _safe_resolve(destination_root, entry.target, "target")
    if not source.exists():
        raise ExportError(f"Source path does not exist: {entry.source}")

    copied: list[CopiedFile] = []
    if source.is_file():
        if not _is_excluded(Path(source.name), entry.exclude):
            copied.append(_copy_file(source, target))
        return copied

    if not source.is_dir():
        raise ExportError(f"Source path is neither file nor directory: {entry.source}")

    for child in sorted(source.rglob("*")):
        relative = child.relative_to(source)
        if _is_excluded(relative, entry.exclude):
            continue
        child_target = target / relative
        if child.is_dir():
            child_target.mkdir(parents=True, exist_ok=True)
        elif child.is_file():
            copied.append(_copy_file(child, child_target))
    return copied


def export_repo(
    manifest: ExportManifest,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = repo_root.resolve() if repo_root is not None else _repo_root()
    destination_root = _safe_resolve(root, manifest.destination, "destination")

    if destination_root.exists():
        if not manifest.overwrite:
            raise ExportError(f"Destination already exists: {manifest.destination}")
        if destination_root.is_file():
            raise ExportError("Destination exists as a file")
    destination_root.mkdir(parents=True, exist_ok=True)

    copied_files: list[CopiedFile] = []
    for entry in manifest.entries:
        LOGGER.info("copying %s -> %s", entry.source, entry.target)
        copied_files.extend(_copy_entry(root, destination_root, entry))

    report = {
        "name": manifest.name,
        "destination": str(destination_root.relative_to(root)),
        "file_count": len(copied_files),
        "bytes": sum(item.size_bytes for item in copied_files),
        "metadata": manifest.metadata,
        "files": [
            {
                "source": str(Path(item.source).relative_to(root)),
                "target": str(Path(item.target).relative_to(destination_root)),
                "size_bytes": item.size_bytes,
                "executable": item.executable,
            }
            for item in copied_files
        ],
    }

    report_relative = (
        manifest.report_path or manifest.destination / "export_report.json"
    )
    report_path = _safe_resolve(root, report_relative, "report_path")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy a manifest-defined export snapshot."
    )
    parser.add_argument("manifest", type=Path, help="Path to a JSON migration manifest")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root. Defaults to the parent of this migration directory.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable info logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    try:
        manifest = parse_manifest(args.manifest)
        report = export_repo(manifest, repo_root=args.repo_root)
    except ExportError as exc:
        LOGGER.error("%s", exc)
        return 2
    print(
        json.dumps(
            {key: report[key] for key in ("name", "destination", "file_count", "bytes")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
