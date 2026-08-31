"""Value-free provenance and schema audit for the official Open Bandit Dataset archive."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import TextIOWrapper
from pathlib import Path
from typing import Any
from zipfile import ZipFile, ZipInfo


PRIMARY_PROPENSITY_FIELD = "propensity_score"
LEGACY_PROPENSITY_FIELD = "action_prob"
REQUIRED_BASE_FIELDS = frozenset({"timestamp", "item_id", "position", "click"})
POLICIES = ("bts", "random")
CAMPAIGN = "all"
EXPECTED_ACTION_IDS = frozenset(range(81))
EXPECTED_RAW_POSITIONS = frozenset({"1", "2", "3"})


@dataclass(frozen=True)
class LoggedFileAudit:
    """Schema/provenance metadata that deliberately excludes outcome aggregates."""

    member: str
    compressed_size: int
    uncompressed_size: int
    sha256: str
    row_count: int
    columns: tuple[str, ...]
    propensity_field: str
    timestamp_min: str
    timestamp_max: str
    raw_positions: tuple[str, ...]
    action_count: int
    action_min: int
    action_max: int


@dataclass(frozen=True)
class ItemContextAudit:
    member: str
    compressed_size: int
    uncompressed_size: int
    sha256: str
    row_count: int
    columns: tuple[str, ...]
    item_count: int
    item_min: int
    item_max: int


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def audit_archive(path: Path, *, enforce_official_contract: bool = True) -> dict[str, Any]:
    """Audit official OBD structure without emitting CTR or reward summaries."""
    if not path.is_file():
        raise FileNotFoundError(path)

    with ZipFile(path) as archive:
        logged: dict[str, LoggedFileAudit] = {}
        contexts: dict[str, ItemContextAudit] = {}
        for policy in POLICIES:
            logged_info = _find_unique_member(archive, f"/{policy}/{CAMPAIGN}/{CAMPAIGN}.csv")
            context_info = _find_unique_member(archive, f"/{policy}/{CAMPAIGN}/item_context.csv")
            logged[policy] = _audit_logged_csv(archive, logged_info)
            contexts[policy] = _audit_item_context(archive, context_info)

    if logged["bts"].action_count != logged["random"].action_count:
        raise ValueError("BTS and Random all-campaign action counts differ.")
    if contexts["bts"].item_count != contexts["random"].item_count:
        raise ValueError("BTS and Random item-context counts differ.")
    if contexts["bts"].item_count != logged["bts"].action_count:
        raise ValueError("logged action count and item-context count disagree.")

    leftmost_raw_position = _leftmost_common_position(logged)
    if max(logged["bts"].timestamp_min, logged["random"].timestamp_min) > min(
        logged["bts"].timestamp_max, logged["random"].timestamp_max
    ):
        raise ValueError("BTS and Random timestamp ranges do not overlap.")

    if enforce_official_contract:
        _enforce_official_contract(logged, contexts)

    return {
        "archive": {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "campaign": CAMPAIGN,
        "logged_files": {key: asdict(value) for key, value in logged.items()},
        "item_context": {key: asdict(value) for key, value in contexts.items()},
        "audited_action_count": logged["bts"].action_count,
        "expected_action_count": len(EXPECTED_ACTION_IDS),
        "leftmost_raw_position": leftmost_raw_position,
        "normalized_leftmost_position": 0,
        "official_contract_enforced": enforce_official_contract,
        "scientific_boundary": (
            "This source audit contains provenance and schema metadata only. It intentionally "
            "omits click counts, CTRs, reward means, OPE estimates, policy rankings, challenger "
            "values, and promotion decisions."
        ),
    }


def _enforce_official_contract(
    logged: dict[str, LoggedFileAudit], contexts: dict[str, ItemContextAudit]
) -> None:
    for policy in POLICIES:
        logged_row = logged[policy]
        context_row = contexts[policy]
        if logged_row.action_count != 81 or logged_row.action_min != 0 or logged_row.action_max != 80:
            raise ValueError(f"{policy} logged all-campaign action universe is not exactly 0..80.")
        if context_row.item_count != 81 or context_row.item_min != 0 or context_row.item_max != 80:
            raise ValueError(f"{policy} item context universe is not exactly 0..80.")
        if set(logged_row.raw_positions) != EXPECTED_RAW_POSITIONS:
            raise ValueError(f"{policy} raw positions are not exactly 1, 2, 3.")
        if logged_row.propensity_field != PRIMARY_PROPENSITY_FIELD:
            raise ValueError(f"{policy} does not use the official propensity_score field.")


def _find_unique_member(archive: ZipFile, suffix: str) -> ZipInfo:
    normalized = suffix.lstrip("/")
    matches = [info for info in archive.infolist() if info.filename.endswith(normalized)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one archive member ending in {suffix!r}; found {len(matches)}")
    return matches[0]


def _member_sha256(archive: ZipFile, info: ZipInfo) -> str:
    digest = sha256()
    with archive.open(info) as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_logged_csv(archive: ZipFile, info: ZipInfo) -> LoggedFileAudit:
    with archive.open(info) as raw:
        reader = csv.DictReader(TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
        columns = tuple(reader.fieldnames or ())
        missing = REQUIRED_BASE_FIELDS.difference(columns)
        if missing:
            raise ValueError(f"{info.filename} missing required columns: {sorted(missing)}")
        propensity = _resolve_propensity_field(columns)

        row_count = 0
        timestamp_min: str | None = None
        timestamp_max: str | None = None
        positions: set[str] = set()
        actions: set[int] = set()
        for row in reader:
            row_count += 1
            timestamp = str(row["timestamp"]).strip()
            if not timestamp:
                raise ValueError(f"{info.filename} contains an empty timestamp.")
            timestamp_min = timestamp if timestamp_min is None else min(timestamp_min, timestamp)
            timestamp_max = timestamp if timestamp_max is None else max(timestamp_max, timestamp)

            action = int(row["item_id"])
            actions.add(action)
            position = str(row["position"]).strip()
            if not position:
                raise ValueError(f"{info.filename} contains an empty position.")
            float(position)
            positions.add(position)

            click = int(row["click"])
            if click not in (0, 1):
                raise ValueError(f"{info.filename} click must be binary.")
            pscore = float(row[propensity])
            if not 0.0 < pscore <= 1.0:
                raise ValueError(f"{info.filename} propensity must lie in (0, 1].")

    if row_count == 0 or not actions or timestamp_min is None or timestamp_max is None:
        raise ValueError(f"{info.filename} contains no usable rows.")

    return LoggedFileAudit(
        member=info.filename,
        compressed_size=info.compress_size,
        uncompressed_size=info.file_size,
        sha256=_member_sha256(archive, info),
        row_count=row_count,
        columns=columns,
        propensity_field=propensity,
        timestamp_min=timestamp_min,
        timestamp_max=timestamp_max,
        raw_positions=tuple(sorted(positions, key=float)),
        action_count=len(actions),
        action_min=min(actions),
        action_max=max(actions),
    )


def _audit_item_context(archive: ZipFile, info: ZipInfo) -> ItemContextAudit:
    with archive.open(info) as raw:
        reader = csv.DictReader(TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
        columns = tuple(reader.fieldnames or ())
        if "item_id" not in columns:
            raise ValueError(f"{info.filename} missing item_id.")
        items: set[int] = set()
        row_count = 0
        for row in reader:
            row_count += 1
            item = int(row["item_id"])
            if item in items:
                raise ValueError(f"{info.filename} contains duplicate item_id={item}.")
            items.add(item)
    if row_count == 0:
        raise ValueError(f"{info.filename} contains no items.")
    return ItemContextAudit(
        member=info.filename,
        compressed_size=info.compress_size,
        uncompressed_size=info.file_size,
        sha256=_member_sha256(archive, info),
        row_count=row_count,
        columns=columns,
        item_count=len(items),
        item_min=min(items),
        item_max=max(items),
    )


def _resolve_propensity_field(columns: tuple[str, ...]) -> str:
    primary = PRIMARY_PROPENSITY_FIELD in columns
    legacy = LEGACY_PROPENSITY_FIELD in columns
    if primary and legacy:
        raise ValueError("both propensity_score and legacy action_prob are present; schema is ambiguous.")
    if primary:
        return PRIMARY_PROPENSITY_FIELD
    if legacy:
        return LEGACY_PROPENSITY_FIELD
    raise ValueError("neither propensity_score nor legacy action_prob is present.")


def _leftmost_common_position(logged: dict[str, LoggedFileAudit]) -> str:
    common = set(logged["bts"].raw_positions).intersection(logged["random"].raw_positions)
    if not common:
        raise ValueError("BTS and Random have no common raw position labels.")
    return min(common, key=float)
