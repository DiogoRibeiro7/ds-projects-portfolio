from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

ARCHIVE_SHA256 = "a" * 64
CODE_SHA = "b" * 40


def _load_selector() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "select_study3_campaign.py"
    spec = importlib.util.spec_from_file_location("select_study3_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Study 3 campaign selector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(
    campaign: str,
    *,
    random_rows: int,
    catalog: list[int] | None = None,
    random_observed: list[int] | None = None,
) -> dict[str, Any]:
    catalog_ids = catalog or [0, 1, 2, 3]
    observed_ids = random_observed or catalog_ids
    return {
        "campaign": campaign,
        "archive": {"sha256": ARCHIVE_SHA256},
        "archive_catalog_action_ids": catalog_ids,
        "click_value_validation_performed": False,
        "logged_files": {
            "bts": {
                "row_count": random_rows * 5,
                "propensity_field": "propensity_score",
                "raw_positions": ["1", "2", "3"],
                "timestamp_min": "2019-11-24 00:00:00",
                "timestamp_max": "2019-11-30 23:59:59",
            },
            "random": {
                "row_count": random_rows,
                "propensity_field": "propensity_score",
                "raw_positions": ["1", "2", "3"],
                "timestamp_min": "2019-11-24 00:00:00",
                "timestamp_max": "2019-11-30 23:59:59",
            },
        },
        "logged_action_support": {
            "bts": {"observed_action_ids": catalog_ids},
            "random": {"observed_action_ids": observed_ids},
        },
    }


def _select(selector: ModuleType, reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = selector.select_campaign(
        reports,
        code_sha=CODE_SHA,
        expected_archive_sha256=ARCHIVE_SHA256,
    )
    return result


def test_select_campaign_uses_largest_qualifying_random_log() -> None:
    selector = _load_selector()

    result = _select(
        selector,
        {
            "men": _report("men", random_rows=100),
            "women": _report("women", random_rows=200),
        },
    )

    assert result["status"] == "selected"
    assert result["selected_campaign"] == "women"
    assert result["outcome_values_parsed"] is False
    assert result["code_sha"] == CODE_SHA
    assert result["archive_sha256"] == ARCHIVE_SHA256


def test_select_campaign_allows_non_empty_subset_action_support() -> None:
    selector = _load_selector()

    result = _select(
        selector,
        {
            "men": _report("men", random_rows=100),
            "women": _report(
                "women",
                random_rows=200,
                catalog=[0, 1, 2, 3],
                random_observed=[0, 1, 2],
            ),
        },
    )

    assert result["selected_campaign"] == "women"


def test_select_campaign_rejects_action_outside_catalog() -> None:
    selector = _load_selector()

    result = _select(
        selector,
        {
            "men": _report("men", random_rows=100),
            "women": _report(
                "women",
                random_rows=200,
                catalog=[0, 1, 2, 3],
                random_observed=[0, 1, 99],
            ),
        },
    )

    assert result["selected_campaign"] == "men"


def test_select_campaign_rejects_swapped_campaign_report() -> None:
    selector = _load_selector()

    result = _select(
        selector,
        {
            "men": _report("women", random_rows=300),
            "women": _report("women", random_rows=200),
        },
    )

    assert result["selected_campaign"] == "women"


def test_select_campaign_uses_lexical_tie_break() -> None:
    selector = _load_selector()

    result = _select(
        selector,
        {
            "men": _report("men", random_rows=200),
            "women": _report("women", random_rows=200),
        },
    )

    assert result["selected_campaign"] == "men"


def test_select_campaign_rejects_reports_that_validated_click_values() -> None:
    selector = _load_selector()
    men = _report("men", random_rows=100)
    women = _report("women", random_rows=200)
    men["click_value_validation_performed"] = True
    women["click_value_validation_performed"] = True

    result = _select(selector, {"men": men, "women": women})

    assert result["status"] == "no_qualifying_campaign"
    assert result["selected_campaign"] is None


def test_select_campaign_rejects_wrong_archive_digest() -> None:
    selector = _load_selector()
    men = _report("men", random_rows=100)
    women = _report("women", random_rows=200)
    men["archive"]["sha256"] = "c" * 64
    women["archive"]["sha256"] = "c" * 64

    result = _select(selector, {"men": men, "women": women})

    assert result["status"] == "no_qualifying_campaign"
