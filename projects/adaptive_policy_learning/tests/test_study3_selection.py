from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_selector() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "select_study3_campaign.py"
    spec = importlib.util.spec_from_file_location("select_study3_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Study 3 campaign selector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(campaign: str, *, random_rows: int, missing: list[int] | None = None) -> dict[str, Any]:
    missing_ids = missing or []
    return {
        "campaign": campaign,
        "click_value_validation_performed": False,
        "logged_files": {
            "bts": {
                "row_count": random_rows * 5,
                "propensity_field": "propensity_score",
                "raw_positions": ["1", "2", "3"],
            },
            "random": {
                "row_count": random_rows,
                "propensity_field": "propensity_score",
                "raw_positions": ["1", "2", "3"],
            },
        },
        "logged_action_support": {
            "bts": {"missing_catalog_action_ids": []},
            "random": {"missing_catalog_action_ids": missing_ids},
        },
    }


def test_select_campaign_uses_largest_qualifying_random_log() -> None:
    selector = _load_selector()

    result = selector.select_campaign(
        {
            "men": _report("men", random_rows=100),
            "women": _report("women", random_rows=200),
        }
    )

    assert result["status"] == "selected"
    assert result["selected_campaign"] == "women"
    assert result["outcome_values_parsed"] is False


def test_select_campaign_excludes_incomplete_action_support() -> None:
    selector = _load_selector()

    result = selector.select_campaign(
        {
            "men": _report("men", random_rows=100),
            "women": _report("women", random_rows=200, missing=[3]),
        }
    )

    assert result["selected_campaign"] == "men"


def test_select_campaign_rejects_reports_that_validated_click_values() -> None:
    selector = _load_selector()
    men = _report("men", random_rows=100)
    women = _report("women", random_rows=200)
    men["click_value_validation_performed"] = True
    women["click_value_validation_performed"] = True

    result = selector.select_campaign({"men": men, "women": women})

    assert result["status"] == "no_qualifying_campaign"
    assert result["selected_campaign"] is None
