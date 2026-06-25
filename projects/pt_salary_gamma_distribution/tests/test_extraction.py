from __future__ import annotations

import pandas as pd

from pt_salary_gamma_distribution.extraction import build_salary_bin_dataset, parse_bin_interval, parse_pt_number


def test_parse_pt_number_handles_malformed_range_tokens() -> None:
    assert pd.isna(parse_pt_number("1--2"))
    assert pd.isna(parse_pt_number("1 - 2"))
    assert parse_pt_number("1 910 957") == 1910957.0
    assert parse_pt_number("1 095,6") == 1095.6


def test_parse_bin_interval_covers_minimum_wage_and_open_top() -> None:
    lower, upper, kind = parse_bin_interval("< RMMG", 2019)
    assert lower == 0.0
    assert upper == 600.0
    assert kind == "below_minimum_wage"

    lower, upper, kind = parse_bin_interval("5 000,00 e + Euros", 2019)
    assert lower == 5000.0
    assert upper == float("inf")
    assert kind == "open_top"


def test_build_salary_bin_dataset_preserves_counts_and_percentages() -> None:
    raw = pd.DataFrame(
        [
            {"year": 2019, "bin_label": "< RMMG", "measure": "counts", "value": 10.0},
            {"year": 2019, "bin_label": "< RMMG", "measure": "percentage", "value": 1.0},
            {"year": 2019, "bin_label": "600,00 - 749,99 €", "measure": "counts", "value": 20.0},
            {"year": 2019, "bin_label": "600,00 - 749,99 €", "measure": "percentage", "value": 2.0},
        ]
    )
    clean = build_salary_bin_dataset(raw)
    assert list(clean["count"]) == [10.0, 20.0]
    assert list(clean["pct"]) == [1.0, 2.0]
    assert set(clean["bin_type"]) == {"below_minimum_wage", "closed_range"}
