from __future__ import annotations

import pandas as pd
import pytest

from supply_chain_resilience.semiconductor_icio import (
    c26_supplier_frame,
    compare_ranked_measures,
    trade_downstream_share_mass,
)


def test_trade_downstream_share_mass_uses_named_material_links_only() -> None:
    links = pd.DataFrame(
        [
            {"partner_code": 1, "partner_desc": "Alpha", "partner_iso": "AAA", "partner_share": 0.4, "is_named_country": True, "material_importer": True},
            {"partner_code": 1, "partner_desc": "Alpha", "partner_iso": "AAA", "partner_share": 0.2, "is_named_country": True, "material_importer": True},
            {"partner_code": 2, "partner_desc": "Residual", "partner_iso": "S19", "partner_share": 0.5, "is_named_country": False, "material_importer": True},
            {"partner_code": 3, "partner_desc": "Beta", "partner_iso": "BBB", "partner_share": 0.7, "is_named_country": True, "material_importer": False},
        ]
    )
    result = trade_downstream_share_mass(links)
    assert result.to_dict("records") == [
        {"partner_iso": "AAA", "partner_code": 1, "partner_desc": "Alpha", "trade_downstream_share_mass": pytest.approx(0.6)}
    ]


def test_c26_supplier_frame_rejects_duplicate_country() -> None:
    frame = pd.DataFrame(
        [
            {"country": "AAA", "activity": "C26", "foreign_intermediate_sales": 1.0, "foreign_downstream_input_share_mass": 0.1},
            {"country": "AAA", "activity": "C26", "foreign_intermediate_sales": 2.0, "foreign_downstream_input_share_mass": 0.2},
        ]
    )
    with pytest.raises(ValueError, match="duplicate country"):
        c26_supplier_frame(frame)


def test_compare_ranked_measures_exact_match_and_deterministic_ranks() -> None:
    trade = pd.DataFrame(
        [
            {"iso": "AAA", "label": "Alpha", "value": 10.0},
            {"iso": "BBB", "label": "Beta", "value": 20.0},
            {"iso": "S19", "label": "Other Asia, nes", "value": 100.0},
        ]
    )
    icio = pd.DataFrame(
        [
            {"country": "AAA", "metric": 1.0},
            {"country": "BBB", "metric": 2.0},
            {"country": "TWN", "metric": 100.0},
        ]
    )
    table, summary = compare_ranked_measures(
        trade,
        icio,
        trade_code_column="iso",
        trade_label_column="label",
        trade_value_column="value",
        icio_value_column="metric",
    )
    assert table["country"].tolist() == ["BBB", "AAA"]
    assert "TWN" not in set(table["country"])
    assert "S19" not in set(table["country"])
    assert summary["matched_country_count"] == 2
    assert summary["spearman_rho"] == pytest.approx(1.0)
    assert summary["top_overlap_count"] == 2


def test_compare_ranked_measures_rank_difference_sign() -> None:
    trade = pd.DataFrame(
        [
            {"iso": "AAA", "label": "Alpha", "value": 3.0},
            {"iso": "BBB", "label": "Beta", "value": 2.0},
            {"iso": "CCC", "label": "Gamma", "value": 1.0},
        ]
    )
    icio = pd.DataFrame(
        [
            {"country": "AAA", "metric": 1.0},
            {"country": "BBB", "metric": 2.0},
            {"country": "CCC", "metric": 3.0},
        ]
    )
    table, _ = compare_ranked_measures(
        trade,
        icio,
        trade_code_column="iso",
        trade_label_column="label",
        trade_value_column="value",
        icio_value_column="metric",
    )
    alpha = table.loc[table["country"].eq("AAA")].iloc[0]
    assert alpha["trade_rank"] == 1
    assert alpha["icio_rank"] == 3
    assert alpha["rank_difference"] == -2
