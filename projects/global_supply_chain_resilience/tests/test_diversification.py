from __future__ import annotations

import pandas as pd
import pytest

from supply_chain_resilience.diversification import (
    baseline_direct_risk,
    optimize_buyer_sourcing,
)


def test_baseline_direct_risk_uses_worst_frozen_supplier() -> None:
    observed = pd.Series({"AAA_X": 60.0, "BBB_X": 30.0, "CCC_X": 10.0})
    risk = baseline_direct_risk(observed, ("AAA_X", "BBB_X"), shock_fraction=0.10)
    assert risk == pytest.approx(0.06)


def test_feasible_diversification_hits_risk_cap_with_minimum_turnover() -> None:
    observed = pd.Series(
        {"AAA_X": 60.0, "BBB_X": 30.0, "CCC_X": 10.0, "DDD_X": 0.0}
    )
    foreign_sales = pd.Series(
        {"AAA_X": 100.0, "BBB_X": 100.0, "CCC_X": 100.0, "DDD_X": 1000.0}
    )
    result = optimize_buyer_sourcing(
        observed,
        foreign_sales,
        ("AAA_X",),
        buyer_node="ZZZ_Y",
        shock_fraction=0.10,
        risk_reduction_target=0.50,
        headroom_fraction=0.10,
    )

    assert result.feasible
    assert result.allocation is not None
    assert result.allocation.loc["AAA_X"] == pytest.approx(30.0)
    assert result.allocation.sum() == pytest.approx(observed.sum())
    assert result.reallocation_burden == pytest.approx(0.30)
    assert result.achieved_worst_case_direct_risk == pytest.approx(0.03)


def test_infeasible_when_headroom_cannot_absorb_required_reallocation() -> None:
    observed = pd.Series({"AAA_X": 80.0, "BBB_X": 20.0})
    foreign_sales = pd.Series({"AAA_X": 0.0, "BBB_X": 0.0})
    result = optimize_buyer_sourcing(
        observed,
        foreign_sales,
        ("AAA_X",),
        buyer_node="ZZZ_Y",
        shock_fraction=0.10,
        risk_reduction_target=0.50,
        headroom_fraction=0.05,
    )
    assert not result.feasible
