from __future__ import annotations

import pandas as pd
import pytest

from supply_chain_resilience.mapping import ICIOBlocks
from supply_chain_resilience.supplier import supplier_importance_metrics


def _blocks() -> ICIOBlocks:
    labels = ["AAA_A", "BBB_B", "CCC_C"]
    z = pd.DataFrame(
        [[20.0, 10.0, 5.0], [4.0, 30.0, 6.0], [8.0, 2.0, 25.0]],
        index=labels,
        columns=labels,
    )
    x = pd.Series([100.0, 100.0, 100.0], index=labels)
    return ICIOBlocks(
        intermediate_use=z,
        gross_output=x,
        final_demand=pd.DataFrame(index=labels),
        value_added=pd.Series([0.0, 0.0, 0.0], index=labels),
        taxes_less_subsidies=pd.Series([0.0, 0.0, 0.0], index=labels),
    )


def test_supplier_importance_preserves_supplier_orientation() -> None:
    metrics = supplier_importance_metrics(_blocks())

    assert metrics.loc["AAA_A", "intermediate_sales"] == pytest.approx(35.0)
    assert metrics.loc["AAA_A", "foreign_intermediate_sales"] == pytest.approx(15.0)
    assert metrics.loc["AAA_A", "foreign_sales_share"] == pytest.approx(15.0 / 35.0)
    assert metrics.loc["AAA_A", "foreign_downstream_input_share_mass"] == pytest.approx(0.15)
    assert metrics.loc["AAA_A", "max_foreign_buyer_input_share"] == pytest.approx(0.10)
    assert metrics.loc["AAA_A", "foreign_buyers_above_1pct"] == 2
