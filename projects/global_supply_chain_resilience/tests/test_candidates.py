from __future__ import annotations

import pandas as pd
import pytest

from supply_chain_resilience.candidates import persistent_top_k, real_country_candidates


def test_persistent_top_k_uses_intersection_and_reference_order() -> None:
    rankings = {
        0.0: pd.Series({"AAA_A": 5.0, "BBB_B": 4.0, "CCC_C": 3.0, "DDD_D": 2.0}),
        0.001: pd.Series({"AAA_A": 6.0, "CCC_C": 5.0, "BBB_B": 1.0, "DDD_D": 0.5}),
        0.01: pd.Series({"CCC_C": 8.0, "AAA_A": 7.0, "DDD_D": 2.0, "BBB_B": 1.0}),
    }

    assert persistent_top_k(rankings, k=2) == ["AAA_A", "CCC_C"]


def test_persistent_top_k_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        persistent_top_k({}, k=2)


def test_real_country_candidates_excludes_rest_of_world() -> None:
    assert real_country_candidates(["ROW_B06", "CHN_C26", "USA_G"]) == ["CHN_C26", "USA_G"]
