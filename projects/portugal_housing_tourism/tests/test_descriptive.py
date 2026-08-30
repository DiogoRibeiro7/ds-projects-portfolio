from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from housing_tourism.descriptive import decompose_episode, index_to_base_year


def test_index_to_base_year_preserves_missing_values() -> None:
    frame = pd.DataFrame({"year": [2017, 2018, 2019], "value": [10.0, 12.0, np.nan]})

    result = index_to_base_year(frame, value_col="value", base_year=2017)

    assert result.iloc[0] == pytest.approx(100.0)
    assert result.iloc[1] == pytest.approx(120.0)
    assert np.isnan(result.iloc[2])


def test_index_to_base_year_requires_one_observed_base() -> None:
    frame = pd.DataFrame({"year": [2017, 2018], "value": [np.nan, 12.0]})

    with pytest.raises(ValueError, match="exactly one observed"):
        index_to_base_year(frame, value_col="value", base_year=2017)


def test_decompose_episode_satisfies_log_identity() -> None:
    frame = pd.DataFrame(
        {
            "year": [2022, 2024],
            "rent_eur_m2": [12.88, 15.93],
            "income_eur": [15128.0, 16278.0],
            "tourism_intensity": [20.886962719298246, 23.664702672065022],
        }
    )

    result = decompose_episode(frame, start_year=2022, end_year=2024)

    assert result.rent_change_pct == pytest.approx(23.680124223602486)
    assert result.income_change_pct == pytest.approx(7.601797990481227)
    assert result.rent_income_ratio_change_pct == pytest.approx(14.942433041303759)
    assert result.affordability_log_gap_pct == pytest.approx(
        result.rent_log_change_pct - result.income_log_change_pct
    )
    assert result.affordability_log_gap_pct == pytest.approx(13.926123021742929)


def test_decompose_episode_rejects_missing_endpoint_values() -> None:
    frame = pd.DataFrame(
        {
            "year": [2024, 2025],
            "rent_eur_m2": [15.93, np.nan],
            "income_eur": [16278.0, np.nan],
            "tourism_intensity": [23.664702672065022, 24.252796869208005],
        }
    )

    with pytest.raises(ValueError, match="finite"):
        decompose_episode(frame, start_year=2024, end_year=2025)
