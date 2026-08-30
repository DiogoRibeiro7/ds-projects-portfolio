"""Tests for municipality two-way fixed-effects helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from housing_tourism.twfe import fit_twfe_bundle, prepare_twfe_sample


def _panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for geo_code, geo_name, scale in [
        ("A", "Alpha", 1.0),
        ("B", "Beta", 1.2),
        ("C", "Gamma", 0.9),
    ]:
        for year, tourism in [(2022, 10.0), (2023, 12.0), (2024, 15.0)]:
            rows.append(
                {
                    "geo_code": geo_code,
                    "geo_name": geo_name,
                    "year": year,
                    "rent_eur_m2": scale * (8.0 + 0.2 * tourism),
                    "income_eur": scale * (12000.0 + 50.0 * tourism),
                    "resident_population": 1000.0 * scale,
                    "overnight_stays": tourism * 1000.0 * scale,
                }
            )
    return pd.DataFrame(rows)


def test_prepare_unbalanced_sample_requires_two_years() -> None:
    frame = _panel()
    frame.loc[(frame["geo_code"] == "C") & (frame["year"] == 2024), "rent_eur_m2"] = None
    sample = prepare_twfe_sample(frame, years=(2022, 2023, 2024), balanced=False)
    assert sample["geo_code"].nunique() == 3
    assert len(sample.loc[sample["geo_code"].eq("C")]) == 2


def test_prepare_balanced_sample_requires_all_years() -> None:
    frame = _panel()
    frame.loc[(frame["geo_code"] == "C") & (frame["year"] == 2024), "income_eur"] = None
    sample = prepare_twfe_sample(frame, years=(2022, 2023, 2024), balanced=True)
    assert set(sample["geo_code"]) == {"A", "B"}
    assert len(sample) == 6


def test_fit_bundle_preserves_log_identity() -> None:
    sample = prepare_twfe_sample(_panel(), years=(2022, 2023, 2024), balanced=True)
    bundle = fit_twfe_bundle(sample)
    assert bundle["coefficient_identity_gap"] == pytest.approx(0.0, abs=1e-10)
    assert bundle["affordability"]["n_observations"] == 9
    assert bundle["affordability"]["municipalities"] == 3


def test_prepare_rejects_nonunique_panel_rows() -> None:
    frame = pd.concat([_panel(), _panel().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="uniquely identify"):
        prepare_twfe_sample(frame, years=(2022, 2023, 2024), balanced=False)
