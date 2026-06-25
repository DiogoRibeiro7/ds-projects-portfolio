from __future__ import annotations

import pandas as pd

from pt_salary_gamma_distribution.fitting import decile_validation_summary, fit_year_models, model_winners


def make_bins() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"year": 2020, "bin_label": "< RMMG", "count": 5.0, "pct": 5.0, "lower": 0.0, "upper": 635.0, "bin_type": "below_minimum_wage"},
            {"year": 2020, "bin_label": "635,00 - 999,99 €", "count": 50.0, "pct": 50.0, "lower": 635.0, "upper": 999.99, "bin_type": "closed_range"},
            {"year": 2020, "bin_label": "1 000,00 - 1 499,99 €", "count": 25.0, "pct": 25.0, "lower": 1000.0, "upper": 1499.99, "bin_type": "closed_range"},
            {"year": 2020, "bin_label": "1 500,00 - 2 499,99 €", "count": 15.0, "pct": 15.0, "lower": 1500.0, "upper": 2499.99, "bin_type": "closed_range"},
            {"year": 2020, "bin_label": "2 500,00 e + Euros", "count": 5.0, "pct": 5.0, "lower": 2500.0, "upper": float("inf"), "bin_type": "open_top"},
        ]
    )


def test_fit_year_models_returns_supported_models() -> None:
    fit_results = fit_year_models(make_bins())
    assert {"gamma", "lognormal", "weibull", "generalized_gamma"} == set(fit_results["model"])
    assert fit_results["aic"].notna().all()
    assert fit_results["bic"].notna().all()


def test_model_winners_and_validation_summary_are_tidy() -> None:
    winners = model_winners(
        pd.DataFrame(
            [
                {"year": 2020, "model": "gamma", "aic": 10.0, "bic": 11.0},
                {"year": 2020, "model": "lognormal", "aic": 9.0, "bic": 12.0},
            ]
        )
    )
    assert winners.iloc[0]["winner_aic"] == "lognormal"
    assert winners.iloc[0]["winner_bic"] == "gamma"

    summary = decile_validation_summary(
        pd.DataFrame(
            [
                {"year": 2020, "model": "gamma", "relative_error": 0.1, "absolute_error": 10.0},
                {"year": 2020, "model": "gamma", "relative_error": -0.2, "absolute_error": -20.0},
            ]
        )
    )
    assert summary.iloc[0]["mean_abs_relative_error"] == 0.15000000000000002
