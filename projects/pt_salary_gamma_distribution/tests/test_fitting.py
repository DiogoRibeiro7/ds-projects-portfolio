from __future__ import annotations

import pandas as pd

from pt_salary_gamma_distribution.fitting import (
    decile_validation_summary,
    fit_lognormal_pareto_splice_all_years,
    fit_sensitivity_scenarios,
    fit_year_models,
    model_winners,
    prepare_bins_for_fit,
    splice_top_share_comparison,
    tail_model_comparison,
)


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


def test_prepare_bins_for_fit_and_sensitivity_scenarios_work() -> None:
    bins = make_bins()
    filtered = prepare_bins_for_fit(bins, drop_open_top=True)
    assert "open_top" not in set(filtered["bin_type"])

    scenarios = fit_sensitivity_scenarios(
        bins,
        {
            "baseline": {"drop_exact_minimum_wage": True},
            "drop_open_top": {"drop_exact_minimum_wage": True, "drop_open_top": True},
        },
    )
    assert {"baseline", "drop_open_top"} == set(scenarios["scenario"])


def test_tail_model_comparison_returns_supported_tail_models() -> None:
    tail = tail_model_comparison(make_bins(), thresholds=[1000.0])
    assert {"lognormal_tail", "pareto_tail"} == set(tail["model"])
    assert tail["aic"].notna().all()


def test_splice_model_and_top_share_comparison_return_rows() -> None:
    splice = fit_lognormal_pareto_splice_all_years(make_bins(), thresholds=[1500.0])
    assert set(splice["model"]) == {"lognormal_pareto_splice"}
    assert splice["bic"].notna().all()

    top_share = splice_top_share_comparison(make_bins(), splice, lower_threshold=1500.0)
    assert {"splice_open_top_share", "splice_top_two_share"} <= set(top_share.columns)
