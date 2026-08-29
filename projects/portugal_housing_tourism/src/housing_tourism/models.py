"""Transparent fixed-effects models and counterfactual utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper

from .indices import external_housing_pressure_index


@dataclass(frozen=True)
class FixedEffectsSpec:
    """Specification for a two-way fixed-effects OLS model."""

    outcome: str = "log_rent"
    regressors: tuple[str, ...] = ("thcr", "log_income", "tourism_intensity")
    entity_col: str = "geo_code"
    time_col: str = "year"

    def __post_init__(self) -> None:
        if not self.outcome.strip():
            raise ValueError("outcome cannot be empty.")
        if not self.regressors:
            raise ValueError("At least one regressor is required.")
        if len(set(self.regressors)) != len(self.regressors):
            raise ValueError("regressors must not contain duplicates.")

    @property
    def formula(self) -> str:
        rhs = " + ".join(self.regressors)
        return f"{self.outcome} ~ {rhs} + C({self.entity_col}) + C({self.time_col})"


def fit_two_way_fixed_effects(
    frame: pd.DataFrame,
    spec: FixedEffectsSpec | None = None,
) -> RegressionResultsWrapper:
    """Fit OLS with entity/year fixed effects and entity-clustered SEs."""
    model_spec = spec or FixedEffectsSpec()
    required = {
        model_spec.outcome,
        *model_spec.regressors,
        model_spec.entity_col,
        model_spec.time_col,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing model columns: {sorted(missing)}")
    analysis = frame[list(required)].dropna().copy()
    if analysis.empty:
        raise ValueError("No complete observations are available for the model.")
    if analysis[model_spec.entity_col].nunique() < 2:
        raise ValueError("At least two entities are required.")
    if analysis[model_spec.time_col].nunique() < 2:
        raise ValueError("At least two time periods are required.")
    model = smf.ols(model_spec.formula, data=analysis)
    return model.fit(
        cov_type="cluster",
        cov_kwds={"groups": analysis[model_spec.entity_col]},
    )


def make_baseline_counterfactual(
    frame: pd.DataFrame,
    *,
    exposure_col: str = "thcr",
    entity_col: str = "geo_code",
    time_col: str = "year",
    base_year: int = 2017,
) -> pd.DataFrame:
    """Hold one exposure at each municipality's base-year value."""
    required = {exposure_col, entity_col, time_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing counterfactual columns: {sorted(missing)}")
    base = (
        frame.loc[frame[time_col] == base_year, [entity_col, exposure_col]]
        .drop_duplicates(entity_col)
        .set_index(entity_col)[exposure_col]
    )
    result = frame.copy()
    result[exposure_col] = result[entity_col].map(base)
    return result


def predict_rent_levels(
    fitted: RegressionResultsWrapper,
    frame: pd.DataFrame,
    *,
    logged_outcome: bool = True,
) -> pd.Series:
    """Predict rent levels from a fitted model."""
    prediction = pd.Series(fitted.predict(frame), index=frame.index, dtype=float)
    if logged_outcome:
        prediction = pd.Series(
            np.exp(prediction.to_numpy()),
            index=prediction.index,
            dtype=float,
        )
    prediction.name = "predicted_rent"
    return prediction


def build_ehpi(
    fitted: RegressionResultsWrapper,
    observed_frame: pd.DataFrame,
    counterfactual_frame: pd.DataFrame,
    *,
    logged_outcome: bool = True,
) -> pd.Series:
    """Compute EHPI from observed and explicitly counterfactual covariates."""
    observed = predict_rent_levels(fitted, observed_frame, logged_outcome=logged_outcome)
    counterfactual = predict_rent_levels(
        fitted, counterfactual_frame, logged_outcome=logged_outcome
    )
    return external_housing_pressure_index(observed, counterfactual)


def coefficient_table(result: RegressionResultsWrapper) -> pd.DataFrame:
    """Return a compact coefficient table for notebook display/export."""
    conf = result.conf_int()
    output = pd.DataFrame(
        {
            "estimate": result.params,
            "std_error": result.bse,
            "p_value": result.pvalues,
            "ci_low": conf.iloc[:, 0],
            "ci_high": conf.iloc[:, 1],
        }
    )
    output.index.name = "term"
    return output.reset_index()
