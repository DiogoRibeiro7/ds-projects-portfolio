"""Two-way fixed-effects association models for municipality housing panels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

REQUIRED_COLUMNS = {
    "geo_code",
    "geo_name",
    "year",
    "rent_eur_m2",
    "income_eur",
    "resident_population",
    "overnight_stays",
}


def prepare_twfe_sample(
    frame: pd.DataFrame,
    *,
    years: tuple[int, ...],
    balanced: bool,
) -> pd.DataFrame:
    """Create the complete repeated municipality sample used by all TWFE equations."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if len(years) < 2 or len(set(years)) != len(years):
        raise ValueError("years must contain at least two unique values.")
    if frame.duplicated(["geo_code", "year"]).any():
        raise ValueError("geo_code and year must uniquely identify panel rows.")

    sample = frame.loc[frame["year"].isin(years)].copy()
    value_columns = [
        "rent_eur_m2",
        "income_eur",
        "resident_population",
        "overnight_stays",
    ]
    for column in value_columns:
        sample[column] = pd.to_numeric(sample[column], errors="coerce")

    complete = sample[value_columns].notna().all(axis=1)
    positive = sample[value_columns].gt(0).all(axis=1)
    sample = sample.loc[complete & positive].copy()
    sample["tourism_intensity"] = sample["overnight_stays"] / sample["resident_population"]
    sample["rent_income_ratio"] = sample["rent_eur_m2"] / sample["income_eur"]

    counts = sample.groupby("geo_code")["year"].nunique()
    required_count = len(years) if balanced else 2
    eligible = counts.loc[counts.ge(required_count)].index
    sample = sample.loc[sample["geo_code"].isin(eligible)].copy()

    sample["log_tourism_intensity"] = np.log(sample["tourism_intensity"])
    sample["log_rent"] = np.log(sample["rent_eur_m2"])
    sample["log_income"] = np.log(sample["income_eur"])
    sample["log_affordability"] = np.log(sample["rent_income_ratio"])
    return sample.sort_values(["geo_code", "year"]).reset_index(drop=True)


def _fit_one(sample: pd.DataFrame, outcome: str) -> dict[str, Any]:
    formula = f"{outcome} ~ log_tourism_intensity + C(geo_code) + C(year)"
    result = smf.ols(formula, data=sample).fit(
        cov_type="cluster",
        cov_kwds={"groups": sample["geo_code"]},
    )
    term = "log_tourism_intensity"
    return {
        "coefficient": float(result.params[term]),
        "std_error_clustered": float(result.bse[term]),
        "p_value": float(result.pvalues[term]),
        "ci_95_low": float(result.conf_int().loc[term, 0]),
        "ci_95_high": float(result.conf_int().loc[term, 1]),
        "n_observations": int(result.nobs),
        "municipalities": int(sample["geo_code"].nunique()),
        "years": sorted(int(year) for year in sample["year"].unique()),
    }


def fit_twfe_bundle(sample: pd.DataFrame) -> dict[str, Any]:
    """Fit affordability, rent, and income TWFE equations on one identical sample."""
    if sample.empty:
        raise ValueError("TWFE sample cannot be empty.")
    if sample["geo_code"].nunique() < 2:
        raise ValueError("TWFE sample requires at least two municipalities.")
    if sample["year"].nunique() < 2:
        raise ValueError("TWFE sample requires at least two years.")

    affordability = _fit_one(sample, "log_affordability")
    rent = _fit_one(sample, "log_rent")
    income = _fit_one(sample, "log_income")
    identity_gap = affordability["coefficient"] - (rent["coefficient"] - income["coefficient"])
    return {
        "affordability": affordability,
        "rent": rent,
        "income": income,
        "coefficient_identity_gap": float(identity_gap),
    }
