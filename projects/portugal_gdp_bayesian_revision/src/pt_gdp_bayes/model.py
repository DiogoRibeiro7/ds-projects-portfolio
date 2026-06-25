"""Bayesian population and GDP revision model.

The core idea is to update the 2025 GDP-per-capita estimate probabilistically.
The new population estimate affects the denominator directly, but it can also
update GDP indirectly through labour supply, consumption and measurement-error
channels. This implementation keeps the first version transparent and
extendable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .bayes import fit_bayesian_linear_regression, normal_normal_update

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PopulationPosteriorResult:
    """Posterior samples and metadata for population."""

    samples: FloatArray
    prior_mean: float
    prior_sd: float
    posterior_mean: float
    posterior_sd: float


@dataclass(frozen=True)
class GdpPosteriorResult:
    """Posterior samples and metadata for GDP."""

    gdp_samples: FloatArray
    gdp_growth_samples: FloatArray
    deflator_growth_samples: FloatArray
    nominal_growth_model_mean: float
    nominal_growth_model_sd: float
    nominal_growth_signal_mean: float
    nominal_growth_signal_sd: float


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _clean_year_series(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    _require_columns(df, ["year", *columns])
    out = df[["year", *columns]].copy()
    out["year"] = out["year"].astype(int)
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values("year").reset_index(drop=True)


def _add_intercept(x: FloatArray) -> FloatArray:
    return np.column_stack([np.ones(x.shape[0]), x])


def estimate_population_posterior_samples(
    df: pd.DataFrame,
    *,
    target_year: int,
    population_col: str = "population",
    observed_population: float,
    observation_relative_sd: float = 0.0015,
    n_samples: int = 20_000,
    random_seed: int = 7,
) -> PopulationPosteriorResult:
    """Estimate posterior population for a target year.

    The prior is a Bayesian AR(1) model for annual log population growth fitted
    to all years before ``target_year``. The likelihood is the new population
    observation, treated as a noisy measurement of latent true population.

    Parameters
    ----------
    df:
        Historical dataframe with columns ``year`` and ``population_col``.
    target_year:
        Year to estimate, e.g. 2025.
    population_col:
        Population column name.
    observed_population:
        New official or revised observation for ``target_year``.
    observation_relative_sd:
        Measurement uncertainty of the revised observation, in relative terms. It
        is kept small (~0.15%) on purpose: the revised INE figure **supersedes**
        the pre-revision trajectory, so the posterior should anchor to it rather
        than be dragged back toward the now-stale AR(1) extrapolation. The AR(1)
        prior's role is to characterise population *growth* (which feeds the GDP
        regression), not to second-guess the official level. Widening this SD
        materially pulls the posterior below the official figure.
    n_samples:
        Number of posterior samples.
    random_seed:
        Random seed for reproducibility.

    Returns
    -------
    PopulationPosteriorResult
        Posterior samples and summary metadata.
    """

    if observed_population <= 0:
        raise ValueError("observed_population must be positive")
    if observation_relative_sd <= 0:
        raise ValueError("observation_relative_sd must be positive")

    rng = np.random.default_rng(random_seed)
    data = _clean_year_series(df, [population_col]).dropna(subset=[population_col]).copy()
    data = data[data["year"] < target_year].copy()
    if len(data) < 10:
        raise ValueError("At least 10 historical population observations are required")

    data["log_population"] = np.log(data[population_col].astype(float))
    data["growth"] = data["log_population"].diff()
    data["lag_growth"] = data["growth"].shift(1)
    reg = data.dropna(subset=["growth", "lag_growth"])
    if reg.empty:
        raise ValueError("Population growth regression has no valid rows")

    X = _add_intercept(reg[["lag_growth"]].to_numpy(dtype=float))
    y = reg["growth"].to_numpy(dtype=float)
    posterior = fit_bayesian_linear_regression(X, y, prior_variance=1.0, alpha0=3.0, beta0=0.0001)

    last_two = data.dropna(subset=["log_population"]).tail(2)
    if len(last_two) < 2:
        raise ValueError("Need at least two latest population observations")
    last_log_population = float(last_two.iloc[-1]["log_population"])
    last_growth = float(last_two.iloc[-1]["log_population"] - last_two.iloc[-2]["log_population"])
    x_new = np.array([1.0, last_growth], dtype=float)

    prior_growth_samples = posterior.sample_predictive(x_new, n_samples, rng=rng)
    prior_log_population_samples = last_log_population + prior_growth_samples
    prior_mean = float(np.mean(prior_log_population_samples))
    prior_sd = float(np.std(prior_log_population_samples, ddof=1))

    observed_log_population = float(np.log(observed_population))
    observed_log_sd = float(observation_relative_sd)
    posterior_normal = normal_normal_update(
        prior_mean=prior_mean,
        prior_sd=prior_sd,
        observation_mean=observed_log_population,
        observation_sd=observed_log_sd,
    )
    posterior_log_samples = rng.normal(
        loc=posterior_normal.mean,
        scale=posterior_normal.sd,
        size=n_samples,
    )
    population_samples = np.exp(posterior_log_samples)

    return PopulationPosteriorResult(
        samples=population_samples,
        prior_mean=float(np.exp(prior_mean)),
        prior_sd=float(np.exp(prior_mean) * prior_sd),
        posterior_mean=float(np.exp(posterior_normal.mean)),
        posterior_sd=float(np.exp(posterior_normal.mean) * posterior_normal.sd),
    )


def _sample_deflator_growth(
    data: pd.DataFrame,
    *,
    target_year: int,
    deflator_col: str,
    n_samples: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Sample target-year GDP-deflator growth using a Bayesian AR(1)."""

    deflator = _clean_year_series(data, [deflator_col]).dropna(subset=[deflator_col]).copy()
    deflator = deflator[deflator["year"] < target_year].copy()
    if len(deflator) < 10:
        # Conservative fallback: 2.5% mean, 2.0pp standard deviation.
        return rng.normal(loc=0.025, scale=0.020, size=n_samples)

    # World Bank deflator values are usually percentages. Convert to log scale.
    deflator["deflator_growth"] = np.log1p(deflator[deflator_col].astype(float) / 100.0)
    deflator["lag_deflator_growth"] = deflator["deflator_growth"].shift(1)
    reg = deflator.dropna(subset=["deflator_growth", "lag_deflator_growth"])
    X = _add_intercept(reg[["lag_deflator_growth"]].to_numpy(dtype=float))
    y = reg["deflator_growth"].to_numpy(dtype=float)
    posterior = fit_bayesian_linear_regression(X, y, prior_variance=1.0, alpha0=3.0, beta0=0.001)
    last_growth = float(reg.iloc[-1]["deflator_growth"])
    return posterior.sample_predictive(np.array([1.0, last_growth]), n_samples, rng=rng)


def estimate_gdp_posterior_samples(
    df: pd.DataFrame,
    *,
    target_year: int,
    population_samples: FloatArray,
    gdp_col: str = "gdp_current_eur",
    population_col: str = "population",
    deflator_col: str = "gdp_deflator_pct",
    known_real_gdp_growth: float | None = None,
    real_growth_observation_sd: float = 0.003,
    n_samples: int = 20_000,
    random_seed: int = 11,
) -> GdpPosteriorResult:
    """Estimate posterior GDP for a target year.

    The model combines two pieces of information:

    1. A historical Bayesian regression for nominal GDP growth.
    2. A direct signal from real GDP growth plus an inferred GDP deflator.

    This avoids the restrictive assumption that GDP is fixed when the
    population estimate is revised.

    ``gdp_col`` should be a **local-currency (EUR)** GDP series so that the
    nominal-growth identity (real growth + deflator growth) holds. Using a
    current-USD series instead would fold euro/dollar exchange-rate movements
    into "GDP growth", which the local deflator and local real-growth signal do
    not describe.
    """

    if len(population_samples) != n_samples:
        raise ValueError("population_samples length must equal n_samples")
    if real_growth_observation_sd <= 0:
        raise ValueError("real_growth_observation_sd must be positive")

    rng = np.random.default_rng(random_seed)
    data = _clean_year_series(df, [gdp_col, population_col, deflator_col]).copy()
    data = data.sort_values("year").reset_index(drop=True)
    historical = data[data["year"] < target_year].dropna(subset=[gdp_col, population_col]).copy()
    if len(historical) < 15:
        raise ValueError("At least 15 historical GDP/population observations are required")

    historical["log_gdp"] = np.log(historical[gdp_col].astype(float))
    historical["log_population"] = np.log(historical[population_col].astype(float))
    historical["gdp_growth"] = historical["log_gdp"].diff()
    historical["lag_gdp_growth"] = historical["gdp_growth"].shift(1)
    historical["population_growth"] = historical["log_population"].diff()
    historical["lag_population_growth"] = historical["population_growth"].shift(1)
    historical["deflator_growth"] = np.log1p(
        pd.to_numeric(historical[deflator_col], errors="coerce") / 100.0
    )
    historical["deflator_growth"] = historical["deflator_growth"].fillna(
        historical["deflator_growth"].median()
    )

    reg = historical.dropna(
        subset=["gdp_growth", "lag_gdp_growth", "population_growth", "lag_population_growth", "deflator_growth"]
    )
    if len(reg) < 10:
        raise ValueError("GDP regression has too few complete observations")

    X = np.column_stack(
        [
            np.ones(len(reg)),
            reg["lag_gdp_growth"].to_numpy(dtype=float),
            reg["population_growth"].to_numpy(dtype=float),
            reg["lag_population_growth"].to_numpy(dtype=float),
            reg["deflator_growth"].to_numpy(dtype=float),
        ]
    )
    y = reg["gdp_growth"].to_numpy(dtype=float)
    posterior = fit_bayesian_linear_regression(X, y, prior_variance=2.0, alpha0=3.0, beta0=0.001)

    latest = historical.dropna(subset=["log_gdp", "log_population"]).tail(2)
    if len(latest) < 2:
        raise ValueError("Need at least two latest GDP/population observations")
    gdp_previous = float(np.exp(latest.iloc[-1]["log_gdp"]))
    lag_gdp_growth = float(latest.iloc[-1]["log_gdp"] - latest.iloc[-2]["log_gdp"])
    population_previous = float(np.exp(latest.iloc[-1]["log_population"]))
    lag_population_growth = float(latest.iloc[-1]["log_population"] - latest.iloc[-2]["log_population"])

    deflator_growth_samples = _sample_deflator_growth(
        data,
        target_year=target_year,
        deflator_col=deflator_col,
        n_samples=n_samples,
        rng=rng,
    )
    population_growth_samples = np.log(population_samples / population_previous)

    beta_samples, sigma_samples = posterior.sample_parameters(n_samples=n_samples, rng=rng)
    X_new = np.column_stack(
        [
            np.ones(n_samples),
            np.full(n_samples, lag_gdp_growth),
            population_growth_samples,
            np.full(n_samples, lag_population_growth),
            deflator_growth_samples,
        ]
    )
    nominal_growth_model_samples = np.sum(X_new * beta_samples, axis=1) + rng.normal(
        loc=0.0,
        scale=sigma_samples,
        size=n_samples,
    )

    model_mean = float(np.mean(nominal_growth_model_samples))
    model_sd = float(np.std(nominal_growth_model_samples, ddof=1))

    if known_real_gdp_growth is None:
        combined_growth_samples = nominal_growth_model_samples
        signal_mean = model_mean
        signal_sd = model_sd
    else:
        # Nominal GDP growth is approximately real growth + deflator growth.
        real_growth_samples = rng.normal(
            loc=np.log1p(known_real_gdp_growth),
            scale=real_growth_observation_sd,
            size=n_samples,
        )
        nominal_growth_signal_samples = real_growth_samples + deflator_growth_samples
        signal_mean = float(np.mean(nominal_growth_signal_samples))
        signal_sd = float(np.std(nominal_growth_signal_samples, ddof=1))

        # Combine the historical model and the direct signal using normal precision weighting.
        combined = normal_normal_update(
            prior_mean=model_mean,
            prior_sd=model_sd,
            observation_mean=signal_mean,
            observation_sd=signal_sd,
        )
        combined_growth_samples = rng.normal(
            loc=combined.mean,
            scale=combined.sd,
            size=n_samples,
        )

    gdp_samples = gdp_previous * np.exp(combined_growth_samples)
    return GdpPosteriorResult(
        gdp_samples=gdp_samples,
        gdp_growth_samples=combined_growth_samples,
        deflator_growth_samples=deflator_growth_samples,
        nominal_growth_model_mean=model_mean,
        nominal_growth_model_sd=model_sd,
        nominal_growth_signal_mean=signal_mean,
        nominal_growth_signal_sd=signal_sd,
    )


def estimate_pps_index_samples(
    *,
    preliminary_index: float,
    population_correction_factor: float,
    gdp_revision_factor_samples: FloatArray | None = None,
    eu_average_revision_factor_samples: FloatArray | None = None,
    n_samples: int = 20_000,
    random_seed: int = 17,
) -> FloatArray:
    """Estimate GDP-per-capita-in-PPS index samples under correction uncertainty.

    This is the transparent sensitivity layer around the Eurostat/AMECO index.
    It generalizes the mechanical correction:

        corrected_index = preliminary_index / population_correction_factor

    by allowing GDP and the EU reference average to move as uncertain quantities.
    """

    if preliminary_index <= 0 or population_correction_factor <= 0:
        raise ValueError("preliminary_index and population_correction_factor must be positive")
    rng = np.random.default_rng(random_seed)

    if gdp_revision_factor_samples is None:
        # Default: GDP may be revised, but the prior is centered on no revision.
        gdp_revision_factor_samples = rng.lognormal(mean=0.0, sigma=0.005, size=n_samples)
    if eu_average_revision_factor_samples is None:
        # Portugal is a small part of the EU average, so this uncertainty is small by default.
        eu_average_revision_factor_samples = rng.lognormal(mean=0.0, sigma=0.001, size=n_samples)

    if len(gdp_revision_factor_samples) != n_samples:
        raise ValueError("gdp_revision_factor_samples length must equal n_samples")
    if len(eu_average_revision_factor_samples) != n_samples:
        raise ValueError("eu_average_revision_factor_samples length must equal n_samples")

    return (
        preliminary_index
        * gdp_revision_factor_samples
        / population_correction_factor
        / eu_average_revision_factor_samples
    )


def backtest_gdp_model(
    df: pd.DataFrame,
    *,
    holdout_years: list[int],
    gdp_col: str = "gdp_current_eur",
    population_col: str = "population",
    deflator_col: str = "gdp_deflator_pct",
    n_samples: int = 4_000,
    credible_mass: float = 0.90,
    random_seed: int = 23,
) -> pd.DataFrame:
    """Out-of-sample check of the nominal-GDP model.

    For each holdout year the model is re-fitted on data *before* that year and
    used to predict nominal GDP, **without** the contemporaneous real-growth
    signal (so nothing from the target year leaks in). The realised value is then
    compared with the predicted median and the central credible interval.

    A well-calibrated model should keep the percentage error small and contain
    the realised value inside the interval about ``credible_mass`` of the time.

    Returns
    -------
    pandas.DataFrame
        One row per holdout year with actual, predicted median, the credible
        interval, percentage error and an ``in_interval`` flag.
    """

    lower_q = (1.0 - credible_mass) / 2.0
    upper_q = 1.0 - lower_q
    rows: list[dict[str, float]] = []

    for year in holdout_years:
        target = df.loc[df["year"] == year]
        if target.empty:
            continue
        actual = pd.to_numeric(target[gdp_col], errors="coerce").iloc[0]
        population = pd.to_numeric(target[population_col], errors="coerce").iloc[0]
        if not (np.isfinite(actual) and np.isfinite(population)):
            continue
        try:
            result = estimate_gdp_posterior_samples(
                df,
                target_year=year,
                population_samples=np.full(n_samples, float(population)),
                gdp_col=gdp_col,
                population_col=population_col,
                deflator_col=deflator_col,
                known_real_gdp_growth=None,
                n_samples=n_samples,
                random_seed=random_seed,
            )
        except ValueError:
            continue
        samples = result.gdp_samples
        median = float(np.median(samples))
        low = float(np.quantile(samples, lower_q))
        high = float(np.quantile(samples, upper_q))
        rows.append(
            {
                "year": int(year),
                "actual": float(actual),
                "predicted_median": median,
                "q_low": low,
                "q_high": high,
                "pct_error": (median - float(actual)) / float(actual) * 100.0,
                "in_interval": bool(low <= float(actual) <= high),
            }
        )

    return pd.DataFrame(rows)


def simulate_labour_channel_samples(
    *,
    extra_residents: float,
    baseline_output_per_employed: float,
    working_age_share: float,
    participation_rate: float,
    employment_rate: float,
    productivity_relative: float,
    n_samples: int,
    rng: np.random.Generator,
    relative_spread: float = 0.12,
) -> FloatArray:
    """Sample the extra GDP produced by the extra residents under one scenario.

    Each labour parameter is drawn around its central value (a normal with
    ``relative_spread`` coefficient of variation, clipped to a sensible range)
    instead of being a fixed point, so the scenario carries its own uncertainty
    rather than borrowing all of it from the GDP posterior. A scenario with every
    central value at zero (the denominator-only case) returns all zeros.

    ``extra_gdp = extra_residents * working_age_share * participation_rate
                  * employment_rate * baseline_output_per_employed
                  * productivity_relative``
    """

    def _draw(center: float, *, upper: float) -> FloatArray:
        if center <= 0.0:
            return np.zeros(n_samples)
        return np.clip(rng.normal(center, center * relative_spread, n_samples), 0.0, upper)

    extra_employed = (
        extra_residents
        * _draw(working_age_share, upper=1.0)
        * _draw(participation_rate, upper=1.0)
        * _draw(employment_rate, upper=1.0)
    )
    return extra_employed * baseline_output_per_employed * _draw(productivity_relative, upper=1.5)
