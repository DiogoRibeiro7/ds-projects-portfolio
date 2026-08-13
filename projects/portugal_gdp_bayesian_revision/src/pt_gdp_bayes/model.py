"""Bayesian population and GDP model for the Portugal 2025 revision.

The model estimates 2025 nominal GDP, resident population and GDP per capita as
posterior distributions rather than point corrections.

Two design rules matter more than the priors:

1. **One population vintage throughout.** GDP, employment and population must
   come from the same statistical vintage. Dividing an INE population by a World
   Bank GDP, or feeding a cross-vintage population *level revision* into a
   regression as if it were one year of demographic *growth*, produces large
   spurious effects. :func:`check_extrapolation` exists to catch exactly that.
2. **Do not credit output twice.** Output produced by workers already inside the
   national accounts is already inside measured GDP. The labour block here is a
   sensitivity on *unmeasured* output only, governed by an explicit statistical
   capture parameter (see :func:`simulate_unmeasured_output_samples`).
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
    demographic_growth_samples: FloatArray
    prior_mean: float
    prior_sd: float
    posterior_mean: float
    posterior_sd: float
    diagnostics: tuple[str, ...] = ()


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
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutturnScore:
    """Scorecard for a posterior against a realised value.

    Attributes
    ----------
    actual:
        Realised value.
    median, mean:
        Posterior central tendency.
    pct_error:
        Percentage error of the posterior median against ``actual``.
    q_low, q_high:
        Bounds of the central credible interval at ``credible_mass``.
    in_interval:
        Whether ``actual`` fell inside that interval.
    pit:
        Probability integral transform, i.e. the posterior CDF evaluated at
        ``actual``. A well-calibrated posterior puts this near the middle of
        ``[0, 1]``; values near 0 or 1 mean the posterior sat entirely on one
        side of the truth.
    """

    actual: float
    median: float
    mean: float
    pct_error: float
    q_low: float
    q_high: float
    in_interval: bool
    pit: float

    def summary(self, *, label: str = "posterior", scale: float = 1.0, unit: str = "") -> str:
        """One-line human-readable scorecard."""

        verdict = "inside" if self.in_interval else "OUTSIDE"
        return (
            f"{label}: median {self.median / scale:,.1f}{unit} vs actual "
            f"{self.actual / scale:,.1f}{unit} ({self.pct_error:+.1f}%), "
            f"{verdict} the credible interval "
            f"[{self.q_low / scale:,.1f}, {self.q_high / scale:,.1f}]{unit}, PIT {self.pit:.2f}"
        )


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


def check_extrapolation(
    name: str,
    value: float,
    training_values: FloatArray,
    *,
    margin: float = 0.25,
) -> str | None:
    """Flag a regressor value that falls outside its training support.

    A linear model asked to predict at a point far outside the data it was fitted
    on will happily return a confident, meaningless number. This is the guard
    that catches a cross-vintage population revision entering a growth regression
    as if it were genuine one-year growth.

    Parameters
    ----------
    name:
        Regressor name, for the message.
    value:
        Value the model is about to be evaluated at.
    training_values:
        Values the model was fitted on.
    margin:
        Tolerated overshoot, as a fraction of the training range.

    Returns
    -------
    str or None
        A warning message, or ``None`` when the value is within support.
    """

    training = np.asarray(training_values, dtype=float)
    training = training[np.isfinite(training)]
    if training.size == 0:
        return None

    low, high = float(training.min()), float(training.max())
    span = high - low
    if span <= 0:
        return None
    allowance = span * margin

    if value < low - allowance or value > high + allowance:
        overshoot = (value - high) / span if value > high else (low - value) / span
        return (
            f"{name}={value:.4g} lies outside the fitted range [{low:.4g}, {high:.4g}] "
            f"by {overshoot:.1f}x the training span. The prediction is an extrapolation; "
            f"check that the series vintages are consistent."
        )
    return None


def estimate_population_posterior_samples(
    df: pd.DataFrame,
    *,
    target_year: int,
    population_col: str = "population",
    observed_population: float,
    observation_relative_sd: float = 0.004,
    n_samples: int = 20_000,
    random_seed: int = 7,
) -> PopulationPosteriorResult:
    """Estimate posterior population for a target year.

    A Bayesian AR(1) on annual log population growth, fitted to all years before
    ``target_year``, supplies the prior; the official figure for ``target_year``
    enters as a noisy observation.

    Both inputs must be on the **same statistical vintage**. When they are, the
    prior and the observation broadly agree and the update is a genuine
    reconciliation of two sources of information. When they are not — for
    instance an international series that has not yet absorbed a national
    revision, paired with the post-revision national figure — the prior is not
    weak evidence about the same quantity, it is a *different quantity*, and no
    choice of ``observation_relative_sd`` makes the blend meaningful. Check
    :attr:`PopulationPosteriorResult.diagnostics` for that condition.

    Parameters
    ----------
    df:
        Historical dataframe with columns ``year`` and ``population_col``.
    target_year:
        Year to estimate, e.g. 2025.
    population_col:
        Population column name.
    observed_population:
        Official figure for ``target_year``.
    observation_relative_sd:
        Measurement uncertainty of the official figure, in relative terms.
        The default (0.4%) reflects that resident-population estimates are
        themselves revised between census benchmarks; it is not survey noise.
    n_samples:
        Number of posterior samples.
    random_seed:
        Random seed for reproducibility.

    Returns
    -------
    PopulationPosteriorResult
        Posterior samples, the AR(1) demographic-growth draws that the GDP model
        consumes, and any vintage diagnostics.
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

    demographic_growth_samples = posterior.sample_predictive(
        np.array([1.0, last_growth], dtype=float), n_samples, rng=rng
    )
    prior_log_population_samples = last_log_population + demographic_growth_samples
    prior_mean = float(np.mean(prior_log_population_samples))
    prior_sd = float(np.std(prior_log_population_samples, ddof=1))

    diagnostics: list[str] = []
    implied_growth = float(np.log(observed_population) - last_log_population)
    warning = check_extrapolation(
        "implied population growth to the observation",
        implied_growth,
        y,
    )
    if warning is not None:
        diagnostics.append(
            "Population vintage mismatch. "
            + warning
            + " The observation is very likely on a different statistical basis than the history."
        )

    posterior_normal = normal_normal_update(
        prior_mean=prior_mean,
        prior_sd=prior_sd,
        observation_mean=float(np.log(observed_population)),
        observation_sd=float(observation_relative_sd),
    )
    population_samples = np.exp(
        rng.normal(loc=posterior_normal.mean, scale=posterior_normal.sd, size=n_samples)
    )

    return PopulationPosteriorResult(
        samples=population_samples,
        demographic_growth_samples=demographic_growth_samples,
        prior_mean=float(np.exp(prior_mean)),
        prior_sd=float(np.exp(prior_mean) * prior_sd),
        posterior_mean=float(np.exp(posterior_normal.mean)),
        posterior_sd=float(np.exp(posterior_normal.mean) * posterior_normal.sd),
        diagnostics=tuple(diagnostics),
    )


def _sample_deflator_growth(
    data: pd.DataFrame,
    *,
    target_year: int,
    deflator_col: str,
    n_samples: int,
    rng: np.random.Generator,
    estimation_start_year: int | None = None,
) -> FloatArray:
    """Sample target-year GDP-deflator growth using a Bayesian AR(1)."""

    deflator = _clean_year_series(data, [deflator_col]).dropna(subset=[deflator_col]).copy()
    deflator = deflator[deflator["year"] < target_year].copy()
    if estimation_start_year is not None:
        deflator = deflator[deflator["year"] >= estimation_start_year].copy()
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
    estimation_start_year: int | None = 1999,
    n_samples: int = 20_000,
    random_seed: int = 11,
) -> GdpPosteriorResult:
    """Estimate posterior nominal GDP for a target year.

    Combines a historical Bayesian regression for nominal GDP growth with, when
    available, a direct signal from realised real growth plus an inferred
    deflator. GDP is therefore not held fixed when population is revised.

    ``gdp_col`` must be a **local-currency (EUR)** GDP series so the identity
    "nominal growth = real growth + deflator growth" holds; a current-USD series
    would fold euro/dollar moves into GDP growth.

    ``population_col`` must be on the same vintage as the population behind
    ``population_samples``. The population-growth regressor is checked against
    its fitted range and any breach is reported in
    :attr:`GdpPosteriorResult.diagnostics` rather than silently extrapolated.

    ``estimation_start_year`` restricts the estimation window, and defaults to
    1999 because Portuguese nominal series contain a decisive structural break at
    euro entry: nominal GDP growth averaged 15.5% a year before 1985 and 3.7%
    from 1999, and the GDP deflator 12.0% against 2.6%. Fitting one AR(1) across
    that break both biases the central prediction upward and inflates its
    residual variance, which in turn lets the weakly-identified regression
    component outvote the direct real-growth signal. Pass ``None`` to fit the
    full history and reproduce that failure.
    """

    if len(population_samples) != n_samples:
        raise ValueError("population_samples length must equal n_samples")
    if real_growth_observation_sd <= 0:
        raise ValueError("real_growth_observation_sd must be positive")

    rng = np.random.default_rng(random_seed)
    data = _clean_year_series(df, [gdp_col, population_col, deflator_col]).copy()
    data = data.sort_values("year").reset_index(drop=True)
    historical = data[data["year"] < target_year].dropna(subset=[gdp_col, population_col]).copy()
    # The lag terms need one observation before the window, so trim to
    # estimation_start_year only after the growth rates have been differenced.
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
        subset=[
            "gdp_growth",
            "lag_gdp_growth",
            "population_growth",
            "lag_population_growth",
            "deflator_growth",
        ]
    )
    if estimation_start_year is not None:
        reg = reg[reg["year"] >= estimation_start_year]
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
    lag_population_growth = float(
        latest.iloc[-1]["log_population"] - latest.iloc[-2]["log_population"]
    )

    deflator_growth_samples = _sample_deflator_growth(
        data,
        target_year=target_year,
        deflator_col=deflator_col,
        n_samples=n_samples,
        rng=rng,
        estimation_start_year=estimation_start_year,
    )
    population_growth_samples = np.log(population_samples / population_previous)

    diagnostics: list[str] = []
    # One growth step is applied from the latest available row to the target year.
    # With an outer-merged multi-source panel the latest row may lag, in which case
    # that single step silently spans more than one year.
    latest_year = int(latest.iloc[-1]["year"])
    if latest_year != target_year - 1:
        diagnostics.append(
            f"Latest available GDP/population row is {latest_year}, not {target_year - 1}. "
            f"The model applies one growth step across a {target_year - latest_year}-year gap, "
            "which understates the change. Check that every input series reaches "
            f"{target_year - 1}."
        )

    warning = check_extrapolation(
        "population growth regressor",
        float(np.median(population_growth_samples)),
        reg["population_growth"].to_numpy(dtype=float),
    )
    if warning is not None:
        diagnostics.append(
            "GDP model is extrapolating on population growth. "
            + warning
            + " A cross-vintage population level revision entering here as one year of growth "
            "will inflate the GDP posterior."
        )

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
        # Nominal GDP growth is approximately real growth + deflator growth, and the
        # deflator enters BOTH the regression (as a regressor) and the direct signal.
        # Precision-weighting the two nominal-growth estimates as though they were
        # independent would cancel that shared term and report a posterior narrower
        # than the deflator uncertainty alone -- which is impossible, since neither
        # input pins the deflator down.
        #
        # So combine only the genuinely independent parts: each source's implied
        # *real* growth, conditional on the deflator draw. The deflator is then added
        # back once, carrying its full uncertainty into the result.
        observed_real_growth = float(np.log1p(known_real_gdp_growth))
        model_real_growth_samples = nominal_growth_model_samples - deflator_growth_samples
        combined_real = normal_normal_update(
            prior_mean=float(np.mean(model_real_growth_samples)),
            prior_sd=float(np.std(model_real_growth_samples, ddof=1)),
            observation_mean=observed_real_growth,
            observation_sd=real_growth_observation_sd,
        )
        combined_growth_samples = (
            rng.normal(loc=combined_real.mean, scale=combined_real.sd, size=n_samples)
            + deflator_growth_samples
        )

        nominal_growth_signal_samples = (
            rng.normal(
                loc=observed_real_growth, scale=real_growth_observation_sd, size=n_samples
            )
            + deflator_growth_samples
        )
        signal_mean = float(np.mean(nominal_growth_signal_samples))
        signal_sd = float(np.std(nominal_growth_signal_samples, ddof=1))

    gdp_samples = gdp_previous * np.exp(combined_growth_samples)
    return GdpPosteriorResult(
        gdp_samples=gdp_samples,
        gdp_growth_samples=combined_growth_samples,
        deflator_growth_samples=deflator_growth_samples,
        nominal_growth_model_mean=model_mean,
        nominal_growth_model_sd=model_sd,
        nominal_growth_signal_mean=signal_mean,
        nominal_growth_signal_sd=signal_sd,
        diagnostics=tuple(diagnostics),
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
    """Sample the mechanical denominator correction under revision uncertainty.

    Generalises ``corrected_index = preliminary_index / population_correction_factor``
    by letting GDP and the EU reference average move as uncertain quantities.

    This function computes what a denominator correction *would* give. It does
    not establish that one is owed — that is an empirical question about which
    population the quoted index was already divided by, answered by
    :func:`pt_gdp_bayes.reconciliation.assess_correction_claim`.
    """

    for name, value in {
        "preliminary_index": preliminary_index,
        "population_correction_factor": population_correction_factor,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
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
    estimation_start_year: int | None = 1999,
    n_samples: int = 4_000,
    credible_mass: float = 0.90,
    random_seed: int = 23,
) -> pd.DataFrame:
    """Out-of-sample check of the nominal-GDP model.

    For each holdout year the model is re-fitted on data *before* that year and
    used to predict nominal GDP, **without** the contemporaneous real-growth
    signal, so nothing from the target year leaks in. The realised value is then
    compared with the predicted median and the central credible interval.

    Target-year population is supplied as known. Population is far more
    predictable than GDP a year ahead, so this isolates the GDP model rather
    than compounding it with demographic forecast error — but it does make the
    reported errors mildly optimistic relative to a true forecast.

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
                estimation_start_year=estimation_start_year,
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


def score_against_outturn(
    samples: FloatArray,
    actual: float,
    *,
    credible_mass: float = 0.90,
) -> OutturnScore:
    """Score a posterior against a realised value.

    Reports the percentage error, interval coverage and the probability integral
    transform, which together say not just *how far off* the posterior was but
    whether it was honestly uncertain about it.
    """

    arr = np.asarray(samples, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("samples must not be empty")
    if not 0 < credible_mass < 1:
        raise ValueError("credible_mass must lie between 0 and 1")
    if not np.isfinite(actual):
        raise ValueError("actual must be finite")

    lower_q = (1.0 - credible_mass) / 2.0
    low = float(np.quantile(arr, lower_q))
    high = float(np.quantile(arr, 1.0 - lower_q))
    median = float(np.median(arr))

    return OutturnScore(
        actual=float(actual),
        median=median,
        mean=float(np.mean(arr)),
        pct_error=(median - float(actual)) / float(actual) * 100.0,
        q_low=low,
        q_high=high,
        in_interval=bool(low <= float(actual) <= high),
        pit=float(np.mean(arr <= actual)),
    )


def _beta_draw(mean: float, relative_sd: float, n_samples: int, rng: np.random.Generator) -> FloatArray:
    """Draw a rate in [0, 1] with the requested mean and relative spread.

    A normal truncated at 0 and 1 — the obvious shortcut — shifts the mean
    whenever the clip bites, which it does for rates near 0.9. Matching a Beta by
    moments keeps the mean exact and the support correct by construction.
    """

    if not 0.0 <= mean <= 1.0:
        raise ValueError("mean must lie in [0, 1]")
    if relative_sd < 0:
        raise ValueError("relative_sd must be non-negative")
    if mean in (0.0, 1.0) or relative_sd == 0.0:
        return np.full(n_samples, float(mean))

    variance = (mean * relative_sd) ** 2
    max_variance = mean * (1.0 - mean)
    if variance >= max_variance:
        # Requested spread is impossible for a Beta with this mean; fall back to
        # the widest admissible one rather than silently returning nonsense.
        variance = max_variance * 0.99

    concentration = max_variance / variance - 1.0
    return rng.beta(mean * concentration, (1.0 - mean) * concentration, size=n_samples)


def _lognormal_draw(
    median: float, relative_sd: float, n_samples: int, rng: np.random.Generator
) -> FloatArray:
    """Draw a positive, unbounded quantity with the requested median and spread."""

    if median < 0:
        raise ValueError("median must be non-negative")
    if median == 0.0 or relative_sd == 0.0:
        return np.full(n_samples, float(median))
    sigma = np.sqrt(np.log1p(relative_sd**2))
    return rng.lognormal(mean=np.log(median), sigma=sigma, size=n_samples)


def simulate_unmeasured_output_samples(
    *,
    extra_residents: float,
    baseline_output_per_employed: float,
    working_age_share: float,
    participation_rate: float,
    employment_rate: float,
    productivity_relative: float,
    statistical_capture: float,
    n_samples: int,
    rng: np.random.Generator,
    relative_spread: float = 0.12,
    capture_relative_spread: float = 0.15,
) -> FloatArray:
    """Sample output produced by extra residents that measured GDP has *missed*.

    .. code-block:: text

        gross_output   = extra_residents * working_age_share * participation_rate
                         * employment_rate * baseline_output_per_employed
                         * productivity_relative
        unmeasured     = gross_output * (1 - statistical_capture)

    ``statistical_capture`` is the crux and is usually assumed away. It is the
    share of that output the national accounts **already record**. GDP is
    compiled from production-side sources — VAT turnover, corporate accounts,
    social-security payroll — which capture a firm's output regardless of whether
    the demographic statistics counted its workers. So evidence that migrants
    *work* says nothing about capture; the relevant evidence is whether
    national-accounts employment tracks the revised population (see
    :func:`pt_gdp_bayes.reconciliation.employment_absorbed_revision`).

    At ``statistical_capture = 1`` this returns zeros: the workers are already
    inside measured GDP and crediting them again would double-count. At
    ``statistical_capture = 0`` it returns the full gross output, which is the
    implicit and rarely defended assumption behind "the extra residents add
    their whole output on top".

    Rates are drawn as Beta variates and productivity as a lognormal, so every
    draw respects its natural support.
    """

    if not 0.0 <= statistical_capture <= 1.0:
        raise ValueError("statistical_capture must lie in [0, 1]")
    if extra_residents < 0:
        raise ValueError("extra_residents must be non-negative")

    gross = (
        extra_residents
        * _beta_draw(working_age_share, relative_spread, n_samples, rng)
        * _beta_draw(participation_rate, relative_spread, n_samples, rng)
        * _beta_draw(employment_rate, relative_spread, n_samples, rng)
        * baseline_output_per_employed
        * _lognormal_draw(productivity_relative, relative_spread, n_samples, rng)
    )
    uncaptured = 1.0 - _beta_draw(statistical_capture, capture_relative_spread, n_samples, rng)
    return gross * uncaptured


def pps_index_samples(
    *,
    gdp_samples: FloatArray,
    population_samples: FloatArray,
    purchasing_power_parity: float,
    reference_gdp_pc_pps: float,
) -> FloatArray:
    """Build the GDP-per-capita-in-PPS index from its definition.

    ``index = (gdp / population / ppp) / reference_gdp_pc_pps * 100``

    Constructing the index from components rather than rescaling a published
    value keeps the numerator and denominator explicit, so it is always visible
    which population the result is divided by.

    Parameters
    ----------
    gdp_samples:
        Posterior draws of nominal GDP in national currency.
    population_samples:
        Posterior draws of the population to divide by. This is the *resident*
        population when the question is living standards per person, which is
        not the same series as the national-accounts population the published
        index currently uses.
    purchasing_power_parity:
        PPP factor, national currency per PPS.
    reference_gdp_pc_pps:
        GDP per capita in PPS of the reference aggregate (the EU average).
    """

    for name, value in {
        "purchasing_power_parity": purchasing_power_parity,
        "reference_gdp_pc_pps": reference_gdp_pc_pps,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if len(gdp_samples) != len(population_samples):
        raise ValueError("gdp_samples and population_samples must have the same length")

    return (
        gdp_samples / population_samples / purchasing_power_parity / reference_gdp_pc_pps * 100.0
    )


def capture_sensitivity_curve(
    *,
    capture_values: FloatArray,
    extra_residents: float,
    baseline_output_per_employed: float,
    working_age_share: float,
    participation_rate: float,
    employment_rate: float,
    productivity_relative: float,
    measured_gdp_samples: FloatArray,
    population_samples: FloatArray,
    purchasing_power_parity: float,
    reference_gdp_pc_pps: float,
    n_samples: int,
    random_seed: int = 41,
) -> pd.DataFrame:
    """Trace the PPS index across the full range of statistical capture.

    Statistical capture is the least identified quantity in this analysis and
    the one that decides the answer, so rather than defending a single value
    this maps the whole curve. A reader can read off the index implied by their
    own view of how much of the uncounted residents' output the national
    accounts already record.

    The two ends are both meaningful bounds:

    ``capture = 1``
        The accounts already record everything those residents produce, so only
        the denominator is wrong. This is the pure denominator correction.
    ``capture = 0``
        The accounts missed their output entirely, so GDP is revised up by the
        full amount and largely offsets the denominator.

    Returns
    -------
    pandas.DataFrame
        Columns ``statistical_capture``, ``extra_gdp_bn_eur``, ``pps_median``,
        ``pps_q5`` and ``pps_q95``.
    """

    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, float]] = []
    for capture in np.asarray(capture_values, dtype=float):
        extra = simulate_unmeasured_output_samples(
            extra_residents=extra_residents,
            baseline_output_per_employed=baseline_output_per_employed,
            working_age_share=working_age_share,
            participation_rate=participation_rate,
            employment_rate=employment_rate,
            productivity_relative=productivity_relative,
            statistical_capture=float(capture),
            n_samples=n_samples,
            rng=rng,
        )
        index = pps_index_samples(
            gdp_samples=measured_gdp_samples + extra,
            population_samples=population_samples,
            purchasing_power_parity=purchasing_power_parity,
            reference_gdp_pc_pps=reference_gdp_pc_pps,
        )
        rows.append(
            {
                "statistical_capture": float(capture),
                "extra_gdp_bn_eur": float(np.mean(extra)) / 1e9,
                "pps_median": float(np.median(index)),
                "pps_q5": float(np.quantile(index, 0.05)),
                "pps_q95": float(np.quantile(index, 0.95)),
            }
        )
    return pd.DataFrame(rows)
