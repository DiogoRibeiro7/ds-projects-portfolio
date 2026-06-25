from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
from scipy import optimize, stats


DistributionName = Literal["gamma", "lognormal", "weibull", "generalized_gamma"]


@dataclass(frozen=True)
class FittedDistribution:
    name: DistributionName
    params: dict[str, float]
    nll: float
    converged: bool
    message: str


def weighted_midpoint_stats(bins: pd.DataFrame) -> tuple[float, float]:
    """Approximate grouped mean and variance from bracket midpoints."""
    midpoint = np.where(
        np.isfinite(bins["upper"]),
        (bins["lower"] + bins["upper"]) / 2.0,
        bins["lower"] * 1.35,
    )
    weights = bins["count"].to_numpy(dtype=float)
    mean = float(np.average(midpoint, weights=weights))
    variance = float(np.average((midpoint - mean) ** 2, weights=weights))
    return mean, max(variance, 1e-6)


def initial_theta(name: DistributionName, bins: pd.DataFrame) -> np.ndarray:
    """Build stable starting values for optimization."""
    mean, variance = weighted_midpoint_stats(bins)
    if name == "gamma":
        shape = max(mean * mean / variance, 0.2)
        scale = max(variance / mean, 1e-3)
        return np.log([shape, scale])
    if name == "lognormal":
        sigma2 = max(math.log(1.0 + variance / (mean * mean)), 1e-6)
        mu = math.log(mean) - 0.5 * sigma2
        return np.array([0.5 * math.log(sigma2), mu])
    if name == "weibull":
        return np.log([1.5, max(mean, 1e-3)])
    if name == "generalized_gamma":
        shape = max(mean * mean / variance, 0.5)
        return np.log([shape, 1.0, max(mean / shape, 1e-3)])
    raise ValueError(f"Unsupported distribution: {name}")


def distribution_cdf(name: DistributionName, theta: np.ndarray, x: np.ndarray | float) -> np.ndarray | float:
    """Evaluate the CDF of a candidate model."""
    if name == "gamma":
        shape = math.exp(theta[0])
        scale = math.exp(theta[1])
        return stats.gamma.cdf(x, a=shape, loc=0.0, scale=scale)
    if name == "lognormal":
        sigma = math.exp(theta[0])
        scale = math.exp(theta[1])
        return stats.lognorm.cdf(x, s=sigma, loc=0.0, scale=scale)
    if name == "weibull":
        shape = math.exp(theta[0])
        scale = math.exp(theta[1])
        return stats.weibull_min.cdf(x, c=shape, loc=0.0, scale=scale)
    if name == "generalized_gamma":
        shape_a = math.exp(theta[0])
        shape_c = math.exp(theta[1])
        scale = math.exp(theta[2])
        return stats.gengamma.cdf(x, a=shape_a, c=shape_c, loc=0.0, scale=scale)
    raise ValueError(f"Unsupported distribution: {name}")


def theta_to_params(name: DistributionName, theta: np.ndarray) -> dict[str, float]:
    """Convert optimization parameters into named model parameters."""
    if name == "gamma":
        return {"shape": float(math.exp(theta[0])), "scale": float(math.exp(theta[1]))}
    if name == "lognormal":
        return {"sigma": float(math.exp(theta[0])), "scale": float(math.exp(theta[1]))}
    if name == "weibull":
        return {"shape": float(math.exp(theta[0])), "scale": float(math.exp(theta[1]))}
    if name == "generalized_gamma":
        return {
            "shape_a": float(math.exp(theta[0])),
            "shape_c": float(math.exp(theta[1])),
            "scale": float(math.exp(theta[2])),
        }
    raise ValueError(f"Unsupported distribution: {name}")


def distribution_frozen(name: DistributionName, params: dict[str, float]):
    """Create a scipy frozen distribution from named parameters."""
    if name == "gamma":
        return stats.gamma(a=params["shape"], loc=0.0, scale=params["scale"])
    if name == "lognormal":
        return stats.lognorm(s=params["sigma"], loc=0.0, scale=params["scale"])
    if name == "weibull":
        return stats.weibull_min(c=params["shape"], loc=0.0, scale=params["scale"])
    if name == "generalized_gamma":
        return stats.gengamma(a=params["shape_a"], c=params["shape_c"], loc=0.0, scale=params["scale"])
    raise ValueError(f"Unsupported distribution: {name}")


def grouped_negative_log_likelihood(name: DistributionName, theta: np.ndarray, bins: pd.DataFrame) -> float:
    """Grouped log-likelihood for interval counts."""
    try:
        lower = bins["lower"].to_numpy(dtype=float)
        upper = bins["upper"].to_numpy(dtype=float)
        count = bins["count"].to_numpy(dtype=float)
        cdf_lower = np.asarray(distribution_cdf(name, theta, lower), dtype=float)
        cdf_upper = np.where(np.isfinite(upper), distribution_cdf(name, theta, upper), 1.0)
        probability = np.clip(cdf_upper - cdf_lower, 1e-12, 1.0)
        if not np.isfinite(probability).all():
            return float("inf")
        return float(-np.sum(count * np.log(probability)))
    except (OverflowError, FloatingPointError, ValueError):
        return float("inf")


def fit_grouped_distribution(name: DistributionName, bins: pd.DataFrame) -> FittedDistribution:
    """Fit one grouped distribution."""
    if name == "generalized_gamma":
        bounds = [(-4.0, 4.0), (-4.0, 4.0), (-2.0, 10.0)]
    else:
        bounds = [(-4.0, 10.0), (-2.0, 10.0)]
    result = optimize.minimize(
        lambda theta: grouped_negative_log_likelihood(name, theta, bins),
        x0=initial_theta(name, bins),
        method="L-BFGS-B",
        bounds=bounds,
    )
    return FittedDistribution(
        name=name,
        params=theta_to_params(name, result.x),
        nll=float(result.fun),
        converged=bool(result.success),
        message=str(result.message),
    )


def expected_counts(fit: FittedDistribution, bins: pd.DataFrame) -> np.ndarray:
    """Compute expected counts by bracket under a fitted model."""
    frozen = distribution_frozen(fit.name, fit.params)
    lower = bins["lower"].to_numpy(dtype=float)
    upper = bins["upper"].to_numpy(dtype=float)
    cdf_lower = frozen.cdf(lower)
    cdf_upper = np.where(np.isfinite(upper), frozen.cdf(upper), 1.0)
    probability = np.clip(cdf_upper - cdf_lower, 1e-12, 1.0)
    return probability * bins["count"].sum()


def fit_year_models(
    bins: pd.DataFrame,
    candidate_models: list[DistributionName] | None = None,
    drop_exact_minimum_wage: bool = True,
) -> pd.DataFrame:
    """Fit candidate grouped models year by year."""
    models = candidate_models or ["gamma", "lognormal", "weibull", "generalized_gamma"]
    rows: list[dict[str, float | int | str | bool]] = []

    for year, group in bins.groupby("year"):
        year_bins = group.copy()
        if drop_exact_minimum_wage:
            year_bins = year_bins.query("bin_type != 'exact_minimum_wage'")
        year_bins = year_bins.query("count > 0 and upper > lower").copy()
        if len(year_bins) < 4:
            continue

        observed = year_bins["count"].to_numpy(dtype=float)
        total_count = float(observed.sum())
        for model in models:
            fit = fit_grouped_distribution(model, year_bins)
            expected = expected_counts(fit, year_bins)
            k = len(fit.params)
            aic = 2 * k + 2 * fit.nll
            bic = math.log(total_count) * k + 2 * fit.nll
            chi_square = float(np.sum((observed - expected) ** 2 / np.maximum(expected, 1e-9)))
            chi_square_df = max(len(observed) - 1 - k, 1)
            rows.append(
                {
                    "year": int(year),
                    "model": model,
                    "n_bins": len(year_bins),
                    "n_workers_used": total_count,
                    "nll": fit.nll,
                    "aic": aic,
                    "bic": bic,
                    "chi_square": chi_square,
                    "chi_square_df": chi_square_df,
                    "chi_square_p_value": float(stats.chi2.sf(chi_square, df=chi_square_df)),
                    "converged": fit.converged,
                    "message": fit.message,
                    **{f"param_{name}": value for name, value in fit.params.items()},
                }
            )
    return pd.DataFrame(rows)


def fit_row_to_object(row: pd.Series) -> FittedDistribution:
    """Recreate a fitted-distribution object from a result row."""
    params = {key.removeprefix("param_"): float(value) for key, value in row.items() if key.startswith("param_") and not pd.isna(value)}
    return FittedDistribution(
        name=row["model"],
        params=params,
        nll=float(row["nll"]),
        converged=bool(row["converged"]),
        message=str(row["message"]),
    )


def grouped_residuals(bins: pd.DataFrame, fit_results: pd.DataFrame) -> pd.DataFrame:
    """Return observed, expected, and residual diagnostics by year, model, and bracket."""
    rows: list[dict[str, float | int | str]] = []
    for year, year_bins in bins.groupby("year"):
        year_bins = year_bins.query("bin_type != 'exact_minimum_wage' and count > 0 and upper > lower").copy()
        if year_bins.empty:
            continue
        for _, fit_row in fit_results.query("year == @year").iterrows():
            fit = fit_row_to_object(fit_row)
            expected = expected_counts(fit, year_bins)
            observed = year_bins["count"].to_numpy(dtype=float)
            for idx, bin_row in enumerate(year_bins.itertuples(index=False)):
                std_resid = (observed[idx] - expected[idx]) / math.sqrt(max(expected[idx], 1e-9))
                rows.append(
                    {
                        "year": int(year),
                        "model": fit.name,
                        "bin_label": bin_row.bin_label,
                        "observed_count": observed[idx],
                        "expected_count": expected[idx],
                        "observed_share": observed[idx] / observed.sum(),
                        "expected_share": expected[idx] / expected.sum(),
                        "pearson_residual": std_resid,
                    }
                )
    return pd.DataFrame(rows)


def conditional_decile_means(frozen, n_deciles: int = 10, integration_points: int = 256) -> np.ndarray:
    """Compute conditional mean earnings inside each model decile.

    A quantile-grid average is much faster and more stable here than repeated
    ``distribution.expect`` calls, especially for generalized Gamma.
    """
    means: list[float] = []
    for decile in range(1, n_deciles + 1):
        p_low = (decile - 1) / n_deciles
        p_high = decile / n_deciles
        epsilon = 1e-6
        grid = np.linspace(max(p_low, epsilon), min(p_high, 1.0 - epsilon), integration_points)
        quantiles = frozen.ppf(grid)
        means.append(float(np.nanmean(quantiles)))
    return np.asarray(means)


def decile_validation(summary_df: pd.DataFrame, fit_results: pd.DataFrame) -> pd.DataFrame:
    """Compare observed mean earnings by decile against each fitted model."""
    observed = summary_df.query("statistic == 'mean_gain_by_decile'").dropna(subset=["decile"]).copy()
    rows: list[dict[str, float | int | str]] = []
    for year, group in observed.groupby("year"):
        for _, fit_row in fit_results.query("year == @year").iterrows():
            fit = fit_row_to_object(fit_row)
            frozen = distribution_frozen(fit.name, fit.params)
            fitted_means = conditional_decile_means(frozen)
            ordered = group.sort_values("decile")
            for obs in ordered.itertuples(index=False):
                decile = int(obs.decile)
                fitted = float(fitted_means[decile - 1])
                observed_value = float(obs.value)
                rows.append(
                    {
                        "year": int(year),
                        "model": fit.name,
                        "decile": decile,
                        "observed_mean_gain": observed_value,
                        "fitted_mean_gain": fitted,
                        "absolute_error": fitted - observed_value,
                        "relative_error": (fitted - observed_value) / observed_value,
                    }
                )
    return pd.DataFrame(rows)


def decile_validation_summary(decile_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate decile validation to year-model summary metrics."""
    if decile_df.empty:
        return pd.DataFrame()
    return (
        decile_df.groupby(["year", "model"], as_index=False)
        .agg(
            mean_abs_relative_error=("relative_error", lambda s: float(np.mean(np.abs(s)))),
            max_abs_relative_error=("relative_error", lambda s: float(np.max(np.abs(s)))),
            rmse=("absolute_error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
        )
    )


def model_winners(fit_results: pd.DataFrame) -> pd.DataFrame:
    """Return the winning model by AIC and BIC for each year."""
    if fit_results.empty:
        return pd.DataFrame()
    rows: list[dict[str, float | int | str]] = []
    for year, group in fit_results.groupby("year"):
        bic_row = group.sort_values("bic").iloc[0]
        aic_row = group.sort_values("aic").iloc[0]
        rows.append(
            {
                "year": int(year),
                "winner_bic": bic_row["model"],
                "winner_aic": aic_row["model"],
                "bic_value": float(bic_row["bic"]),
                "aic_value": float(aic_row["aic"]),
            }
        )
    return pd.DataFrame(rows)


def gamma_parameter_trend(fit_results: pd.DataFrame) -> pd.DataFrame:
    """Build a tidy time series for Gamma parameters and implied mean."""
    gamma = fit_results.query("model == 'gamma'").copy()
    if gamma.empty:
        return gamma
    gamma["fitted_mean"] = gamma["param_shape"] * gamma["param_scale"]
    gamma["fitted_variance"] = gamma["param_shape"] * np.square(gamma["param_scale"])
    return gamma.sort_values("year").reset_index(drop=True)


def optional_microdata_fit(private_dir: Path) -> pd.DataFrame:
    """Fit microdata if a local anonymized CSV is available; otherwise return empty."""
    candidates = list(private_dir.glob("*.csv"))
    if not candidates:
        return pd.DataFrame()
    path = candidates[0]
    microdata = pd.read_csv(path)
    if "monthly_earnings" not in microdata.columns:
        return pd.DataFrame()
    rows: list[dict[str, float | int | str]] = []
    for year, group in microdata.groupby("year"):
        sample = group["monthly_earnings"].dropna().to_numpy(dtype=float)
        if len(sample) < 100:
            continue
        gamma_a, _, gamma_scale = stats.gamma.fit(sample, floc=0.0)
        lognorm_sigma, _, lognorm_scale = stats.lognorm.fit(sample, floc=0.0)
        weibull_c, _, weibull_scale = stats.weibull_min.fit(sample, floc=0.0)
        rows.extend(
            [
                {"year": int(year), "model": "gamma", "param_shape": gamma_a, "param_scale": gamma_scale},
                {"year": int(year), "model": "lognormal", "param_sigma": lognorm_sigma, "param_scale": lognorm_scale},
                {"year": int(year), "model": "weibull", "param_shape": weibull_c, "param_scale": weibull_scale},
            ]
        )
    return pd.DataFrame(rows)


def pareto_tail_diagnostics(bins: pd.DataFrame, top_bin_count: int = 2) -> pd.DataFrame:
    """Estimate a simple Pareto tail index from the highest brackets as a tail-only check.

    This is not used in the main grouped model competition. It is a lightweight diagnostic
    for whether the top tail is materially heavier than the body-only models imply.
    """
    rows: list[dict[str, float | int]] = []
    for year, group in bins.groupby("year"):
        tail = group.sort_values("lower").tail(top_bin_count).copy()
        tail = tail.query("count > 0").copy()
        if len(tail) < 2:
            continue
        xmin = float(tail["lower"].min())
        midpoint = np.where(np.isfinite(tail["upper"]), (tail["lower"] + tail["upper"]) / 2.0, tail["lower"] * 1.35)
        repeated = np.repeat(midpoint.astype(float), tail["count"].round().astype(int))
        if len(repeated) < 10:
            continue
        alpha = 1.0 + len(repeated) / np.sum(np.log(np.maximum(repeated / xmin, 1.000001)))
        rows.append(
            {
                "year": int(year),
                "tail_threshold": xmin,
                "pareto_alpha": float(alpha),
                "tail_worker_count": float(tail["count"].sum()),
                "tail_worker_share": float(tail["count"].sum() / group["count"].sum()),
            }
        )
    return pd.DataFrame(rows)
