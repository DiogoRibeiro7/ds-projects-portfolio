from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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


@dataclass(frozen=True)
class SpliceDistribution:
    threshold: float
    body_params: dict[str, float]
    tail_params: dict[str, float]
    tail_weight: float
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


def conditional_body_probability(
    body_fit: FittedDistribution,
    lower: np.ndarray,
    upper: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Compute interval probabilities under the body model conditional on X < threshold."""
    frozen = distribution_frozen(body_fit.name, body_fit.params)
    cdf_threshold = max(float(frozen.cdf(threshold)), 1e-12)
    cdf_lower = frozen.cdf(lower)
    cdf_upper = np.where(np.isfinite(upper), frozen.cdf(np.minimum(upper, threshold)), frozen.cdf(threshold))
    probability = np.clip((cdf_upper - cdf_lower) / cdf_threshold, 1e-12, 1.0)
    return probability


def prepare_bins_for_fit(
    bins: pd.DataFrame,
    drop_exact_minimum_wage: bool = True,
    drop_open_top: bool = False,
    min_lower: float | None = None,
    max_lower: float | None = None,
) -> pd.DataFrame:
    """Apply common grouped-fit filters to a salary-bin table."""
    out = bins.copy()
    if drop_exact_minimum_wage:
        out = out.query("bin_type != 'exact_minimum_wage'")
    if drop_open_top:
        out = out.query("bin_type != 'open_top'")
    if min_lower is not None:
        out = out.loc[out["lower"] >= min_lower].copy()
    if max_lower is not None:
        out = out.loc[out["lower"] <= max_lower].copy()
    out = out.query("count > 0 and upper > lower").copy()
    return out.sort_values(["year", "lower", "upper"]).reset_index(drop=True)


def fit_year_models(
    bins: pd.DataFrame,
    candidate_models: list[DistributionName] | None = None,
    drop_exact_minimum_wage: bool = True,
    drop_open_top: bool = False,
    min_lower: float | None = None,
    max_lower: float | None = None,
) -> pd.DataFrame:
    """Fit candidate grouped models year by year."""
    models = candidate_models or ["gamma", "lognormal", "weibull", "generalized_gamma"]
    rows: list[dict[str, float | int | str | bool]] = []

    for year, group in bins.groupby("year"):
        year_bins = prepare_bins_for_fit(
            group,
            drop_exact_minimum_wage=drop_exact_minimum_wage,
            drop_open_top=drop_open_top,
            min_lower=min_lower,
            max_lower=max_lower,
        )
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


def fit_sensitivity_scenarios(
    bins: pd.DataFrame,
    scenarios: dict[str, dict[str, float | bool | None]],
    candidate_models: list[DistributionName] | None = None,
) -> pd.DataFrame:
    """Run a family of grouped-fit sensitivity scenarios."""
    frames: list[pd.DataFrame] = []
    for scenario_name, options in scenarios.items():
        frame = fit_year_models(
            bins,
            candidate_models=candidate_models,
            drop_exact_minimum_wage=bool(options.get("drop_exact_minimum_wage", True)),
            drop_open_top=bool(options.get("drop_open_top", False)),
            min_lower=options.get("min_lower"),  # type: ignore[arg-type]
            max_lower=options.get("max_lower"),  # type: ignore[arg-type]
        )
        if frame.empty:
            continue
        frame["scenario"] = scenario_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
        year_bins = prepare_bins_for_fit(year_bins)
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


def bootstrap_grouped_parameter_ranges(
    bins: pd.DataFrame,
    years: list[int],
    models: list[DistributionName] | None = None,
    n_boot: int = 12,
    seed: int = 0,
) -> pd.DataFrame:
    """Bootstrap grouped-fit parameter ranges for selected years and models."""
    rng = np.random.default_rng(seed)
    candidate_models = models or ["gamma", "lognormal"]
    rows: list[dict[str, float | int | str]] = []

    for year in years:
        year_bins = prepare_bins_for_fit(bins.query("year == @year"))
        if year_bins.empty:
            continue
        observed = year_bins["count"].to_numpy(dtype=float)
        total = int(observed.sum())
        probabilities = observed / observed.sum()

        for model in candidate_models:
            samples: list[dict[str, float]] = []
            for _ in range(n_boot):
                boot_counts = rng.multinomial(total, probabilities)
                boot_bins = year_bins.copy()
                boot_bins["count"] = boot_counts
                fit = fit_grouped_distribution(model, boot_bins)
                if fit.converged:
                    samples.append(fit.params)
            if not samples:
                continue
            params_df = pd.DataFrame(samples)
            for column in params_df.columns:
                rows.append(
                    {
                        "year": int(year),
                        "model": model,
                        "parameter": column,
                        "n_boot": int(len(params_df)),
                        "p10": float(params_df[column].quantile(0.10)),
                        "p50": float(params_df[column].quantile(0.50)),
                        "p90": float(params_df[column].quantile(0.90)),
                    }
                )
    return pd.DataFrame(rows)


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


def conditional_tail_probability(
    name: Literal["lognormal"],
    theta: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    xmin: float,
) -> np.ndarray:
    """Compute conditional interval probabilities given X >= xmin for a tail model."""
    cdf_xmin = np.asarray(distribution_cdf(name, theta, xmin), dtype=float)
    cdf_lower = np.asarray(distribution_cdf(name, theta, lower), dtype=float)
    cdf_upper = np.where(np.isfinite(upper), distribution_cdf(name, theta, upper), 1.0)
    denominator = np.clip(1.0 - cdf_xmin, 1e-12, 1.0)
    return np.clip((cdf_upper - cdf_lower) / denominator, 1e-12, 1.0)


def pareto_interval_probability(alpha: float, lower: np.ndarray, upper: np.ndarray, xmin: float) -> np.ndarray:
    """Compute interval probabilities under a Pareto tail with support x >= xmin."""
    lower_term = np.power(np.maximum(lower / xmin, 1.0), -alpha)
    upper_term = np.where(np.isfinite(upper), np.power(np.maximum(upper / xmin, 1.0), -alpha), 0.0)
    return np.clip(lower_term - upper_term, 1e-12, 1.0)


def fit_lognormal_pareto_splice_year(bins: pd.DataFrame, threshold: float) -> SpliceDistribution | None:
    """Fit a lognormal body plus Pareto tail splice model for one year."""
    year_bins = prepare_bins_for_fit(bins)
    body_bins = year_bins.loc[year_bins["upper"] <= threshold].copy()
    tail_bins = year_bins.loc[year_bins["lower"] >= threshold].copy()
    if len(body_bins) < 3 or len(tail_bins) < 2:
        return None

    total_count = float(year_bins["count"].sum())
    tail_weight = float(tail_bins["count"].sum() / total_count)
    body_weight = 1.0 - tail_weight
    if not (0.0 < tail_weight < 1.0):
        return None

    body_fit = fit_grouped_distribution("lognormal", body_bins)
    lower_tail = tail_bins["lower"].to_numpy(dtype=float)
    upper_tail = tail_bins["upper"].to_numpy(dtype=float)
    count_tail = tail_bins["count"].to_numpy(dtype=float)

    pareto_result = optimize.minimize(
        lambda theta: float(-np.sum(count_tail * np.log(pareto_interval_probability(math.exp(theta[0]), lower_tail, upper_tail, threshold)))),
        x0=np.array([math.log(3.0)]),
        method="L-BFGS-B",
        bounds=[(-4.0, 6.0)],
    )
    pareto_alpha = float(math.exp(pareto_result.x[0]))

    lower = year_bins["lower"].to_numpy(dtype=float)
    upper = year_bins["upper"].to_numpy(dtype=float)
    count = year_bins["count"].to_numpy(dtype=float)

    probability = np.empty(len(year_bins), dtype=float)
    body_mask = year_bins["upper"].to_numpy(dtype=float) <= threshold
    tail_mask = year_bins["lower"].to_numpy(dtype=float) >= threshold

    probability[body_mask] = body_weight * conditional_body_probability(body_fit, lower[body_mask], upper[body_mask], threshold)
    probability[tail_mask] = tail_weight * pareto_interval_probability(pareto_alpha, lower[tail_mask], upper[tail_mask], threshold)
    probability = np.clip(probability, 1e-12, 1.0)
    nll = float(-np.sum(count * np.log(probability)))

    return SpliceDistribution(
        threshold=threshold,
        body_params=body_fit.params,
        tail_params={"alpha": pareto_alpha},
        tail_weight=tail_weight,
        nll=nll,
        converged=bool(body_fit.converged and pareto_result.success),
        message=f"body={body_fit.message}; tail={pareto_result.message}",
    )


def fit_lognormal_pareto_splice_all_years(
    bins: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Fit a lognormal body plus Pareto tail splice model across years."""
    rows: list[dict[str, float | int | str | bool]] = []
    thresholds_to_check = thresholds or [2500.0, 3750.0]
    for year, group in bins.groupby("year"):
        year_bins = prepare_bins_for_fit(group)
        if year_bins.empty:
            continue
        total_count = float(year_bins["count"].sum())
        for threshold in thresholds_to_check:
            fit = fit_lognormal_pareto_splice_year(group, threshold)
            if fit is None:
                continue
            k = 4  # body sigma, body scale, Pareto alpha, tail weight
            rows.append(
                {
                    "year": int(year),
                    "model": "lognormal_pareto_splice",
                    "tail_threshold": threshold,
                    "n_bins": len(year_bins),
                    "n_workers_used": total_count,
                    "nll": fit.nll,
                    "aic": 2 * k + 2 * fit.nll,
                    "bic": math.log(total_count) * k + 2 * fit.nll,
                    "converged": fit.converged,
                    "message": fit.message,
                    "param_body_sigma": fit.body_params["sigma"],
                    "param_body_scale": fit.body_params["scale"],
                    "param_tail_alpha": fit.tail_params["alpha"],
                    "param_tail_weight": fit.tail_weight,
                }
            )
    return pd.DataFrame(rows)


def expected_counts_splice(splice_row: pd.Series, bins: pd.DataFrame) -> np.ndarray:
    """Compute expected counts under a fitted lognormal-Pareto splice model."""
    year_bins = prepare_bins_for_fit(bins)
    threshold = float(splice_row["tail_threshold"])
    body_fit = FittedDistribution(
        name="lognormal",
        params={"sigma": float(splice_row["param_body_sigma"]), "scale": float(splice_row["param_body_scale"])},
        nll=float(splice_row["nll"]),
        converged=bool(splice_row["converged"]),
        message=str(splice_row["message"]),
    )
    tail_weight = float(splice_row["param_tail_weight"])
    body_weight = 1.0 - tail_weight
    pareto_alpha = float(splice_row["param_tail_alpha"])

    lower = year_bins["lower"].to_numpy(dtype=float)
    upper = year_bins["upper"].to_numpy(dtype=float)
    probability = np.empty(len(year_bins), dtype=float)
    body_mask = year_bins["upper"].to_numpy(dtype=float) <= threshold
    tail_mask = year_bins["lower"].to_numpy(dtype=float) >= threshold
    probability[body_mask] = body_weight * conditional_body_probability(body_fit, lower[body_mask], upper[body_mask], threshold)
    probability[tail_mask] = tail_weight * pareto_interval_probability(pareto_alpha, lower[tail_mask], upper[tail_mask], threshold)
    probability = np.clip(probability, 1e-12, 1.0)
    return probability * year_bins["count"].sum()


def splice_sample(row: pd.Series, size: int = 200000, seed: int = 0) -> np.ndarray:
    """Draw samples from a fitted lognormal-Pareto splice model."""
    rng = np.random.default_rng(seed + int(row["year"]) + int(row["tail_threshold"]))
    sigma = float(row["param_body_sigma"])
    scale = float(row["param_body_scale"])
    alpha = float(row["param_tail_alpha"])
    tail_weight = float(row["param_tail_weight"])
    threshold = float(row["tail_threshold"])

    body_size = int(round(size * (1.0 - tail_weight)))
    tail_size = max(size - body_size, 1)

    body_frozen = stats.lognorm(s=sigma, loc=0.0, scale=scale)
    cdf_threshold = max(float(body_frozen.cdf(threshold)), 1e-12)
    body_uniform = rng.uniform(1e-6, max(cdf_threshold - 1e-6, 2e-6), size=max(body_size, 1))
    body_sample = body_frozen.ppf(body_uniform)

    tail_uniform = rng.uniform(1e-6, 1.0 - 1e-6, size=tail_size)
    tail_sample = threshold * np.power(1.0 - tail_uniform, -1.0 / alpha)
    sample = np.concatenate([body_sample, tail_sample])
    return sample[np.isfinite(sample)]


def splice_top_decile_diagnostics(
    summary_df: pd.DataFrame,
    splice_results: pd.DataFrame,
    size: int = 200000,
) -> pd.DataFrame:
    """Compare splice-model top-decile diagnostics against published decile summaries."""
    top_cutpoints = summary_df.query("statistic == 'decile_cutpoint' and decile == 9")[["year", "value"]].rename(columns={"value": "observed_p90_cutpoint"})
    top_means = summary_df.query("statistic == 'mean_gain_by_decile' and decile == 10")[["year", "value"]].rename(columns={"value": "observed_decile10_mean"})
    observed = top_means.merge(top_cutpoints, on="year", how="left")

    rows: list[dict[str, float | int]] = []
    for row in splice_results.itertuples(index=False):
        observed_match = observed.loc[observed["year"] == row.year]
        if observed_match.empty:
            continue
        sample = splice_sample(pd.Series(row._asdict()), size=size)
        p90 = float(np.quantile(sample, 0.9))
        decile10_sample = sample[sample >= p90]
        mean_decile10 = float(np.mean(decile10_sample))
        obs = observed_match.iloc[0]
        observed_p90_cutpoint = float(obs["observed_p90_cutpoint"]) if not pd.isna(obs["observed_p90_cutpoint"]) else float("nan")
        rows.append(
            {
                "year": int(row.year),
                "tail_threshold": float(row.tail_threshold),
                "observed_p90_cutpoint": observed_p90_cutpoint,
                "splice_p90_cutpoint": p90,
                "observed_decile10_mean": float(obs["observed_decile10_mean"]),
                "splice_decile10_mean": mean_decile10,
                "p90_relative_error": (
                    (p90 - observed_p90_cutpoint) / observed_p90_cutpoint if not math.isnan(observed_p90_cutpoint) else float("nan")
                ),
                "decile10_mean_relative_error": (mean_decile10 - float(obs["observed_decile10_mean"])) / float(obs["observed_decile10_mean"]),
            }
        )
    return pd.DataFrame(rows)


def splice_top_share_comparison(
    bins: pd.DataFrame,
    splice_results: pd.DataFrame,
    lower_threshold: float = 3750.0,
) -> pd.DataFrame:
    """Compare observed and splice-implied top shares."""
    rows: list[dict[str, float | int]] = []
    for year, year_bins in bins.groupby("year"):
        fit_bins = prepare_bins_for_fit(year_bins)
        observed_total = float(fit_bins["count"].sum())
        observed_open_top = float(fit_bins.loc[fit_bins["bin_type"] == "open_top", "count"].sum() / observed_total)
        observed_top_two = float(fit_bins.loc[fit_bins["lower"] >= lower_threshold, "count"].sum() / observed_total)

        for _, splice_row in splice_results.query("year == @year").iterrows():
            expected = expected_counts_splice(splice_row, year_bins)
            fit_bins_local = fit_bins.copy()
            fit_bins_local["expected_count"] = expected
            expected_total = float(fit_bins_local["expected_count"].sum())
            rows.append(
                {
                    "year": int(year),
                    "tail_threshold": float(splice_row["tail_threshold"]),
                    "observed_open_top_share": observed_open_top,
                    "splice_open_top_share": float(fit_bins_local.loc[fit_bins_local["bin_type"] == "open_top", "expected_count"].sum() / expected_total),
                    "observed_top_two_share": observed_top_two,
                    "splice_top_two_share": float(fit_bins_local.loc[fit_bins_local["lower"] >= lower_threshold, "expected_count"].sum() / expected_total),
                }
            )
    return pd.DataFrame(rows)


def tail_model_comparison(
    bins: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Compare lognormal and Pareto fits on the upper tail only."""
    thresholds_to_check = thresholds or [2500.0, 3750.0]
    rows: list[dict[str, float | int | str | bool]] = []

    for year, year_bins_full in bins.groupby("year"):
        for xmin in thresholds_to_check:
            year_bins = prepare_bins_for_fit(year_bins_full, min_lower=xmin)
            if len(year_bins) < 2:
                continue
            lower = year_bins["lower"].to_numpy(dtype=float)
            upper = year_bins["upper"].to_numpy(dtype=float)
            count = year_bins["count"].to_numpy(dtype=float)
            total_count = float(count.sum())

            lognormal_fit = fit_grouped_distribution("lognormal", year_bins)
            theta_lognormal = np.log([lognormal_fit.params["sigma"], lognormal_fit.params["scale"]])
            lognormal_prob = conditional_tail_probability("lognormal", theta_lognormal, lower, upper, xmin)
            lognormal_nll = float(-np.sum(count * np.log(lognormal_prob)))

            pareto_result = optimize.minimize(
                lambda theta: float(-np.sum(count * np.log(pareto_interval_probability(math.exp(theta[0]), lower, upper, xmin)))),
                x0=np.array([math.log(3.0)]),
                method="L-BFGS-B",
                bounds=[(-4.0, 6.0)],
            )
            pareto_alpha = float(math.exp(pareto_result.x[0]))
            pareto_prob = pareto_interval_probability(pareto_alpha, lower, upper, xmin)
            pareto_nll = float(-np.sum(count * np.log(pareto_prob)))

            rows.extend(
                [
                    {
                        "year": int(year),
                        "tail_threshold": xmin,
                        "model": "lognormal_tail",
                        "n_bins": len(year_bins),
                        "n_workers_used": total_count,
                        "nll": lognormal_nll,
                        "aic": 2 * 2 + 2 * lognormal_nll,
                        "bic": math.log(total_count) * 2 + 2 * lognormal_nll,
                        "converged": lognormal_fit.converged,
                        "message": lognormal_fit.message,
                        "param_sigma": lognormal_fit.params["sigma"],
                        "param_scale": lognormal_fit.params["scale"],
                    },
                    {
                        "year": int(year),
                        "tail_threshold": xmin,
                        "model": "pareto_tail",
                        "n_bins": len(year_bins),
                        "n_workers_used": total_count,
                        "nll": pareto_nll,
                        "aic": 2 * 1 + 2 * pareto_nll,
                        "bic": math.log(total_count) * 1 + 2 * pareto_nll,
                        "converged": bool(pareto_result.success),
                        "message": str(pareto_result.message),
                        "param_alpha": pareto_alpha,
                    },
                ]
            )
    return pd.DataFrame(rows)


def top_share_fit_comparison(
    bins: pd.DataFrame,
    fit_results: pd.DataFrame,
    lower_threshold: float = 3750.0,
) -> pd.DataFrame:
    """Compare observed and fitted top-bracket shares by model and year."""
    rows: list[dict[str, float | int | str]] = []
    for year, year_bins in bins.groupby("year"):
        fit_bins = prepare_bins_for_fit(year_bins)
        observed_total = float(fit_bins["count"].sum())
        observed_open_top = float(fit_bins.loc[fit_bins["bin_type"] == "open_top", "count"].sum() / observed_total)
        observed_top_two = float(fit_bins.loc[fit_bins["lower"] >= lower_threshold, "count"].sum() / observed_total)

        for _, fit_row in fit_results.query("year == @year").iterrows():
            fit = fit_row_to_object(fit_row)
            expected = expected_counts(fit, fit_bins)
            fit_bins_local = fit_bins.copy()
            fit_bins_local["expected_count"] = expected
            expected_total = float(fit_bins_local["expected_count"].sum())
            expected_open_top = float(fit_bins_local.loc[fit_bins_local["bin_type"] == "open_top", "expected_count"].sum() / expected_total)
            expected_top_two = float(fit_bins_local.loc[fit_bins_local["lower"] >= lower_threshold, "expected_count"].sum() / expected_total)
            rows.append(
                {
                    "year": int(year),
                    "model": fit.name,
                    "observed_open_top_share": observed_open_top,
                    "expected_open_top_share": expected_open_top,
                    "observed_top_two_share": observed_top_two,
                    "expected_top_two_share": expected_top_two,
                }
            )
    return pd.DataFrame(rows)
