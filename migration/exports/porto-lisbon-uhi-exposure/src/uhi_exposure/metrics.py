"""Metrics for urban heat island exposure and representation analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from uhi_exposure.schema import ExposureColumns, validate_exposure_cells

GROUP_COLUMNS: dict[str, str] = {
    "children_0_14": "age_0_14",
    "older_65_plus": "age_65_plus",
    "not_employed": "not_employed",
    "born_outside_eu": "born_outside_eu",
}


def add_exposure_flag(
    df: pd.DataFrame,
    threshold: float = 2.0,
    columns: ExposureColumns | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with a boolean urban heat exposure flag.

    Parameters
    ----------
    df:
        Cell-level input table.
    threshold:
        UHI intensity threshold in degrees Celsius. Cells with intensity greater
        than or equal to this threshold are marked as exposed.
    columns:
        Optional column mapping.

    Returns
    -------
    pandas.DataFrame
        A validated copy of the input with ``is_uhi_exposed`` as a boolean column.
    """
    columns = columns or ExposureColumns()
    validate_exposure_cells(df, columns)

    result = df.copy()
    if columns.is_uhi_exposed in result.columns:
        result[columns.is_uhi_exposed] = result[columns.is_uhi_exposed].astype(bool)
    else:
        result[columns.is_uhi_exposed] = result[columns.uhi_intensity_celsius] >= threshold

    result["not_employed"] = result[columns.population_total] - result[columns.employed]
    return result


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide two numbers and return NaN for zero denominators."""
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def compute_city_exposure_summary(
    df: pd.DataFrame,
    threshold: float = 2.0,
    columns: ExposureColumns | None = None,
) -> pd.DataFrame:
    """Compute city-level population exposure summary.

    Parameters
    ----------
    df:
        Cell-level input table.
    threshold:
        UHI intensity threshold used when ``is_uhi_exposed`` is absent.
    columns:
        Optional column mapping.

    Returns
    -------
    pandas.DataFrame
        One row per city with total population, exposed population, and exposure share.
    """
    columns = columns or ExposureColumns()
    prepared = add_exposure_flag(df, threshold=threshold, columns=columns)

    records: list[dict[str, float | str]] = []
    for city, city_df in prepared.groupby(columns.city, sort=True):
        total_population = float(city_df[columns.population_total].sum())
        exposed_population = float(
            city_df.loc[city_df[columns.is_uhi_exposed], columns.population_total].sum()
        )
        records.append(
            {
                "city": city,
                "total_population": total_population,
                "exposed_population": exposed_population,
                "exposed_population_share": _safe_divide(exposed_population, total_population),
                "mean_uhi_intensity_celsius": float(
                    np.average(
                        city_df[columns.uhi_intensity_celsius],
                        weights=city_df[columns.population_total],
                    )
                ),
                "mean_uhi_intensity_exposed_celsius": float(
                    np.average(
                        city_df.loc[
                            city_df[columns.is_uhi_exposed], columns.uhi_intensity_celsius
                        ],
                        weights=city_df.loc[
                            city_df[columns.is_uhi_exposed], columns.population_total
                        ],
                    )
                )
                if exposed_population > 0
                else float("nan"),
            }
        )

    return pd.DataFrame.from_records(records)


def compute_group_representation(
    df: pd.DataFrame,
    threshold: float = 2.0,
    columns: ExposureColumns | None = None,
) -> pd.DataFrame:
    """Compute group representation inside exposed UHI areas.

    For each city and population group, this function compares the group's share
    inside UHI-exposed cells with the group's share in the city as a whole.

    Parameters
    ----------
    df:
        Cell-level input table.
    threshold:
        UHI intensity threshold used when ``is_uhi_exposed`` is absent.
    columns:
        Optional column mapping.

    Returns
    -------
    pandas.DataFrame
        Long table with one row per city and group.
    """
    columns = columns or ExposureColumns()
    prepared = add_exposure_flag(df, threshold=threshold, columns=columns)

    records: list[dict[str, float | str]] = []
    for city, city_df in prepared.groupby(columns.city, sort=True):
        total_population = float(city_df[columns.population_total].sum())
        exposed_df = city_df.loc[city_df[columns.is_uhi_exposed]].copy()
        exposed_population = float(exposed_df[columns.population_total].sum())

        for group_name, group_column in GROUP_COLUMNS.items():
            full_group_count = float(city_df[group_column].sum())
            exposed_group_count = float(exposed_df[group_column].sum())
            city_share = _safe_divide(full_group_count, total_population)
            exposed_share = _safe_divide(exposed_group_count, exposed_population)
            difference_pp = (exposed_share - city_share) * 100
            ratio = _safe_divide(exposed_share, city_share)

            records.append(
                {
                    "city": city,
                    "group": group_name,
                    "city_group_count": full_group_count,
                    "exposed_group_count": exposed_group_count,
                    "city_group_share": city_share,
                    "exposed_group_share": exposed_share,
                    "difference_percentage_points": difference_pp,
                    "representation_ratio": ratio,
                    "is_overrepresented": bool(ratio > 1.0) if not np.isnan(ratio) else False,
                }
            )

    return pd.DataFrame.from_records(records)


def compute_threshold_sensitivity(
    df: pd.DataFrame,
    thresholds: list[float] | np.ndarray,
    columns: ExposureColumns | None = None,
) -> pd.DataFrame:
    """Compute exposed population share for a range of UHI thresholds."""
    columns = columns or ExposureColumns()
    validate_exposure_cells(df, columns)

    records: list[dict[str, float | str]] = []
    for city, city_df in df.groupby(columns.city, sort=True):
        total_population = float(city_df[columns.population_total].sum())
        for threshold in thresholds:
            exposed_population = float(
                city_df.loc[
                    city_df[columns.uhi_intensity_celsius] >= float(threshold),
                    columns.population_total,
                ].sum()
            )
            records.append(
                {
                    "city": city,
                    "threshold_celsius": float(threshold),
                    "exposed_population": exposed_population,
                    "exposed_population_share": _safe_divide(exposed_population, total_population),
                }
            )

    return pd.DataFrame.from_records(records)


def compute_group_representation_by_threshold(
    df: pd.DataFrame,
    thresholds: list[float] | np.ndarray,
    columns: ExposureColumns | None = None,
) -> pd.DataFrame:
    """Compute representation ratios for each group across multiple thresholds."""
    columns = columns or ExposureColumns()
    records: list[pd.DataFrame] = []
    for threshold in thresholds:
        threshold_result = compute_group_representation(df, threshold=float(threshold), columns=columns)
        threshold_result["threshold_celsius"] = float(threshold)
        records.append(threshold_result)

    return pd.concat(records, ignore_index=True)


def compute_exposure_band_decomposition(
    df: pd.DataFrame,
    threshold: float = 2.0,
    band_width: float = 0.1,
    columns: ExposureColumns | None = None,
) -> pd.DataFrame:
    """Decompose exposed population into fine-grained UHI bands above the threshold."""
    columns = columns or ExposureColumns()
    prepared = add_exposure_flag(df, threshold=threshold, columns=columns)
    exposed = prepared.loc[prepared[columns.is_uhi_exposed]].copy()
    if exposed.empty:
        return pd.DataFrame(
            columns=[
                "city",
                "uhi_band",
                "band_population",
                "band_population_share_of_exposed",
                "heat_excess_population",
                "heat_excess_share_of_exposed",
            ]
        )

    max_value = float(exposed[columns.uhi_intensity_celsius].max())
    upper = np.ceil(max_value / band_width) * band_width + band_width
    bins = np.arange(threshold, upper + band_width, band_width)
    labels = [f"{start:.1f}-{end:.1f}" for start, end in zip(bins[:-1], bins[1:])]

    exposed["uhi_band"] = pd.cut(
        exposed[columns.uhi_intensity_celsius],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    exposed["heat_excess_population"] = (
        (exposed[columns.uhi_intensity_celsius] - threshold) * exposed[columns.population_total]
    )

    summary = (
        exposed.groupby([columns.city, "uhi_band"], observed=False)
        .agg(
            band_population=(columns.population_total, "sum"),
            heat_excess_population=("heat_excess_population", "sum"),
        )
        .reset_index()
    )
    summary = summary.loc[summary["band_population"] > 0].copy()
    summary["band_population_share_of_exposed"] = summary.groupby(columns.city)[
        "band_population"
    ].transform(lambda values: values / values.sum())
    summary["heat_excess_share_of_exposed"] = summary.groupby(columns.city)[
        "heat_excess_population"
    ].transform(lambda values: values / values.sum() if values.sum() > 0 else np.nan)
    summary = summary.sort_values(
        [columns.city, "band_population_share_of_exposed"],
        ascending=[True, False],
    )
    return summary


def bootstrap_group_representation(
    df: pd.DataFrame,
    threshold: float = 2.0,
    n_boot: int = 2000,
    ci: float = 0.90,
    columns: ExposureColumns | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Bootstrap confidence intervals for the group representation ratios.

    The representation ratio (group share inside exposed cells / group share in the
    whole city) is a point estimate computed on a small exposed tail, so it needs an
    uncertainty band before any "over-represented" claim can be trusted. We resample
    **cells** with replacement within each city and recompute the ratio, giving a
    bootstrap distribution per group.

    A ratio is reported as statistically meaningful when the central ``ci`` interval
    excludes 1.0. ``prob_overrepresented`` is the bootstrap probability the ratio
    exceeds 1. (Cells are spatially autocorrelated, so these intervals are mildly
    optimistic; treat them as a screen, not an exact test.)

    Returns
    -------
    pandas.DataFrame
        One row per city and group with the observed ratio, the ``ci`` interval,
        ``prob_overrepresented`` and a ``significant`` flag.
    """
    columns = columns or ExposureColumns()
    prepared = add_exposure_flag(df, threshold=threshold, columns=columns)
    observed = compute_group_representation(df, threshold=threshold, columns=columns)
    rng = np.random.default_rng(random_state)
    lo_q, hi_q = (1.0 - ci) / 2.0, 1.0 - (1.0 - ci) / 2.0

    rows: list[dict[str, float | str | bool]] = []
    for city, city_df in prepared.groupby(columns.city, sort=True):
        n = len(city_df)
        pop = city_df[columns.population_total].to_numpy(dtype=float)
        exposed = city_df[columns.is_uhi_exposed].to_numpy(dtype=bool)
        group_arrays = {g: city_df[col].to_numpy(dtype=float) for g, col in GROUP_COLUMNS.items()}
        boot = {g: np.empty(n_boot, dtype=float) for g in GROUP_COLUMNS}

        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            p = pop[idx]
            e = exposed[idx]
            total_pop = p.sum()
            exposed_pop = p[e].sum()
            for g, arr in group_arrays.items():
                gc = arr[idx]
                city_share = gc.sum() / total_pop if total_pop > 0 else np.nan
                exposed_share = gc[e].sum() / exposed_pop if exposed_pop > 0 else np.nan
                boot[g][b] = exposed_share / city_share if city_share and city_share > 0 else np.nan

        for g in GROUP_COLUMNS:
            draws = boot[g][~np.isnan(boot[g])]
            obs = float(
                observed.loc[
                    (observed["city"] == city) & (observed["group"] == g),
                    "representation_ratio",
                ].iloc[0]
            )
            low, high = (float(np.quantile(draws, lo_q)), float(np.quantile(draws, hi_q))) if draws.size else (np.nan, np.nan)
            rows.append(
                {
                    "city": city,
                    "group": g,
                    "representation_ratio": obs,
                    "ci_low": low,
                    "ci_high": high,
                    "prob_overrepresented": float(np.mean(draws > 1.0)) if draws.size else np.nan,
                    "significant": bool(low > 1.0 or high < 1.0) if draws.size else False,
                }
            )

    return pd.DataFrame.from_records(rows)


def _weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Population-weighted Pearson correlation between two arrays."""
    w_sum = w.sum()
    if w_sum == 0:
        return float("nan")
    mx = np.average(x, weights=w)
    my = np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    vx = np.average((x - mx) ** 2, weights=w)
    vy = np.average((y - my) ** 2, weights=w)
    denom = np.sqrt(vx * vy)
    return float(cov / denom) if denom > 0 else float("nan")


def compute_dose_response(
    df: pd.DataFrame,
    columns: ExposureColumns | None = None,
) -> pd.DataFrame:
    """Threshold-free association between UHI intensity and each group's local share.

    Instead of a binary "exposed / not exposed" cut, this measures how each group's
    *local share* of the cell population moves with the cell's UHI intensity, using
    **all** cells and the full heat gradient. We report a population-weighted Pearson
    correlation and a weighted slope (percentage points of group share per +1 °C).

    A positive value means the group is concentrated in hotter cells — corroborating
    (or contradicting) the threshold-based representation ratio without depending on
    where the threshold is drawn.
    """
    columns = columns or ExposureColumns()
    prepared = add_exposure_flag(df, columns=columns)

    rows: list[dict[str, float | str]] = []
    for city, city_df in prepared.groupby(columns.city, sort=True):
        uhi = city_df[columns.uhi_intensity_celsius].to_numpy(dtype=float)
        pop = city_df[columns.population_total].to_numpy(dtype=float)
        ok = pop > 0
        for group_name, group_column in GROUP_COLUMNS.items():
            share = np.full(len(pop), np.nan)
            np.divide(city_df[group_column].to_numpy(dtype=float), pop, out=share, where=pop > 0)
            xu, ys, wp = uhi[ok], share[ok], pop[ok]
            corr = _weighted_corr(xu, ys, wp)
            mx = np.average(xu, weights=wp)
            var_x = np.average((xu - mx) ** 2, weights=wp)
            my = np.average(ys, weights=wp)
            slope = (
                np.average((xu - mx) * (ys - my), weights=wp) / var_x if var_x > 0 else np.nan
            )
            rows.append(
                {
                    "city": city,
                    "group": group_name,
                    "weighted_corr_uhi_share": corr,
                    "slope_pp_share_per_degree": float(slope * 100) if np.isfinite(slope) else np.nan,
                }
            )

    return pd.DataFrame.from_records(rows)


def compute_double_burden(
    df: pd.DataFrame,
    columns: ExposureColumns | None = None,
) -> pd.DataFrame:
    """Cross-classify cells by heat and by heat-vulnerable-age share (the double burden).

    Heat exposure and demographic vulnerability are usually reported separately. Here we
    split each city's cells at the **population-weighted median** of UHI intensity and at the
    median share of physiologically heat-vulnerable residents (children 0-14 plus adults
    65+), then report how much population falls in each of the four quadrants. The
    "high heat + high vulnerability" quadrant is the double burden — where heat and the
    least heat-resilient residents coincide.
    """
    columns = columns or ExposureColumns()
    validate_exposure_cells(df, columns)

    rows: list[dict[str, float | str]] = []
    for city, city_df in df.groupby(columns.city, sort=True):
        work = city_df.copy()
        pop = work[columns.population_total].to_numpy(dtype=float)
        total_pop = pop.sum()
        vulnerable_share = np.divide(
            (work["age_0_14"] + work["age_65_plus"]).to_numpy(dtype=float),
            pop, out=np.zeros(len(pop)), where=pop > 0,
        )
        uhi = work[columns.uhi_intensity_celsius].to_numpy(dtype=float)

        # Population-weighted medians as the split points.
        def _wmedian(values: np.ndarray) -> float:
            order = np.argsort(values)
            cum = np.cumsum(pop[order])
            return float(values[order][np.searchsorted(cum, 0.5 * total_pop)])

        hot = uhi >= _wmedian(uhi)
        vulnerable = vulnerable_share >= _wmedian(vulnerable_share)
        labels = {
            ("high heat + high vulnerability (double burden)"): hot & vulnerable,
            ("high heat + low vulnerability"): hot & ~vulnerable,
            ("low heat + high vulnerability"): ~hot & vulnerable,
            ("low heat + low vulnerability"): ~hot & ~vulnerable,
        }
        for label, mask in labels.items():
            quadrant_pop = float(pop[mask].sum())
            rows.append(
                {
                    "city": city,
                    "quadrant": label,
                    "population": quadrant_pop,
                    "population_share": _safe_divide(quadrant_pop, total_pop),
                    "mean_uhi_celsius": float(np.average(uhi[mask], weights=pop[mask]))
                    if quadrant_pop > 0 else np.nan,
                }
            )

    return pd.DataFrame.from_records(rows)


def compute_green_uhi_relationship(
    df: pd.DataFrame,
    green_column: str = "green_fraction",
    columns: ExposureColumns | None = None,
) -> pd.DataFrame:
    """Relate green cover to UHI intensity per city (the parks-cool-the-city test).

    For each city we measure how a cell's modelled UHI intensity moves with the share
    of the cell that is green (parks/vegetation): a population-weighted correlation and a
    weighted regression slope expressed in **°C per +10 percentage points of green cover**.
    A negative relationship means greener cells are cooler.
    """
    columns = columns or ExposureColumns()
    if green_column not in df.columns:
        raise ValueError(f"Expected a {green_column!r} column; run attach_green_fraction first.")

    rows: list[dict[str, float | str]] = []
    for city, city_df in df.groupby(columns.city, sort=True):
        work = city_df.dropna(subset=[green_column]).copy()
        green = work[green_column].to_numpy(dtype=float)
        uhi = work[columns.uhi_intensity_celsius].to_numpy(dtype=float)
        pop = work[columns.population_total].to_numpy(dtype=float)
        ok = pop > 0
        corr = _weighted_corr(green[ok], uhi[ok], pop[ok])
        mg = np.average(green[ok], weights=pop[ok])
        var_g = np.average((green[ok] - mg) ** 2, weights=pop[ok])
        mu = np.average(uhi[ok], weights=pop[ok])
        slope = (
            np.average((green[ok] - mg) * (uhi[ok] - mu), weights=pop[ok]) / var_g
            if var_g > 0 else np.nan
        )
        rows.append(
            {
                "city": city,
                "n_cells": int(ok.sum()),
                "mean_green_fraction": float(mg),
                "weighted_corr_green_uhi": corr,
                "celsius_per_10pp_green": float(slope * 0.10) if np.isfinite(slope) else np.nan,
            }
        )

    return pd.DataFrame.from_records(rows)


def compute_uhi_by_green_band(
    df: pd.DataFrame,
    green_column: str = "green_fraction",
    bands: tuple[float, ...] = (0.0, 0.05, 0.15, 0.30, 1.01),
    labels: tuple[str, ...] = ("<5%", "5-15%", "15-30%", ">30%"),
    columns: ExposureColumns | None = None,
) -> pd.DataFrame:
    """Population-weighted mean UHI intensity by green-cover band, per city."""
    columns = columns or ExposureColumns()
    work = df.dropna(subset=[green_column]).copy()
    work["green_band"] = pd.cut(
        work[green_column], bins=list(bands), labels=list(labels), include_lowest=True
    )

    rows: list[dict[str, float | str]] = []
    for (city, band), band_df in work.groupby([columns.city, "green_band"], observed=True):
        pop = band_df[columns.population_total].to_numpy(dtype=float)
        if pop.sum() <= 0:
            continue
        rows.append(
            {
                "city": city,
                "green_band": str(band),
                "population": float(pop.sum()),
                "mean_uhi_celsius": float(
                    np.average(band_df[columns.uhi_intensity_celsius], weights=pop)
                ),
            }
        )

    return pd.DataFrame.from_records(rows)
