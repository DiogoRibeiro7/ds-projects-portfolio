"""Plotting helpers for UHI exposure analysis."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_exposed_population_share(summary: pd.DataFrame, output_path: str | Path | None = None) -> None:
    """Plot the share of population living in UHI-exposed cells by city."""
    ax = summary.sort_values("city").plot.bar(
        x="city",
        y="exposed_population_share",
        legend=False,
        figsize=(7, 4),
    )
    ax.set_xlabel("")
    ax.set_ylabel("Share of population exposed")
    ax.set_title("Population exposed to stronger urban heat island effect")
    ax.set_ylim(0, max(1.0, float(summary["exposed_population_share"].max()) * 1.15))
    ax.yaxis.set_major_formatter(lambda value, _position: f"{value:.0%}")
    plt.xticks(rotation=0)
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160)
    plt.close()


def plot_representation_ratios(
    representation: pd.DataFrame,
    output_path: str | Path | None = None,
) -> None:
    """Plot representation ratios for each group and city."""
    pivot = representation.pivot(index="group", columns="city", values="representation_ratio")
    ax = pivot.plot.bar(figsize=(9, 5))
    ax.axhline(1.0, linewidth=1)
    ax.set_xlabel("")
    ax.set_ylabel("Representation ratio")
    ax.set_title("Group representation in UHI-exposed areas")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160)
    plt.close()


def plot_group_share_comparison(
    representation: pd.DataFrame,
    city: str,
    output_path: str | Path | None = None,
) -> None:
    """Plot full-city vs exposed-area group shares for one city."""
    data = representation.loc[representation["city"] == city].copy()
    plot_df = data.set_index("group")[["city_group_share", "exposed_group_share"]]
    ax = plot_df.plot.bar(figsize=(9, 5))
    ax.set_xlabel("")
    ax.set_ylabel("Population share")
    ax.set_title(f"{city}: group shares in city vs UHI-exposed areas")
    ax.yaxis.set_major_formatter(lambda value, _position: f"{value:.0%}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160)
    plt.close()


def plot_population_weighted_uhi_distribution(
    cells: pd.DataFrame,
    output_path: str | Path | None = None,
    bin_width: float = 0.25,
) -> None:
    """Plot the population-weighted UHI distribution for each city."""
    min_value = float(cells["uhi_intensity_celsius"].min())
    max_value = float(cells["uhi_intensity_celsius"].max())
    lower = np.floor(min_value / bin_width) * bin_width
    upper = np.ceil(max_value / bin_width) * bin_width + bin_width
    bins = np.arange(lower, upper + bin_width, bin_width)

    fig, ax = plt.subplots(figsize=(9, 5))
    for city, city_cells in cells.groupby("city", sort=True):
        weights = city_cells["population_total"] / city_cells["population_total"].sum()
        ax.hist(
            city_cells["uhi_intensity_celsius"],
            bins=bins,
            weights=weights,
            histtype="step",
            linewidth=2,
            label=city,
        )

    ax.axvline(2.0, color="black", linestyle="--", linewidth=1, label="2.0 °C threshold")
    ax.set_xlabel("UHI intensity (°C)")
    ax.set_ylabel("Share of city population")
    ax.set_title("Population-weighted distribution of modelled UHI intensity")
    ax.yaxis.set_major_formatter(lambda value, _position: f"{value:.0%}")
    ax.legend()
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160)
    plt.close()


def plot_threshold_sensitivity(
    cells: pd.DataFrame,
    output_path: str | Path | None = None,
    thresholds: np.ndarray | None = None,
) -> None:
    """Plot exposed population share by UHI threshold for each city."""
    thresholds = thresholds if thresholds is not None else np.arange(0.0, 2.76, 0.25)

    records: list[dict[str, float | str]] = []
    for city, city_cells in cells.groupby("city", sort=True):
        total_population = float(city_cells["population_total"].sum())
        for threshold in thresholds:
            exposed_population = float(
                city_cells.loc[
                    city_cells["uhi_intensity_celsius"] >= float(threshold), "population_total"
                ].sum()
            )
            records.append(
                {
                    "city": city,
                    "threshold": float(threshold),
                    "exposed_population_share": exposed_population / total_population,
                }
            )

    sensitivity = pd.DataFrame.from_records(records)
    fig, ax = plt.subplots(figsize=(9, 5))
    for city, city_data in sensitivity.groupby("city", sort=True):
        ax.plot(
            city_data["threshold"],
            city_data["exposed_population_share"],
            marker="o",
            linewidth=2,
            label=city,
        )

    ax.axvline(2.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("UHI threshold (°C)")
    ax.set_ylabel("Share of city population above threshold")
    ax.set_title("Threshold sensitivity of exposed population share")
    ax.yaxis.set_major_formatter(lambda value, _position: f"{value:.0%}")
    ax.legend()
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160)
    plt.close()


def plot_population_uhi_band_comparison(
    cells: pd.DataFrame,
    output_path: str | Path | None = None,
) -> None:
    """Plot population shares by broad UHI band for each city."""
    bins = [-np.inf, 0.0, 0.5, 1.0, 1.5, 2.0, np.inf]
    labels = ["<0.0", "0.0-0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", ">=2.0"]
    banded = cells.copy()
    banded["uhi_band"] = pd.cut(
        banded["uhi_intensity_celsius"],
        bins=bins,
        labels=labels,
        right=False,
    )
    summary = (
        banded.groupby(["city", "uhi_band"], observed=False)["population_total"]
        .sum()
        .groupby(level=0)
        .transform(lambda values: values / values.sum())
        .rename("population_share")
        .reset_index()
    )
    pivot = summary.pivot(index="city", columns="uhi_band", values="population_share").fillna(0.0)

    ax = pivot.plot.bar(stacked=True, figsize=(10, 5), width=0.7)
    ax.set_xlabel("")
    ax.set_ylabel("Share of city population")
    ax.set_title("Where residents fall in the UHI intensity range")
    ax.yaxis.set_major_formatter(lambda value, _position: f"{value:.0%}")
    ax.legend(title="UHI band", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160)
    plt.close()


def plot_representation_heatmap(
    representation: pd.DataFrame,
    output_path: str | Path | None = None,
    value_column: str = "representation_ratio",
) -> None:
    """Plot a compact heatmap of representation metrics by group and city."""
    pivot = representation.pivot(index="group", columns="city", values=value_column)
    values = pivot.to_numpy()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    image = ax.imshow(values, aspect="auto", cmap="RdYlBu_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Representation ratios by city and group")

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            ax.text(
                column_index,
                row_index,
                f"{values[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(value_column.replace("_", " ").title())
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160)
    plt.close()


def plot_group_threshold_small_multiples(
    representation_by_threshold: pd.DataFrame,
    output_path: str | Path | None = None,
) -> None:
    """Plot representation ratios across thresholds for each group."""
    groups = representation_by_threshold["group"].drop_duplicates().tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for axis, group in zip(axes_flat, groups):
        group_data = representation_by_threshold.loc[
            representation_by_threshold["group"] == group
        ].copy()
        for city, city_data in group_data.groupby("city", sort=True):
            axis.plot(
                city_data["threshold_celsius"],
                city_data["representation_ratio"],
                marker="o",
                linewidth=2,
                label=city,
            )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(group)
        axis.set_xlabel("Threshold (°C)")
        axis.set_ylabel("Representation ratio")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("How group overrepresentation changes with the UHI threshold", y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160)
    plt.close()


def plot_exposure_band_decomposition(
    decomposition: pd.DataFrame,
    output_path: str | Path | None = None,
) -> None:
    """Plot how exposed population is distributed across fine UHI bands."""
    pivot = (
        decomposition.pivot(index="uhi_band", columns="city", values="band_population_share_of_exposed")
        .fillna(0.0)
    )
    pivot = pivot.sort_index(ascending=False)

    ax = pivot.plot.barh(figsize=(9, 5))
    ax.set_xlabel("Share of exposed population")
    ax.set_ylabel("UHI band (°C)")
    ax.set_title("Where exposed population comes from within the hot tail")
    ax.xaxis.set_major_formatter(lambda value, _position: f"{value:.0%}")
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160)
    plt.close()


def plot_hotspot_map(
    cells_geo: gpd.GeoDataFrame,
    output_path: str | Path | None = None,
    top_n: int = 25,
) -> None:
    """Plot a pseudo-map of UHI intensity with the hottest populated fragments highlighted."""
    top_cells = cells_geo.loc[cells_geo["population_total"] > 0].nlargest(
        top_n, "uhi_intensity_celsius"
    )
    cities = sorted(cells_geo["city"].unique().tolist())
    fig, axes = plt.subplots(1, len(cities), figsize=(6 * len(cities), 6))
    if len(cities) == 1:
        axes = [axes]

    for axis, city in zip(axes, cities):
        city_cells = cells_geo.loc[cells_geo["city"] == city].copy()
        city_top = top_cells.loc[top_cells["city"] == city].copy()
        city_cells.plot(
            ax=axis,
            column="uhi_intensity_celsius",
            cmap="inferno",
            linewidth=0.1,
            edgecolor="none",
            legend=False,
        )
        if not city_top.empty:
            city_top.boundary.plot(ax=axis, color="cyan", linewidth=1.0)
            top_centroids = city_top.geometry.centroid
            axis.scatter(
                top_centroids.x,
                top_centroids.y,
                s=np.clip(city_top["population_total"] / 30, 10, 120),
                facecolors="none",
                edgecolors="white",
                linewidths=0.8,
            )
        axis.set_title(f"{city}: UHI surface and hottest populated fragments")
        axis.set_axis_off()

    sm = plt.cm.ScalarMappable(
        cmap="inferno",
        norm=plt.Normalize(
            vmin=float(cells_geo["uhi_intensity_celsius"].min()),
            vmax=float(cells_geo["uhi_intensity_celsius"].max()),
        ),
    )
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=axes, shrink=0.8)
    colorbar.set_label("UHI intensity (°C)")
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160)
    plt.close()


def plot_representation_with_ci(
    bootstrap_df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> None:
    """Forest plot of representation ratios with bootstrap confidence intervals.

    Significant over-representation (CI entirely above 1) is drawn in red, significant
    under-representation (CI entirely below 1) in blue, and non-significant ratios in grey,
    so the eye separates robust findings from noise.
    """
    df = bootstrap_df.copy()
    df["label"] = df["city"] + " - " + df["group"]
    df = df.sort_values(["city", "representation_ratio"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(df) + 1.5))
    for i, row in df.iterrows():
        if row["significant"] and row["representation_ratio"] > 1:
            color = "#A4161A"
        elif row["significant"]:
            color = "#1f6aa5"
        else:
            color = "0.6"
        ax.plot([row["ci_low"], row["ci_high"]], [i, i], color=color, lw=2.2)
        ax.plot(row["representation_ratio"], i, "o", color=color, ms=7)
    ax.axvline(1.0, color="black", ls="--", lw=1)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label"])
    ax.set_title("Group representation in heat-exposed cells (ratio with 90% bootstrap CI)")
    ax.set_xlabel("Representation ratio  (1 = proportional, >1 = over-represented)")
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160)
    plt.close()


def plot_dose_response(
    dose_df: pd.DataFrame,
    output_path: str | Path | None = None,
    value_column: str = "weighted_corr_uhi_share",
) -> None:
    """Bar chart of the threshold-free UHI-vs-group-share association by city and group."""
    pivot = dose_df.pivot(index="group", columns="city", values=value_column)

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="barh", ax=ax, color={"Lisbon": "#1f6aa5", "Porto": "#E6892B"})
    ax.axvline(0.0, color="black", lw=1)
    ax.set_title("Dose-response: association of group share with UHI intensity (all cells)")
    ax.set_xlabel("Population-weighted correlation  (>0 = group concentrated in hotter cells)")
    ax.set_ylabel("")
    ax.legend(title="City", frameon=False)
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160)
    plt.close()
