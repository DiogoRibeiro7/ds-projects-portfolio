"""Plotting helpers for UHI exposure analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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
