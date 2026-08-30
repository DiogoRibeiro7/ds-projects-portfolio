"""Recompute decision summaries under the shared prospective headline contract."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mobility_optimization.evaluation import filter_headline_policy_results

PANEL_PATH = Path("data/processed/mobility_demand_hourly.parquet")
RESULT_ROOT = Path("data/results")


def _aggregate(frame: pd.DataFrame, *, include_regret: bool) -> list[dict[str, float | str]]:
    """Aggregate one headline-eligible policy-hour table."""
    aggregations: dict[str, tuple[str, str]] = {
        "mean_total_cost": ("total_cost", "mean"),
        "mean_relocation_cost": ("relocation_cost", "mean"),
        "mean_unmet_cost": ("unmet_cost", "mean"),
        "mean_idle_cost": ("idle_cost", "mean"),
        "mean_service_level": ("service_level", "mean"),
    }
    if include_regret:
        aggregations["mean_regret"] = ("regret", "mean")
    summary = (
        frame.groupby("policy", as_index=False)
        .agg(**aggregations)
        .sort_values("mean_total_cost", ignore_index=True)
    )
    return summary.to_dict(orient="records")


def _rewrite_summary(
    directory: str,
    panel: pd.DataFrame,
    *,
    include_regret: bool,
) -> None:
    """Rewrite one saved summary using only headline-eligible hours."""
    result_dir = RESULT_ROOT / directory
    hourly_path = result_dir / "hourly_policy_results.csv"
    summary_path = result_dir / "summary.json"
    if not hourly_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Missing decision artifacts for {directory}.")

    hourly = pd.read_csv(hourly_path)
    headline = filter_headline_policy_results(hourly, panel)
    metadata = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata["headline_exclusion"] = "DST transition target hours excluded prospectively at summary time"
    metadata["headline_hours"] = int(headline["timestamp"].nunique())
    metadata["summary"] = _aggregate(headline, include_regret=include_regret)
    summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    """Align every decision-layer summary after full trajectories have executed."""
    panel = pd.read_parquet(PANEL_PATH)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    for directory, include_regret in (
        ("allocation", True),
        ("spatial_allocation", True),
        ("rolling_allocation", False),
        ("duration_aware_allocation", False),
    ):
        _rewrite_summary(directory, panel, include_regret=include_regret)


if __name__ == "__main__":
    main()
