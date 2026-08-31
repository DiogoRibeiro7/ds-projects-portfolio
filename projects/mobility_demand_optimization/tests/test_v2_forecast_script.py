"""End-to-end smoke test for the preregistered v2 forecast runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mobility_optimization.frozen_inputs import FROZEN_ZONES


def _load_runner() -> object:
    """Load the script as a module without executing its ``__main__`` block."""
    script = Path("scripts/run_v2_forecasts.py")
    spec = importlib.util.spec_from_file_location("run_v2_forecasts_smoke", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load run_v2_forecasts.py for smoke testing.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_forecast_script_writes_complete_june_outputs(tmp_path: Path) -> None:
    """The real v2 forecast script must execute end-to-end on synthetic data."""
    hours = pd.date_range(
        start="2026-05-25T00:00:00",
        end="2026-07-01T00:00:00",
        freq="h",
        inclusive="left",
    )
    index = pd.MultiIndex.from_product(
        [hours, FROZEN_ZONES],
        names=["timestamp", "zone_id"],
    )
    panel = index.to_frame(index=False)
    hour_of_week = panel["timestamp"].dt.dayofweek * 24 + panel["timestamp"].dt.hour
    panel["demand"] = ((hour_of_week + panel["zone_id"]) % 17 + 1).astype(np.int64)

    panel_path = tmp_path / "synthetic_panel.parquet"
    output_dir = tmp_path / "forecasts"
    panel.to_parquet(panel_path, index=False)

    runner = _load_runner()
    runner.PANEL_PATH = panel_path
    runner.OUTPUT_DIR = output_dir
    runner.main()

    forecast_path = output_dir / "june_poisson_hour_of_week.parquet"
    summary_path = output_dir / "summary.json"
    assert forecast_path.exists()
    assert summary_path.exists()

    forecasts = pd.read_parquet(forecast_path)
    assert len(forecasts) == 30 * 24 * 30
    assert forecasts["timestamp"].min() == pd.Timestamp("2026-06-01T00:00:00")
    assert forecasts["timestamp"].max() == pd.Timestamp("2026-06-30T23:00:00")
    assert tuple(sorted(int(value) for value in forecasts["zone_id"].unique())) == FROZEN_ZONES
    assert forecasts["y_pred"].notna().all()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["design"] == "v2.0-preregistered"
    assert summary["origins"] == 30
    assert summary["horizon_hours"] == 24
    assert summary["rows"] == 21_600
    assert summary["negative_binomial_alpha_for_scenarios"] == 0.05
    assert summary["alpha_reselected"] is False
