"""Run Porto/Lisbon UHI exposure analysis from a prepared CSV or Parquet file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from uhi_exposure.io import load_exposure_cells, write_dataframe
from uhi_exposure.metrics import compute_city_exposure_summary, compute_group_representation
from uhi_exposure.plotting import plot_exposed_population_share, plot_representation_ratios


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input CSV or Parquet with cell-level data.")
    parser.add_argument("--output", required=True, help="Output CSV or Parquet for representation metrics.")
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional output path for city-level exposure summary.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="UHI intensity threshold used to mark exposed cells.",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Optional directory where plots will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the analysis."""
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    threshold = float(args.threshold)

    cells = load_exposure_cells(input_path)
    summary = compute_city_exposure_summary(cells, threshold=threshold)
    representation = compute_group_representation(cells, threshold=threshold)

    write_dataframe(representation, output_path)

    summary_output = args.summary_output
    if summary_output is None:
        summary_output = output_path.with_name(output_path.stem + "_city_summary" + output_path.suffix)
    write_dataframe(summary, summary_output)

    if args.plot_dir is not None:
        plot_dir = Path(args.plot_dir)
        plot_exposed_population_share(summary, plot_dir / "exposed_population_share.png")
        plot_representation_ratios(representation, plot_dir / "representation_ratios.png")

    with pd.option_context("display.max_columns", None, "display.width", 120):
        print("\nCity exposure summary")
        print(summary)
        print("\nGroup representation")
        print(representation)


if __name__ == "__main__":
    main()
