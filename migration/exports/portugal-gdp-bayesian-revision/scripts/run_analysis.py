"""Run the Portugal GDP revision analysis from the command line.

Reads cached data from ``data/raw/`` when present (pass ``--refresh`` to
re-download), runs the reconciliation and the Bayesian model, writes the output
tables to ``outputs/`` and prints the headline findings.

Usage
-----
    python scripts/run_analysis.py
    python scripts/run_analysis.py --refresh --samples 50000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pt_gdp_bayes.pipeline import TARGET_YEAR, run_analysis  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true", help="re-download every series instead of using the cache"
    )
    parser.add_argument(
        "--samples", type=int, default=20_000, help="posterior sample size (default: 20000)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_analysis(PROJECT_ROOT, refresh=args.refresh, n_samples=args.samples)

    print("=" * 78)
    print("1. IS THE DENOMINATOR CORRECTION OWED?")
    print("=" * 78)
    print(result.verdict.summary())

    print()
    print("=" * 78)
    print("2. BUT IS THE NUMERATOR ON THE SAME BASIS? THE CAPTURE QUESTION")
    print("=" * 78)
    check = result.employment_check
    first, last = check.iloc[0], check.iloc[-1]
    population_growth = (last["na_population"] / first["na_population"] - 1) * 100
    employment_growth = (last["na_employment"] / first["na_employment"] - 1) * 100
    print(
        f"Residents the accounts have not counted: {result.uncounted_residents:,.0f} "
        f"-> roughly {result.missing_workers:,.0f} workers."
    )
    print(
        f"On the accounts' OWN population base, {int(first['year'])}-{int(last['year'])}: population "
        f"+{population_growth:.1f}%, employment +{employment_growth:.1f}%."
    )
    print(
        "That looks like absorption but is not evidence of it: the employment count is grossed "
        "up on the same un-rebenchmarked base, and Portugal's employment-rate rise is "
        "unremarkable beside the EU27, Spain and Ireland. Capture is NOT identified by the data."
    )
    curve = result.capture_curve
    print(
        f"Index across the full capture range: "
        f"{curve['pps_median'].iloc[0]:.1f} (phi=0, accounts missed all their output) to "
        f"{curve['pps_median'].iloc[-1]:.1f} (phi=1, pure denominator correction). "
        f"Production-side compilation argues for the latter."
    )

    print()
    print("=" * 78)
    print(f"3. MODEL vs THE PUBLISHED {TARGET_YEAR} OUTTURN (conditional nowcast)")
    print("=" * 78)
    print(result.gdp_score.summary(label="Nominal GDP", scale=1e9, unit=" bn EUR"))
    print("Conditional on INE's announced real growth, so the miss is deflator "
          "error, not forecast error. The unconditional test is the back-test below.")
    coverage = result.backtest["in_interval"].mean()
    mae = result.backtest["pct_error"].abs().mean()
    print(f"Back-test {result.backtest['year'].min()}-{result.backtest['year'].max()}: "
          f"{coverage:.0%} coverage of the 90% interval, {mae:.1f}% mean absolute error.")

    for label, diagnostics in [
        ("population", result.population_result.diagnostics),
        ("GDP", result.gdp_result.diagnostics),
    ]:
        for message in diagnostics:
            print(f"  [!] {label}: {message}")

    print()
    print("=" * 78)
    print("4. POSTERIOR SUMMARY")
    print("=" * 78)
    print(result.summary.to_string(float_format=lambda x: f"{x:,.3f}"))

    print()
    print(f"Output tables written to {PROJECT_ROOT / 'outputs'}")


if __name__ == "__main__":
    main()
