from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def project_root() -> Path:
    """Return the project root from the script location."""
    return Path(__file__).resolve().parents[1]


def load_required_csv(data_dir: Path, filename: str) -> pd.DataFrame:
    """Load one required processed CSV."""
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required processed file: {path}")
    return pd.read_csv(path)


def build_summary_payload(processed_dir: Path) -> dict[str, object]:
    """Build a compact summary payload from processed notebook outputs."""
    main_findings = load_required_csv(processed_dir, "main_findings_first.csv")
    winners = load_required_csv(processed_dir, "model_winners_by_year.csv")
    decile_summary = load_required_csv(processed_dir, "decile_fit_validation_summary.csv")
    sensitivity = load_required_csv(processed_dir, "sensitivity_summary.csv")
    tail_winners = load_required_csv(processed_dir, "tail_model_winners.csv")

    gamma_median_error = float(
        decile_summary.loc[decile_summary["model"] == "gamma", "mean_abs_relative_error"].median()
    )
    lognormal_median_error = float(
        decile_summary.loc[decile_summary["model"] == "lognormal", "mean_abs_relative_error"].median()
    )
    bic_winner_counts = winners["winner_bic"].value_counts().to_dict()
    sensitivity_winner_counts = (
        sensitivity.groupby("scenario")["winner_bic"].agg(lambda series: series.value_counts().to_dict()).to_dict()
    )
    tail_winner_counts = (
        tail_winners.groupby("tail_threshold")["winner_bic"].agg(lambda series: series.value_counts().to_dict()).to_dict()
    )

    findings_map = dict(zip(main_findings["finding"], main_findings["value"], strict=False))
    return {
        "public_grouped_window": findings_map["Public grouped-analysis window"],
        "grouped_year_count": int(findings_map["Most frequent BIC winner"] != "" and len(winners)),
        "most_frequent_bic_winner": findings_map["Most frequent BIC winner"],
        "gamma_median_decile_error": gamma_median_error,
        "lognormal_median_decile_error": lognormal_median_error,
        "bic_winner_counts": bic_winner_counts,
        "sensitivity_winner_counts": sensitivity_winner_counts,
        "tail_winner_counts": tail_winner_counts,
    }


def build_markdown_summary(payload: dict[str, object]) -> str:
    """Render a compact markdown summary from the notebook outputs."""
    lines = [
        "# Notebook Summary",
        "",
        "## Headline",
        "",
        f"- Public grouped-data window: `{payload['public_grouped_window']}`",
        f"- Most frequent BIC winner: `{payload['most_frequent_bic_winner']}`",
        f"- Median Gamma decile error: `{payload['gamma_median_decile_error']:.4f}`",
        f"- Median Lognormal decile error: `{payload['lognormal_median_decile_error']:.4f}`",
        "",
        "## Model wins",
        "",
    ]

    for model, count in sorted(payload["bic_winner_counts"].items()):  # type: ignore[union-attr]
        lines.append(f"- `{model}`: {count} yearly BIC wins")

    lines.extend(
        [
            "",
            "## Sensitivity scenarios",
            "",
        ]
    )
    sensitivity_counts: dict[str, dict[str, int]] = payload["sensitivity_winner_counts"]  # type: ignore[assignment]
    for scenario, counts in sensitivity_counts.items():
        pieces = ", ".join(f"{model}={count}" for model, count in sorted(counts.items()))
        lines.append(f"- `{scenario}`: {pieces}")

    lines.extend(
        [
            "",
            "## Tail-only winners",
            "",
        ]
    )
    tail_counts: dict[str, dict[str, int]] = payload["tail_winner_counts"]  # type: ignore[assignment]
    for threshold, counts in tail_counts.items():
        pieces = ", ".join(f"{model}={count}" for model, count in sorted(counts.items()))
        lines.append(f"- `xmin={threshold}`: {pieces}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The current grouped-data evidence supports Lognormal as the strongest full-distribution benchmark in this notebook.",
            "- Gamma remains useful as an interpretable comparison model, especially for the body of the distribution.",
            "- The sensitivity and tail sections should be checked alongside the headline winner before making a strong distributional claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Generate a markdown and JSON summary from notebook processed outputs."""
    root = project_root()
    processed_dir = root / "data" / "processed"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    payload = build_summary_payload(processed_dir)
    markdown = build_markdown_summary(payload)

    markdown_path = reports_dir / "notebook_summary.md"
    json_path = reports_dir / "notebook_summary.json"
    markdown_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {markdown_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
