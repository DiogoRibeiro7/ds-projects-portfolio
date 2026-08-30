"""Build deterministic Lisbon descriptive decomposition tables."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from housing_tourism.descriptive import decompose_episode, index_to_base_year  # noqa: E402

INPUT_PATH = ROOT / "results" / "processed" / "lisbon_longitudinal_bridged.csv"
ANNUAL_PATH = ROOT / "results" / "processed" / "lisbon_descriptive_annual.csv"
EPISODE_PATH = ROOT / "results" / "processed" / "lisbon_descriptive_episodes.csv"

EPISODES = (
    ("pre_pandemic", 2017, 2019),
    ("pandemic", 2019, 2021),
    ("recovery", 2021, 2024),
    ("post_2022", 2022, 2024),
    ("rent_acceleration", 2022, 2023),
    ("recent_stabilisation", 2023, 2024),
)


def build_annual_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Build annual indexed levels and observed platform exposure columns."""
    result = pd.DataFrame({"year": frame["year"]})
    result["rent_index_2017"] = index_to_base_year(frame, value_col="rent_eur_m2", base_year=2017)
    result["income_index_2017"] = index_to_base_year(frame, value_col="income_eur", base_year=2017)
    result["lhdi"] = frame["lhdi"]
    result["tourism_intensity"] = frame["tourism_intensity"]
    result["tourism_index_2019"] = index_to_base_year(
        frame, value_col="tourism_intensity", base_year=2019
    )
    for column in (
        "listed_units",
        "entire_home_units",
        "listed_units_per_1000_residents",
        "entire_home_per_1000_residents",
    ):
        result[column] = frame[column]
    return result


def build_episode_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Build named episode decompositions from observed endpoint values."""
    rows: list[dict[str, str | int | float]] = []
    for name, start_year, end_year in EPISODES:
        record: dict[str, str | int | float] = {"episode": name}
        record.update(
            decompose_episode(
                frame,
                start_year=start_year,
                end_year=end_year,
            ).as_record()
        )
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    """Regenerate the Lisbon descriptive tables from the bridged evidence."""
    frame = pd.read_csv(INPUT_PATH)
    annual = build_annual_table(frame)
    episodes = build_episode_table(frame)

    annual.to_csv(ANNUAL_PATH, index=False, float_format="%.6f")
    episodes.to_csv(EPISODE_PATH, index=False, float_format="%.6f")

    print("Annual descriptive table:")
    print(annual.to_string(index=False))
    print("\nEpisode decomposition:")
    print(episodes.to_string(index=False))


if __name__ == "__main__":
    main()
