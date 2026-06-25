"""Run the Portugal Bayesian GDP revision model from the command line.

This script downloads long-run World Bank data, patches the revised INE 2025
population estimate, and estimates posterior distributions for 2025 population,
GDP, GDP per capita and a PPS-index sensitivity estimate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pt_gdp_bayes.bayes import summarize_samples
from pt_gdp_bayes.data_sources import (
    fetch_world_bank_panel,
    get_known_value,
    load_known_observations,
    patch_manual_observation,
)
from pt_gdp_bayes.model import (
    estimate_gdp_posterior_samples,
    estimate_population_posterior_samples,
    estimate_pps_index_samples,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWN_OBSERVATIONS_PATH = PROJECT_ROOT / "data" / "known_observations.csv"

WORLD_BANK_INDICATORS = {
    "population": "SP.POP.TOTL",
    "gdp_current_usd": "NY.GDP.MKTP.CD",
    "gdp_constant_2015_usd": "NY.GDP.MKTP.KD",
    "real_gdp_growth_pct": "NY.GDP.MKTP.KD.ZG",
    "gdp_deflator_pct": "NY.GDP.DEFL.KD.ZG",
    "unemployment_pct": "SL.UEM.TOTL.ZS",
    "labor_force": "SL.TLF.TOTL.IN",
    "exports_current_usd": "NE.EXP.GNFS.CD",
    "imports_current_usd": "NE.IMP.GNFS.CD",
    "household_consumption_current_usd": "NE.CON.PRVT.CD",
    "gross_capital_formation_current_usd": "NE.GDI.TOTL.CD",
}


def main() -> None:
    observations = load_known_observations(KNOWN_OBSERVATIONS_PATH)
    revised_population_2025 = get_known_value(observations, "resident_population", 2025)
    real_gdp_growth_2025 = get_known_value(observations, "real_gdp_growth", 2025)
    preliminary_pps_index = get_known_value(observations, "preliminary_gdp_pc_pps_index", 2025)
    population_correction_factor = get_known_value(observations, "population_correction_factor", 2025)

    panel = fetch_world_bank_panel(
        "PRT",
        WORLD_BANK_INDICATORS,
        start_year=1960,
        end_year=2025,
    )
    panel = patch_manual_observation(
        panel,
        year=2025,
        column="population",
        value=revised_population_2025,
        note="Patched with revised INE 2025 resident population estimate",
    )

    population_result = estimate_population_posterior_samples(
        panel,
        target_year=2025,
        population_col="population",
        observed_population=revised_population_2025,
        observation_relative_sd=0.001,
        n_samples=20_000,
    )

    gdp_result = estimate_gdp_posterior_samples(
        panel,
        target_year=2025,
        population_samples=population_result.samples,
        gdp_col="gdp_current_usd",
        population_col="population",
        deflator_col="gdp_deflator_pct",
        known_real_gdp_growth=real_gdp_growth_2025,
        n_samples=20_000,
    )

    gdp_pc_samples = gdp_result.gdp_samples / population_result.samples
    pps_index_samples = estimate_pps_index_samples(
        preliminary_index=preliminary_pps_index,
        population_correction_factor=population_correction_factor,
        n_samples=20_000,
    )

    summary = pd.DataFrame(
        {
            "population_2025": summarize_samples(population_result.samples),
            "gdp_2025_current_usd": summarize_samples(gdp_result.gdp_samples),
            "gdp_pc_2025_current_usd": summarize_samples(gdp_pc_samples),
            "gdp_pc_pps_index_sensitivity": summarize_samples(pps_index_samples),
        }
    ).T
    print(summary.to_string(float_format=lambda x: f"{x:,.3f}"))


if __name__ == "__main__":
    main()
