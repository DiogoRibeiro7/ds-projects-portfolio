"""End-to-end analysis pipeline.

The notebook and the command-line script both call into this module, so the
panel construction, the estimation window and the scenario definitions cannot
drift apart between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .bayes import summarize_samples
from .data_sources import (
    default_cache_dir,
    fetch_eurostat_national_accounts_employment,
    fetch_eurostat_national_accounts_population,
    fetch_eurostat_population,
    fetch_eurostat_pps_inputs,
    fetch_oecd_employment_rate_by_birthplace,
    fetch_world_bank_panel,
    get_known_value,
    load_known_observations,
)
from .model import (
    backtest_gdp_model,
    capture_sensitivity_curve,
    estimate_gdp_posterior_samples,
    estimate_population_posterior_samples,
    pps_index_samples,
    score_against_outturn,
)
from .reconciliation import (
    assess_correction_claim,
    compare_population_bases,
    derive_purchasing_power_parity,
    employment_absorbed_revision,
    implied_missing_workers,
)

TARGET_YEAR = 2025

#: Euro entry marks a decisive break in Portuguese nominal series. Nominal GDP
#: growth averaged 15.5% a year before 1985 and 3.7% from 1999; the GDP deflator
#: 12.0% against 2.6%. Fitting across that break biases the level up and inflates
#: the residual variance.
ESTIMATION_START_YEAR = 1999

#: World Bank supplies the long GDP and labour history. Its population series
#: happens to track the *national-accounts* population rather than the revised
#: demographic one, which is why it is named ``wb_population`` and never used as
#: "the population" without qualification.
WORLD_BANK_INDICATORS = {
    "wb_population": "SP.POP.TOTL",
    "gdp_current_eur": "NY.GDP.MKTP.CN",
    "gdp_current_usd": "NY.GDP.MKTP.CD",
    "real_gdp_growth_pct": "NY.GDP.MKTP.KD.ZG",
    "gdp_deflator_pct": "NY.GDP.DEFL.KD.ZG",
    "unemployment_pct": "SL.UEM.TOTL.ZS",
    "labor_force": "SL.TLF.TOTL.IN",
    "working_age_pct": "SP.POP.1564.TO.ZS",
}


@dataclass(frozen=True)
class AnalysisResult:
    """Everything the notebook and the CLI need, computed once."""

    panel: pd.DataFrame
    pps_inputs: pd.DataFrame
    na_employment: pd.DataFrame
    migrant_employment: pd.DataFrame
    vintages: pd.DataFrame
    verdict: object
    employment_check: pd.DataFrame
    population_result: object
    gdp_result: object
    gdp_pc_samples: np.ndarray
    backtest: pd.DataFrame
    capture_curve: pd.DataFrame
    gdp_score: object
    summary: pd.DataFrame
    uncounted_residents: float
    missing_workers: float
    purchasing_power_parity: float
    reference_gdp_pc_pps: float
    actual_gdp: float


def build_panel(
    *,
    cache_dir: str | Path | None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Assemble the modelling panel carrying **both** population bases.

    Portugal has two official population series for 2025 that differ by 5.6%,
    and they answer different questions. The panel keeps them apart by name
    rather than picking one and calling it "population":

    ``na_population``
        National-accounts population (Eurostat ``nama_10_pe``). Compiled with
        GDP, and the denominator of every published per-capita national-accounts
        figure. This is the series the GDP growth model uses, because it is the
        one GDP is vintage-consistent with.
    ``resident_population``
        Revised demographic estimate (Eurostat ``demo_gind``). This is how many
        people actually live in Portugal, and therefore the right denominator
        when the question is output per person.

    Before 1995 only the World Bank series is available; no revision is at stake
    that far back, so it backfills both.
    """

    world_bank = fetch_world_bank_panel(
        "PRT", WORLD_BANK_INDICATORS, start_year=1960, end_year=TARGET_YEAR,
        cache_dir=cache_dir, refresh=refresh,
    )
    demographic = fetch_eurostat_population("PT", cache_dir=cache_dir, refresh=refresh)
    national_accounts = fetch_eurostat_national_accounts_population(
        "PT", cache_dir=cache_dir, refresh=refresh
    )

    panel = (
        world_bank
        .merge(demographic[["year", "population_avg"]], on="year", how="outer")
        .merge(national_accounts, on="year", how="outer")
        .sort_values("year")
        .reset_index(drop=True)
    )

    panel["na_population"] = panel["na_population"].fillna(panel["wb_population"])
    panel["resident_population"] = panel["population_avg"].fillna(panel["wb_population"])
    panel["population_gap_pct"] = (
        panel["resident_population"] / panel["na_population"] - 1.0
    ) * 100.0
    return panel


def run_analysis(
    project_root: str | Path,
    *,
    refresh: bool = False,
    n_samples: int = 20_000,
    write_outputs: bool = True,
) -> AnalysisResult:
    """Run the full analysis and optionally write the output tables."""

    project_root = Path(project_root)
    cache_dir = default_cache_dir(project_root)
    observations = load_known_observations(project_root / "data" / "known_observations.csv")

    def anchor(name: str) -> float:
        return get_known_value(observations, name, TARGET_YEAR)

    panel = build_panel(cache_dir=cache_dir, refresh=refresh)
    pps_inputs = fetch_eurostat_pps_inputs("PT", cache_dir=cache_dir, refresh=refresh)
    na_employment = fetch_eurostat_national_accounts_employment(
        "PT", cache_dir=cache_dir, refresh=refresh
    )
    na_population = fetch_eurostat_national_accounts_population(
        "PT", cache_dir=cache_dir, refresh=refresh
    )
    migrant_employment = fetch_oecd_employment_rate_by_birthplace(
        "PRT", cache_dir=cache_dir, refresh=refresh
    )

    def _value(frame: pd.DataFrame, column: str, *, source: str) -> float:
        """Target-year lookup that fails with a diagnosable message, not an IndexError."""
        rows = frame.loc[frame["year"] == TARGET_YEAR]
        if rows.empty or pd.isna(rows[column].iloc[0]):
            raise ValueError(
                f"{source} has no usable {column!r} for {TARGET_YEAR}. The series has not been "
                f"published yet or the cache is stale; re-run with refresh=True."
            )
        return float(rows[column].iloc[0])

    target_rows = pps_inputs.loc[pps_inputs["year"] == TARGET_YEAR]
    if target_rows.empty:
        raise ValueError(f"Eurostat PPS inputs have no row for {TARGET_YEAR}")
    target = target_rows.iloc[0]

    # The per-capita index divides by an ANNUAL AVERAGE population, so the comparison
    # is against demo_gind's average (11,405,627) rather than INE's 31 December figure
    # (11,424,031). They describe the same revision at different points in the year;
    # mixing them would label two different numbers "the revised population".
    resident_population = _value(panel, "resident_population", source="Eurostat demo_gind")
    accounts_population = _value(panel, "na_population", source="Eurostat nama_10_pe")
    actual_gdp = _value(pps_inputs, "gdp_meur", source="Eurostat nama_10_gdp") * 1e6
    ppp = derive_purchasing_power_parity(
        gdp_pc_eur=float(target["gdp_pc_eur"]), gdp_pc_pps=float(target["gdp_pc_pps"])
    )
    reference_gdp_pc_pps = float(target["eu_gdp_pc_pps"])

    # --- Step 1: which population is inside the published index? ---------------
    vintages = compare_population_bases(
        pps_inputs, revised_anchor=resident_population, anchor_year=TARGET_YEAR
    )
    verdict = assess_correction_claim(
        pps_inputs,
        year=TARGET_YEAR,
        quoted_index=anchor("preliminary_gdp_pc_pps_index"),
        revised_population=resident_population,
    )
    # Both frames are on the national-accounts basis, so the ratio is meaningful.
    employment_check = employment_absorbed_revision(
        na_employment=na_employment, na_population=na_population, years=(2019, TARGET_YEAR)
    )

    # --- Step 2: the Bayesian model -------------------------------------------
    # GDP growth is regressed on the national-accounts population, the series GDP
    # is vintage-consistent with. The resident population is modelled separately
    # because it is the denominator the living-standards question asks about.
    gdp_result = estimate_gdp_posterior_samples(
        panel,
        target_year=TARGET_YEAR,
        population_samples=np.full(n_samples, accounts_population),
        population_col="na_population",
        known_real_gdp_growth=anchor("real_gdp_growth"),
        estimation_start_year=ESTIMATION_START_YEAR,
        n_samples=n_samples,
    )
    population_result = estimate_population_posterior_samples(
        panel,
        target_year=TARGET_YEAR,
        population_col="resident_population",
        observed_population=resident_population,
        n_samples=n_samples,
    )
    gdp_pc_samples = gdp_result.gdp_samples / population_result.samples

    # --- Step 3: validation ----------------------------------------------------
    backtest = backtest_gdp_model(
        panel,
        holdout_years=list(range(2010, TARGET_YEAR)),
        population_col="na_population",
        estimation_start_year=ESTIMATION_START_YEAR,
        n_samples=max(4_000, n_samples // 5),
    )
    gdp_score = score_against_outturn(gdp_result.gdp_samples, actual_gdp)

    # --- Step 4: the capture question -----------------------------------------
    uncounted_residents = resident_population - accounts_population
    employed_target_year = _value(
        na_employment, "na_employment", source="Eurostat nama_10_a10_e"
    )
    output_per_employed = actual_gdp / employed_target_year
    migrant_rate = float(
        migrant_employment.loc[migrant_employment["year"] >= 2022, "foreign_born"].mean()
    ) / 100.0
    activity_rate = anchor("migrant_activity_rate")

    missing_workers = implied_missing_workers(
        extra_residents=uncounted_residents,
        working_age_share=anchor("migrant_working_age_share"),
        activity_rate=activity_rate,
        employment_rate=migrant_rate / activity_rate,
    )
    capture_curve = capture_sensitivity_curve(
        capture_values=np.linspace(0.0, 1.0, 21),
        extra_residents=uncounted_residents,
        baseline_output_per_employed=output_per_employed,
        working_age_share=anchor("migrant_working_age_share"),
        participation_rate=activity_rate,
        employment_rate=migrant_rate / activity_rate,
        productivity_relative=anchor("migrant_relative_productivity"),
        measured_gdp_samples=np.full(n_samples, actual_gdp),
        population_samples=population_result.samples,
        purchasing_power_parity=ppp,
        reference_gdp_pc_pps=reference_gdp_pc_pps,
        n_samples=n_samples,
    )

    full_capture_index = pps_index_samples(
        gdp_samples=np.full(n_samples, actual_gdp),
        population_samples=population_result.samples,
        purchasing_power_parity=ppp,
        reference_gdp_pc_pps=reference_gdp_pc_pps,
    )

    summary = pd.DataFrame(
        {
            "resident_population_2025": summarize_samples(population_result.samples),
            "gdp_2025_current_eur": summarize_samples(gdp_result.gdp_samples),
            "gdp_pc_2025_current_eur": summarize_samples(gdp_pc_samples),
            "pps_index_full_capture": summarize_samples(full_capture_index),
        }
    ).T

    result = AnalysisResult(
        panel=panel,
        pps_inputs=pps_inputs,
        na_employment=na_employment,
        migrant_employment=migrant_employment,
        vintages=vintages,
        verdict=verdict,
        employment_check=employment_check,
        population_result=population_result,
        gdp_result=gdp_result,
        gdp_pc_samples=gdp_pc_samples,
        backtest=backtest,
        capture_curve=capture_curve,
        gdp_score=gdp_score,
        summary=summary,
        uncounted_residents=uncounted_residents,
        missing_workers=missing_workers,
        purchasing_power_parity=ppp,
        reference_gdp_pc_pps=reference_gdp_pc_pps,
        actual_gdp=actual_gdp,
    )

    if write_outputs:
        write_analysis_outputs(result, project_root / "outputs")
    return result


def write_analysis_outputs(result: AnalysisResult, outputs_dir: str | Path) -> list[Path]:
    """Write the analysis tables to ``outputs_dir`` and return the paths written."""

    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    verdict = result.verdict
    tables = {
        "population_bases.csv": result.vintages.dropna(
            subset=["na_population", "demographic_population"], how="all"
        ),
        "correction_verdict.csv": pd.DataFrame(
            [
                {
                    "year": verdict.year,
                    "quoted_index": verdict.quoted_index,
                    "index_population": verdict.index_population,
                    "revised_population": verdict.revised_population,
                    "ratio": verdict.ratio,
                    "denominator_only_index": verdict.denominator_only_index,
                    "correction_is_owed": verdict.correction_is_owed,
                    "uncounted_residents": result.uncounted_residents,
                    "implied_missing_workers": result.missing_workers,
                }
            ]
        ),
        "employment_absorption_check.csv": result.employment_check,
        "gdp_backtest.csv": result.backtest,
        "capture_sensitivity_curve.csv": result.capture_curve,
        "posterior_summary.csv": result.summary,
        "outturn_score.csv": pd.DataFrame(
            [
                {
                    "quantity": "nominal_gdp_2025_eur",
                    "actual": result.gdp_score.actual,
                    "median": result.gdp_score.median,
                    "pct_error": result.gdp_score.pct_error,
                    "q_low": result.gdp_score.q_low,
                    "q_high": result.gdp_score.q_high,
                    "in_interval": result.gdp_score.in_interval,
                    "pit": result.gdp_score.pit,
                }
            ]
        ),
    }

    written: list[Path] = []
    for name, frame in tables.items():
        path = outputs_dir / name
        frame.to_csv(path, index=name == "posterior_summary.csv")
        written.append(path)
    return written
