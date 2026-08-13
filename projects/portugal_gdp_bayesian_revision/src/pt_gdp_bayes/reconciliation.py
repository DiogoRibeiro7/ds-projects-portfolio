"""Which population is inside the published GDP-per-capita index?

Portugal has two official population figures for 2025 and they differ by 5.6%:

======================================  ============  ==============================
series                                  2025          what it is
======================================  ============  ==============================
Eurostat ``demo_gind`` (AVG)            11,405,627    demographic estimate, revised
Eurostat ``nama_10_pe`` (POP_NC)        10,804,160    national-accounts population
======================================  ============  ==============================

Every published per-capita national-accounts aggregate — including GDP per
capita in PPS, and therefore the EU-comparison index — divides by the **second**
one. The national accounts have not been rebenchmarked onto the revised
population, so the index is still computed on the pre-revision basis while the
demographic statistics have moved on.

That makes a denominator correction genuinely owed. It also opens a second
question the correction alone does not answer: if the accounts are built on a
population that excludes ~600,000 residents, is those residents' **output** also
missing from the GDP numerator? See
:func:`pt_gdp_bayes.model.simulate_unmeasured_output_samples`.

The functions here recover the population inside the index from published data
only, without assuming which series it is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorrectionVerdict:
    """Whether a denominator correction to a quoted index is still owed.

    Attributes
    ----------
    year:
        Year under test.
    quoted_index:
        Index value being examined.
    index_population:
        Population actually inside the published index, recovered as
        ``GDP / published GDP per capita``.
    revised_population:
        Revised resident population the correction would substitute in.
    ratio:
        ``index_population / revised_population``. Near 1.0 means the index is
        already on the revised basis and no correction is owed.
    denominator_only_index:
        The index recomputed on ``revised_population``, holding GDP fixed. This
        is a *bound*, not an answer: it assumes measured GDP already captures
        everything the extra residents produce.
    correction_is_owed:
        ``True`` when the index is on a materially different population basis.
    """

    year: int
    quoted_index: float
    index_population: float
    revised_population: float
    ratio: float
    denominator_only_index: float
    correction_is_owed: bool

    def summary(self) -> str:
        """Human-readable verdict, suitable for printing in a notebook."""

        if not self.correction_is_owed:
            return (
                f"{self.year}: the quoted index {self.quoted_index:.2f} is already computed on a "
                f"population of {self.index_population:,.0f}, i.e. {self.ratio:.4f} x the revised "
                f"{self.revised_population:,.0f}. No denominator correction is owed; applying one "
                f"would double-count a revision the index has already absorbed."
            )
        return (
            f"{self.year}: the quoted index {self.quoted_index:.2f} is computed on a population of "
            f"{self.index_population:,.0f}, which is {(1 - self.ratio) * 100:.1f}% BELOW the revised "
            f"{self.revised_population:,.0f}. The national accounts have not been rebenchmarked. "
            f"Holding GDP fixed, the index on the revised population is "
            f"{self.denominator_only_index:.2f} — but that holds only if measured GDP already "
            f"captures the extra residents' output, which is a separate question."
        )


def derive_purchasing_power_parity(*, gdp_pc_eur: float, gdp_pc_pps: float) -> float:
    """PPP conversion factor from two published per-capita figures.

    ``ppp = gdp_pc_eur / gdp_pc_pps``, in national currency per PPS.

    Both inputs are published, so this involves no population at all. Deriving
    the PPP as ``gdp / population / gdp_pc_pps`` instead would silently bake in
    an assumption about *which* population the index used — which is precisely
    the quantity under test.
    """

    for name, value in {"gdp_pc_eur": gdp_pc_eur, "gdp_pc_pps": gdp_pc_pps}.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    return gdp_pc_eur / gdp_pc_pps


def population_inside_index(*, gdp: float, gdp_pc_eur: float) -> float:
    """Recover the population a published per-capita figure was divided by.

    Parameters
    ----------
    gdp:
        Nominal GDP in national currency.
    gdp_pc_eur:
        Published GDP per capita in the same currency.

    Returns
    -------
    float
        ``gdp / gdp_pc_eur``.
    """

    for name, value in {"gdp": gdp, "gdp_pc_eur": gdp_pc_eur}.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    return gdp / gdp_pc_eur


def assess_correction_claim(
    pps_inputs: pd.DataFrame,
    *,
    year: int,
    quoted_index: float,
    revised_population: float,
    tolerance: float = 0.01,
) -> CorrectionVerdict:
    """Test whether a denominator correction to ``quoted_index`` is owed.

    Recovers the population inside the published index from ``gdp_meur`` and
    ``gdp_pc_eur`` — both published — and compares it with
    ``revised_population``.

    Parameters
    ----------
    pps_inputs:
        Frame from :func:`~pt_gdp_bayes.data_sources.fetch_eurostat_pps_inputs`,
        containing ``year``, ``gdp_meur`` and ``gdp_pc_eur``.
    year:
        Year under test.
    quoted_index:
        Index value the correction would be applied to.
    revised_population:
        Revised resident population.
    tolerance:
        Relative distance from 1.0 within which the two bases count as the same.

    Returns
    -------
    CorrectionVerdict
        Structured verdict, with :meth:`CorrectionVerdict.summary` for display.
    """

    required = {"year", "gdp_meur", "gdp_pc_eur"}
    missing = required.difference(pps_inputs.columns)
    if missing:
        raise ValueError(f"pps_inputs is missing columns: {sorted(missing)}")
    if not np.isfinite(revised_population) or revised_population <= 0:
        raise ValueError("revised_population must be positive and finite")

    rows = pps_inputs.loc[pps_inputs["year"] == year]
    if rows.empty:
        raise KeyError(f"No PPS inputs available for {year}")
    row = rows.iloc[0]

    index_population = population_inside_index(
        gdp=float(row["gdp_meur"]) * 1e6, gdp_pc_eur=float(row["gdp_pc_eur"])
    )
    ratio = index_population / revised_population

    return CorrectionVerdict(
        year=int(year),
        quoted_index=float(quoted_index),
        index_population=float(index_population),
        revised_population=float(revised_population),
        ratio=float(ratio),
        denominator_only_index=float(quoted_index * ratio),
        correction_is_owed=bool(abs(ratio - 1.0) > tolerance),
    )


def compare_population_bases(
    pps_inputs: pd.DataFrame,
    *,
    revised_anchor: float | None = None,
    anchor_year: int | None = None,
) -> pd.DataFrame:
    """Put the national-accounts and demographic population bases side by side.

    Also recovers the population implied by the published per-capita figure, so
    that the claim "the index uses the national-accounts population" is shown
    rather than asserted.

    Returns
    -------
    pandas.DataFrame
        Columns ``year``, ``na_population``, ``demographic_population``,
        ``implied_by_published_index``, ``gap``, ``gap_pct`` and, when an anchor
        is supplied, ``official_revised``.
    """

    required = {"year", "gdp_meur", "gdp_pc_eur", "na_population", "population_avg"}
    missing = required.difference(pps_inputs.columns)
    if missing:
        raise ValueError(f"pps_inputs is missing columns: {sorted(missing)}")

    out = pps_inputs[["year", "na_population", "population_avg"]].rename(
        columns={"population_avg": "demographic_population"}
    ).copy()
    out["implied_by_published_index"] = (
        pps_inputs["gdp_meur"] * 1e6 / pps_inputs["gdp_pc_eur"]
    )
    out["gap"] = out["demographic_population"] - out["na_population"]
    out["gap_pct"] = out["gap"] / out["na_population"] * 100.0

    if revised_anchor is not None:
        if anchor_year is None:
            raise ValueError("anchor_year is required when revised_anchor is given")
        out["official_revised"] = np.where(out["year"] == anchor_year, revised_anchor, np.nan)

    return out.sort_values("year").reset_index(drop=True)


def employment_absorbed_revision(
    *,
    na_employment: pd.DataFrame,
    na_population: pd.DataFrame,
    years: tuple[int, int],
    employment_col: str = "na_employment",
    population_col: str = "na_population",
) -> pd.DataFrame:
    """Employment relative to the population base the accounts actually use.

    Both inputs must be on the **national-accounts** basis. Comparing
    national-accounts employment against the *demographic* population mixes two
    bases that differ by 5.6% and produces a ratio whose movements are an
    artefact of the mismatch rather than a fact about the labour market.

    A stable ratio on the national-accounts basis says the accounts are
    internally consistent — which is evidence that they have *not* absorbed the
    extra residents on either side, and therefore that those residents' output
    may be missing from GDP as well as their headcount from the denominator.

    Returns
    -------
    pandas.DataFrame
        Per-year employment, population, their ratio and growth rates.
    """

    merged = (
        na_employment[["year", employment_col]]
        .merge(na_population[["year", population_col]], on="year", how="inner")
        .sort_values("year")
        .reset_index(drop=True)
    )
    merged = merged[merged["year"].between(*years)].copy()
    merged["employment_population_ratio"] = merged[employment_col] / merged[population_col]
    merged["employment_growth_pct"] = merged[employment_col].pct_change() * 100.0
    merged["population_growth_pct"] = merged[population_col].pct_change() * 100.0
    return merged.reset_index(drop=True)


def employment_rate_excess_over_peers(
    rates: pd.DataFrame,
    *,
    country: str,
    peers: list[str],
    base_year: int,
    target_year: int,
) -> pd.DataFrame:
    """Split a country's employment-rate rise into common trend and excess.

    A tempting but wrong way to estimate statistical capture is to treat *all* of
    the rise in employment relative to a stale population base as evidence that
    the accounts are absorbing uncounted workers. Employment rates rose across
    Europe over the same period for reasons that have nothing to do with
    Portuguese population benchmarks, and that common trend has to come out
    first.

    Parameters
    ----------
    rates:
        Frame with a ``year`` column and one column per geography, holding
        published employment rates in percent.
    country, peers:
        Column names for the country under test and its comparators.
    base_year, target_year:
        Endpoints of the comparison.

    Returns
    -------
    pandas.DataFrame
        One row per peer with the country's change, the peer's change, and the
        excess in percentage points.
    """

    missing = {country, *peers}.difference(rates.columns)
    if missing:
        raise ValueError(f"rates is missing columns: {sorted(missing)}")

    indexed = rates.set_index("year")
    for year in (base_year, target_year):
        if year not in indexed.index:
            raise KeyError(f"No employment-rate data for {year}")

    country_change = float(indexed.loc[target_year, country] - indexed.loc[base_year, country])
    rows = []
    for peer in peers:
        peer_change = float(indexed.loc[target_year, peer] - indexed.loc[base_year, peer])
        rows.append(
            {
                "peer": peer,
                "country_change_pp": country_change,
                "peer_change_pp": peer_change,
                "excess_pp": country_change - peer_change,
            }
        )
    return pd.DataFrame(rows)


def implied_missing_workers(
    *,
    extra_residents: float,
    working_age_share: float,
    activity_rate: float,
    employment_rate: float,
) -> float:
    """Workers implied by residents the national accounts have not counted.

    A headcount, not an output estimate: how many of the uncounted residents
    would be working, at the observed migrant participation and employment
    rates. Whether their output is nonetheless already inside GDP — because
    production-side sources capture a firm's turnover regardless of whether the
    demographic count knew about its staff — is the separate capture question.
    """

    for name, value in {
        "working_age_share": working_age_share,
        "activity_rate": activity_rate,
        "employment_rate": employment_rate,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if extra_residents < 0:
        raise ValueError("extra_residents must be non-negative")

    return extra_residents * working_age_share * activity_rate * employment_rate
