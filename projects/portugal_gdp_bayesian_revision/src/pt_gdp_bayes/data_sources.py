"""Data loading utilities for the Portugal GDP Bayesian revision project.

Every remote series is fetched through :func:`cached_frame`, which writes the
parsed result to ``data/raw/`` on first use and reads from that copy afterwards.
The project therefore runs offline once the cache is populated, and a reviewer
who clones the repository reproduces the published numbers without holding API
credentials or waiting on three separate statistical agencies.

Pass ``refresh=True`` to bypass the cache and re-download.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd
import requests

DEFAULT_USER_AGENT = "pt-gdp-bayes/1.0"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_RETRIES = 3


@dataclass(frozen=True)
class WorldBankIndicator:
    """World Bank indicator metadata.

    Attributes
    ----------
    column_name:
        Name to use in the returned dataframe.
    code:
        World Bank indicator code.
    """

    column_name: str
    code: str


class DataDownloadError(RuntimeError):
    """Raised when an external data source cannot be downloaded."""


def default_cache_dir(project_root: str | Path) -> Path:
    """Return the conventional raw-data cache directory for the project."""

    return Path(project_root) / "data" / "raw"


def http_get(
    url: str,
    *,
    params: Mapping[str, str] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    backoff_seconds: float = 2.0,
) -> requests.Response:
    """GET a URL, retrying on transport errors with linear backoff.

    The statistical APIs used here (World Bank, OECD SDMX, Eurostat) are all
    prone to occasional slow responses, so a single timeout is not evidence that
    the endpoint is down.
    """

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=dict(params) if params else None,
                timeout=timeout_seconds,
                headers={"User-Agent": DEFAULT_USER_AGENT},
            )
            response.raise_for_status()
            return response
        except Exception as exc:  # pragma: no cover - depends on network
            last_error = exc
            if attempt < retries - 1:
                time.sleep(backoff_seconds * (attempt + 1))
    raise DataDownloadError(f"Could not download {url}: {last_error}") from last_error


def cached_frame(
    loader: Callable[[], pd.DataFrame],
    *,
    cache_dir: str | Path | None,
    key: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return ``loader()`` output, memoised on disk as ``{cache_dir}/{key}.csv``.

    Parameters
    ----------
    loader:
        Zero-argument callable performing the actual download and parsing.
    cache_dir:
        Directory for cached CSVs. ``None`` disables caching entirely.
    key:
        Cache file stem. Should encode every parameter that changes the result.
    refresh:
        Re-download and overwrite the cached copy.

    Returns
    -------
    pandas.DataFrame
        Either the cached frame or a freshly downloaded one.
    """

    if cache_dir is None:
        return loader()

    path = Path(cache_dir) / f"{key}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)

    frame = loader()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def load_known_observations(path: str | Path) -> pd.DataFrame:
    """Load manually anchored observations used by the notebook.

    Parameters
    ----------
    path:
        Path to ``known_observations.csv``.

    Returns
    -------
    pandas.DataFrame
        Normalized table with variable, country, year, value and source fields.
    """

    df = pd.read_csv(path)
    required = {"variable", "country", "year", "value", "unit", "source"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in known observations: {missing}")
    df["year"] = df["year"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="raise")
    return df


def get_known_value(observations: pd.DataFrame, variable: str, year: int) -> float:
    """Extract a single numeric value from the known-observations table."""

    rows = observations[(observations["variable"] == variable) & (observations["year"] == year)]
    if rows.empty:
        raise KeyError(f"No known observation found for variable={variable!r}, year={year}")
    if len(rows) > 1:
        raise ValueError(f"Multiple observations found for variable={variable!r}, year={year}")
    return float(rows.iloc[0]["value"])


def fetch_world_bank_indicator(
    country: str,
    indicator: WorldBankIndicator,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch one World Bank indicator as a tidy dataframe.

    Parameters
    ----------
    country:
        ISO-3 or World Bank country code, e.g. ``"PRT"``.
    indicator:
        Indicator metadata.
    start_year, end_year:
        Optional inclusive year bounds.
    cache_dir:
        Directory for the on-disk cache. ``None`` disables caching.
    refresh:
        Force a re-download.

    Returns
    -------
    pandas.DataFrame
        Dataframe with columns ``year`` and ``indicator.column_name``.
    """

    def _load() -> pd.DataFrame:
        url = (
            f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator.code}"
            "?format=json&per_page=20000"
        )
        payload = http_get(url).json()
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            raise DataDownloadError(f"Unexpected World Bank response for {indicator.code}")

        records = [
            {"year": int(item["date"]), indicator.column_name: float(item["value"])}
            for item in payload[1]
            if item.get("value") is not None
        ]
        return pd.DataFrame.from_records(records).sort_values("year").reset_index(drop=True)

    frame = cached_frame(
        _load,
        cache_dir=cache_dir,
        key=f"worldbank_{country}_{indicator.code}",
        refresh=refresh,
    )
    frame = frame.rename(columns={frame.columns[1]: indicator.column_name})
    if start_year is not None:
        frame = frame[frame["year"] >= start_year]
    if end_year is not None:
        frame = frame[frame["year"] <= end_year]
    return frame.sort_values("year").reset_index(drop=True)


def fetch_world_bank_panel(
    country: str,
    indicators: Mapping[str, str],
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch and merge several World Bank indicators.

    Parameters
    ----------
    country:
        Country code, e.g. ``"PRT"``.
    indicators:
        Mapping from output column name to World Bank indicator code.
    start_year, end_year:
        Optional inclusive year bounds.
    cache_dir:
        Directory for the on-disk cache. ``None`` disables caching.
    refresh:
        Force a re-download of every indicator.

    Returns
    -------
    pandas.DataFrame
        One row per year and one column per requested indicator.
    """

    frames: list[pd.DataFrame] = []
    for column_name, code in indicators.items():
        frames.append(
            fetch_world_bank_indicator(
                country,
                WorldBankIndicator(column_name=column_name, code=code),
                start_year=start_year,
                end_year=end_year,
                cache_dir=cache_dir,
                refresh=refresh,
            )
        )

    if not frames:
        raise ValueError("At least one indicator is required")

    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on="year", how="outer")

    return panel.sort_values("year").reset_index(drop=True)


def fetch_oecd_employment_rate_by_birthplace(
    country: str = "PRT",
    *,
    start_year: int = 2005,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Employment rate (15-64, %) by place of birth from the OECD migration database.

    The SDMX key is filtered to ``country`` rather than downloading every
    reporting area, and only the ``Total`` education level and both-sexes series
    are kept. The rate is ``employed / population aged 15-64`` for each group,
    which is the base the migrant working-age share in the model is paired with.

    Returns
    -------
    pandas.DataFrame
        Columns ``year``, ``foreign_born``, ``native_born`` (percentages).
    """

    def _load() -> pd.DataFrame:
        url = (
            "https://sdmx.oecd.org/public/rest/data/OECD.ELS.IMD,DSD_MIG@DF_MIG_EMP_EDU,/"
            f"{country}.......?startPeriod={start_year}"
            "&format=csvfilewithlabels&dimensionAtObservation=AllDimensions"
        )
        raw = pd.read_csv(io.StringIO(http_get(url).text))
        subset = raw[
            (raw["REF_AREA"] == country)
            & (raw["SEX"] == "_T")
            & (raw["Education level"] == "Total")
        ]
        if subset.empty:
            raise DataDownloadError(f"OECD migration data returned no rows for {country}")
        wide = subset.pivot_table(
            index="TIME_PERIOD", columns="Place of birth", values="OBS_VALUE"
        )
        missing = {"Foreign-born", "Native-born"}.difference(wide.columns)
        if missing:
            raise DataDownloadError(f"OECD migration data missing place-of-birth groups: {missing}")
        return (
            wide.rename(columns={"Foreign-born": "foreign_born", "Native-born": "native_born"})
            .rename_axis("year")
            .reset_index()[["year", "foreign_born", "native_born"]]
            .sort_values("year")
            .reset_index(drop=True)
        )

    frame = cached_frame(
        _load,
        cache_dir=cache_dir,
        key=f"oecd_employment_by_birthplace_{country}_{start_year}",
        refresh=refresh,
    )
    frame["year"] = frame["year"].astype(int)
    return frame


def fetch_eurostat_series(
    dataset: str,
    *,
    value_name: str,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    **filters: str,
) -> pd.DataFrame:
    """Fetch one Eurostat series as a tidy ``year``/``value_name`` dataframe.

    Parameters
    ----------
    dataset:
        Eurostat dataset code, e.g. ``"nama_10_gdp"``.
    value_name:
        Name for the value column in the returned frame.
    cache_dir, refresh:
        On-disk cache controls, as in :func:`cached_frame`.
    **filters:
        Dimension filters passed straight to the dissemination API, e.g.
        ``geo="PT", na_item="B1GQ", unit="CP_MEUR"``.

    Returns
    -------
    pandas.DataFrame
        Columns ``year`` and ``value_name``.
    """

    def _load() -> pd.DataFrame:
        url = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}"
        payload = http_get(url, params={**filters, "format": "JSON"}).json()
        if "value" not in payload:
            raise DataDownloadError(
                f"Eurostat {dataset} returned no values for {filters}: {payload.get('error', payload)}"
            )
        time_index = payload["dimension"]["time"]["category"]["index"]
        position_to_year = {position: int(year) for year, position in time_index.items()}
        records = [
            {"year": position_to_year[int(position)], value_name: float(value)}
            for position, value in payload["value"].items()
            if int(position) in position_to_year
        ]
        if not records:
            raise DataDownloadError(f"Eurostat {dataset} returned an empty series for {filters}")
        return pd.DataFrame.from_records(records).sort_values("year").reset_index(drop=True)

    # The value column name is part of the cache key: two calls with identical
    # filters but different value_name are different frames, and sharing a cache
    # file between them silently returns the wrong column.
    slug = "_".join(str(filters[k]) for k in sorted(filters))
    frame = cached_frame(
        _load,
        cache_dir=cache_dir,
        key=f"eurostat_{dataset}_{slug}_{value_name}",
        refresh=refresh,
    )
    frame = frame.rename(columns={frame.columns[1]: value_name})
    frame["year"] = frame["year"].astype(int)
    return frame


def fetch_eurostat_national_accounts_population(
    geo: str = "PT",
    *,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Total population, national concept, from Eurostat ``nama_10_pe``.

    This is the population **inside the national accounts**, and it is the
    denominator of every published per-capita national-accounts aggregate,
    including GDP per capita in PPS.

    It is *not* the same series as the resident population in ``demo_gind``.
    For Portugal in 2025 the national-accounts population is 10.80 million while
    the demographic estimate is 11.41 million — a 5.6% divergence, because the
    national accounts have not yet been rebenchmarked onto the revised
    population. Using the wrong one of these two is the single easiest way to
    get this analysis wrong in either direction.

    Returns
    -------
    pandas.DataFrame
        Columns ``year`` and ``na_population`` (persons).
    """

    frame = fetch_eurostat_series(
        "nama_10_pe", value_name="na_population_ths", cache_dir=cache_dir, refresh=refresh,
        geo=geo, na_item="POP_NC", unit="THS_PER",
    )
    frame["na_population"] = frame["na_population_ths"] * 1_000.0
    return frame[["year", "na_population"]]


def fetch_eurostat_pps_inputs(
    geo: str = "PT",
    *,
    reference: str = "EU27_2020",
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch every input behind the published GDP-per-capita-in-PPS index.

    Pulling the components rather than only the headline index is what makes the
    index *auditable*. Two of these columns matter more than the rest:

    ``gdp_pc_eur``
        Eurostat's **published** GDP per capita in euro. Dividing GDP by it
        recovers the population actually inside the index. Deriving that
        population any other way — for instance by assuming it equals the
        demographic estimate — assumes the very thing under test.
    ``na_population`` vs ``population_avg``
        The national-accounts population and the demographic population. They
        differ by 5.6% for Portugal in 2025, and only the first one is inside the
        published per-capita figures.

    Returns
    -------
    pandas.DataFrame
        Columns ``year``, ``gdp_meur``, ``gdp_pc_eur``, ``gdp_pc_pps``,
        ``eu_gdp_pc_pps``, ``published_index``, ``na_population``,
        ``population_avg``.
    """

    unit_pps = f"CP_PPS_{reference}_HAB"
    frames = [
        fetch_eurostat_series(
            "nama_10_gdp", value_name="gdp_meur", cache_dir=cache_dir, refresh=refresh,
            geo=geo, na_item="B1GQ", unit="CP_MEUR",
        ),
        fetch_eurostat_series(
            "nama_10_pc", value_name="gdp_pc_eur", cache_dir=cache_dir, refresh=refresh,
            geo=geo, na_item="B1GQ", unit="CP_EUR_HAB",
        ),
        fetch_eurostat_series(
            "nama_10_pc", value_name="gdp_pc_pps", cache_dir=cache_dir, refresh=refresh,
            geo=geo, na_item="B1GQ", unit=unit_pps,
        ),
        fetch_eurostat_series(
            "nama_10_pc", value_name="eu_gdp_pc_pps", cache_dir=cache_dir, refresh=refresh,
            geo=reference, na_item="B1GQ", unit=unit_pps,
        ),
        fetch_eurostat_series(
            "tec00114", value_name="published_index", cache_dir=cache_dir, refresh=refresh,
            geo=geo,
        ),
        fetch_eurostat_national_accounts_population(geo, cache_dir=cache_dir, refresh=refresh),
        fetch_eurostat_series(
            "demo_gind", value_name="population_avg", cache_dir=cache_dir, refresh=refresh,
            geo=geo, indic_de="AVG",
        ),
    ]

    merged: pd.DataFrame | None = None
    for frame in frames:
        merged = frame if merged is None else merged.merge(frame, on="year", how="outer")
    assert merged is not None
    return merged.sort_values("year").reset_index(drop=True)


def fetch_eurostat_population(
    geo: str = "PT",
    *,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Eurostat resident population on 1 January and annual average.

    This is the series the Commission's per-capita indicators divide by, and it
    is *not* the same vintage as the World Bank population series.

    Returns
    -------
    pandas.DataFrame
        Columns ``year``, ``population_jan1``, ``population_avg``.
    """

    jan = fetch_eurostat_series(
        "demo_gind", value_name="population_jan1", cache_dir=cache_dir, refresh=refresh,
        geo=geo, indic_de="JAN",
    )
    avg = fetch_eurostat_series(
        "demo_gind", value_name="population_avg", cache_dir=cache_dir, refresh=refresh,
        geo=geo, indic_de="AVG",
    )
    return jan.merge(avg, on="year", how="outer").sort_values("year").reset_index(drop=True)


def fetch_eurostat_employment_rate(
    geo: str = "PT",
    *,
    age: str = "Y15-64",
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Published Labour Force Survey employment rate, % of population aged ``age``.

    Both the numerator and the denominator of this rate come from the LFS, which
    is grossed up on its own population benchmark. The rate is therefore
    internally consistent, but it describes the *counted* population only — which
    is exactly why it cannot, on its own, reveal how many uncounted residents are
    working. Comparing it against other countries separates the common European
    employment trend from anything specific to Portugal.

    Returns
    -------
    pandas.DataFrame
        Columns ``year`` and ``employment_rate_pct``.
    """

    return fetch_eurostat_series(
        "lfsi_emp_a", value_name="employment_rate_pct", cache_dir=cache_dir, refresh=refresh,
        geo=geo, sex="T", age=age, unit="PC_POP", indic_em="EMP_LFS",
    )


def fetch_eurostat_national_accounts_employment(
    geo: str = "PT",
    *,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Total employment, domestic concept, from Eurostat ``nama_10_a10_e``.

    This is the employment measure compiled *inside* the national accounts and
    therefore consistent with the GDP figure. If it tracks the revised
    population, the workers behind the revision are already inside measured GDP.

    Returns
    -------
    pandas.DataFrame
        Columns ``year`` and ``na_employment`` (persons).
    """

    frame = fetch_eurostat_series(
        "nama_10_a10_e", value_name="na_employment_ths", cache_dir=cache_dir, refresh=refresh,
        geo=geo, na_item="EMP_DC", unit="THS_PER", nace_r2="TOTAL",
    )
    frame["na_employment"] = frame["na_employment_ths"] * 1_000.0
    return frame[["year", "na_employment"]]


def patch_manual_observation(
    df: pd.DataFrame,
    *,
    year: int,
    column: str,
    value: float,
    source_column: str = "manual_patch_notes",
    note: str | None = None,
) -> pd.DataFrame:
    """Patch or append a manually anchored observation.

    This is used for the revised 2025 INE population estimate, because many
    international APIs can lag official national revisions.
    """

    out = df.copy()
    if "year" not in out.columns:
        raise ValueError("Input dataframe must contain a 'year' column")
    if column not in out.columns:
        out[column] = pd.NA
    if source_column not in out.columns:
        out[source_column] = ""

    mask = out["year"] == year
    if mask.any():
        out.loc[mask, column] = float(value)
        out.loc[mask, source_column] = note or "manual observation patch"
    else:
        new_row = {col: pd.NA for col in out.columns}
        new_row["year"] = year
        new_row[column] = float(value)
        new_row[source_column] = note or "manual observation patch"
        out = pd.concat([out, pd.DataFrame([new_row])], ignore_index=True)

    return out.sort_values("year").reset_index(drop=True)
