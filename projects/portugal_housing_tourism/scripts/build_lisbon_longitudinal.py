"""Build the first observed Lisbon housing-tourism longitudinal dataset."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from housing_tourism.data import (  # noqa: E402
    canonicalise_ine_measure,
    flatten_ine_payload,
    infer_year,
)

INE_MIRROR_ENDPOINT = "https://gateway.pipeworx.io/ine-pt/mcp"
VALUE_NAMES = {
    "rent": "rent_eur_m2",
    "income": "income_eur",
    "housing_stock": "housing_stock",
    "population": "resident_population",
    "overnight_stays": "overnight_stays",
}


def _load_sources() -> dict[str, object]:
    with (ROOT / "config" / "sources.toml").open("rb") as handle:
        return tomllib.load(handle)


def _mirror_call(tool_name: str, arguments: dict[str, object]) -> Any:
    """Call a public transport proxy for an INE endpoint and decode its JSON payload."""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    response = requests.post(INE_MIRROR_ENDPOINT, json=request, timeout=30.0)
    response.raise_for_status()
    body = response.json()
    if "error" in body:
        raise RuntimeError(f"INE proxy error: {body['error']}")
    try:
        text = body["result"]["content"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Unexpected INE proxy response shape.") from exc
    return json.loads(text)


def _metadata(indicator: str) -> dict[str, Any]:
    payload = _mirror_call("indicator_meta", {"varcd": indicator, "lang": "PT"})
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise ValueError(f"Unexpected metadata payload for INE indicator {indicator}.")
    return payload[0]


def _categories(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten INE's category groups into one list of dimension-value records."""
    try:
        groups = metadata["Dimensoes"]["Categoria_Dim"]
    except (KeyError, TypeError) as exc:
        raise ValueError("INE metadata does not contain dimension categories.") from exc
    if not isinstance(groups, list):
        raise ValueError("INE dimension categories must be a list.")

    records: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        for values in group.values():
            if isinstance(values, list):
                records.extend(value for value in values if isinstance(value, dict))
    return records


def _dimension_numbers(
    metadata: dict[str, Any],
    categories: list[dict[str, Any]],
) -> tuple[str, str, list[str]]:
    """Identify time/geography dimensions from labels and category semantics."""
    descriptions = metadata.get("Dimensoes", {}).get("Descricao_Dim", [])
    if not isinstance(descriptions, list):
        raise ValueError("INE metadata dimension descriptions must be a list.")

    all_dims = [
        str(item["dim_num"])
        for item in descriptions
        if isinstance(item, dict) and item.get("dim_num") is not None
    ]
    if not all_dims:
        raise ValueError("INE metadata contains no dimensions.")

    time_candidates = {
        str(item["dim_num"])
        for item in descriptions
        if isinstance(item, dict)
        and item.get("dim_num") is not None
        and ("período" in str(item.get("abrv", "")).casefold() or "periodo" in str(item.get("abrv", "")).casefold())
    }
    if len(time_candidates) != 1:
        time_candidates = {
            str(entry.get("dim_num"))
            for entry in categories
            if str(entry.get("categ_dsg", "")).strip().isdigit()
            and len(str(entry.get("categ_dsg", "")).strip()) == 4
        }
    if len(time_candidates) != 1:
        raise ValueError(f"Could not identify a unique INE time dimension: {sorted(time_candidates)}")
    time_dim = next(iter(time_candidates))

    geography_candidates = {
        str(entry.get("dim_num"))
        for entry in categories
        if str(entry.get("categ_dsg", "")).strip().casefold() == "lisboa"
        and str(entry.get("categ_nivel")) == "5"
    }
    if len(geography_candidates) != 1:
        raise ValueError(
            "Could not identify a unique municipality geography dimension from Lisboa "
            f"level-5 metadata: {sorted(geography_candidates)}"
        )
    geography_dim = next(iter(geography_candidates))
    other_dims = [dim for dim in all_dims if dim not in {time_dim, geography_dim}]
    return time_dim, geography_dim, other_dims


def _lisboa_code(categories: list[dict[str, Any]], geography_dim: str) -> str:
    matches = {
        str(entry["categ_cod"])
        for entry in categories
        if str(entry.get("dim_num")) == geography_dim
        and str(entry.get("categ_dsg", "")).strip().casefold() == "lisboa"
        and str(entry.get("categ_nivel")) == "5"
    }
    if len(matches) != 1:
        raise ValueError(f"Expected one Lisboa municipality code, found {sorted(matches)}")
    return next(iter(matches))


def _total_codes(
    categories: list[dict[str, Any]],
    other_dims: list[str],
) -> dict[str, str]:
    """Select one explicit aggregate category for every non-time/geography dimension."""
    selected: dict[str, str] = {}
    for dim in other_dims:
        records = [record for record in categories if str(record.get("dim_num")) == dim]
        totals = {
            str(record["categ_cod"])
            for record in records
            if str(record.get("categ_dsg", "")).strip().casefold().startswith("total")
        }
        if len(totals) == 1:
            selected[dim] = next(iter(totals))
        elif len(records) == 1:
            selected[dim] = str(records[0]["categ_cod"])
        else:
            labels = sorted({str(record.get("categ_dsg", "")) for record in records})[:15]
            raise ValueError(
                f"Cannot select an unambiguous aggregate for INE dimension {dim}. "
                f"Observed labels include {labels!r}."
            )
    return selected


def _period_codes(categories: list[dict[str, Any]], time_dim: str) -> list[tuple[int, str]]:
    periods = {
        (int(label), str(record["categ_cod"]))
        for record in categories
        if str(record.get("dim_num")) == time_dim
        if (label := str(record.get("categ_dsg", "")).strip()).isdigit()
        if len(label) == 4 and int(label) >= 2017
    }
    result = sorted(periods)
    if not result:
        raise ValueError("INE metadata contains no annual periods from 2017 onward.")
    return result


def _fetch_lisbon_indicator(
    measure: str,
    indicator: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fetch every advertised annual period for Lisbon municipality."""
    metadata = _metadata(indicator)
    categories = _categories(metadata)
    time_dim, geography_dim, other_dims = _dimension_numbers(metadata, categories)
    lisboa = _lisboa_code(categories, geography_dim)
    totals = _total_codes(categories, other_dims)

    observations: list[pd.DataFrame] = []
    extraction_dates: list[str] = []
    for year, period_code in _period_codes(categories, time_dim):
        dimensions = {
            f"Dim{time_dim}": period_code,
            f"Dim{geography_dim}": lisboa,
            **{f"Dim{dim}": code for dim, code in totals.items()},
        }
        payload = _mirror_call(
            "get_indicator",
            {"varcd": indicator, "dims": dimensions, "lang": "PT"},
        )
        flat = flatten_ine_payload(payload)
        flat["year"] = infer_year(flat["period"])
        canonical = canonicalise_ine_measure(
            flat,
            value_name=VALUE_NAMES[measure],
            minimum_year=2017,
        )
        canonical = canonical.loc[canonical["year"] == year]
        if canonical.empty:
            print(f"{measure}: no observation returned for {year}; leaving it missing")
            continue
        if canonical["geo_code"].nunique() != 1 or str(canonical["geo_code"].iloc[0]) != lisboa:
            raise ValueError(f"INE returned an unexpected geography for {measure} in {year}.")
        observations.append(canonical[["year", VALUE_NAMES[measure]]])
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            extraction_dates.append(str(payload[0].get("DataExtracao", "")))

    if not observations:
        raise ValueError(f"No observations returned for configured INE measure {measure}.")
    result = pd.concat(observations, ignore_index=True).sort_values("year").reset_index(drop=True)
    if result.duplicated("year").any():
        raise ValueError(f"Duplicate annual observations returned for {measure}.")

    provenance: dict[str, object] = {
        "measure": measure,
        "indicator_code": indicator,
        "indicator_name": metadata.get("IndicadorNome"),
        "first_period": metadata.get("PrimeiroPeriodo"),
        "last_period": metadata.get("UltimoPeriodo"),
        "last_update": metadata.get("DataUltimaAtualizacao"),
        "geo_code": lisboa,
        "geo_name": "Lisboa",
        "statistical_source": "Instituto Nacional de Estatística (INE), Portugal",
        "transport": "Pipeworx INE proxy",
        "transport_url": INE_MIRROR_ENDPOINT,
        "extraction_date": max(extraction_dates) if extraction_dates else metadata.get("DataExtracao"),
        "total_dimension_codes": json.dumps(totals, ensure_ascii=False, sort_keys=True),
    }
    print(
        f"{measure}: indicator={indicator}, Lisboa={lisboa}, "
        f"years={result['year'].min()}-{result['year'].max()}, rows={len(result)}"
    )
    return result, provenance


def _build_ine_series() -> tuple[pd.DataFrame, pd.DataFrame]:
    sources = _load_sources()
    configured = sources.get("ine")
    if not isinstance(configured, dict):
        raise TypeError("config/sources.toml must define an [ine] table.")

    series: list[pd.DataFrame] = []
    provenance: list[dict[str, object]] = []
    for measure, source in configured.items():
        if measure not in VALUE_NAMES:
            continue
        if not isinstance(source, dict):
            raise TypeError(f"INE source {measure!r} must be a table.")
        frame, source_provenance = _fetch_lisbon_indicator(measure, str(source["indicator"]))
        series.append(frame)
        provenance.append(source_provenance)

    if not series:
        raise ValueError("No configured INE series were loaded.")
    result = series[0]
    for frame in series[1:]:
        result = result.merge(frame, on="year", how="outer", validate="one_to_one")
    return result.sort_values("year").reset_index(drop=True), pd.DataFrame(provenance)


def _add_indices(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["rent_income_ratio"] = result["rent_eur_m2"] / result["income_eur"]
    baseline = result.loc[result["year"] == 2017, "rent_income_ratio"]
    if baseline.empty or pd.isna(baseline.iloc[0]) or float(baseline.iloc[0]) <= 0:
        raise ValueError("A valid 2017 rent/income baseline is required for LHDI.")
    result["lhdi"] = 100.0 * result["rent_income_ratio"] / float(baseline.iloc[0])
    result["tourism_intensity"] = result["overnight_stays"] / result["resident_population"]
    return result


def _join_airbnb(frame: pd.DataFrame) -> pd.DataFrame:
    path = ROOT / "results" / "processed" / "inside_airbnb_lisbon_annual.csv"
    airbnb = pd.read_csv(path, parse_dates=["snapshot_date"])
    required = {"year", "listed_units", "entire_home_units"}
    missing = required.difference(airbnb.columns)
    if missing:
        raise KeyError(f"Inside Airbnb result is missing columns: {sorted(missing)}")
    airbnb["entire_home_share"] = airbnb["entire_home_units"] / airbnb["listed_units"]
    result = frame.merge(airbnb, on="year", how="outer", validate="one_to_one")
    result["entire_home_per_1000_dwellings"] = (
        1000.0 * result["entire_home_units"] / result["housing_stock"]
    )
    return result.sort_values("year").reset_index(drop=True)


def _save_figures(frame: pd.DataFrame) -> None:
    figure_dir = ROOT / "results" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    lhd = frame.dropna(subset=["lhdi"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lhd["year"], lhd["lhdi"], marker="o")
    ax.axhline(100.0, linewidth=1)
    ax.set(title="Lisbon local housing decoupling index", xlabel="Year", ylabel="LHDI (2017 = 100)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "lisbon_lhdi.png", dpi=180)
    plt.close(fig)

    tourism = frame.dropna(subset=["tourism_intensity"])
    platform = frame.dropna(subset=["listed_units"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tourism["year"], tourism["tourism_intensity"], marker="o")
    ax.set(xlabel="Year", ylabel="Tourist overnight stays per resident")
    ax.grid(alpha=0.25)
    if not platform.empty:
        secondary = ax.twinx()
        secondary.scatter(platform["year"], platform["listed_units"], marker="s")
        secondary.set_ylabel("Observed platform listings")
    ax.set_title("Lisbon tourism intensity and observed platform listings")
    fig.tight_layout()
    fig.savefig(figure_dir / "lisbon_tourism_platform_exposure.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Build, export and print the observed Lisbon longitudinal dataset."""
    ine, provenance = _build_ine_series()
    result = _join_airbnb(_add_indices(ine))
    output_dir = ROOT / "results" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "lisbon_longitudinal.csv", index=False)
    provenance.to_csv(output_dir / "lisbon_ine_provenance.csv", index=False)
    _save_figures(result)

    columns = [
        "year",
        "rent_eur_m2",
        "income_eur",
        "lhdi",
        "tourism_intensity",
        "listed_units",
        "entire_home_units",
        "entire_home_per_1000_dwellings",
    ]
    print("\nLisbon longitudinal series:")
    print(result[columns].to_string(index=False))
    print("\nINE provenance:")
    print(provenance.to_string(index=False))


if __name__ == "__main__":
    main()
