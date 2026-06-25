from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


TITLE_BRACKET_PATTERNS = (
    "trabalhadores por conta de outrem",
    "escalao de remuneracao mensal ganho",
)
TITLE_SUMMARY_PATTERNS = (
    "ganho mensal",
    "mediano",
)
DECILE_RE = re.compile(r"^(?P<decil>10|[1-9])(?:\s*[\.\-]?\s*[a-z])?\s*decil")


RMMG_BY_YEAR: dict[int, float] = {
    1999: 305.76,
    2000: 318.23,
    2001: 334.19,
    2002: 348.01,
    2003: 356.60,
    2004: 365.60,
    2005: 374.70,
    2006: 385.90,
    2007: 403.00,
    2008: 426.00,
    2009: 450.00,
    2010: 475.00,
    2011: 485.00,
    2012: 485.00,
    2013: 485.00,
    2014: 505.00,
    2015: 505.00,
    2016: 530.00,
    2017: 557.00,
    2018: 580.00,
    2019: 600.00,
    2020: 635.00,
    2021: 665.00,
    2022: 705.00,
    2023: 760.00,
    2024: 820.00,
}


@dataclass(frozen=True)
class DownloadResult:
    source_id: str
    start_year: int
    end_year: int
    local_path: str
    status: str
    error: str
    url: str


def normalize_text(value: object) -> str:
    """Normalize spreadsheet text for robust matching."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\xa0", " ").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text)
    return text


def parse_pt_number(value: object) -> float:
    """Parse Portuguese-formatted numbers and return NaN for malformed tokens."""
    if value is None or pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).replace("\xa0", " ").strip()
    if not text or text in {"-", "--", "..", "x"}:
        return float("nan")

    text = re.sub(r"[^0-9,\.\- ]", "", text).strip()
    if not text:
        return float("nan")
    if text.count("-") > 1 or re.fullmatch(r"\d+\s*-\s*\d+", text):
        return float("nan")

    try:
        if "," in text:
            return float(text.replace(".", "").replace(" ", "").replace(",", "."))
        return float(text.replace(".", "").replace(" ", ""))
    except ValueError:
        return float("nan")


def maybe_year(value: object) -> int | None:
    """Return an integer year for cells that look like a year."""
    number = parse_pt_number(value)
    if np.isnan(number):
        return None
    year = int(round(number))
    if 1990 <= year <= 2035 and abs(number - year) < 1e-9:
        return year
    return None


def row_text(df: pd.DataFrame, row_idx: int, stop_col: int | None = None) -> str:
    """Join the normalized text in a row, optionally only to the left of a column."""
    if df.empty or row_idx >= len(df):
        return ""
    values = df.iloc[row_idx, :stop_col].tolist() if stop_col is not None else df.iloc[row_idx].tolist()
    parts = [normalize_text(value) for value in values]
    return " ".join(part for part in parts if part)


def find_year_map(df: pd.DataFrame, min_years: int = 5) -> tuple[int, dict[int, int]]:
    """Find the row containing year labels and map column index to year."""
    for row_idx in range(len(df)):
        year_map = {
            col_idx: year
            for col_idx, value in enumerate(df.iloc[row_idx].tolist())
            if (year := maybe_year(value)) is not None
        }
        if len(year_map) >= min_years:
            return row_idx, year_map
    raise ValueError("Could not detect a row with year labels.")


def read_excel_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read a workbook sheet without inferring a header."""
    return pd.read_excel(path, sheet_name=sheet_name, header=None, dtype=object, engine="openpyxl")


def download_file(url: str, output_path: Path, overwrite: bool = False) -> Path:
    """Download one source file."""
    if output_path.exists() and not overwrite:
        return output_path

    headers = {"User-Agent": "Mozilla/5.0 salary-distribution-research/1.0"}
    response = requests.get(url, headers=headers, timeout=90)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if "html" in content_type:
        raise ValueError(f"Expected a spreadsheet, received HTML from {url}")

    output_path.write_bytes(response.content)
    return output_path


def download_manifest(manifest_df: pd.DataFrame, raw_dir: Path, overwrite: bool = False) -> pd.DataFrame:
    """Download every source listed in the manifest."""
    rows: list[dict[str, Any]] = []
    for record in manifest_df.itertuples(index=False):
        output_path = raw_dir / f"{record.source_id}.xlsx"
        try:
            local_path = download_file(str(record.url), output_path, overwrite=overwrite)
            status = "ok"
            error = ""
        except Exception as exc:
            local_path = output_path
            status = "failed"
            error = str(exc)
        rows.append(
            {
                "source_id": record.source_id,
                "start_year": int(record.start_year),
                "end_year": int(record.end_year),
                "kind": record.kind,
                "url": record.url,
                "notes": record.notes,
                "local_path": str(local_path),
                "status": status,
                "error": error,
            }
        )
    return pd.DataFrame(rows)


def matches_all_patterns(text: str, patterns: tuple[str, ...]) -> bool:
    """Return True if all normalized patterns are present in the text."""
    return all(pattern in text for pattern in patterns)


def is_salary_distribution_sheet(df: pd.DataFrame) -> bool:
    """Detect the public grouped salary-bracket table."""
    if df.empty:
        return False
    return matches_all_patterns(row_text(df, 0), TITLE_BRACKET_PATTERNS)


def is_salary_summary_sheet(df: pd.DataFrame) -> bool:
    """Detect the public salary summary table with median and deciles."""
    if df.empty:
        return False
    title = row_text(df, 0)
    return matches_all_patterns(title, TITLE_SUMMARY_PATTERNS) and (
        "media por decil" in normalize_text(" ".join(df.iloc[:20, 0].astype(str).tolist()))
        or "decis" in title
    )


def extract_values_by_year(df: pd.DataFrame, row_idx: int, year_map: dict[int, int]) -> dict[int, float]:
    """Extract values from one row according to a detected year map."""
    out: dict[int, float] = {}
    for col_idx, year in year_map.items():
        out[year] = parse_pt_number(df.iat[row_idx, col_idx])
    return out


def clean_bin_label(label: str) -> str:
    """Normalize bracket labels while preserving meaning."""
    label = label.replace("�", "€")
    label = re.sub(r"\s+", " ", label).strip()
    label = label.replace("Euros", "euros")
    return label


def extract_salary_bins_from_sheet(df: pd.DataFrame, source_id: str, sheet_name: str) -> pd.DataFrame:
    """Extract grouped salary counts and percentages from one detected sheet."""
    if not is_salary_distribution_sheet(df):
        return pd.DataFrame()

    header_row, year_map = find_year_map(df)
    first_year_col = min(year_map)
    section = "counts"
    rows: list[dict[str, Any]] = []

    for row_idx in range(header_row + 1, len(df)):
        label = row_text(df, row_idx, stop_col=first_year_col)
        if not label:
            continue
        if label == "percentagem":
            section = "percentage"
            continue
        if label == "total":
            continue
        if "fonte:" in label:
            break
        if not (
            "< rmmg" in label
            or "= rmmg" in label
            or ">rmmg" in label
            or bool(re.search(r"\d[\d\s\.,]*\s*-\s*\d[\d\s\.,]*", label))
            or bool(re.search(r"\d[\d\s\.,]*\s*e\s*\+", label))
        ):
            continue

        values = extract_values_by_year(df, row_idx, year_map)
        for year, value in values.items():
            if np.isnan(value):
                continue
            rows.append(
                {
                    "source_id": source_id,
                    "sheet_name": sheet_name,
                    "year": int(year),
                    "bin_label": clean_bin_label(str(df.iat[row_idx, 0])),
                    "measure": section,
                    "value": float(value),
                }
            )
    return pd.DataFrame(rows)


def extract_salary_summary_from_sheet(df: pd.DataFrame, source_id: str, sheet_name: str) -> pd.DataFrame:
    """Extract totals, median, decile cutpoints, and mean earnings by decile."""
    if not is_salary_summary_sheet(df):
        return pd.DataFrame()

    header_row, year_map = find_year_map(df)
    first_year_col = min(year_map)
    rows: list[dict[str, Any]] = []
    current_statistic: str | None = None

    for row_idx in range(header_row + 1, len(df)):
        label = row_text(df, row_idx, stop_col=first_year_col)
        if not label:
            continue
        if "fonte:" in label:
            break
        if "trabalhadores por conta de outrem" in label:
            values = extract_values_by_year(df, row_idx, year_map)
            for year, value in values.items():
                if not np.isnan(value):
                    rows.append(
                        {
                            "source_id": source_id,
                            "sheet_name": sheet_name,
                            "year": int(year),
                            "statistic": "total_workers",
                            "decile": np.nan,
                            "value": float(value),
                        }
                    )
            continue
        if "ganho mensal mediano" in label:
            values = extract_values_by_year(df, row_idx, year_map)
            for year, value in values.items():
                if not np.isnan(value):
                    rows.append(
                        {
                            "source_id": source_id,
                            "sheet_name": sheet_name,
                            "year": int(year),
                            "statistic": "median_gain",
                            "decile": np.nan,
                            "value": float(value),
                        }
                    )
            continue
        if "ganho mensal - decis" in label:
            current_statistic = "decile_cutpoint"
            continue
        if "ganho mensal - media por decil" in label:
            current_statistic = "mean_gain_by_decile"
            continue
        if "limiar de baixos salarios" in label:
            current_statistic = None
            continue

        decile_match = DECILE_RE.search(label)
        if current_statistic and decile_match:
            decile = int(decile_match.group("decil"))
            values = extract_values_by_year(df, row_idx, year_map)
            for year, value in values.items():
                if not np.isnan(value):
                    rows.append(
                        {
                            "source_id": source_id,
                            "sheet_name": sheet_name,
                            "year": int(year),
                            "statistic": current_statistic,
                            "decile": decile,
                            "value": float(value),
                        }
                    )
    return pd.DataFrame(rows)


def extract_all_sources(valid_sources: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract bracket and summary rows from all successfully downloaded workbooks."""
    bracket_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []

    for source in valid_sources.itertuples(index=False):
        path = Path(source.local_path)
        xls = pd.ExcelFile(path, engine="openpyxl")
        for sheet_name in xls.sheet_names:
            normalized_sheet_name = normalize_text(sheet_name)
            if normalized_sheet_name not in {"q23", "q24", "q32", "q33"}:
                continue
            sheet = read_excel_sheet(path, sheet_name)
            bracket_df = extract_salary_bins_from_sheet(sheet, source.source_id, sheet_name)
            summary_df = extract_salary_summary_from_sheet(sheet, source.source_id, sheet_name)
            if not bracket_df.empty:
                bracket_frames.append(bracket_df)
            if not summary_df.empty:
                summary_frames.append(summary_df)

    all_brackets = pd.concat(bracket_frames, ignore_index=True) if bracket_frames else pd.DataFrame()
    all_summaries = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    return all_brackets, all_summaries


def deduplicate_brackets(df: pd.DataFrame, source_priority: dict[str, int]) -> pd.DataFrame:
    """Prefer one most recent publication window per year.

    This avoids mixing incompatible bracket definitions from overlapping workbooks.
    """
    if df.empty:
        return df
    out = df.copy()
    out["source_end_year"] = out["source_id"].map(source_priority)
    preferred = (
        out[["year", "source_id", "source_end_year"]]
        .drop_duplicates()
        .sort_values(["year", "source_end_year"], ascending=[True, False])
        .drop_duplicates(["year"], keep="first")
    )
    out = out.merge(preferred[["year", "source_id"]], on=["year", "source_id"], how="inner")
    return out.drop(columns=["source_end_year"]).reset_index(drop=True)


def deduplicate_summaries(df: pd.DataFrame, source_priority: dict[str, int]) -> pd.DataFrame:
    """Prefer one latest publication window per year for summary statistics."""
    if df.empty:
        return df
    out = df.copy()
    out["source_end_year"] = out["source_id"].map(source_priority)
    preferred = (
        out[["year", "source_id", "source_end_year"]]
        .drop_duplicates()
        .sort_values(["year", "source_end_year"], ascending=[True, False])
        .drop_duplicates(["year"], keep="first")
    )
    out = out.merge(preferred[["year", "source_id"]], on=["year", "source_id"], how="inner")
    return out.drop(columns=["source_end_year"]).reset_index(drop=True)


def parse_bin_interval(label: str, year: int, exact_width: float = 1.0) -> tuple[float, float, str]:
    """Convert a published bracket label into numeric lower and upper bounds."""
    text = normalize_text(label)
    rmmg = RMMG_BY_YEAR.get(int(year), float("nan"))

    if re.search(r"^<\s*rmmg", text):
        return 0.0, rmmg, "below_minimum_wage"
    if re.search(r"^=\s*rmmg", text):
        half = exact_width / 2.0
        return max(0.0, rmmg - half), rmmg + half, "exact_minimum_wage"
    if re.search(r"^>\s*rmmg", text):
        numbers = re.findall(r"\d[\d\s\.,]*", text)
        upper = parse_pt_number(numbers[-1]) if numbers else float("nan")
        return rmmg, upper, "above_minimum_wage_to_threshold"
    if re.search(r"e\s*\+|\+", text):
        numbers = re.findall(r"\d[\d\s\.,]*", text)
        if numbers:
            return parse_pt_number(numbers[0]), float("inf"), "open_top"
    numbers = re.findall(r"\d[\d\s\.,]*", text)
    if "-" in text and len(numbers) >= 2:
        return parse_pt_number(numbers[0]), parse_pt_number(numbers[1]), "closed_range"
    return float("nan"), float("nan"), "unparsed"


def build_salary_bin_dataset(raw_brackets: pd.DataFrame) -> pd.DataFrame:
    """Pivot counts and percentages into one model-ready bracket dataset."""
    if raw_brackets.empty:
        return raw_brackets
    wide = (
        raw_brackets.pivot_table(
            index=["year", "bin_label"],
            columns="measure",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    intervals = wide.apply(
        lambda row: parse_bin_interval(str(row["bin_label"]), int(row["year"])),
        axis=1,
        result_type="expand",
    )
    intervals.columns = ["lower", "upper", "bin_type"]
    out = pd.concat([wide, intervals], axis=1).rename(columns={"counts": "count", "percentage": "pct"})
    out = out.dropna(subset=["count", "lower", "upper"]).query("bin_type != 'unparsed'").copy()
    out["count"] = out["count"].astype(float)
    return out.sort_values(["year", "lower", "upper"]).reset_index(drop=True)


def build_year_totals(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Extract the yearly worker totals from the summary table."""
    totals = summary_df.query("statistic == 'total_workers'").copy()
    if totals.empty:
        return pd.DataFrame(columns=["year", "total_workers"])
    return totals[["year", "value"]].rename(columns={"value": "total_workers"}).drop_duplicates("year")


def validate_against_percentages(bin_df: pd.DataFrame) -> pd.DataFrame:
    """Compare count-implied shares with published shares when both exist."""
    if bin_df.empty or "pct" not in bin_df.columns:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for year, group in bin_df.groupby("year"):
        total = float(group["count"].sum())
        for row in group.itertuples(index=False):
            if pd.isna(getattr(row, "pct", np.nan)):
                continue
            implied_pct = 100.0 * float(row.count) / total
            records.append(
                {
                    "year": int(year),
                    "bin_label": row.bin_label,
                    "published_pct": float(row.pct),
                    "implied_pct": implied_pct,
                    "pct_point_error": implied_pct - float(row.pct),
                }
            )
    return pd.DataFrame(records)


def workbook_cell_value(path: Path, sheet_name: str, row_idx_zero_based: int, year: int) -> float:
    """Fetch a numeric value from a validated workbook row and year column."""
    df = read_excel_sheet(path, sheet_name)
    _, year_map = find_year_map(df)
    target_col = next(col for col, mapped_year in year_map.items() if mapped_year == year)
    return parse_pt_number(df.iat[row_idx_zero_based, target_col])


def build_manual_validation_checks(raw_dir: Path) -> pd.DataFrame:
    """Create explicit cross-checks against known workbook cells for three years."""
    checks = [
        {
            "validation_year": 2007,
            "source_id": "seriesqp_2007_2017",
            "sheet_name": "q23",
            "row_idx": 7,
            "metric": "count_2007_gt_rmmg_to_599_99",
        },
        {
            "validation_year": 2014,
            "source_id": "seriesqp_2014_2024",
            "sheet_name": "q33 ",
            "row_idx": 17,
            "metric": "mean_decile_1_2014",
        },
        {
            "validation_year": 2024,
            "source_id": "seriesqp_2014_2024",
            "sheet_name": "q33 ",
            "row_idx": 5,
            "metric": "median_gain_2024",
        },
    ]
    rows: list[dict[str, Any]] = []
    for check in checks:
        path = raw_dir / f"{check['source_id']}.xlsx"
        rows.append(
            {
                **check,
                "workbook_value": workbook_cell_value(path, str(check["sheet_name"]), int(check["row_idx"]), int(check["validation_year"])),
            }
        )
    return pd.DataFrame(rows)
