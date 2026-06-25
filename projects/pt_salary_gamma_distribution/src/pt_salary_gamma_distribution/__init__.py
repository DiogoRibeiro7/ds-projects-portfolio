"""Utilities for Portugal salary distribution extraction and grouped fitting."""

from .extraction import (
    RMMG_BY_YEAR,
    build_manual_validation_checks,
    build_salary_bin_dataset,
    build_year_totals,
    deduplicate_brackets,
    deduplicate_summaries,
    download_manifest,
    extract_all_sources,
    normalize_text,
    parse_pt_number,
    validate_against_percentages,
)
from .fitting import (
    decile_validation,
    decile_validation_summary,
    fit_year_models,
    gamma_parameter_trend,
    grouped_residuals,
    model_winners,
    optional_microdata_fit,
    pareto_tail_diagnostics,
)

__all__ = [
    "RMMG_BY_YEAR",
    "build_manual_validation_checks",
    "build_salary_bin_dataset",
    "build_year_totals",
    "decile_validation",
    "decile_validation_summary",
    "deduplicate_brackets",
    "deduplicate_summaries",
    "download_manifest",
    "extract_all_sources",
    "fit_year_models",
    "gamma_parameter_trend",
    "grouped_residuals",
    "model_winners",
    "normalize_text",
    "optional_microdata_fit",
    "pareto_tail_diagnostics",
    "parse_pt_number",
    "validate_against_percentages",
]
