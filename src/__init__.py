"""
Data Science Portfolio Package

A comprehensive toolkit for A/B testing, statistical analysis, and experimental design.
"""

__version__ = "1.0.0"
__author__ = "Diogo Ribeiro"
__email__ = "dfr@esmad.ipp.pt"

from .data_processing.cleaning import (
    apply_cuped,
    clean_ab_data,
    validate_experiment_data,
)

# Core imports for easy access
from .statistics.core import (
    ExperimentAnalyzer,
    bootstrap_ci_diff,
    calculate_sample_size,
    two_prop_ztest,
)
from .vizualization.plots import (
    ExperimentDashboard,
    plot_conversion_funnel,
    plot_experiment_results,
)

__all__ = [
    "ExperimentAnalyzer",
    "two_prop_ztest",
    "bootstrap_ci_diff",
    "calculate_sample_size",
    "clean_ab_data",
    "validate_experiment_data",
    "apply_cuped",
    "plot_experiment_results",
    "plot_conversion_funnel",
    "ExperimentDashboard",
]
