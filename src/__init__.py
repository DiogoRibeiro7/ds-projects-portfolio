"""Data Science Portfolio Package

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

from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .vizualization.plots import ExperimentDashboard as _ExperimentDashboard

ExperimentDashboard: Optional[type["_ExperimentDashboard"]]
plot_conversion_funnel: Optional[Callable[..., Any]]
plot_experiment_results: Optional[Callable[..., Any]]

try:
    from .vizualization.plots import (
        ExperimentDashboard,
        plot_conversion_funnel,
        plot_experiment_results,
    )
except Exception:
    ExperimentDashboard = None
    plot_conversion_funnel = None
    plot_experiment_results = None

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
