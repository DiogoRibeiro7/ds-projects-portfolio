"""
Visualization utilities for A/B testing and experimental analysis.
"""

from .plots import (
    BRAND_COLORS,
    ExperimentDashboard,
    ThemeManager,
    create_memory_efficient_plots,
    create_publication_ready_plot,
    plot_conversion_funnel,
    plot_experiment_results,
    plot_statistical_power,
    plot_time_series_analysis,
    set_publication_style,
)

# Export all functions for easy access
__all__ = [
    "plot_experiment_results",
    "plot_conversion_funnel",
    "plot_time_series_analysis",
    "plot_statistical_power",
    "ExperimentDashboard",
    "set_publication_style",
    "create_publication_ready_plot",
    "create_memory_efficient_plots",
    "ThemeManager",
    "BRAND_COLORS",
]
