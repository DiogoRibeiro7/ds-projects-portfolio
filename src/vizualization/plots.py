"""
Visualization utilities for A/B testing and experimental analysis.

This module provides standardized plotting functions for experiment results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings


def plot_experiment_results(df: pd.DataFrame,
                           metric_col: str,
                           group_col: str = 'group',
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """Create comprehensive visualization of A/B test results.
    
    # TODO: Add interactive plots using Plotly for better user experience
    # Static matplotlib plots are limited for exploratory analysis
    # ASSIGNEE: @diogoribeiro7
    # LABELS: enhancement, interactive
    # PRIORITY: medium
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Distribution comparison (top-left)
    for group in df[group_col].unique():
        group_data = df[df[group_col] == group][metric_col]
        axes[0, 0].hist(group_data, alpha=0.6, label=group, bins=30)
    
    axes[0, 0].set_title('Distribution Comparison')
    axes[0, 0].set_xlabel(metric_col)
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # Box plot comparison (top-right)
    df.boxplot(column=metric_col, by=group_col, ax=axes[0, 1])
    axes[0, 1].set_title('Box Plot Comparison')
    
    # TODO: Add violin plots for better distribution visualization
    # Box plots don't show the full distribution shape
    # LABELS: enhancement, distribution-viz
    # PRIORITY: low
    
    # Sample sizes (bottom-left)
    group_counts = df[group_col].value_counts()
    axes[1, 0].bar(group_counts.index, group_counts.values)
    axes[1, 0].set_title('Sample Sizes by Group')
    axes[1, 0].set_ylabel('Count')
    
    # FIXME: Bottom-right subplot is currently empty
    # Should add a meaningful statistical summary plot
    axes[1, 1].text(0.5, 0.5, 'Statistical Summary\n(To be implemented)', 
                    ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Statistical Summary')
    
    plt.tight_layout()
    return fig


def plot_conversion_funnel(df: pd.DataFrame,
                          steps: List[str],
                          group_col: str = 'group') -> plt.Figure:
    """Plot conversion funnel analysis by experimental group.
    
    # TODO: Add funnel drop-off analysis with statistical tests
    # Should test if drop-off rates differ significantly between groups
    # ASSIGNEE: @diogoribeiro7
    # LABELS: funnel-analysis, statistics
    # PRIORITY: high
    """
    if not all(step in df.columns for step in steps):
        missing_steps = [step for step in steps if step not in df.columns]
        raise ValueError(f"Missing funnel steps: {missing_steps}")
    
    # Calculate conversion rates for each step
    funnel_data = []
    
    for group in df[group_col].unique():
        group_df = df[df[group_col] == group]
        rates = []
        
        for step in steps:
            if step == steps[0]:
                # First step is always 100% for users in the dataset
                rate = 1.0
            else:
                # BUG: This calculation assumes binary columns
                # Should handle different data types more gracefully
                rate = group_df[step].mean()
            rates.append(rate)
        
        funnel_data.append({
            'group': group,
            'steps': steps,
            'rates': rates,
            'counts': [len(group_df) * rate for rate in rates]
        })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Conversion rates plot
    for data in funnel_data:
        ax1.plot(data['steps'], data['rates'], marker='o', 
                label=data['group'], linewidth=2, markersize=8)
    
    ax1.set_title('Conversion Funnel Rates')
    ax1.set_ylabel('Conversion Rate')
    ax1.set_xlabel('Funnel Step')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # TODO: Add error bars showing confidence intervals for each rate
    # Current plot doesn't show uncertainty in the estimates
    # LABELS: uncertainty, error-bars
    # PRIORITY: medium
    
    # Absolute counts plot
    x_pos = np.arange(len(steps))
    width = 0.35
    
    for i, data in enumerate(funnel_data):
        offset = (i - len(funnel_data)/2 + 0.5) * width
        ax2.bar(x_pos + offset, data['counts'], width, 
               label=data['group'], alpha=0.8)
    
    ax2.set_title('Funnel Counts')
    ax2.set_ylabel('User Count')
    ax2.set_xlabel('Funnel Step')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(steps)
    ax2.legend()
    
    plt.tight_layout()
    return fig


def plot_time_series_analysis(df: pd.DataFrame,
                             date_col: str,
                             metric_col: str,
                             group_col: str = 'group',
                             agg_level: str = 'day') -> plt.Figure:
    """Plot time series analysis of experiment metrics.
    
    # TODO: Add seasonality detection and decomposition
    # Important for understanding if treatment effects vary by time
    # ASSIGNEE: @diogoribeiro7
    # LABELS: time-series, seasonality
    # PRIORITY: medium
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found")
    
    # Convert date column if it's not datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        # FIXME: Date parsing is not robust - should handle multiple formats
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Aggregate data by time period
    if agg_level == 'day':
        df['time_period'] = df[date_col].dt.date
    elif agg_level == 'hour':
        df['time_period'] = df[date_col].dt.floor('H')
    else:
        raise ValueError(f"Unsupported aggregation level: {agg_level}")
    
    # Calculate daily metrics by group
    time_series = df.groupby(['time_period', group_col])[metric_col].agg(['mean', 'count']).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot mean metric over time
    for group in df[group_col].unique():
        group_data = time_series[time_series[group_col] == group]
        ax1.plot(group_data['time_period'], group_data['mean'], 
                marker='o', label=f'{group} (mean)', linewidth=2)
    
    ax1.set_title(f'{metric_col} Over Time by Group')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'Mean {metric_col}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # TODO: Add statistical tests for time period differences
    # Should test if there are significant changes over time within groups
    # LABELS: statistics, temporal-tests
    # PRIORITY: high
    
    # Plot sample sizes over time
    for group in df[group_col].unique():
        group_data = time_series[time_series[group_col] == group]
        ax2.plot(group_data['time_period'], group_data['count'], 
                marker='s', label=f'{group} (count)', linewidth=2)
    
    ax2.set_title('Sample Sizes Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Daily Sample Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # NOTE: Large fluctuations in sample size could indicate data quality issues
    # or problems with the randomization system
    
    plt.tight_layout()
    return fig


def plot_statistical_power(effect_sizes: np.ndarray,
                          sample_sizes: np.ndarray,
                          alpha: float = 0.05,
                          baseline_rate: float = 0.1) -> plt.Figure:
    """Plot statistical power curves for experiment planning.
    
    # TODO: Add support for different statistical tests
    # Currently only implements power for two-proportion z-test
    # ASSIGNEE: @diogoribeiro7
    # LABELS: power-analysis, test-types
    # PRIORITY: medium
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Power vs Effect Size (for fixed sample size)
    fixed_n = sample_sizes[len(sample_sizes)//2]  # Use middle sample size
    
    # HACK: Using simplified power calculation
    # Should use proper statistical power formulas
    z_alpha = 1.96  # For alpha = 0.05, two-sided
    
    powers_vs_effect = []
    for effect in effect_sizes:
        # Simplified power calculation - TODO: implement proper formula
        # LABELS: implementation, power-formula
        z_beta = effect * np.sqrt(fixed_n) / (2 * np.sqrt(baseline_rate * (1 - baseline_rate)))
        power = 1 - 0.5 * (1 + np.sign(z_beta) * np.minimum(np.abs(z_beta), 3))  # Crude approximation
        powers_vs_effect.append(max(0, min(1, power)))
    
    ax1.plot(effect_sizes, powers_vs_effect, 'b-', linewidth=2)
    ax1.axhline(y=0.8, color='r', linestyle='--', label='80% Power')
    ax1.set_title(f'Power vs Effect Size (n={fixed_n} per group)')
    ax1.set_xlabel('Effect Size (absolute)')
    ax1.set_ylabel('Statistical Power')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Power vs Sample Size (for fixed effect size)
    fixed_effect = effect_sizes[len(effect_sizes)//2]  # Use middle effect size
    
    powers_vs_sample = []
    for n in sample_sizes:
        # Simplified power calculation
        z_beta = fixed_effect * np.sqrt(n) / (2 * np.sqrt(baseline_rate * (1 - baseline_rate)))
        power = 1 - 0.5 * (1 + np.sign(z_beta) * np.minimum(np.abs(z_beta), 3))
        powers_vs_sample.append(max(0, min(1, power)))
    
    ax2.plot(sample_sizes, powers_vs_sample, 'g-', linewidth=2)
    ax2.axhline(y=0.8, color='r', linestyle='--', label='80% Power')
    ax2.set_title(f'Power vs Sample Size (effect={fixed_effect:.3f})')
    ax2.set_xlabel('Sample Size per Group')
    ax2.set_ylabel('Statistical Power')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig


class ExperimentDashboard:
    """Interactive dashboard for experiment monitoring.
    
    # TODO: Implement real-time dashboard with live data updates
    # Current implementation is static - should refresh automatically
    # ASSIGNEE: @diogoribeiro7
    # LABELS: dashboard, real-time
    # PRIORITY: high
    """
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.figures = {}
        
        # TODO: Add configuration options for dashboard layout
        # Should be customizable based on experiment type
        # LABELS: configuration, customization
        # PRIORITY: low
    
    def generate_summary_dashboard(self, df: pd.DataFrame, 
                                  config: Dict) -> Dict[str, plt.Figure]:
        """Generate comprehensive experiment dashboard.
        
        # TODO: Add automated interpretation of results
        # Dashboard should highlight significant findings and concerns
        # LABELS: automation, interpretation
        # PRIORITY: medium
        """
        dashboard_figures = {}
        
        # Main results plot
        if 'primary_metric' in config:
            dashboard_figures['main_results'] = plot_experiment_results(
                df, config['primary_metric']
            )
        
        # Time series if date column available
        if 'date_col' in config and config['date_col'] in df.columns:
            dashboard_figures['time_series'] = plot_time_series_analysis(
                df, config['date_col'], config.get('primary_metric', 'converted')
            )
        
        # BUG: Dashboard doesn't handle missing configuration gracefully
        # Should provide sensible defaults and clear error messages
        
        # TODO: Add funnel analysis if funnel steps are configured
        # LABELS: funnel, conditional-plots
        
        self.figures = dashboard_figures
        return dashboard_figures
    
    def export_dashboard(self, output_dir: str = 'experiment_dashboard'):
        """Export dashboard plots to files.
        
        # TODO: Add support for multiple export formats (PNG, PDF, HTML)
        # LABELS: export, formats
        # PRIORITY: low
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in self.figures.items():
            fig.savefig(f'{output_dir}/{self.experiment_name}_{name}.png', 
                       dpi=300, bbox_inches='tight')


# Global plotting configuration
# NOTE: Should be moved to a proper configuration system
plt.style.use('seaborn-v0_8')  # TODO: Update to newer seaborn style
sns.set_palette("husl")

# TODO: Create theme system for consistent plotting across all visualizations
# Should include company branding, color schemes, font choices
# ASSIGNEE: @diogoribeiro7
# LABELS: theming, branding
# PRIORITY: low

def create_publication_ready_plot(df: pd.DataFrame,
                                 plot_type: str,
                                 **kwargs) -> plt.Figure:
    """Create publication-ready plots with proper formatting.
    
    # TODO: Implement this function with academic journal standards
    # Should follow specific formatting guidelines for different journals
    # LABELS: publication, formatting
    # PRIORITY: low
    """
    # Placeholder implementation
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, f'Publication plot: {plot_type}\n(To be implemented)', 
           ha='center', va='center', transform=ax.transAxes)
    return fig


# FIXME: Memory usage could be high with large datasets and many plots
# Should implement plot caching and lazy loading for better performance
