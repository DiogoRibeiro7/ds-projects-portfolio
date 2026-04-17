"""Visualization optimization utilities for efficient plotting and rendering.
"""

import warnings
from functools import lru_cache
from typing import Any

import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


class VisualizationOptimizer:
    """Optimization techniques for data visualization."""

    @staticmethod
    def sample_for_visualization(
        df: pd.DataFrame, sample_size: int = 10000, stratify_col: str | None = None
    ) -> pd.DataFrame:
        """Sample data for exploratory visualization.

        Args:
            df: Input DataFrame
            sample_size: Number of samples
            stratify_col: Column for stratified sampling

        Returns:
            Sampled DataFrame
        """
        if len(df) <= sample_size:
            return df

        if stratify_col and stratify_col in df.columns:
            # Stratified sampling
            return df.groupby(stratify_col, group_keys=False).apply(
                lambda x: x.sample(
                    min(len(x), sample_size // df[stratify_col].nunique())
                )
            )
        else:
            # Random sampling
            return df.sample(n=min(sample_size, len(df)))

    @staticmethod
    def adaptive_sampling(
        df: pd.DataFrame, x_col: str, y_col: str, max_points: int = 5000
    ) -> pd.DataFrame:
        """Adaptive sampling based on data density.

        Args:
            df: Input DataFrame
            x_col, y_col: Columns for x and y axes
            max_points: Maximum number of points

        Returns:
            Adaptively sampled DataFrame
        """
        if len(df) <= max_points:
            return df

        # Calculate density
        x_range = df[x_col].max() - df[x_col].min()
        y_range = df[y_col].max() - df[y_col].min()

        # Create bins
        n_bins = int(np.sqrt(max_points))
        x_bins = pd.cut(df[x_col], bins=n_bins)
        y_bins = pd.cut(df[y_col], bins=n_bins)

        # Sample from each bin
        sampled = df.groupby([x_bins, y_bins], observed=True).apply(
            lambda x: x.sample(min(len(x), max(1, max_points // (n_bins * n_bins))))
        )

        return sampled.reset_index(drop=True)

    @staticmethod
    def create_aggregated_heatmap(
        df: pd.DataFrame, x_col: str, y_col: str, value_col: str, bins: int = 50
    ) -> go.Figure:
        """Create heatmap using aggregated data for large datasets.

        Args:
            df: Input DataFrame
            x_col, y_col: Columns for x and y axes
            value_col: Column for values
            bins: Number of bins for aggregation

        Returns:
            Plotly figure
        """
        # Create bins
        x_bins = pd.cut(df[x_col], bins=bins)
        y_bins = pd.cut(df[y_col], bins=bins)

        # Aggregate data
        aggregated = (
            df.groupby([x_bins, y_bins], observed=True)[value_col].mean().reset_index()
        )

        # Create pivot table for heatmap
        pivot = aggregated.pivot_table(
            index=y_bins.name, columns=x_bins.name, values=value_col, aggfunc="mean"
        )

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=[str(col) for col in pivot.columns],
                y=[str(idx) for idx in pivot.index],
                colorscale="Viridis",
            )
        )

        fig.update_layout(
            title=f"Aggregated Heatmap: {value_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
        )

        return fig

    @staticmethod
    def create_datashader_plot(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        category_col: str | None = None,
        plot_width: int = 800,
        plot_height: int = 600,
    ) -> Any:
        """Create datashader plot for millions of points.

        Args:
            df: Input DataFrame
            x_col, y_col: Columns for x and y axes
            category_col: Optional categorical column for coloring
            plot_width, plot_height: Plot dimensions

        Returns:
            Datashader image
        """
        # Create canvas
        canvas = ds.Canvas(plot_width=plot_width, plot_height=plot_height)

        # Aggregate points
        if category_col:
            agg = canvas.points(df, x_col, y_col, ds.count_cat(category_col))
            img = tf.shade(agg, color_key=["blue", "red", "green", "yellow", "purple"])
        else:
            agg = canvas.points(df, x_col, y_col)
            img = tf.shade(agg, cmap=["lightblue", "darkblue"])

        return img

    @staticmethod
    def progressive_loading_plot(
        df: pd.DataFrame, x_col: str, y_col: str, chunk_size: int = 1000
    ) -> go.Figure:
        """Create plot with progressive data loading.

        Args:
            df: Input DataFrame
            x_col, y_col: Columns for plotting
            chunk_size: Size of each chunk to load

        Returns:
            Plotly figure with progressive loading
        """
        fig = go.Figure()

        # Initial plot with first chunk
        initial_data = df.iloc[:chunk_size]

        fig.add_trace(
            go.Scattergl(
                x=initial_data[x_col],
                y=initial_data[y_col],
                mode="markers",
                marker=dict(size=3),
                name="Data Points",
            )
        )

        # Add slider for progressive loading
        steps = []
        for i in range(chunk_size, len(df), chunk_size):
            step = dict(
                method="update",
                args=[
                    {"x": [df[x_col].iloc[:i].values], "y": [df[y_col].iloc[:i].values]}
                ],
                label=f"{i} points",
            )
            steps.append(step)

        sliders = [
            dict(
                active=0,
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                currentvalue=dict(
                    prefix="Points loaded: ", visible=True, xanchor="right"
                ),
                steps=steps,
            )
        ]

        fig.update_layout(
            sliders=sliders,
            title=f"Progressive Loading: {x_col} vs {y_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
        )

        return fig


class PlotCaching:
    """Caching system for expensive plots."""

    def __init__(self):
        """Initialize plot cache."""
        self.cache = {}

    @lru_cache(maxsize=32)
    def _get_plot_key(self, *args) -> str:
        """Generate cache key for plot."""
        return str(hash(args))

    def get_or_create_plot(self, plot_func: callable, *args, **kwargs) -> Any:
        """Get plot from cache or create new one.

        Args:
            plot_func: Function to create plot
            *args, **kwargs: Arguments for plot function

        Returns:
            Plot object
        """
        # Generate cache key
        cache_key = self._get_plot_key(str(args), str(kwargs))

        if cache_key in self.cache:
            print("[PLOT CACHE HIT] Returning cached plot")
            return self.cache[cache_key]

        # Create new plot
        print("[PLOT CACHE MISS] Creating new plot")
        plot = plot_func(*args, **kwargs)
        self.cache[cache_key] = plot

        return plot

    def clear_cache(self):
        """Clear plot cache."""
        self.cache.clear()
        self._get_plot_key.cache_clear()


class EfficientPlotting:
    """Efficient plotting techniques for different scenarios."""

    @staticmethod
    def create_line_plot_decimated(
        df: pd.DataFrame, x_col: str, y_col: str, max_points: int = 1000
    ) -> go.Figure:
        """Create line plot with decimated data for performance.

        Args:
            df: Input DataFrame
            x_col, y_col: Columns for plotting
            max_points: Maximum points to display

        Returns:
            Plotly figure
        """
        if len(df) <= max_points:
            x_data = df[x_col].values
            y_data = df[y_col].values
        else:
            # Decimate data using LTTB algorithm (simplified version)
            indices = np.linspace(0, len(df) - 1, max_points, dtype=int)
            x_data = df[x_col].iloc[indices].values
            y_data = df[y_col].iloc[indices].values

        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=x_data,
                y=y_data,
                mode="lines",
                name="Decimated Data",
                line=dict(width=1),
            )
        )

        fig.update_layout(
            title=f"{y_col} vs {x_col} (Decimated to {len(x_data)} points)",
            xaxis_title=x_col,
            yaxis_title=y_col,
            hovermode="x",
        )

        return fig

    @staticmethod
    def create_scatter_with_density(
        df: pd.DataFrame, x_col: str, y_col: str, sample_size: int = 5000
    ) -> go.Figure:
        """Create scatter plot with density contours for large datasets.

        Args:
            df: Input DataFrame
            x_col, y_col: Columns for plotting
            sample_size: Number of points to show

        Returns:
            Plotly figure
        """
        # Sample data for scatter
        if len(df) > sample_size:
            sample_df = df.sample(n=sample_size)
        else:
            sample_df = df

        fig = go.Figure()

        # Add density contours using all data
        fig.add_trace(
            go.Histogram2dContour(
                x=df[x_col],
                y=df[y_col],
                colorscale="Blues",
                showscale=False,
                contours=dict(coloring="lines"),
                line=dict(width=1),
                name="Density",
            )
        )

        # Add scatter points using sampled data
        fig.add_trace(
            go.Scattergl(
                x=sample_df[x_col],
                y=sample_df[y_col],
                mode="markers",
                marker=dict(size=3, opacity=0.5),
                name=f"Sample ({len(sample_df)} points)",
            )
        )

        fig.update_layout(
            title=f"{y_col} vs {x_col} with Density Contours",
            xaxis_title=x_col,
            yaxis_title=y_col,
        )

        return fig

    @staticmethod
    def create_binned_scatter(
        df: pd.DataFrame, x_col: str, y_col: str, bins: int = 100
    ) -> go.Figure:
        """Create binned scatter plot (hexbin-like) for large datasets.

        Args:
            df: Input DataFrame
            x_col, y_col: Columns for plotting
            bins: Number of bins

        Returns:
            Plotly figure
        """
        # Create hexbin-like aggregation
        x_bins = pd.cut(df[x_col], bins=bins)
        y_bins = pd.cut(df[y_col], bins=bins)

        # Count points in each bin
        binned = (
            df.groupby([x_bins, y_bins], observed=True).size().reset_index(name="count")
        )

        # Get bin centers
        binned["x_center"] = binned[x_col].apply(lambda x: x.mid)
        binned["y_center"] = binned[y_col].apply(lambda x: x.mid)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=binned["x_center"],
                y=binned["y_center"],
                mode="markers",
                marker=dict(
                    size=binned["count"] / binned["count"].max() * 20,
                    color=binned["count"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Count"),
                ),
                text=binned["count"],
                hovertemplate="<b>Count:</b> %{text}<br>"
                + f"<b>{x_col}:</b> %{{x}}<br>"
                + f"<b>{y_col}:</b> %{{y}}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"Binned Scatter: {y_col} vs {x_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
        )

        return fig

    @staticmethod
    def create_rasterized_plot(
        df: pd.DataFrame, x_col: str, y_col: str, resolution: int = 500
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create rasterized matplotlib plot for PDF export.

        Args:
            df: Input DataFrame
            x_col, y_col: Columns for plotting
            resolution: Rasterization resolution

        Returns:
            Matplotlib figure and axes
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot with rasterization for large datasets
        ax.scatter(df[x_col], df[y_col], alpha=0.5, s=1, rasterized=True)

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col} (Rasterized)")

        # Set rasterization resolution
        fig.set_dpi(resolution)

        return fig, ax


class DashboardOptimizer:
    """Optimization techniques for dashboards."""

    @staticmethod
    def create_lazy_loading_dashboard_data(
        df: pd.DataFrame, page_size: int = 100
    ) -> dict[str, Any]:
        """Prepare data for lazy loading in dashboard.

        Args:
            df: Input DataFrame
            page_size: Number of rows per page

        Returns:
            Dictionary with paginated data
        """
        n_pages = (len(df) - 1) // page_size + 1

        return {
            "total_rows": len(df),
            "page_size": page_size,
            "n_pages": n_pages,
            "pages": {
                i: df.iloc[i * page_size : (i + 1) * page_size].to_dict("records")
                for i in range(min(5, n_pages))  # Pre-load first 5 pages
            },
        }

    @staticmethod
    def create_aggregated_dashboard_metrics(
        df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]
    ) -> pd.DataFrame:
        """Create pre-aggregated metrics for dashboard.

        Args:
            df: Input DataFrame
            group_cols: Columns to group by
            metric_cols: Columns to aggregate

        Returns:
            Aggregated DataFrame
        """
        agg_funcs = {col: ["sum", "mean", "min", "max", "count"] for col in metric_cols}

        aggregated = df.groupby(group_cols).agg(agg_funcs)
        aggregated.columns = [
            "_".join(col).strip() for col in aggregated.columns.values
        ]

        return aggregated.reset_index()

    @staticmethod
    def optimize_dashboard_queries(
        df: pd.DataFrame, common_filters: dict[str, list[Any]]
    ) -> dict[str, pd.DataFrame]:
        """Pre-compute common dashboard queries.

        Args:
            df: Input DataFrame
            common_filters: Dictionary of common filter combinations

        Returns:
            Dictionary of pre-filtered DataFrames
        """
        cached_queries = {}

        for filter_name, filter_values in common_filters.items():
            if isinstance(filter_values, dict):
                # Multiple column filters
                query_df = df.copy()
                for col, vals in filter_values.items():
                    query_df = query_df[query_df[col].isin(vals)]
            else:
                # Single column filter
                query_df = df[df[filter_name].isin(filter_values)]

            cached_queries[filter_name] = query_df

        return cached_queries
