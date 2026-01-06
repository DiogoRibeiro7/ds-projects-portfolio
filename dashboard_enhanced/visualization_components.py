"""
Interactive Visualization Components

Advanced visualization components using Plotly and Bokeh with animations,
custom interactions, and accessibility features.

Author: Portfolio Team
Date: 2024
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Plotly imports
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# Bokeh imports
from bokeh.plotting import figure, output_file, show
from bokeh.models import (
    HoverTool, ColumnDataSource, CustomJS, Slider,
    RangeSlider, Select, DatePicker, CheckboxGroup,
    DataTable, DateFormatter, TableColumn,
    WheelZoomTool, ResetTool, PanTool, SaveTool,
    BoxZoomTool, TapTool, CrosshairTool
)
from bokeh.layouts import column, row, gridplot
from bokeh.palettes import Category20, Viridis256
from bokeh.transform import cumsum, linear_cmap
from bokeh.models.widgets import Button, TextInput
from bokeh.events import ButtonClick

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractiveVisualizations:
    """
    Advanced interactive visualization components with animations,
    custom interactions, and accessibility features.
    """

    def __init__(self):
        """Initialize visualization components."""
        self.theme = "plotly_white"
        self.color_palette = px.colors.qualitative.Plotly
        self.accessibility_mode = True
        self.animation_duration = 500

        logger.info("Interactive Visualizations initialized")

    def create_animated_time_series(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_cols: List[str],
        title: str = "Animated Time Series",
        play_button: bool = True,
        range_slider: bool = True
    ) -> go.Figure:
        """
        Create animated time series visualization.

        Args:
            df: DataFrame with time series data
            x_col: Column for x-axis (time)
            y_cols: Columns for y-axis values
            title: Chart title
            play_button: Show play button for animation
            range_slider: Show range slider

        Returns:
            Animated Plotly figure
        """
        # Create figure
        fig = go.Figure()

        # Add traces for each y column
        for i, col in enumerate(y_cols):
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[col],
                    mode='lines+markers',
                    name=col,
                    line=dict(
                        color=self.color_palette[i % len(self.color_palette)],
                        width=2
                    ),
                    marker=dict(size=6),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  'Date: %{x|%Y-%m-%d}<br>' +
                                  'Value: %{y:.2f}<br>' +
                                  '<extra></extra>'
                )
            )

        # Create frames for animation
        frames = []
        for k in range(len(df)):
            frame_data = []
            for col in y_cols:
                frame_data.append(
                    go.Scatter(
                        x=df[x_col][:k+1],
                        y=df[col][:k+1],
                        mode='lines+markers',
                        name=col
                    )
                )
            frames.append(go.Frame(data=frame_data, name=str(k)))

        fig.frames = frames

        # Update layout
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis=dict(
                title=x_col,
                rangeslider=dict(visible=range_slider) if range_slider else None,
                type='date' if pd.api.types.is_datetime64_any_dtype(df[x_col]) else None
            ),
            yaxis=dict(title='Value'),
            hovermode='x unified',
            template=self.theme,
            height=500
        )

        # Add animation controls
        if play_button:
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        buttons=[
                            dict(
                                label="▶ Play",
                                method="animate",
                                args=[None, {
                                    "frame": {"duration": self.animation_duration, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 100}
                                }]
                            ),
                            dict(
                                label="⏸ Pause",
                                method="animate",
                                args=[[None], {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}
                                }]
                            )
                        ],
                        x=0.1,
                        y=1.15,
                        xanchor="left",
                        yanchor="top"
                    )
                ],
                sliders=[{
                    'steps': [
                        {
                            'args': [
                                [f.name],
                                {'frame': {'duration': self.animation_duration, 'redraw': True},
                                 'mode': 'immediate',
                                 'transition': {'duration': 100}}
                            ],
                            'label': str(k),
                            'method': 'animate'
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                    'active': 0,
                    'y': -0.1,
                    'len': 0.9,
                    'x': 0.05,
                    'xanchor': 'left',
                    'y': 0,
                    'yanchor': 'top',
                    'transition': {'duration': 100},
                    'currentvalue': {
                        'font': {'size': 12},
                        'prefix': 'Frame: ',
                        'visible': True,
                        'xanchor': 'center'
                    }
                }]
            )

        # Add accessibility features
        if self.accessibility_mode:
            fig.update_layout(
                font=dict(size=14),
                hoverlabel=dict(
                    font_size=14,
                    font_family="Arial"
                )
            )

        return fig

    def create_interactive_3d_scatter(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str,
        color_col: Optional[str] = None,
        size_col: Optional[str] = None,
        title: str = "3D Scatter Plot"
    ) -> go.Figure:
        """
        Create interactive 3D scatter plot.

        Args:
            df: DataFrame with data
            x_col, y_col, z_col: Column names for axes
            color_col: Optional column for color mapping
            size_col: Optional column for size mapping
            title: Chart title

        Returns:
            3D Plotly figure
        """
        # Prepare marker properties
        marker_dict = {
            'size': 5,
            'opacity': 0.8,
            'line': {'width': 0.5, 'color': 'white'}
        }

        if size_col:
            marker_dict['size'] = df[size_col]
            marker_dict['sizemode'] = 'diameter'
            marker_dict['sizeref'] = 2. * max(df[size_col]) / (40.**2)
            marker_dict['sizemin'] = 4

        if color_col:
            marker_dict['color'] = df[color_col]
            marker_dict['colorscale'] = 'Viridis'
            marker_dict['showscale'] = True
            marker_dict['colorbar'] = dict(
                title=color_col,
                thickness=15,
                len=0.7,
                x=1.02,
                y=0.5
            )

        # Create 3D scatter
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=df[x_col],
                    y=df[y_col],
                    z=df[z_col],
                    mode='markers',
                    marker=marker_dict,
                    text=df.index,
                    hovertemplate=f'<b>{x_col}</b>: %{{x:.2f}}<br>' +
                                  f'<b>{y_col}</b>: %{{y:.2f}}<br>' +
                                  f'<b>{z_col}</b>: %{{z:.2f}}<br>' +
                                  '<extra></extra>'
                )
            ]
        )

        # Update layout
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center'
            },
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col,
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            template=self.theme,
            height=600,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        return fig

    def create_sankey_diagram(
        self,
        df: pd.DataFrame,
        source_col: str,
        target_col: str,
        value_col: str,
        title: str = "Sankey Diagram"
    ) -> go.Figure:
        """
        Create interactive Sankey diagram for flow visualization.

        Args:
            df: DataFrame with flow data
            source_col: Source node column
            target_col: Target node column
            value_col: Flow value column
            title: Chart title

        Returns:
            Sankey Plotly figure
        """
        # Get unique nodes
        all_nodes = pd.concat([df[source_col], df[target_col]]).unique()
        node_dict = {node: i for i, node in enumerate(all_nodes)}

        # Create node colors
        node_colors = [self.color_palette[i % len(self.color_palette)]
                      for i in range(len(all_nodes))]

        # Prepare Sankey data
        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=all_nodes,
                        color=node_colors,
                        hovertemplate='<b>%{label}</b><br>' +
                                      'Total: %{value}<br>' +
                                      '<extra></extra>'
                    ),
                    link=dict(
                        source=[node_dict[s] for s in df[source_col]],
                        target=[node_dict[t] for t in df[target_col]],
                        value=df[value_col],
                        hovertemplate='%{source.label} → %{target.label}<br>' +
                                      'Value: %{value}<br>' +
                                      '<extra></extra>'
                    )
                )
            ]
        )

        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center'
            },
            font_size=12,
            template=self.theme,
            height=500
        )

        return fig

    def create_sunburst_chart(
        self,
        df: pd.DataFrame,
        path_cols: List[str],
        value_col: str,
        title: str = "Sunburst Chart"
    ) -> go.Figure:
        """
        Create interactive sunburst chart for hierarchical data.

        Args:
            df: DataFrame with hierarchical data
            path_cols: Columns defining the hierarchy path
            value_col: Column with values
            title: Chart title

        Returns:
            Sunburst Plotly figure
        """
        fig = px.sunburst(
            df,
            path=path_cols,
            values=value_col,
            title=title,
            color=value_col,
            color_continuous_scale='RdYlBu_r',
            hover_data={value_col: ':.2f'}
        )

        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>' +
                          'Value: %{value:.2f}<br>' +
                          'Percentage: %{percentRoot}<br>' +
                          '<extra></extra>',
            textinfo="label+percent parent"
        )

        fig.update_layout(
            height=600,
            template=self.theme
        )

        return fig

    def create_parallel_coordinates(
        self,
        df: pd.DataFrame,
        dimensions: List[str],
        color_col: Optional[str] = None,
        title: str = "Parallel Coordinates"
    ) -> go.Figure:
        """
        Create parallel coordinates plot for multi-dimensional data.

        Args:
            df: DataFrame with multi-dimensional data
            dimensions: List of dimension columns
            color_col: Optional column for color coding
            title: Chart title

        Returns:
            Parallel coordinates Plotly figure
        """
        # Prepare dimensions
        dims = []
        for dim in dimensions:
            if df[dim].dtype == 'object':
                # Categorical dimension
                vals = df[dim].unique()
                dims.append(
                    dict(
                        label=dim,
                        values=[list(vals).index(v) for v in df[dim]],
                        tickvals=list(range(len(vals))),
                        ticktext=vals
                    )
                )
            else:
                # Numerical dimension
                dims.append(
                    dict(
                        label=dim,
                        values=df[dim],
                        range=[df[dim].min(), df[dim].max()]
                    )
                )

        # Create figure
        line_dict = {}
        if color_col:
            line_dict = dict(
                color=df[color_col],
                colorscale='Viridis',
                showscale=True,
                cmin=df[color_col].min(),
                cmax=df[color_col].max()
            )

        fig = go.Figure(
            data=go.Parcoords(
                line=line_dict,
                dimensions=dims
            )
        )

        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center'
            },
            template=self.theme,
            height=500
        )

        return fig

    def create_bokeh_interactive_plot(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str = "Interactive Bokeh Plot"
    ) -> figure:
        """
        Create interactive Bokeh plot with tools and widgets.

        Args:
            df: DataFrame with data
            x_col: X-axis column
            y_col: Y-axis column
            title: Plot title

        Returns:
            Bokeh figure
        """
        # Create ColumnDataSource
        source = ColumnDataSource(df)

        # Create figure with tools
        tools = [
            PanTool(),
            WheelZoomTool(),
            BoxZoomTool(),
            ResetTool(),
            SaveTool(),
            CrosshairTool(),
            TapTool(),
            HoverTool(
                tooltips=[
                    (x_col, f"@{x_col}{{0.00}}"),
                    (y_col, f"@{y_col}{{0.00}}")
                ]
            )
        ]

        p = figure(
            title=title,
            x_axis_label=x_col,
            y_axis_label=y_col,
            tools=tools,
            width=800,
            height=400,
            toolbar_location="above"
        )

        # Add circle glyph
        p.circle(
            x_col, y_col,
            source=source,
            size=10,
            color="navy",
            alpha=0.7,
            hover_color="red",
            hover_alpha=1.0
        )

        # Add line
        p.line(
            x_col, y_col,
            source=source,
            line_width=2,
            color="darkblue",
            alpha=0.5
        )

        # Style the plot
        p.title.text_font_size = "16pt"
        p.xaxis.axis_label_text_font_size = "12pt"
        p.yaxis.axis_label_text_font_size = "12pt"

        # Add grid
        p.grid.grid_line_alpha = 0.3

        return p

    def create_data_storytelling_template(
        self,
        story_data: List[Dict[str, Any]],
        title: str = "Data Story"
    ) -> go.Figure:
        """
        Create a data storytelling template with multiple visualizations.

        Args:
            story_data: List of dictionaries with story elements
            title: Story title

        Returns:
            Multi-panel Plotly figure
        """
        # Calculate grid dimensions
        n_stories = len(story_data)
        n_cols = min(2, n_stories)
        n_rows = (n_stories + n_cols - 1) // n_cols

        # Create subplots
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[s.get('title', f'Story {i+1}')
                          for i, s in enumerate(story_data)],
            specs=[[{'type': s.get('type', 'scatter')}
                   for s in story_data[i*n_cols:(i+1)*n_cols]]
                  for i in range(n_rows)],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # Add each story element
        for idx, story in enumerate(story_data):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            if story.get('type') == 'bar':
                fig.add_trace(
                    go.Bar(
                        x=story['x'],
                        y=story['y'],
                        name=story.get('name', ''),
                        marker_color=story.get('color', self.color_palette[idx])
                    ),
                    row=row, col=col
                )
            else:  # Default to scatter
                fig.add_trace(
                    go.Scatter(
                        x=story['x'],
                        y=story['y'],
                        mode=story.get('mode', 'lines+markers'),
                        name=story.get('name', ''),
                        line=dict(color=story.get('color', self.color_palette[idx]))
                    ),
                    row=row, col=col
                )

            # Add annotations for storytelling
            if 'annotation' in story:
                fig.add_annotation(
                    text=story['annotation'],
                    xref=f"x{idx+1}",
                    yref=f"y{idx+1}",
                    x=story.get('ann_x', story['x'][len(story['x'])//2]),
                    y=story.get('ann_y', max(story['y'])),
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="black",
                    ax=20,
                    ay=-30,
                    bordercolor="black",
                    borderwidth=1,
                    bgcolor="white",
                    opacity=0.9
                )

        # Update layout
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            showlegend=True,
            template=self.theme,
            height=400 * n_rows,
            hovermode='closest'
        )

        return fig

    def create_accessibility_chart(
        self,
        df: pd.DataFrame,
        chart_type: str = "bar",
        title: str = "Accessible Chart",
        **kwargs
    ) -> go.Figure:
        """
        Create a chart with enhanced accessibility features.

        Args:
            df: DataFrame with data
            chart_type: Type of chart (bar, line, scatter)
            title: Chart title
            **kwargs: Additional chart parameters

        Returns:
            Accessible Plotly figure
        """
        # High contrast color palette for accessibility
        accessible_colors = [
            '#000000',  # Black
            '#0066CC',  # Blue
            '#CC0000',  # Red
            '#00CC00',  # Green
            '#CC6600',  # Orange
            '#6600CC',  # Purple
            '#CC0066',  # Magenta
            '#00CCCC',  # Cyan
        ]

        # Create base figure based on type
        if chart_type == "bar":
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=df.index,
                        y=df.iloc[:, 0],
                        marker_color=accessible_colors[0],
                        marker_pattern_shape="/"  # Add pattern for colorblind users
                    )
                ]
            )
        elif chart_type == "line":
            fig = go.Figure()
            for i, col in enumerate(df.columns):
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col],
                        mode='lines+markers',
                        name=col,
                        line=dict(
                            color=accessible_colors[i % len(accessible_colors)],
                            width=3,
                            dash=['solid', 'dash', 'dot'][i % 3]  # Different line styles
                        ),
                        marker=dict(
                            size=10,
                            symbol=['circle', 'square', 'diamond'][i % 3]  # Different markers
                        )
                    )
                )
        else:  # scatter
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=df.iloc[:, 0],
                        y=df.iloc[:, 1],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color=accessible_colors[0],
                            symbol='circle-open',
                            line=dict(width=2)
                        )
                    )
                ]
            )

        # Update layout for accessibility
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif'}
            },
            font=dict(
                size=14,
                family='Arial, sans-serif'
            ),
            xaxis=dict(
                title=kwargs.get('x_title', 'X Axis'),
                titlefont=dict(size=16),
                tickfont=dict(size=14),
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray',
                showline=True,
                linewidth=2,
                linecolor='Black'
            ),
            yaxis=dict(
                title=kwargs.get('y_title', 'Y Axis'),
                titlefont=dict(size=16),
                tickfont=dict(size=14),
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray',
                showline=True,
                linewidth=2,
                linecolor='Black'
            ),
            paper_bgcolor='white',
            plot_bgcolor='white',
            hovermode='closest',
            hoverlabel=dict(
                bgcolor="white",
                font_size=16,
                font_family="Arial, sans-serif",
                bordercolor="black"
            ),
            template=None,  # No template for maximum control
            height=500,
            # Add aria-label
            title_text=f"<span aria-label='{title} chart'>{title}</span>"
        )

        # Add text annotations for screen readers
        if kwargs.get('add_annotations', True):
            # Add data summary
            summary = f"Chart showing {len(df)} data points. "
            if chart_type == "bar":
                summary += f"Maximum value: {df.iloc[:, 0].max():.2f}, "
                summary += f"Minimum value: {df.iloc[:, 0].min():.2f}"

            fig.add_annotation(
                text=f"<span class='sr-only'>{summary}</span>",
                xref="paper",
                yref="paper",
                x=0,
                y=1.1,
                showarrow=False,
                font=dict(size=1)  # Invisible but present for screen readers
            )

        return fig

    def export_chart_with_description(
        self,
        fig: go.Figure,
        filename: str,
        alt_text: str,
        long_description: str
    ) -> Dict[str, str]:
        """
        Export chart with accessibility descriptions.

        Args:
            fig: Plotly figure to export
            filename: Output filename
            alt_text: Alternative text for the image
            long_description: Detailed description of the chart

        Returns:
            Dictionary with file paths and descriptions
        """
        # Export as static image
        img_path = f"{filename}.png"
        fig.write_image(img_path)

        # Export as interactive HTML with ARIA labels
        html_path = f"{filename}.html"
        html_string = fig.to_html(
            include_plotlyjs='cdn',
            div_id="accessible-chart",
            config={'displayModeBar': True}
        )

        # Add ARIA labels and descriptions
        accessible_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{alt_text}</title>
        </head>
        <body>
            <div role="img" aria-label="{alt_text}">
                <p class="sr-only">{long_description}</p>
                {html_string}
            </div>
            <details>
                <summary>Chart Description</summary>
                <p>{long_description}</p>
            </details>
        </body>
        </html>
        """

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(accessible_html)

        # Create text description file
        txt_path = f"{filename}_description.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Alternative Text: {alt_text}\n\n")
            f.write(f"Detailed Description:\n{long_description}\n")

        logger.info(f"Exported accessible chart to {filename}")

        return {
            'image': img_path,
            'html': html_path,
            'description': txt_path,
            'alt_text': alt_text,
            'long_description': long_description
        }


# Example usage and demo
def create_demo_visualizations():
    """Create demo visualizations showcasing all features."""

    # Initialize visualization component
    viz = InteractiveVisualizations()

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=30)

    df_time = pd.DataFrame({
        'date': dates,
        'metric_1': np.cumsum(np.random.randn(30)) + 100,
        'metric_2': np.cumsum(np.random.randn(30)) + 150,
        'metric_3': np.cumsum(np.random.randn(30)) + 200
    })

    # Create animated time series
    fig_animated = viz.create_animated_time_series(
        df_time,
        x_col='date',
        y_cols=['metric_1', 'metric_2', 'metric_3'],
        title='Animated Metrics Over Time'
    )

    # Create 3D scatter
    df_3d = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100),
        'z': np.random.randn(100),
        'value': np.random.uniform(0, 100, 100)
    })

    fig_3d = viz.create_interactive_3d_scatter(
        df_3d,
        x_col='x',
        y_col='y',
        z_col='z',
        color_col='value',
        title='3D Scatter Visualization'
    )

    # Create Sankey diagram
    df_flow = pd.DataFrame({
        'source': ['A', 'A', 'B', 'B', 'C'],
        'target': ['X', 'Y', 'X', 'Z', 'Y'],
        'value': [10, 15, 20, 10, 25]
    })

    fig_sankey = viz.create_sankey_diagram(
        df_flow,
        source_col='source',
        target_col='target',
        value_col='value',
        title='Flow Visualization'
    )

    # Create data story
    story_data = [
        {
            'title': 'Revenue Growth',
            'x': list(range(10)),
            'y': np.cumsum(np.random.uniform(5, 15, 10)),
            'type': 'scatter',
            'annotation': 'Strong growth trend',
            'ann_x': 7,
            'ann_y': 80
        },
        {
            'title': 'User Segments',
            'x': ['Segment A', 'Segment B', 'Segment C'],
            'y': [45, 30, 25],
            'type': 'bar',
            'annotation': 'Segment A dominates'
        }
    ]

    fig_story = viz.create_data_storytelling_template(
        story_data,
        title='Q1 2024 Business Overview'
    )

    # Create accessible chart
    df_access = pd.DataFrame({
        'Category': ['A', 'B', 'C', 'D'],
        'Value': [23, 45, 12, 67]
    })

    fig_accessible = viz.create_accessibility_chart(
        df_access.set_index('Category'),
        chart_type='bar',
        title='Category Performance',
        x_title='Categories',
        y_title='Performance Score'
    )

    return {
        'animated': fig_animated,
        '3d_scatter': fig_3d,
        'sankey': fig_sankey,
        'story': fig_story,
        'accessible': fig_accessible
    }


if __name__ == "__main__":
    # Create demo visualizations
    demo_figs = create_demo_visualizations()

    # Show one of the figures
    demo_figs['animated'].show()

    logger.info("Demo visualizations created successfully")