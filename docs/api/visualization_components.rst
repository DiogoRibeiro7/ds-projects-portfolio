Visualization Components
========================

.. currentmodule:: dashboard_enhanced

Reusable visualization components for interactive dashboards.

Visualization Components Module
--------------------------------

.. automodule:: dashboard_enhanced.visualization_components
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Key Components
--------------

ChartBuilder
~~~~~~~~~~~~

.. autoclass:: dashboard_enhanced.visualization_components.ChartBuilder
   :members:
   :special-members: __init__
   :show-inheritance:

InteractiveDashboard
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dashboard_enhanced.visualization_components.InteractiveDashboard
   :members:
   :special-members: __init__
   :show-inheritance:

Usage Examples
--------------

Creating Charts
~~~~~~~~~~~~~~~

.. code-block:: python

    from dashboard_enhanced.visualization_components import ChartBuilder

    # Initialize chart builder
    builder = ChartBuilder()

    # Create line chart
    line_chart = builder.line_chart(
        data=df,
        x='date',
        y='metric',
        title='Metric Over Time',
        color='category'
    )

    # Create bar chart
    bar_chart = builder.bar_chart(
        data=df,
        x='category',
        y='value',
        title='Values by Category',
        orientation='horizontal'
    )

    # Create heatmap
    heatmap = builder.heatmap(
        data=correlation_matrix,
        title='Feature Correlations',
        colorscale='RdBu'
    )

Interactive Dashboard
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from dashboard_enhanced.visualization_components import InteractiveDashboard

    # Initialize dashboard
    dashboard = InteractiveDashboard(title='Analytics Dashboard')

    # Add components
    dashboard.add_chart(
        chart=line_chart,
        position='top-left',
        width=6
    )

    dashboard.add_chart(
        chart=bar_chart,
        position='top-right',
        width=6
    )

    dashboard.add_filter(
        column='date',
        type='date_range',
        default_range=('2024-01-01', '2024-12-31')
    )

    # Render dashboard
    dashboard.render(port=8050)