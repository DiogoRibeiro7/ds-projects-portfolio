"""
Example Dashboard Application

Complete example demonstrating all enhanced dashboard features including
real-time data, interactive visualizations, export capabilities, and API integration.

Author: Portfolio Team
Date: 2024
"""

import logging
import threading
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from flask import Flask

# Import our custom components
from dashboard_framework import EnhancedDashboard, DashboardConfig
from visualization_components import InteractiveVisualizations
from api_infrastructure import DashboardAPI, APIConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioDashboard:
    """
    Complete example dashboard showcasing all features:
    - Real-time streaming data
    - Interactive visualizations
    - Export functionality
    - Mobile-responsive design
    - Dark mode support
    - API integration
    """

    def __init__(self):
        """Initialize the portfolio dashboard."""

        # Configure dashboard
        self.dashboard_config = DashboardConfig(
            app_name="Data Science Portfolio Dashboard",
            enable_realtime=True,
            enable_export=True,
            enable_filtering=True,
            enable_dark_mode=True,
            enable_mobile=True,
            enable_accessibility=True,
            data_refresh_interval=5000
        )

        # Configure API
        self.api_config = APIConfig(
            host="0.0.0.0",
            port=5000,
            api_title="Portfolio Dashboard API",
            api_version="v1"
        )

        # Initialize components
        self.dashboard = EnhancedDashboard(self.dashboard_config)
        self.visualizations = InteractiveVisualizations()
        self.api = DashboardAPI(self.api_config)

        # Sample data storage
        self.data_store = {}

        # Initialize data sources
        self._setup_data_sources()

        # Build custom layout
        self._build_custom_layout()

        # Setup callbacks
        self._setup_custom_callbacks()

        logger.info("Portfolio Dashboard initialized")

    def _setup_data_sources(self):
        """Set up data sources for the dashboard."""

        # Register real-time metrics data source
        def get_realtime_metrics(filters=None):
            """Generate real-time metrics data."""
            current_time = datetime.now()
            time_range = pd.date_range(
                end=current_time,
                periods=100,
                freq='1min'
            )

            data = pd.DataFrame({
                'timestamp': time_range,
                'cpu_usage': np.random.uniform(20, 80, 100) + np.sin(np.linspace(0, 4*np.pi, 100)) * 10,
                'memory_usage': np.random.uniform(30, 70, 100) + np.cos(np.linspace(0, 4*np.pi, 100)) * 5,
                'network_traffic': np.random.uniform(100, 500, 100),
                'active_users': np.random.poisson(50, 100),
                'response_time': np.random.exponential(50, 100)
            })

            if filters:
                # Apply filters if provided
                if 'start_date' in filters and 'end_date' in filters:
                    mask = (data['timestamp'] >= filters['start_date']) & \
                           (data['timestamp'] <= filters['end_date'])
                    data = data[mask]

            return data

        # Register ML model performance data source
        def get_model_performance(filters=None):
            """Get ML model performance metrics."""
            models = ['XGBoost', 'LightGBM', 'CatBoost', 'Neural Network', 'Random Forest']
            metrics = pd.DataFrame({
                'model': models,
                'accuracy': [0.92, 0.91, 0.93, 0.89, 0.88],
                'precision': [0.89, 0.88, 0.91, 0.86, 0.85],
                'recall': [0.94, 0.93, 0.92, 0.91, 0.90],
                'f1_score': [0.91, 0.90, 0.91, 0.88, 0.87],
                'auc_roc': [0.95, 0.94, 0.96, 0.92, 0.91],
                'training_time': [12.5, 8.3, 15.2, 45.6, 10.1]
            })
            return metrics

        # Register business KPIs data source
        def get_business_kpis(filters=None):
            """Get business KPI data."""
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            kpis = pd.DataFrame({
                'date': dates,
                'revenue': np.cumsum(np.random.uniform(10000, 50000, 30)),
                'customers': np.cumsum(np.random.poisson(100, 30)),
                'conversion_rate': np.random.uniform(0.02, 0.05, 30),
                'churn_rate': np.random.uniform(0.05, 0.10, 30),
                'avg_order_value': np.random.uniform(100, 200, 30)
            })
            return kpis

        # Register customer segmentation data
        def get_customer_segments(filters=None):
            """Get customer segmentation data."""
            segments = pd.DataFrame({
                'segment': ['Premium', 'Regular', 'Basic', 'Inactive'],
                'count': [1200, 3500, 2800, 500],
                'avg_revenue': [5000, 2000, 500, 0],
                'avg_lifetime_value': [15000, 6000, 1500, 0],
                'churn_probability': [0.05, 0.10, 0.20, 0.80]
            })
            return segments

        # Register data sources
        self.dashboard.register_data_source('realtime_metrics', get_realtime_metrics)
        self.dashboard.register_data_source('model_performance', get_model_performance)
        self.dashboard.register_data_source('business_kpis', get_business_kpis)
        self.dashboard.register_data_source('customer_segments', get_customer_segments)

    def _build_custom_layout(self):
        """Build custom dashboard layout."""

        # Override the dashboard layout with custom design
        self.dashboard.app.layout = html.Div([
            # Header with navigation
            dbc.NavbarSimple(
                children=[
                    dbc.NavItem(dbc.NavLink("Overview", href="/", active="exact")),
                    dbc.NavItem(dbc.NavLink("ML Models", href="/models", active="exact")),
                    dbc.NavItem(dbc.NavLink("Analytics", href="/analytics", active="exact")),
                    dbc.NavItem(dbc.NavLink("Real-time", href="/realtime", active="exact")),
                    dbc.DropdownMenu(
                        children=[
                            dbc.DropdownMenuItem("Export PDF", id="export-pdf-btn"),
                            dbc.DropdownMenuItem("Export PowerPoint", id="export-pptx-btn"),
                            dbc.DropdownMenuItem("Export Data", id="export-data-btn"),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("API Docs", href="/api/docs", external_link=True),
                        ],
                        nav=True,
                        in_navbar=True,
                        label="Export & Tools",
                    ),
                ],
                brand="📊 Data Science Portfolio",
                brand_href="/",
                color="primary",
                dark=True,
                fluid=True,
                id="navbar",
                className="mb-3"
            ),

            # Dark mode toggle
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        dbc.Switch(
                            id="dark-mode-switch",
                            label="🌙 Dark Mode",
                            value=False,
                            className="float-end"
                        )
                    ], width=12)
                ], className="mb-2")
            ], fluid=True),

            # Main content area
            dcc.Location(id="url", refresh=False),
            dbc.Container([
                # Alert for notifications
                html.Div(id="notification-area"),

                # Filter panel (collapsible)
                dbc.Collapse([
                    dbc.Card([
                        dbc.CardHeader("🔍 Filters"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Date Range"),
                                    dcc.DatePickerRange(
                                        id="date-filter",
                                        start_date=datetime.now() - timedelta(days=30),
                                        end_date=datetime.now(),
                                        display_format='YYYY-MM-DD',
                                        className="w-100"
                                    )
                                ], md=4),
                                dbc.Col([
                                    dbc.Label("Model Type"),
                                    dcc.Dropdown(
                                        id="model-filter",
                                        options=[
                                            {"label": "All Models", "value": "all"},
                                            {"label": "Tree-based", "value": "tree"},
                                            {"label": "Neural Networks", "value": "nn"},
                                            {"label": "Linear", "value": "linear"}
                                        ],
                                        value="all",
                                        className="w-100"
                                    )
                                ], md=4),
                                dbc.Col([
                                    dbc.Label("Metric"),
                                    dcc.Dropdown(
                                        id="metric-filter",
                                        options=[
                                            {"label": "Accuracy", "value": "accuracy"},
                                            {"label": "AUC-ROC", "value": "auc_roc"},
                                            {"label": "F1 Score", "value": "f1_score"}
                                        ],
                                        value="accuracy",
                                        className="w-100"
                                    )
                                ], md=4),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button(
                                        "Apply Filters",
                                        id="apply-filters",
                                        color="primary",
                                        className="mt-3 w-100"
                                    )
                                ], md=12)
                            ])
                        ])
                    ], className="mb-3")
                ], id="filter-collapse", is_open=False),

                # Toggle button for filters
                dbc.Button(
                    "🔽 Show Filters",
                    id="toggle-filters",
                    color="secondary",
                    size="sm",
                    className="mb-3"
                ),

                # Page content
                html.Div(id="page-content"),

                # Footer
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.P([
                            "© 2024 Data Science Portfolio | ",
                            html.A("API Documentation", href="/api/docs"),
                            " | ",
                            html.A("GitHub", href="https://github.com"),
                            " | Real-time updates enabled"
                        ], className="text-center text-muted small")
                    ])
                ])
            ], fluid=True),

            # Interval components for real-time updates
            dcc.Interval(id='interval-fast', interval=1000, n_intervals=0),  # 1 second
            dcc.Interval(id='interval-medium', interval=5000, n_intervals=0),  # 5 seconds
            dcc.Interval(id='interval-slow', interval=30000, n_intervals=0),  # 30 seconds

            # Store components
            dcc.Store(id='theme-store', storage_type='local'),
            dcc.Store(id='filter-store', storage_type='session'),
            dcc.Store(id='realtime-store', storage_type='memory'),
        ], id="main-container")

    def _setup_custom_callbacks(self):
        """Set up custom callbacks for interactivity."""

        # Page routing callback
        @self.dashboard.app.callback(
            Output('page-content', 'children'),
            [Input('url', 'pathname')]
        )
        def display_page(pathname):
            if pathname == '/models':
                return self._create_models_page()
            elif pathname == '/analytics':
                return self._create_analytics_page()
            elif pathname == '/realtime':
                return self._create_realtime_page()
            else:
                return self._create_overview_page()

        # Dark mode toggle
        @self.dashboard.app.callback(
            [Output('main-container', 'style'),
             Output('theme-store', 'data')],
            [Input('dark-mode-switch', 'value')],
            [State('theme-store', 'data')]
        )
        def toggle_dark_mode(dark_mode, stored_theme):
            if dark_mode:
                style = {
                    'backgroundColor': '#1e1e1e',
                    'color': '#ffffff',
                    'minHeight': '100vh'
                }
            else:
                style = {
                    'backgroundColor': '#ffffff',
                    'color': '#212529',
                    'minHeight': '100vh'
                }
            return style, dark_mode

        # Filter collapse toggle
        @self.dashboard.app.callback(
            [Output("filter-collapse", "is_open"),
             Output("toggle-filters", "children")],
            [Input("toggle-filters", "n_clicks")],
            [State("filter-collapse", "is_open")]
        )
        def toggle_filters(n, is_open):
            if n:
                is_open = not is_open
            btn_text = "🔼 Hide Filters" if is_open else "🔽 Show Filters"
            return is_open, btn_text

        # Export callbacks
        @self.dashboard.app.callback(
            Output('notification-area', 'children'),
            [Input('export-pdf-btn', 'n_clicks'),
             Input('export-pptx-btn', 'n_clicks'),
             Input('export-data-btn', 'n_clicks')]
        )
        def handle_export(pdf_clicks, pptx_clicks, data_clicks):
            ctx = callback_context
            if not ctx.triggered:
                return ""

            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if button_id == 'export-pdf-btn':
                # Export to PDF
                figures = self._collect_current_figures()
                filepath = self.dashboard.export_to_pdf(figures, "dashboard_export.pdf")
                return dbc.Alert(
                    f"✅ PDF exported successfully to {filepath}",
                    color="success",
                    dismissable=True,
                    duration=4000
                )
            elif button_id == 'export-pptx-btn':
                # Export to PowerPoint
                figures = self._collect_current_figures()
                filepath = self.dashboard.export_to_powerpoint(figures, "dashboard_export.pptx")
                return dbc.Alert(
                    f"✅ PowerPoint exported successfully to {filepath}",
                    color="success",
                    dismissable=True,
                    duration=4000
                )
            elif button_id == 'export-data-btn':
                # Export data to Excel
                self._export_all_data()
                return dbc.Alert(
                    "✅ Data exported successfully to dashboard_data.xlsx",
                    color="success",
                    dismissable=True,
                    duration=4000
                )

            return ""

    def _create_overview_page(self):
        """Create the overview page."""
        # Get data
        kpis = self.dashboard.get_data('business_kpis')
        segments = self.dashboard.get_data('customer_segments')

        # Calculate KPI values
        total_revenue = kpis['revenue'].iloc[-1]
        total_customers = kpis['customers'].iloc[-1]
        avg_conversion = kpis['conversion_rate'].mean()
        avg_churn = kpis['churn_rate'].mean()

        # Create KPI cards
        kpi_cards = dbc.Row([
            dbc.Col([
                self.dashboard.create_kpi_card(
                    "Total Revenue",
                    f"${total_revenue:,.0f}",
                    change=12.5,
                    icon="fas fa-dollar-sign"
                )
            ], xs=12, sm=6, md=3),
            dbc.Col([
                self.dashboard.create_kpi_card(
                    "Total Customers",
                    f"{total_customers:,}",
                    change=8.3,
                    icon="fas fa-users"
                )
            ], xs=12, sm=6, md=3),
            dbc.Col([
                self.dashboard.create_kpi_card(
                    "Avg Conversion",
                    f"{avg_conversion:.2%}",
                    change=-2.1,
                    icon="fas fa-chart-line"
                )
            ], xs=12, sm=6, md=3),
            dbc.Col([
                self.dashboard.create_kpi_card(
                    "Avg Churn Rate",
                    f"{avg_churn:.2%}",
                    change=-5.2,
                    icon="fas fa-user-slash"
                )
            ], xs=12, sm=6, md=3),
        ])

        # Create revenue trend chart
        revenue_fig = px.area(
            kpis,
            x='date',
            y='revenue',
            title='Revenue Trend',
            labels={'revenue': 'Revenue ($)', 'date': 'Date'}
        )
        revenue_fig.update_layout(height=400)

        # Create customer segmentation sunburst
        segments_fig = self.visualizations.create_sunburst_chart(
            segments,
            path_cols=['segment'],
            value_col='count',
            title='Customer Segmentation'
        )

        # Create conversion funnel
        funnel_data = pd.DataFrame({
            'stage': ['Visitors', 'Leads', 'Prospects', 'Customers'],
            'count': [10000, 3000, 1000, 300]
        })
        funnel_fig = px.funnel(
            funnel_data,
            x='count',
            y='stage',
            title='Conversion Funnel'
        )

        return html.Div([
            html.H2("📊 Dashboard Overview"),
            html.Hr(),
            kpi_cards,
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=revenue_fig, id='revenue-chart')
                ], md=8),
                dbc.Col([
                    dcc.Graph(figure=segments_fig, id='segments-chart')
                ], md=4)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=funnel_fig, id='funnel-chart')
                ], md=12)
            ])
        ])

    def _create_models_page(self):
        """Create the ML models page."""
        # Get model performance data
        models_df = self.dashboard.get_data('model_performance')

        # Create model comparison radar chart
        categories = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']

        radar_fig = go.Figure()
        for _, model in models_df.iterrows():
            radar_fig.add_trace(go.Scatterpolar(
                r=[model[cat] for cat in categories],
                theta=categories,
                fill='toself',
                name=model['model']
            ))

        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0.8, 1.0]
                )
            ),
            showlegend=True,
            title="Model Performance Comparison"
        )

        # Create training time comparison
        time_fig = px.bar(
            models_df,
            x='model',
            y='training_time',
            title='Model Training Time Comparison',
            labels={'training_time': 'Time (seconds)'}
        )

        # Create performance heatmap
        perf_matrix = models_df[categories].T
        perf_matrix.columns = models_df['model']

        heatmap_fig = go.Figure(data=go.Heatmap(
            z=perf_matrix.values,
            x=perf_matrix.columns,
            y=perf_matrix.index,
            colorscale='RdYlGn',
            text=perf_matrix.values,
            texttemplate='%{text:.3f}',
            textfont={"size": 12}
        ))
        heatmap_fig.update_layout(title="Performance Metrics Heatmap")

        return html.Div([
            html.H2("🤖 ML Model Performance"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=radar_fig, id='radar-chart')
                ], md=6),
                dbc.Col([
                    dcc.Graph(figure=time_fig, id='time-chart')
                ], md=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=heatmap_fig, id='heatmap-chart')
                ], md=12)
            ])
        ])

    def _create_analytics_page(self):
        """Create the analytics page."""
        # Get business KPIs data
        kpis_df = self.dashboard.get_data('business_kpis')

        # Create animated time series
        animated_fig = self.visualizations.create_animated_time_series(
            kpis_df,
            x_col='date',
            y_cols=['revenue', 'customers'],
            title='Animated Business Metrics'
        )

        # Create 3D scatter plot
        scatter_3d_data = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
            'z': np.random.randn(100),
            'cluster': np.random.randint(0, 3, 100)
        })

        scatter_3d_fig = self.visualizations.create_interactive_3d_scatter(
            scatter_3d_data,
            x_col='x',
            y_col='y',
            z_col='z',
            color_col='cluster',
            title='3D Customer Clustering'
        )

        # Create parallel coordinates
        parallel_data = pd.DataFrame({
            'Revenue': np.random.uniform(1000, 10000, 50),
            'Customers': np.random.uniform(10, 100, 50),
            'Conversion': np.random.uniform(0.01, 0.05, 50),
            'Satisfaction': np.random.uniform(1, 5, 50),
            'Segment': np.random.choice(['Premium', 'Regular', 'Basic'], 50)
        })

        parallel_fig = self.visualizations.create_parallel_coordinates(
            parallel_data,
            dimensions=['Revenue', 'Customers', 'Conversion', 'Satisfaction'],
            color_col='Revenue',
            title='Multi-dimensional Analysis'
        )

        return html.Div([
            html.H2("📈 Advanced Analytics"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=animated_fig, id='animated-chart')
                ], md=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=scatter_3d_fig, id='3d-scatter')
                ], md=6),
                dbc.Col([
                    dcc.Graph(figure=parallel_fig, id='parallel-coords')
                ], md=6)
            ])
        ])

    def _create_realtime_page(self):
        """Create the real-time monitoring page."""
        return html.Div([
            html.H2("⚡ Real-time Monitoring"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("System Metrics"),
                        dbc.CardBody([
                            dcc.Graph(id='realtime-metrics-chart')
                        ])
                    ])
                ], md=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Live Statistics"),
                        dbc.CardBody([
                            html.Div(id='live-stats')
                        ])
                    ])
                ], md=4)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Activity Feed"),
                        dbc.CardBody([
                            html.Div(id='activity-feed')
                        ])
                    ])
                ], md=12)
            ])
        ])

    def setup_realtime_callbacks(self):
        """Set up real-time update callbacks."""

        @self.dashboard.app.callback(
            Output('realtime-metrics-chart', 'figure'),
            [Input('interval-fast', 'n_intervals')]
        )
        def update_realtime_metrics(n):
            # Get latest metrics
            metrics = self.dashboard.get_data('realtime_metrics')
            latest = metrics.tail(50)

            # Create figure with subplots
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('CPU Usage', 'Memory Usage', 'Network Traffic', 'Response Time')
            )

            # CPU Usage
            fig.add_trace(
                go.Scatter(x=latest['timestamp'], y=latest['cpu_usage'],
                          mode='lines', name='CPU', line=dict(color='#FF6B6B')),
                row=1, col=1
            )

            # Memory Usage
            fig.add_trace(
                go.Scatter(x=latest['timestamp'], y=latest['memory_usage'],
                          mode='lines', name='Memory', line=dict(color='#4ECDC4')),
                row=1, col=2
            )

            # Network Traffic
            fig.add_trace(
                go.Scatter(x=latest['timestamp'], y=latest['network_traffic'],
                          mode='lines', name='Network', line=dict(color='#45B7D1')),
                row=2, col=1
            )

            # Response Time
            fig.add_trace(
                go.Scatter(x=latest['timestamp'], y=latest['response_time'],
                          mode='lines', name='Response', line=dict(color='#96CEB4')),
                row=2, col=2
            )

            fig.update_layout(height=600, showlegend=False)
            return fig

        @self.dashboard.app.callback(
            Output('live-stats', 'children'),
            [Input('interval-medium', 'n_intervals')]
        )
        def update_live_stats(n):
            # Get latest metrics
            metrics = self.dashboard.get_data('realtime_metrics')
            latest_row = metrics.iloc[-1]

            stats = html.Div([
                html.H5("Current Metrics", className="mb-3"),
                html.Div([
                    html.P([
                        html.Strong("CPU: "),
                        f"{latest_row['cpu_usage']:.1f}%"
                    ], className="mb-2"),
                    html.P([
                        html.Strong("Memory: "),
                        f"{latest_row['memory_usage']:.1f}%"
                    ], className="mb-2"),
                    html.P([
                        html.Strong("Active Users: "),
                        f"{latest_row['active_users']:,}"
                    ], className="mb-2"),
                    html.P([
                        html.Strong("Avg Response: "),
                        f"{latest_row['response_time']:.2f}ms"
                    ], className="mb-2"),
                ]),
                html.Hr(),
                html.Small(f"Updated: {latest_row['timestamp'].strftime('%H:%M:%S')}")
            ])

            return stats

        @self.dashboard.app.callback(
            Output('activity-feed', 'children'),
            [Input('interval-slow', 'n_intervals')]
        )
        def update_activity_feed(n):
            # Generate sample activity feed
            activities = [
                f"🟢 New user registration at {datetime.now().strftime('%H:%M:%S')}",
                f"📊 Report generated for Q1 2024",
                f"🚀 Model deployment completed successfully",
                f"⚠️ Anomaly detected in network traffic",
                f"✅ Backup completed successfully"
            ]

            feed_items = [
                html.Div([
                    html.P(activity, className="mb-1"),
                    html.Hr(className="my-1")
                ]) for activity in activities[:5]
            ]

            return html.Div(feed_items)

    def _collect_current_figures(self):
        """Collect all current figures for export."""
        figures = []

        # Get sample figures from different pages
        kpis = self.dashboard.get_data('business_kpis')
        revenue_fig = px.area(
            kpis, x='date', y='revenue',
            title='Revenue Trend'
        )
        figures.append(revenue_fig)

        models = self.dashboard.get_data('model_performance')
        model_fig = px.bar(
            models, x='model', y='accuracy',
            title='Model Accuracy Comparison'
        )
        figures.append(model_fig)

        return figures

    def _export_all_data(self):
        """Export all dashboard data to Excel."""
        with pd.ExcelWriter('dashboard_data.xlsx', engine='xlsxwriter') as writer:
            # Export each data source
            for source_name in ['business_kpis', 'model_performance', 'customer_segments']:
                df = self.dashboard.get_data(source_name)
                df.to_excel(writer, sheet_name=source_name, index=False)

            # Add metadata sheet
            metadata = pd.DataFrame({
                'Property': ['Export Date', 'Dashboard Version', 'Data Sources'],
                'Value': [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '1.0.0', '4']
            })
            metadata.to_excel(writer, sheet_name='Metadata', index=False)

    def run(self):
        """Run the complete dashboard application."""
        # Set up real-time callbacks
        self.setup_realtime_callbacks()

        # Start API server in a separate thread
        api_thread = threading.Thread(target=self.api.run, daemon=True)
        api_thread.start()

        # Run dashboard
        logger.info("="*60)
        logger.info("PORTFOLIO DASHBOARD STARTING")
        logger.info("="*60)
        logger.info(f"Dashboard URL: http://localhost:8050")
        logger.info(f"API URL: http://localhost:5000")
        logger.info(f"API Docs: http://localhost:5000/api/docs")
        logger.info("="*60)

        self.dashboard.run()


if __name__ == "__main__":
    # Create and run the example dashboard
    portfolio = PortfolioDashboard()
    portfolio.run()