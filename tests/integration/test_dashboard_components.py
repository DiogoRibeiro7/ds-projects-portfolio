"""
Test suite for dashboard components.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash import Dash, html, dcc
from dash.testing import DashComposite
from dashboard_enhanced.dashboard_framework import (
    DashboardBase,
    InteractiveDashboard,
    RealtimeDashboard,
    ChartBuilder,
    DataConnector,
)
from dashboard_enhanced.app import create_app
from dashboard_enhanced.graphql_api import schema, Query

pytestmark = pytest.mark.integration


class TestDashboardBase:
    """Test suite for base dashboard functionality."""

    @pytest.fixture
    def dashboard(self):
        """Create dashboard instance."""
        return DashboardBase(name="test_dashboard")

    def test_dashboard_initialization(self, dashboard):
        """Test dashboard initialization."""
        assert dashboard.name == "test_dashboard"
        assert dashboard.app is not None
        assert isinstance(dashboard.app, Dash)

    def test_add_component(self, dashboard):
        """Test adding components to dashboard."""
        component = html.Div("Test Component")
        dashboard.add_component(component, "test_div")

        assert "test_div" in dashboard.components
        assert dashboard.components["test_div"] == component

    def test_create_layout(self, dashboard):
        """Test layout creation."""
        dashboard.add_component(html.H1("Title"), "title")
        dashboard.add_component(dcc.Graph(id="graph"), "graph")

        layout = dashboard.create_layout()

        assert layout is not None
        assert len(layout.children) == 2

    def test_add_callback(self, dashboard):
        """Test callback registration."""

        @dashboard.callback(output="graph-output", inputs=["input-value"])
        def update_graph(value):
            return {"data": [{"x": [1, 2], "y": [value, value * 2]}]}

        assert len(dashboard.callbacks) == 1

    def test_theme_configuration(self, dashboard):
        """Test theme configuration."""
        dashboard.set_theme(
            {
                "primary_color": "#007bff",
                "background_color": "#f8f9fa",
                "font_family": "Arial",
            }
        )

        assert dashboard.theme["primary_color"] == "#007bff"


class TestInteractiveDashboard:
    """Test suite for interactive dashboard features."""

    @pytest.fixture
    def interactive_dashboard(self):
        """Create interactive dashboard."""
        return InteractiveDashboard(name="interactive_test")

    def test_add_filter(self, interactive_dashboard):
        """Test adding filters."""
        interactive_dashboard.add_filter(
            filter_id="date-filter",
            filter_type="date_range",
            options={"start_date": "2024-01-01", "end_date": "2024-12-31"},
        )

        assert "date-filter" in interactive_dashboard.filters
        assert interactive_dashboard.filters["date-filter"]["type"] == "date_range"

    def test_add_dropdown(self, interactive_dashboard):
        """Test adding dropdown menu."""
        interactive_dashboard.add_dropdown(
            dropdown_id="metric-dropdown",
            options=[
                {"label": "Revenue", "value": "revenue"},
                {"label": "Users", "value": "users"},
            ],
            default_value="revenue",
        )

        assert "metric-dropdown" in interactive_dashboard.dropdowns

    def test_cross_filtering(self, interactive_dashboard):
        """Test cross-filtering between components."""
        interactive_dashboard.enable_cross_filtering(["chart1", "chart2", "table1"])

        assert interactive_dashboard.cross_filtering_enabled
        assert len(interactive_dashboard.cross_filter_components) == 3

    def test_export_functionality(self, interactive_dashboard):
        """Test data export functionality."""
        data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        with patch("pandas.DataFrame.to_csv") as mock_csv:
            interactive_dashboard.export_data(data, format="csv")
            mock_csv.assert_called_once()

        with patch("pandas.DataFrame.to_excel") as mock_excel:
            interactive_dashboard.export_data(data, format="excel")
            mock_excel.assert_called_once()


class TestRealtimeDashboard:
    """Test suite for real-time dashboard functionality."""

    @pytest.fixture
    def realtime_dashboard(self):
        """Create real-time dashboard."""
        return RealtimeDashboard(
            name="realtime_test",
            update_interval=1000,  # 1 second
        )

    def test_websocket_connection(self, realtime_dashboard):
        """Test WebSocket connection setup."""
        with patch("asyncio.create_task") as mock_task:
            realtime_dashboard.start_websocket_server()
            mock_task.assert_called()

    def test_data_streaming(self, realtime_dashboard):
        """Test data streaming functionality."""
        mock_data_source = Mock()
        mock_data_source.get_data.return_value = {"value": 42}

        realtime_dashboard.add_data_stream(
            stream_id="metrics", data_source=mock_data_source, update_interval=500
        )

        assert "metrics" in realtime_dashboard.data_streams

    def test_auto_refresh(self, realtime_dashboard):
        """Test auto-refresh mechanism."""
        component = dcc.Graph(id="live-graph")
        realtime_dashboard.add_auto_refresh_component(component, refresh_interval=1000)

        assert component in realtime_dashboard.auto_refresh_components

    def test_buffer_management(self, realtime_dashboard):
        """Test data buffer management."""
        realtime_dashboard.set_buffer_size(100)

        for i in range(150):
            realtime_dashboard.add_to_buffer("test", i)

        buffer_data = realtime_dashboard.get_buffer("test")
        assert len(buffer_data) == 100
        assert buffer_data[-1] == 149


class TestChartBuilder:
    """Test suite for chart building utilities."""

    @pytest.fixture
    def chart_builder(self):
        """Create chart builder instance."""
        return ChartBuilder()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for charts."""
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=30),
                "value": np.random.randn(30).cumsum() + 100,
                "category": np.random.choice(["A", "B", "C"], 30),
            }
        )

    def test_line_chart(self, chart_builder, sample_data):
        """Test line chart creation."""
        fig = chart_builder.create_line_chart(
            data=sample_data, x="date", y="value", title="Test Line Chart"
        )

        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Test Line Chart"

    def test_bar_chart(self, chart_builder, sample_data):
        """Test bar chart creation."""
        fig = chart_builder.create_bar_chart(
            data=sample_data.groupby("category")["value"].sum().reset_index(),
            x="category",
            y="value",
            title="Test Bar Chart",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_scatter_plot(self, chart_builder, sample_data):
        """Test scatter plot creation."""
        fig = chart_builder.create_scatter(
            data=sample_data,
            x="date",
            y="value",
            color="category",
            size="value",
            title="Test Scatter",
        )

        assert isinstance(fig, go.Figure)
        assert fig.data[0].mode == "markers"

    def test_heatmap(self, chart_builder):
        """Test heatmap creation."""
        correlation_matrix = pd.DataFrame(
            np.random.rand(5, 5),
            columns=["A", "B", "C", "D", "E"],
            index=["A", "B", "C", "D", "E"],
        )

        fig = chart_builder.create_heatmap(
            data=correlation_matrix, title="Correlation Heatmap"
        )

        assert isinstance(fig, go.Figure)
        assert fig.data[0].type == "heatmap"

    def test_custom_styling(self, chart_builder, sample_data):
        """Test custom chart styling."""
        style = {
            "color_palette": ["#FF6B6B", "#4ECDC4", "#45B7D1"],
            "font_size": 14,
            "show_grid": False,
        }

        fig = chart_builder.create_line_chart(
            data=sample_data, x="date", y="value", style=style
        )

        assert fig.layout.font.size == 14
        assert not fig.layout.xaxis.showgrid


class TestDataConnector:
    """Test suite for data connector functionality."""

    @pytest.fixture
    def data_connector(self):
        """Create data connector instance."""
        return DataConnector()

    def test_database_connection(self, data_connector):
        """Test database connection."""
        with patch("sqlalchemy.create_engine") as mock_engine:
            data_connector.connect_database("postgresql://user:pass@localhost/db")
            mock_engine.assert_called_once()

    def test_api_connection(self, data_connector):
        """Test API connection."""
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = {"data": [1, 2, 3]}

            data = data_connector.fetch_from_api("https://api.example.com/data")

            assert data["data"] == [1, 2, 3]

    def test_file_loading(self, data_connector):
        """Test file loading capabilities."""
        with patch("pandas.read_csv") as mock_csv:
            mock_csv.return_value = pd.DataFrame({"A": [1, 2]})

            data = data_connector.load_file("data.csv")

            assert isinstance(data, pd.DataFrame)
            mock_csv.assert_called_once()

    def test_caching(self, data_connector):
        """Test data caching mechanism."""
        data_connector.enable_caching(ttl=3600)

        with patch.object(data_connector, "fetch_from_api") as mock_fetch:
            mock_fetch.return_value = {"data": "test"}

            # First call should fetch
            result1 = data_connector.get_data("api_key")
            # Second call should use cache
            result2 = data_connector.get_data("api_key")

            mock_fetch.assert_called_once()
            assert result1 == result2


class TestGraphQLAPI:
    """Test suite for GraphQL API."""

    @pytest.fixture
    def graphql_client(self):
        """Create GraphQL test client."""
        from graphene.test import Client

        return Client(schema)

    def test_dashboard_query(self, graphql_client):
        """Test dashboard data query."""
        query = """
            query {
                dashboard(id: "test") {
                    name
                    components {
                        id
                        type
                    }
                }
            }
        """

        with patch.object(Query, "resolve_dashboard") as mock_resolve:
            mock_resolve.return_value = {
                "name": "Test Dashboard",
                "components": [{"id": "chart1", "type": "line"}],
            }

            result = graphql_client.execute(query)

            assert not result.get("errors")
            assert result["data"]["dashboard"]["name"] == "Test Dashboard"

    def test_metrics_query(self, graphql_client):
        """Test metrics data query."""
        query = """
            query {
                metrics(
                    startDate: "2024-01-01",
                    endDate: "2024-01-31",
                    granularity: "daily"
                ) {
                    date
                    value
                }
            }
        """

        with patch.object(Query, "resolve_metrics") as mock_resolve:
            mock_resolve.return_value = [
                {"date": "2024-01-01", "value": 100},
                {"date": "2024-01-02", "value": 110},
            ]

            result = graphql_client.execute(query)

            assert len(result["data"]["metrics"]) == 2

    def test_mutation_update_dashboard(self, graphql_client):
        """Test dashboard update mutation."""
        mutation = """
            mutation {
                updateDashboard(
                    id: "test",
                    name: "Updated Dashboard"
                ) {
                    success
                    dashboard {
                        name
                    }
                }
            }
        """

        result = graphql_client.execute(mutation)

        assert result["data"]["updateDashboard"]["success"]


class TestDashboardIntegration:
    """Integration tests for dashboard functionality."""

    @pytest.fixture
    def app(self):
        """Create test application."""
        return create_app()

    def test_full_dashboard_creation(self, app):
        """Test creating a complete dashboard."""
        dashboard = InteractiveDashboard("integration_test")

        # Add components
        dashboard.add_component(html.H1("Dashboard Title"), "title")
        dashboard.add_component(dcc.Graph(id="main-chart"), "chart")
        dashboard.add_filter("date-filter", "date_range")

        # Set up callbacks
        @dashboard.callback(output="main-chart", inputs=["date-filter"])
        def update_chart(date_range):
            return {"data": [{"x": [1, 2], "y": [3, 4]}]}

        layout = dashboard.create_layout()

        assert layout is not None
        assert len(dashboard.components) == 2
        assert len(dashboard.filters) == 1

    def test_dashboard_performance(self):
        """Test dashboard performance with large dataset."""
        dashboard = DashboardBase("performance_test")

        # Create large dataset
        large_data = pd.DataFrame({"x": range(10000), "y": np.random.randn(10000)})

        chart_builder = ChartBuilder()
        fig = chart_builder.create_line_chart(large_data, "x", "y")

        # Measure rendering time
        import time

        start = time.time()
        dashboard.add_component(dcc.Graph(figure=fig), "large-chart")
        elapsed = time.time() - start

        assert elapsed < 1.0  # Should render in less than 1 second


class TestDashboardSecurity:
    """Test suite for dashboard security features."""

    def test_authentication(self):
        """Test dashboard authentication."""
        from dashboard_enhanced.auth import authenticate_user

        with patch("dashboard_enhanced.auth.verify_credentials") as mock_verify:
            mock_verify.return_value = True

            result = authenticate_user("user", "password")

            assert result["authenticated"]
            assert "token" in result

    def test_authorization(self):
        """Test dashboard authorization."""
        from dashboard_enhanced.auth import check_permission

        user = {"role": "admin"}
        assert check_permission(user, "edit_dashboard")

        user = {"role": "viewer"}
        assert not check_permission(user, "edit_dashboard")

    def test_input_sanitization(self):
        """Test input sanitization."""
        from dashboard_enhanced.security import sanitize_input

        malicious_input = "<script>alert('XSS')</script>"
        clean_input = sanitize_input(malicious_input)

        assert "<script>" not in clean_input
        assert "alert" not in clean_input
