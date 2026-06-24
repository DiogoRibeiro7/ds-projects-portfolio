"""Corrected tests for dashboard components based on actual implementation."""

import base64
import io
import json
import os
import sys
import tempfile
import types
import unittest
import time
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import components
from app import DashboardConfig, DataStreamer, ExportManager, VisualizationEngine, app


class TestVisualizationEngine(unittest.TestCase):
    """Test the VisualizationEngine component."""

    def setUp(self):
        """Create a visualization engine fixture per test."""
        self.viz = VisualizationEngine()

    def test_create_time_series(self):
        """Test time series creation."""
        data = pd.DataFrame(
            {"value1": np.random.randn(10), "value2": np.random.randn(10)}
        )
        chart_html = self.viz.create_time_series(data, title="Test Time Series")
        self.assertIsInstance(chart_html, str)
        self.assertIn("time-series-chart", chart_html)
        self.assertIn("plotly", chart_html)

    def test_create_heatmap(self):
        """Test heatmap creation."""
        data = pd.DataFrame(
            np.random.randn(5, 5),
            columns=["A", "B", "C", "D", "E"],
            index=["Row1", "Row2", "Row3", "Row4", "Row5"],
        )
        chart_html = self.viz.create_heatmap(data, title="Test Heatmap")
        self.assertIsInstance(chart_html, str)
        self.assertIn("heatmap-chart", chart_html)

    def test_create_3d_scatter(self):
        """Test 3D scatter plot creation."""
        data = pd.DataFrame(
            {
                "x": np.random.randn(20),
                "y": np.random.randn(20),
                "z": np.random.randn(20),
            }
        )
        chart_html = self.viz.create_3d_scatter(data, title="Test 3D Scatter")
        self.assertIsInstance(chart_html, str)
        self.assertIn("scatter3d-chart", chart_html)

    def test_create_dashboard_layout(self):
        """Test dashboard layout creation."""
        charts = ["<div>Chart1</div>", "<div>Chart2</div>"]

        # Test grid layout
        layout_html = self.viz.create_dashboard_layout(charts, layout="grid")
        self.assertIn("dashboard-grid", layout_html)
        self.assertIn("Chart1", layout_html)
        self.assertIn("Chart2", layout_html)

        # Test stacked layout
        layout_html = self.viz.create_dashboard_layout(charts, layout="stacked")
        self.assertIn("dashboard-stacked", layout_html)


class TestDataStreamer(unittest.TestCase):
    """Test the DataStreamer component."""

    def setUp(self):
        """Prepare a DataStreamer without starting background threads."""
        self.streamer = DataStreamer()

    def test_initialization(self):
        """Test DataStreamer initialization."""
        self.assertFalse(self.streamer.streaming)
        self.assertIsNone(self.streamer.stream_thread)
        self.assertEqual(len(self.streamer.data_generators), 0)

    def test_start_stream(self):
        """Test stream starting (without actual threading)."""
        with patch("app.socketio.emit") as mock_emit:
            self.streamer.start_stream("test-stream", interval=0)

            self.assertTrue(self.streamer.streaming)
            self.assertIn("test-stream", self.streamer.data_generators)
            self.assertIsNotNone(self.streamer.stream_thread)
            self.assertTrue(self.streamer.stream_thread.is_alive())

            time.sleep(0.01)
            self.assertGreaterEqual(mock_emit.call_count, 1)

            self.streamer.stop_stream("test-stream")
            self.streamer.stream_thread.join(timeout=1.0)

            self.assertNotIn("test-stream", self.streamer.data_generators)
            self.assertFalse(self.streamer.stream_thread.is_alive())


class TestDashboardConfig(unittest.TestCase):
    """Test DashboardConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = DashboardConfig()
        self.assertEqual(config.theme, "light")
        self.assertTrue(config.auto_refresh)
        self.assertEqual(config.refresh_interval, 5000)
        self.assertEqual(config.export_formats, ["pdf", "png", "pptx"])
        self.assertEqual(config.mobile_breakpoint, 768)

    def test_custom_config(self):
        """Test custom configuration."""
        config = DashboardConfig(
            theme="dark",
            auto_refresh=False,
            refresh_interval=10000,
            export_formats=["pdf"],
            mobile_breakpoint=1024,
        )
        self.assertEqual(config.theme, "dark")
        self.assertFalse(config.auto_refresh)
        self.assertEqual(config.refresh_interval, 10000)
        self.assertEqual(config.export_formats, ["pdf"])
        self.assertEqual(config.mobile_breakpoint, 1024)


class TestFlaskApp(unittest.TestCase):
    """Test Flask application endpoints."""

    def setUp(self):
        """Build a Flask test client to exercise HTTP routes."""
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_index_route(self):
        """Test the index route."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_api_auth_login(self):
        """Test the login endpoint."""
        payload = {"username": "testuser", "password": "testpass"}
        response = self.client.post(
            "/api/auth/login", data=json.dumps(payload), content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("access_token", data)

    def test_api_data_endpoint(self):
        """Test the /api/data/<dataset_name> endpoint."""
        response = self.client.get("/api/data/default")
        # Since we need JWT, this will return 401 without token
        self.assertIn(response.status_code, [401, 200])

    def test_api_visualizations(self):
        """Test the visualizations endpoint."""
        payload = {"data": [[1, 2, 3], [4, 5, 6]], "title": "Test Chart"}
        response = self.client.post(
            "/api/visualizations/time_series",
            data=json.dumps(payload),
            content_type="application/json",
        )
        # Since we need JWT, this will return 401 without token
        self.assertIn(response.status_code, [401, 200])

    def test_api_config(self):
        """Test the config endpoint."""
        response = self.client.get("/api/config")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("config", data)


class TestExportManager(unittest.TestCase):
    """Test the ExportManager component."""

    def setUp(self):
        """Instantiate an export manager for each scenario."""
        self.exporter = ExportManager()

    def test_export_to_pdf(self):
        """Test PDF export."""
        html_content = "<div>Test Chart</div>"
        fake_pdfkit = types.ModuleType("pdfkit")
        fake_pdfkit.from_string = Mock(return_value=b"fake-pdf-bytes")

        with patch.dict("sys.modules", {"pdfkit": fake_pdfkit}):
            result = self.exporter.export_to_pdf(html_content)

        self.assertEqual(result, b"fake-pdf-bytes")

    def test_export_to_pptx(self):
        """Test PowerPoint export with proper structure."""
        chart_image = base64.b64encode(b"fake_image_1").decode("utf-8")
        charts = [{"title": "Chart 1", "image": chart_image}]

        saved_presentations = []

        class FakePlaceholder:
            def __init__(self):
                self.text = ""

        class FakeShapes:
            def __init__(self):
                self.title = Mock()
                self.title.text = ""
                self.added_picture = None

            def add_picture(self, stream, left, top, width=None):
                self.added_picture = {
                    "data": stream.read(),
                    "left": left,
                    "top": top,
                    "width": width,
                }
                return Mock()

        class FakeSlide:
            def __init__(self):
                self.shapes = FakeShapes()
                self.placeholders = [Mock(), FakePlaceholder()]

        class FakeSlides:
            def __init__(self):
                self.slides = []

            def add_slide(self, layout):
                slide = FakeSlide()
                self.slides.append(slide)
                return slide

        class FakePresentation:
            def __init__(self):
                self.slides = FakeSlides()
                self.slide_layouts = list(range(10))
                saved_presentations.append(self)

            def save(self, output):
                output.write(b"fake-pptx-bytes")

        fake_pptx = types.ModuleType("pptx")
        fake_pptx.__path__ = ["pptx"]
        fake_pptx.Presentation = FakePresentation

        fake_pptx_util = types.ModuleType("pptx.util")
        fake_pptx_util.Inches = lambda value: value

        fake_modules = {
            "pptx": fake_pptx,
            "pptx.util": fake_pptx_util,
        }

        with patch.dict("sys.modules", fake_modules):
            result = self.exporter.export_to_pptx(charts)

        self.assertEqual(result, b"fake-pptx-bytes")
        self.assertEqual(len(saved_presentations), 1)
        prs = saved_presentations[0]
        self.assertEqual(len(prs.slides.slides), 2)
        self.assertEqual(prs.slides.slides[0].shapes.title.text, "Dashboard Export")
        self.assertEqual(prs.slides.slides[1].shapes.title.text, "Chart 1")
        self.assertEqual(prs.slides.slides[1].shapes.added_picture["data"], b"fake_image_1")


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)
