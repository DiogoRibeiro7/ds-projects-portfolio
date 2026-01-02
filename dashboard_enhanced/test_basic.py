"""Basic tests for dashboard components to verify they work correctly."""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import components
from app import DataStreamer, ExportManager, VisualizationEngine, app


class TestVisualizationEngine(unittest.TestCase):
    """Test the VisualizationEngine component."""

    def setUp(self):
        self.viz = VisualizationEngine()

    def test_create_line_chart(self):
        """Test line chart creation."""
        data = {"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]}
        chart = self.viz.create_line_chart(data, title="Test Line Chart")
        self.assertIn("data", chart)
        self.assertIn("layout", chart)

    def test_create_bar_chart(self):
        """Test bar chart creation."""
        data = {"categories": ["A", "B", "C"], "values": [10, 20, 30]}
        chart = self.viz.create_bar_chart(data, title="Test Bar Chart")
        self.assertIn("data", chart)
        self.assertIn("layout", chart)

    def test_create_heatmap(self):
        """Test heatmap creation."""
        data = {"z": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}
        chart = self.viz.create_heatmap(data, title="Test Heatmap")
        self.assertIn("data", chart)
        self.assertIn("layout", chart)


class TestDataStreamer(unittest.TestCase):
    """Test the DataStreamer component."""

    def setUp(self):
        self.streamer = DataStreamer()

    def test_create_data_generator(self):
        """Test data generator creation."""
        self.streamer.create_data_generator("test_stream", "line")
        self.assertIn("test_stream", self.streamer.data_generators)

    def test_start_stop_stream(self):
        """Test streaming start/stop."""
        self.streamer.create_data_generator("test_stream", "line")
        # Just test that methods run without error
        self.streamer.start_stream("test_stream")
        self.assertTrue(self.streamer.streaming)
        self.streamer.stop_stream()
        self.assertFalse(self.streamer.streaming)


class TestExportManager(unittest.TestCase):
    """Test the ExportManager component."""

    def setUp(self):
        self.exporter = ExportManager()

    @patch("app.plotly.io.to_image")
    def test_export_to_pdf(self, mock_to_image):
        """Test PDF export."""
        mock_to_image.return_value = b"fake_image_data"

        charts = [
            {"data": [], "layout": {"title": {"text": "Chart 1"}}},
            {"data": [], "layout": {"title": {"text": "Chart 2"}}},
        ]

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            output_path = tmp.name

        try:
            result = self.exporter.export_to_pdf(charts, output_path)
            self.assertEqual(result, output_path)
            self.assertTrue(os.path.exists(output_path))
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    @patch("app.plotly.io.to_image")
    def test_export_to_powerpoint(self, mock_to_image):
        """Test PowerPoint export."""
        mock_to_image.return_value = b"fake_image_data"

        charts = [
            {"data": [], "layout": {"title": {"text": "Chart 1"}}},
            {"data": [], "layout": {"title": {"text": "Chart 2"}}},
        ]

        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as tmp:
            output_path = tmp.name

        try:
            result = self.exporter.export_to_powerpoint(charts, output_path)
            self.assertEqual(result, output_path)
            self.assertTrue(os.path.exists(output_path))
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


class TestFlaskApp(unittest.TestCase):
    """Test Flask application endpoints."""

    def setUp(self):
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_index_route(self):
        """Test the index route."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_api_data_endpoint(self):
        """Test the /api/data endpoint."""
        response = self.client.get("/api/data")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("data", data)
        self.assertIn("timestamp", data)

    def test_api_chart_endpoint(self):
        """Test the /api/chart endpoint."""
        payload = {"type": "line", "data": {"x": [1, 2, 3], "y": [4, 5, 6]}}
        response = self.client.post(
            "/api/chart", data=json.dumps(payload), content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("chart", data)

    def test_api_export_endpoint(self):
        """Test the /api/export endpoint."""
        payload = {
            "charts": [{"data": [], "layout": {"title": {"text": "Test Chart"}}}],
            "format": "pdf",
        }
        response = self.client.post(
            "/api/export", data=json.dumps(payload), content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("file", data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
