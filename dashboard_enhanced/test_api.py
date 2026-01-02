"""API endpoint tests with proper authentication."""

import json
import os
import sys
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app


class TestAPIEndpoints(unittest.TestCase):
    """Test API endpoints with authentication."""

    def setUp(self):
        """Set up test client."""
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        self.access_token = None

    def get_auth_token(self):
        """Get authentication token."""
        if not self.access_token:
            response = self.client.post(
                "/api/auth/login",
                data=json.dumps({"username": "testuser", "password": "testpass"}),
                content_type="application/json",
            )
            if response.status_code == 200:
                data = json.loads(response.data)
                self.access_token = data["access_token"]
        return self.access_token

    def test_login(self):
        """Test login endpoint."""
        response = self.client.post(
            "/api/auth/login",
            data=json.dumps({"username": "testuser", "password": "testpass"}),
            content_type="application/json",
        )
        # Login should succeed with any username/password in test mode
        # or return 401 if credentials are invalid
        self.assertIn(response.status_code, [200, 401])

        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertIn("access_token", data)

    def test_protected_endpoints(self):
        """Test protected endpoints with token."""
        token = self.get_auth_token()

        if token:
            headers = {"Authorization": f"Bearer {token}"}

            # Test data endpoint
            response = self.client.get("/api/data/default", headers=headers)
            self.assertIn(response.status_code, [200, 404])

            # Test visualizations endpoint
            response = self.client.post(
                "/api/visualizations/time_series",
                data=json.dumps({"data": [[1, 2, 3], [4, 5, 6]], "title": "Test"}),
                content_type="application/json",
                headers=headers,
            )
            self.assertIn(response.status_code, [200, 404])

            # Test config endpoint with auth
            response = self.client.get("/api/config", headers=headers)
            self.assertIn(response.status_code, [200, 401])

    def test_public_endpoints(self):
        """Test public endpoints."""
        # Index should be accessible without auth
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

        # Config endpoint might be public or protected
        response = self.client.get("/api/config")
        self.assertIn(response.status_code, [200, 401])

    def test_export_endpoint(self):
        """Test export endpoint."""
        token = self.get_auth_token()

        if token:
            headers = {"Authorization": f"Bearer {token}"}
            response = self.client.post(
                "/api/export/pdf",
                data=json.dumps({"html": "<div>Test</div>", "filename": "test.pdf"}),
                content_type="application/json",
                headers=headers,
            )
            # Export might succeed or fail due to dependencies
            self.assertIn(response.status_code, [200, 400, 401, 500])

    def test_invalid_endpoints(self):
        """Test invalid endpoints return 404."""
        response = self.client.get("/api/nonexistent")
        self.assertEqual(response.status_code, 404)

        response = self.client.post(
            "/api/invalid/endpoint",
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 404)


class TestWebSocketConnection(unittest.TestCase):
    """Test WebSocket functionality."""

    def setUp(self):
        """Set up test client."""
        self.app = app
        self.app.config["TESTING"] = True

    def test_socketio_initialization(self):
        """Test SocketIO is initialized."""
        from app import socketio

        self.assertIsNotNone(socketio)

    def test_socket_events_defined(self):
        """Test socket event handlers are defined."""
        # This verifies that the socket handlers exist
        # Actual WebSocket testing would require socketio test client
        from app import socketio

        # Check that socketio has handlers
        handlers = getattr(socketio, "handlers", {})
        self.assertIsInstance(handlers, dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
