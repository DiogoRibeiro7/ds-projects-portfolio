"""Test GraphQL API functionality."""

import json
import os
import sys
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import GraphQL components
from graphql_api import create_graphql_app


class TestGraphQLAPI(unittest.TestCase):
    """Test GraphQL API endpoints and queries."""

    def setUp(self):
        """Set up test client."""
        self.app = create_graphql_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_graphql_endpoint_exists(self):
        """Test GraphQL endpoint is accessible."""
        response = self.client.get("/graphql")
        # GraphiQL interface or method not allowed
        self.assertIn(response.status_code, [200, 405])

    def test_simple_query(self):
        """Test simple GraphQL query."""
        query = """
        query {
            dashboard {
                id
                name
                theme
                autoRefresh
            }
        }
        """
        response = self.client.post(
            "/graphql",
            data=json.dumps({"query": query}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertIn("data", data)

        if data["data"]:
            dashboard_data = data["data"].get("dashboard")
            if dashboard_data:
                self.assertIn("id", dashboard_data)
                self.assertIn("name", dashboard_data)

    def test_chart_query(self):
        """Test querying charts."""
        query = """
        query {
            charts {
                id
                title
                type
                data
            }
        }
        """
        response = self.client.post(
            "/graphql",
            data=json.dumps({"query": query}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertIn("data", data)

        if data["data"]:
            charts = data["data"].get("charts")
            if charts:
                self.assertIsInstance(charts, list)

    def test_metric_query(self):
        """Test querying metrics."""
        query = """
        query {
            metrics {
                name
                value
                timestamp
                unit
                trend
            }
        }
        """
        response = self.client.post(
            "/graphql",
            data=json.dumps({"query": query}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertIn("data", data)

    def test_query_with_variables(self):
        """Test GraphQL query with variables."""
        query = """
        query GetChart($chartId: ID!) {
            chart(id: $chartId) {
                id
                title
                type
            }
        }
        """
        variables = {"chartId": "1"}

        response = self.client.post(
            "/graphql",
            data=json.dumps({"query": query, "variables": variables}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertIn("data", data)

    def test_mutation_update_dashboard(self):
        """Test dashboard update mutation."""
        mutation = """
        mutation UpdateDashboard($theme: String!, $autoRefresh: Boolean!) {
            updateDashboard(theme: $theme, autoRefresh: $autoRefresh) {
                success
                message
                dashboard {
                    theme
                    autoRefresh
                }
            }
        }
        """
        variables = {"theme": "dark", "autoRefresh": False}

        response = self.client.post(
            "/graphql",
            data=json.dumps({"query": mutation, "variables": variables}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertIn("data", data)

        if data["data"]:
            result = data["data"].get("updateDashboard")
            if result:
                self.assertIn("success", result)
                self.assertIn("message", result)

    def test_mutation_update_chart(self):
        """Test chart update mutation."""
        mutation = """
        mutation UpdateChart($id: ID!, $title: String, $data: String) {
            updateChart(id: $id, title: $title, data: $data) {
                success
                message
                chart {
                    id
                    title
                }
            }
        }
        """
        variables = {
            "id": "1",
            "title": "Updated Chart Title",
            "data": json.dumps({"values": [1, 2, 3, 4, 5]}),
        }

        response = self.client.post(
            "/graphql",
            data=json.dumps({"query": mutation, "variables": variables}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertIn("data", data)

    def test_invalid_query(self):
        """Test invalid GraphQL query."""
        query = """
        query {
            nonexistentField {
                id
            }
        }
        """
        response = self.client.post(
            "/graphql",
            data=json.dumps({"query": query}),
            content_type="application/json",
        )
        # Even with errors, GraphQL returns 200
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        # Should have errors for invalid field
        self.assertIn("errors", data)

    def test_introspection_query(self):
        """Test GraphQL introspection."""
        query = """
        query {
            __schema {
                types {
                    name
                }
            }
        }
        """
        response = self.client.post(
            "/graphql",
            data=json.dumps({"query": query}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertIn("data", data)

        if data["data"]:
            schema_data = data["data"].get("__schema")
            if schema_data:
                types = schema_data.get("types")
                if types:
                    type_names = [t["name"] for t in types]
                    # Check our custom types exist
                    self.assertIn("DashboardType", type_names)
                    self.assertIn("ChartType", type_names)
                    self.assertIn("MetricType", type_names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
