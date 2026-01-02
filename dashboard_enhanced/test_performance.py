"""Performance tests for dashboard using Locust."""

import json
import random

from locust import HttpUser, between, task


class DashboardUser(HttpUser):
    """Simulate a user interacting with the dashboard."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    def on_start(self):
        """Called when a user starts."""
        # Try to login (might fail but that's ok for testing)
        response = self.client.post(
            "/api/auth/login", json={"username": "testuser", "password": "testpass"}
        )
        if response.status_code == 200:
            data = response.json()
            self.token = data.get("access_token")
        else:
            self.token = None

    @task(5)
    def view_homepage(self):
        """View the homepage."""
        self.client.get("/")

    @task(3)
    def get_config(self):
        """Get dashboard configuration."""
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self.client.get("/api/config", headers=headers)

    @task(2)
    def get_data(self):
        """Get data from API."""
        dataset = random.choice(["default", "timeseries", "categories"])
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self.client.get(f"/api/data/{dataset}", headers=headers)

    @task(1)
    def create_visualization(self):
        """Create a visualization."""
        viz_type = random.choice(["time_series", "heatmap", "3d_scatter"])
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        data = {
            "data": [[random.random() for _ in range(10)] for _ in range(3)],
            "title": f"Test {viz_type}",
        }

        self.client.post(f"/api/visualizations/{viz_type}", json=data, headers=headers)


class GraphQLUser(HttpUser):
    """Simulate a user using GraphQL API."""

    wait_time = between(1, 2)

    @task(3)
    def query_dashboard(self):
        """Query dashboard data."""
        query = """
        query {
            dashboard {
                id
                name
                theme
                charts {
                    id
                    title
                }
            }
        }
        """
        self.client.post(
            "/graphql",
            json={"query": query},
            headers={"Content-Type": "application/json"},
        )

    @task(2)
    def query_metrics(self):
        """Query metrics."""
        query = """
        query {
            metrics {
                name
                value
                trend
            }
        }
        """
        self.client.post(
            "/graphql",
            json={"query": query},
            headers={"Content-Type": "application/json"},
        )

    @task(1)
    def mutation_update(self):
        """Update dashboard settings."""
        mutation = """
        mutation {
            updateDashboard(theme: "dark", autoRefresh: true) {
                success
                message
            }
        }
        """
        self.client.post(
            "/graphql",
            json={"query": mutation},
            headers={"Content-Type": "application/json"},
        )


class MobileUser(HttpUser):
    """Simulate mobile user with different behavior."""

    wait_time = between(2, 5)  # Mobile users might be slower

    def on_start(self):
        """Set mobile headers."""
        self.client.headers.update(
            {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)"}
        )

    @task(10)
    def view_homepage(self):
        """Mobile users mainly view the homepage."""
        self.client.get("/")

    @task(2)
    def get_config(self):
        """Occasionally check config."""
        self.client.get("/api/config")


# Run with: locust -f test_performance.py --host http://localhost:5000
# Or headless: locust -f test_performance.py --host http://localhost:5000 --headless -u 10 -r 2 -t 30s
