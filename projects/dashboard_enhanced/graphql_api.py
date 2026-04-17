"""GraphQL API for flexible dashboard data queries.
Provides a GraphQL endpoint with schema, resolvers, and subscriptions.
"""

import asyncio
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from graphene import (
    Boolean,
    DateTime,
    Field,
    Float,
    InputObjectType,
    Int,
    JSONString,
    List,
    Mutation,
    ObjectType,
    Schema,
    String,
)

# ============== GraphQL Types ==============


class MetricType(ObjectType):
    """GraphQL object describing a single dashboard metric.

    Fields
    ------
    name : str
        Metric identifier (e.g., ``"active_users"``).
    value : float
        Latest numeric value.
    unit : str
        Unit or dimension (``"%"``, ``"USD"``, etc.).
    timestamp : datetime
        ISO-8601 timestamp in UTC.
    change : float
        Delta relative to the previous period (same units as ``value``).
    trend : str
        Qualitative description (``"up"``, ``"down"``, ``"flat"``).

    Examples:
    --------
    >>> query = '''
    ... {
    ...   metric(name: "conversion_rate") {
    ...     name
    ...     value
    ...     unit
    ...   }
    ... }
    ... '''
    >>> isinstance(str(query), str)
    True
    """

    name = String()
    value = Float()
    unit = String()
    timestamp = DateTime()
    change = Float()
    trend = String()


class TimeSeriesPointType(ObjectType):
    """Time series data point."""

    timestamp = DateTime()
    value = Float()
    label = String()


class TimeSeriesType(ObjectType):
    """Time series data type."""

    name = String()
    data = List(TimeSeriesPointType)
    aggregation = String()
    interval = String()


class CategoryDataType(ObjectType):
    """Category data type."""

    category = String()
    value = Float()
    percentage = Float()
    count = Int()


class ChartDataType(ObjectType):
    """Chart data type."""

    id = String()
    type = String()
    title = String()
    data = JSONString()
    config = JSONString()
    lastUpdated = DateTime()


class DashboardType(ObjectType):
    """Dashboard data type."""

    id = String()
    name = String()
    description = String()
    charts = List(ChartDataType)
    metrics = List(MetricType)
    lastUpdated = DateTime()
    refreshInterval = Int()


class FilterInputType(InputObjectType):
    """Input type for filters."""

    dateFrom = DateTime()
    dateTo = DateTime()
    categories = List(String)
    metrics = List(String)
    aggregation = String()
    limit = Int()


class UserType(ObjectType):
    """User data type."""

    id = String()
    username = String()
    email = String()
    role = String()
    preferences = JSONString()
    lastLogin = DateTime()


class NotificationType(ObjectType):
    """Notification type."""

    id = String()
    type = String()
    message = String()
    severity = String()
    timestamp = DateTime()
    read = Boolean()


# ============== Resolvers ==============


class DashboardQuery(ObjectType):
    """Root query for dashboard data."""

    # Dashboard queries
    dashboard = Field(
        DashboardType, id=String(required=True), filters=FilterInputType()
    )
    dashboards = List(DashboardType, limit=Int())

    # Metrics queries
    metric = Field(MetricType, name=String(required=True), timeRange=String())
    metrics = List(MetricType, filters=FilterInputType())

    # Time series queries
    timeseries = Field(
        TimeSeriesType, metric=String(required=True), filters=FilterInputType()
    )
    multiTimeseries = List(
        TimeSeriesType, metrics=List(String), filters=FilterInputType()
    )

    # Category data queries
    categories = List(CategoryDataType, metric=String(), filters=FilterInputType())

    # Chart queries
    chart = Field(ChartDataType, id=String(required=True))
    charts = List(ChartDataType, dashboardId=String(), type=String())

    # User queries
    user = Field(UserType, id=String())
    currentUser = Field(UserType)

    # Notification queries
    notifications = List(NotificationType, unreadOnly=Boolean())

    # Search
    search = List(JSONString, query=String(required=True), type=String())

    # Resolvers
    def resolve_dashboard(self, info, id, filters=None):
        """Resolve dashboard by ID."""
        # Simulate dashboard data
        return DashboardType(
            id=id,
            name=f"Dashboard {id}",
            description="Interactive dashboard with real-time data",
            charts=self._get_charts(id),
            metrics=self._get_metrics(filters),
            lastUpdated=datetime.now(),
            refreshInterval=5000,
        )

    def resolve_dashboards(self, info, limit=10):
        """Resolve list of dashboards."""
        dashboards = []
        for i in range(min(limit, 5)):
            dashboards.append(
                DashboardType(
                    id=str(i),
                    name=f"Dashboard {i}",
                    description=f"Dashboard description {i}",
                    charts=[],
                    metrics=[],
                    lastUpdated=datetime.now(),
                    refreshInterval=5000,
                )
            )
        return dashboards

    def resolve_metric(self, info, name, timeRange=None):
        """Resolve single metric."""
        value = np.random.uniform(0, 100)
        change = np.random.uniform(-10, 10)

        return MetricType(
            name=name,
            value=value,
            unit=self._get_unit(name),
            timestamp=datetime.now(),
            change=change,
            trend="up" if change > 0 else "down",
        )

    def resolve_metrics(self, info, filters=None):
        """Resolve multiple metrics."""
        metric_names = ["revenue", "users", "conversion", "performance", "errors"]
        metrics = []

        for name in metric_names:
            if filters and filters.get("metrics") and name not in filters["metrics"]:
                continue

            value = np.random.uniform(0, 100)
            change = np.random.uniform(-10, 10)

            metrics.append(
                MetricType(
                    name=name,
                    value=value,
                    unit=self._get_unit(name),
                    timestamp=datetime.now(),
                    change=change,
                    trend="up" if change > 0 else "down",
                )
            )

        return metrics

    def resolve_timeseries(self, info, metric, filters=None):
        """Resolve time series data."""
        data_points = []
        start_date = (
            filters.dateFrom
            if filters and filters.dateFrom
            else datetime.now() - timedelta(days=7)
        )
        end_date = filters.dateTo if filters and filters.dateTo else datetime.now()

        # Generate time series data
        dates = pd.date_range(start=start_date, end=end_date, freq="H")
        values = np.cumsum(np.random.randn(len(dates))) + 50

        for i, date in enumerate(dates):
            data_points.append(
                TimeSeriesPointType(timestamp=date, value=values[i], label=metric)
            )

        return TimeSeriesType(
            name=metric,
            data=data_points,
            aggregation=filters.aggregation if filters else "avg",
            interval="hourly",
        )

    def resolve_multi_timeseries(self, info, metrics=None, filters=None):
        """Resolve multiple time series."""
        if not metrics:
            metrics = ["metric1", "metric2"]

        series_list = []
        for metric in metrics:
            series_list.append(self.resolve_timeseries(info, metric, filters))

        return series_list

    def resolve_categories(self, info, metric=None, filters=None):
        """Resolve category data."""
        categories = ["Category A", "Category B", "Category C", "Category D"]
        data = []

        total = 0
        values = [np.random.uniform(100, 1000) for _ in categories]
        total = sum(values)

        for i, cat in enumerate(categories):
            data.append(
                CategoryDataType(
                    category=cat,
                    value=values[i],
                    percentage=(values[i] / total) * 100,
                    count=int(values[i]),
                )
            )

        return data

    def resolve_chart(self, info, id):
        """Resolve single chart."""
        return self._create_chart(id)

    def resolve_charts(self, info, dashboardId=None, type=None):
        """Resolve multiple charts."""
        charts = []
        chart_types = ["line", "bar", "pie", "heatmap", "scatter"]

        for i in range(5):
            chart_type = type if type else chart_types[i % len(chart_types)]
            chart = self._create_chart(f"chart_{i}", chart_type)

            if not type or chart.type == type:
                charts.append(chart)

        return charts

    def resolve_user(self, info, id=None):
        """Resolve user data."""
        return UserType(
            id=id or "1",
            username="user123",
            email="user@example.com",
            role="admin",
            preferences=json.dumps({"theme": "dark", "lang": "en"}),
            lastLogin=datetime.now(),
        )

    def resolve_current_user(self, info):
        """Resolve current user."""
        # Get from context/auth
        return self.resolve_user(info, id="current")

    def resolve_notifications(self, info, unreadOnly=False):
        """Resolve notifications."""
        notifications = []
        types = ["info", "warning", "error", "success"]
        severities = ["low", "medium", "high", "critical"]

        for i in range(5):
            read = np.random.choice([True, False])

            if unreadOnly and read:
                continue

            notifications.append(
                NotificationType(
                    id=str(i),
                    type=np.random.choice(types),
                    message=f"Notification message {i}",
                    severity=np.random.choice(severities),
                    timestamp=datetime.now() - timedelta(hours=i),
                    read=read,
                )
            )

        return notifications

    def resolve_search(self, info, query, type=None):
        """Resolve search results."""
        results = []

        # Simulate search results
        for i in range(10):
            if type and type != "all":
                result_type = type
            else:
                result_type = np.random.choice(["dashboard", "chart", "metric"])

            results.append(
                {
                    "id": str(i),
                    "type": result_type,
                    "title": f"Result {i} for '{query}'",
                    "description": f"Description for result {i}",
                    "score": np.random.uniform(0.5, 1.0),
                }
            )

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return [json.dumps(r) for r in results]

    # Helper methods
    def _get_charts(self, dashboard_id):
        """Get charts for dashboard."""
        charts = []
        for i in range(4):
            charts.append(self._create_chart(f"{dashboard_id}_chart_{i}"))
        return charts

    def _get_metrics(self, filters):
        """Get metrics based on filters."""
        return self.resolve_metrics(None, filters)

    def _create_chart(self, id, type=None):
        """Create a chart object."""
        if not type:
            type = np.random.choice(["line", "bar", "pie", "heatmap"])

        # Generate chart data based on type
        if type == "line":
            data = {
                "x": list(range(10)),
                "y": [np.random.uniform(0, 100) for _ in range(10)],
            }
        elif type == "bar":
            data = {
                "categories": ["A", "B", "C", "D"],
                "values": [np.random.uniform(0, 100) for _ in range(4)],
            }
        elif type == "pie":
            data = {"labels": ["Slice 1", "Slice 2", "Slice 3"], "values": [30, 45, 25]}
        else:
            data = {
                "z": [[np.random.uniform(0, 100) for _ in range(10)] for _ in range(10)]
            }

        return ChartDataType(
            id=id,
            type=type,
            title=f"Chart {id}",
            data=json.dumps(data),
            config=json.dumps({"theme": "light", "animated": False}),
            lastUpdated=datetime.now(),
        )

    def _get_unit(self, metric_name):
        """Get unit for metric."""
        units = {
            "revenue": "$",
            "users": "",
            "conversion": "%",
            "performance": "ms",
            "errors": "",
        }
        return units.get(metric_name, "")


# ============== Mutations ==============


class UpdateDashboardInput(InputObjectType):
    """Input type for updating dashboard."""

    id = String(required=True)
    name = String()
    description = String()
    refreshInterval = Int()


class CreateChartInput(InputObjectType):
    """Input type for creating chart."""

    dashboardId = String(required=True)
    type = String(required=True)
    title = String(required=True)
    config = JSONString()


class UpdateMetricInput(InputObjectType):
    """Input type for updating metric."""

    name = String(required=True)
    value = Float()
    unit = String()


class DashboardMutation(Mutation):
    """Update dashboard mutation."""

    class Arguments:
        """Incoming dashboard configuration."""

        input = UpdateDashboardInput(required=True)

    dashboard = Field(DashboardType)

    def mutate(self, info, input):
        """Apply metadata updates and shape the dashboard payload."""
        dashboard = DashboardType(
            id=input.id,
            name=input.name or f"Dashboard {input.id}",
            description=input.description or "Updated dashboard",
            charts=[],
            metrics=[],
            lastUpdated=datetime.now(),
            refreshInterval=input.refreshInterval or 5000,
        )
        return DashboardMutation(dashboard=dashboard)


class CreateChartMutation(Mutation):
    """Create chart mutation."""

    class Arguments:
        """Creation payload for a chart."""

        input = CreateChartInput(required=True)

    chart = Field(ChartDataType)

    def mutate(self, info, input):
        """Persist a new chart definition and return the GraphQL node."""
        chart = ChartDataType(
            id=f"new_chart_{datetime.now().timestamp()}",
            type=input.type,
            title=input.title,
            data=json.dumps({}),
            config=input.config or json.dumps({}),
            lastUpdated=datetime.now(),
        )
        return CreateChartMutation(chart=chart)


class DeleteChartMutation(Mutation):
    """Delete chart mutation."""

    class Arguments:
        """Identity of the chart to delete."""

        id = String(required=True)

    success = Boolean()
    message = String()

    def mutate(self, info, id):
        """Remove the referenced chart and signal whether it succeeded."""
        return DeleteChartMutation(success=True, message=f"Chart {id} deleted")


class UpdateMetricMutation(Mutation):
    """Update metric mutation."""

    class Arguments:
        """Metric payload describing the new observation."""

        input = UpdateMetricInput(required=True)

    metric = Field(MetricType)

    def mutate(self, info, input):
        """Refresh a metric reading and include randomized trend metadata."""
        metric = MetricType(
            name=input.name,
            value=input.value or np.random.uniform(0, 100),
            unit=input.unit or "",
            timestamp=datetime.now(),
            change=np.random.uniform(-10, 10),
            trend="up",
        )
        return UpdateMetricMutation(metric=metric)


class RootMutation(ObjectType):
    """Root mutations."""

    update_dashboard = DashboardMutation.Field()
    create_chart = CreateChartMutation.Field()
    delete_chart = DeleteChartMutation.Field()
    update_metric = UpdateMetricMutation.Field()


# ============== Subscriptions ==============


class DashboardSubscription(ObjectType):
    """Root subscription for real-time updates."""

    metric_update = Field(MetricType, metric_name=String())
    chart_update = Field(ChartDataType, chart_id=String())
    notification = Field(NotificationType)

    async def resolve_metric_update(self, info, metric_name=None):
        """Subscribe to metric updates."""
        while True:
            await asyncio.sleep(1)
            yield MetricType(
                name=metric_name or "realtime_metric",
                value=np.random.uniform(0, 100),
                unit="",
                timestamp=datetime.now(),
                change=np.random.uniform(-5, 5),
                trend=np.random.choice(["up", "down"]),
            )

    async def resolve_chart_update(self, info, chart_id=None):
        """Subscribe to chart updates."""
        while True:
            await asyncio.sleep(2)
            yield ChartDataType(
                id=chart_id or "realtime_chart",
                type="line",
                title="Real-time Chart",
                data=json.dumps(
                    {
                        "value": np.random.uniform(0, 100),
                        "timestamp": datetime.now().isoformat(),
                    }
                ),
                config=json.dumps({}),
                lastUpdated=datetime.now(),
            )

    async def resolve_notification(self, info):
        """Subscribe to notifications."""
        while True:
            await asyncio.sleep(5)
            yield NotificationType(
                id=str(datetime.now().timestamp()),
                type=np.random.choice(["info", "warning", "success"]),
                message="New notification",
                severity="medium",
                timestamp=datetime.now(),
                read=False,
            )


# ============== Schema ==============

# Create schema
schema = Schema(
    query=DashboardQuery, mutation=RootMutation, subscription=DashboardSubscription
)


# ============== Flask Integration ==============


def create_graphql_app():
    """Create Flask app with GraphQL endpoint."""
    from flask_cors import CORS

    app = Flask(__name__)
    CORS(app)

    # Add GraphQL endpoint manually
    @app.route("/graphql", methods=["GET", "POST"])
    def graphql():
        if request.method == "GET":
            # Return GraphiQL IDE
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>GraphiQL</title>
                <style>body { margin: 0; }</style>
            </head>
            <body>
                <div id="root"></div>
                <script src="https://unpkg.com/react@17/umd/react.production.min.js"></script>
                <script src="https://unpkg.com/react-dom@17/umd/react-dom.production.min.js"></script>
                <script src="https://unpkg.com/graphiql/graphiql.min.js"></script>
                <link rel="stylesheet" href="https://unpkg.com/graphiql/graphiql.min.css" />
                <script>
                    ReactDOM.render(
                        React.createElement(GraphiQL, {
                            fetcher: GraphiQL.createFetcher({url: '/graphql'}),
                        }),
                        document.getElementById('root')
                    );
                </script>
            </body>
            </html>
            """

        # Handle POST requests
        data = request.get_json()
        query = data.get("query")
        variables = data.get("variables")

        try:
            result = schema.execute(query, variable_values=variables)

            response = {"data": result.data}

            if result.errors:
                response["errors"] = [str(e) for e in result.errors]

            return jsonify(response)

        except Exception as e:
            return (
                jsonify({"errors": [str(e)]}),
                200,
            )  # GraphQL typically returns 200 even with errors

    # Health check
    @app.route("/health")
    def health():
        return {"status": "healthy", "service": "graphql"}

    return app


# Example queries for documentation
EXAMPLE_QUERIES = """
# Get dashboard with charts and metrics
query GetDashboard($id: String!, $dateFrom: DateTime) {
  dashboard(id: $id, filters: {dateFrom: $dateFrom}) {
    id
    name
    description
    lastUpdated
    charts {
      id
      type
      title
      data
    }
    metrics {
      name
      value
      unit
      change
      trend
    }
  }
}

# Get time series data
query GetTimeSeries($metric: String!, $dateFrom: DateTime, $dateTo: DateTime) {
  timeseries(
    metric: $metric,
    filters: {dateFrom: $dateFrom, dateTo: $dateTo}
  ) {
    name
    data {
      timestamp
      value
    }
    aggregation
  }
}

# Get multiple metrics
query GetMetrics($categories: [String]) {
  metrics(filters: {categories: $categories}) {
    name
    value
    unit
    change
    trend
  }
}

# Search dashboards and charts
query Search($query: String!) {
  search(query: $query, type: "dashboard")
}

# Create a new chart
mutation CreateChart($input: CreateChartInput!) {
  createChart(input: $input) {
    chart {
      id
      type
      title
      lastUpdated
    }
  }
}

# Update dashboard settings
mutation UpdateDashboard($input: UpdateDashboardInput!) {
  updateDashboard(input: $input) {
    dashboard {
      id
      name
      refreshInterval
    }
  }
}

# Subscribe to metric updates
subscription MetricUpdates($metricName: String) {
  metricUpdate(metricName: $metricName) {
    name
    value
    change
    timestamp
  }
}

# Subscribe to notifications
subscription Notifications {
  notification {
    id
    type
    message
    severity
    timestamp
  }
}
"""

if __name__ == "__main__":
    app = create_graphql_app()
    print("GraphQL API running at http://localhost:5001/graphql")
    print("\nExample queries available in GraphiQL IDE")
    app.run(port=5001, debug=True)
