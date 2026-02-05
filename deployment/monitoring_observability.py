"""
Monitoring and Observability Infrastructure

Comprehensive monitoring, logging, and alerting for ML systems.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import asyncio
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    CollectorRegistry,
    push_to_gateway,
)
import grafana_api
from elasticsearch import Elasticsearch
from logstash import TCPLogstashHandler
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
import statsd
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Metrics Registry
registry = CollectorRegistry()

# Define metrics
model_inference_duration = Histogram(
    "model_inference_duration_seconds",
    "Model inference duration",
    ["model_name", "model_version", "endpoint"],
    registry=registry,
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5],
)

model_request_count = Counter(
    "model_request_total",
    "Total model requests",
    ["model_name", "model_version", "status", "endpoint"],
    registry=registry,
)

model_feature_values = Summary(
    "model_feature_values",
    "Feature value distributions",
    ["model_name", "feature_name"],
    registry=registry,
)

model_prediction_values = Summary(
    "model_prediction_values",
    "Prediction value distributions",
    ["model_name", "model_version"],
    registry=registry,
)

data_quality_score = Gauge(
    "data_quality_score",
    "Data quality score",
    ["model_name", "check_type"],
    registry=registry,
)

model_performance_metric = Gauge(
    "model_performance_metric",
    "Model performance metrics",
    ["model_name", "model_version", "metric_name"],
    registry=registry,
)

alert_triggered = Counter(
    "alert_triggered_total",
    "Total alerts triggered",
    ["alert_name", "severity", "model_name"],
    registry=registry,
)

cache_performance = Summary(
    "cache_performance",
    "Cache performance metrics",
    ["operation", "model_name"],
    registry=registry,
)


@dataclass
class MonitoringConfig:
    """Configuration for monitoring system."""

    prometheus_gateway: str = "prometheus-pushgateway:9091"
    elasticsearch_host: str = "elasticsearch:9200"
    grafana_host: str = "grafana:3000"
    jaeger_host: str = "jaeger:14268"
    statsd_host: str = "statsd:8125"
    sentry_dsn: Optional[str] = None
    alert_webhook_url: Optional[str] = None
    monitoring_interval: int = 60  # seconds
    retention_days: int = 30
    enable_profiling: bool = True
    enable_tracing: bool = True
    enable_metrics: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    custom_dashboards: List[str] = field(default_factory=list)


class ModelMonitoringService:
    """Central monitoring service for ML models."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.es_client = None
        self.grafana_client = None
        self.statsd_client = None
        self.tracer = None

        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize monitoring clients."""
        # Elasticsearch for logs
        if self.config.enable_logging:
            self.es_client = Elasticsearch([self.config.elasticsearch_host])

            # Setup Logstash handler
            logstash_handler = TCPLogstashHandler(host="logstash", port=5000, version=1)
            logger.addHandler(logstash_handler)

        # Sentry for error tracking
        if self.config.sentry_dsn:
            sentry_logging = LoggingIntegration(
                level=logging.INFO, event_level=logging.ERROR
            )
            sentry_sdk.init(
                dsn=self.config.sentry_dsn,
                integrations=[sentry_logging],
                traces_sample_rate=0.1,
            )

        # Grafana API client
        if self.config.grafana_host:
            self.grafana_client = grafana_api.GrafanaApi(
                host=self.config.grafana_host,
                auth=("admin", "admin"),  # Use proper auth in production
            )

        # StatsD client for custom metrics
        if self.config.statsd_host:
            self.statsd_client = statsd.StatsClient(
                self.config.statsd_host.split(":")[0],
                int(self.config.statsd_host.split(":")[1]),
            )

        # Jaeger tracing
        if self.config.enable_tracing:
            jaeger_exporter = JaegerExporter(
                agent_host_name=self.config.jaeger_host.split(":")[0],
                agent_port=int(self.config.jaeger_host.split(":")[1]),
            )
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            self.tracer = trace.get_tracer(__name__)

    async def monitor_inference(
        self,
        model_name: str,
        model_version: str,
        input_data: pd.DataFrame,
        predictions: np.ndarray,
        latency: float,
        endpoint: str = "default",
    ):
        """Monitor model inference."""
        with tracer.start_as_current_span("monitor_inference") as span:
            span.set_attribute("model.name", model_name)
            span.set_attribute("model.version", model_version)
            span.set_attribute("inference.latency", latency)
            span.set_attribute("batch.size", len(input_data))

            try:
                # Record metrics
                model_inference_duration.labels(
                    model_name=model_name,
                    model_version=model_version,
                    endpoint=endpoint,
                ).observe(latency)

                model_request_count.labels(
                    model_name=model_name,
                    model_version=model_version,
                    status="success",
                    endpoint=endpoint,
                ).inc()

                # Monitor feature distributions
                for column in input_data.columns:
                    model_feature_values.labels(
                        model_name=model_name, feature_name=column
                    ).observe(input_data[column].mean())

                # Monitor prediction distributions
                model_prediction_values.labels(
                    model_name=model_name, model_version=model_version
                ).observe(predictions.mean())

                # Data quality checks
                quality_issues = await self._check_data_quality(input_data, model_name)

                # Performance monitoring
                await self._monitor_performance(model_name, model_version, predictions)

                # Push metrics to Prometheus
                push_to_gateway(
                    self.config.prometheus_gateway,
                    job=f"{model_name}_inference",
                    registry=registry,
                )

                # Log to Elasticsearch
                if self.es_client:
                    await self._log_inference(
                        model_name,
                        model_version,
                        input_data,
                        predictions,
                        latency,
                        quality_issues,
                    )

                # Custom StatsD metrics
                if self.statsd_client:
                    self.statsd_client.incr(f"{model_name}.requests")
                    self.statsd_client.timing(f"{model_name}.latency", latency * 1000)

            except Exception as e:
                logger.error(f"Error monitoring inference: {e}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))

                model_request_count.labels(
                    model_name=model_name,
                    model_version=model_version,
                    status="error",
                    endpoint=endpoint,
                ).inc()

    async def _check_data_quality(
        self, data: pd.DataFrame, model_name: str
    ) -> List[Dict]:
        """Check data quality and record issues."""
        issues = []

        # Missing values check
        missing_pct = data.isnull().sum() / len(data)
        for column, pct in missing_pct.items():
            if pct > 0.1:  # More than 10% missing
                issues.append(
                    {
                        "type": "missing_values",
                        "column": column,
                        "percentage": float(pct),
                        "severity": "warning" if pct < 0.3 else "error",
                    }
                )

                data_quality_score.labels(
                    model_name=model_name, check_type="missing_values"
                ).set(1 - pct)

        # Outliers check
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            z_scores = np.abs(stats.zscore(data[column].dropna()))
            outlier_pct = (z_scores > 3).sum() / len(z_scores)

            if outlier_pct > 0.05:  # More than 5% outliers
                issues.append(
                    {
                        "type": "outliers",
                        "column": column,
                        "percentage": float(outlier_pct),
                        "severity": "warning",
                    }
                )

                data_quality_score.labels(
                    model_name=model_name, check_type="outliers"
                ).set(1 - outlier_pct)

        # Data type consistency
        for column in data.columns:
            if data[column].dtype == "object":
                try:
                    pd.to_numeric(data[column])
                    issues.append(
                        {
                            "type": "type_inconsistency",
                            "column": column,
                            "expected": "numeric",
                            "actual": "object",
                            "severity": "warning",
                        }
                    )
                except:
                    pass

        return issues

    async def _monitor_performance(
        self, model_name: str, model_version: str, predictions: np.ndarray
    ):
        """Monitor model performance metrics."""
        # Basic statistics
        metrics = {
            "prediction_mean": float(predictions.mean()),
            "prediction_std": float(predictions.std()),
            "prediction_min": float(predictions.min()),
            "prediction_max": float(predictions.max()),
        }

        # Record in Prometheus
        for metric_name, value in metrics.items():
            model_performance_metric.labels(
                model_name=model_name,
                model_version=model_version,
                metric_name=metric_name,
            ).set(value)

    async def _log_inference(
        self,
        model_name: str,
        model_version: str,
        input_data: pd.DataFrame,
        predictions: np.ndarray,
        latency: float,
        quality_issues: List[Dict],
    ):
        """Log inference details to Elasticsearch."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "model_version": model_version,
            "batch_size": len(input_data),
            "latency_ms": latency * 1000,
            "prediction_stats": {
                "mean": float(predictions.mean()),
                "std": float(predictions.std()),
                "min": float(predictions.min()),
                "max": float(predictions.max()),
            },
            "data_quality_issues": quality_issues,
            "feature_stats": {},
        }

        # Add feature statistics
        for column in input_data.select_dtypes(include=[np.number]).columns:
            log_entry["feature_stats"][column] = {
                "mean": float(input_data[column].mean()),
                "std": float(input_data[column].std()),
                "min": float(input_data[column].min()),
                "max": float(input_data[column].max()),
            }

        # Index to Elasticsearch
        try:
            self.es_client.index(
                index=f"ml-inference-{datetime.now().strftime('%Y.%m')}", body=log_entry
            )
        except Exception as e:
            logger.error(f"Failed to log to Elasticsearch: {e}")

    def create_dashboard(self, model_name: str) -> str:
        """Create Grafana dashboard for model monitoring."""
        dashboard = {
            "dashboard": {
                "title": f"{model_name} Model Monitoring",
                "panels": [
                    self._create_panel(
                        "Requests per Second",
                        f'rate(model_request_total{{model_name="{model_name}"}}[1m])',
                        "graph",
                        1,
                        1,
                    ),
                    self._create_panel(
                        "Latency (p95)",
                        f'histogram_quantile(0.95, model_inference_duration_seconds{{model_name="{model_name}"}})',
                        "graph",
                        1,
                        2,
                    ),
                    self._create_panel(
                        "Error Rate",
                        f'rate(model_request_total{{model_name="{model_name}",status="error"}}[5m])',
                        "graph",
                        2,
                        1,
                    ),
                    self._create_panel(
                        "Data Quality Score",
                        f'data_quality_score{{model_name="{model_name}"}}',
                        "graph",
                        2,
                        2,
                    ),
                    self._create_panel(
                        "Prediction Distribution",
                        f'model_prediction_values{{model_name="{model_name}"}}',
                        "heatmap",
                        3,
                        1,
                    ),
                    self._create_panel(
                        "Feature Drift",
                        f'data_drift_score{{model_name="{model_name}"}}',
                        "graph",
                        3,
                        2,
                    ),
                    self._create_panel(
                        "Cache Hit Rate",
                        f'rate(cache_hits_total{{model_name="{model_name}"}}[5m])',
                        "stat",
                        4,
                        1,
                    ),
                    self._create_panel(
                        "Active Alerts",
                        f'alert_triggered_total{{model_name="{model_name}"}}',
                        "stat",
                        4,
                        2,
                    ),
                ],
                "refresh": "5s",
                "time": {"from": "now-6h", "to": "now"},
            },
            "overwrite": True,
        }

        # Create dashboard via API
        response = self.grafana_client.dashboard.create_dashboard(dashboard)
        return response["url"]

    def _create_panel(
        self, title: str, query: str, panel_type: str, row: int, col: int
    ) -> Dict:
        """Create Grafana panel configuration."""
        return {
            "title": title,
            "type": panel_type,
            "gridPos": {"h": 8, "w": 12, "x": (col - 1) * 12, "y": (row - 1) * 8},
            "targets": [{"expr": query, "refId": "A"}],
        }


class AlertingService:
    """Alerting service for model monitoring."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alert_rules = {}
        self.alert_history = []

    def register_alert_rule(self, rule: "AlertRule"):
        """Register an alert rule."""
        self.alert_rules[rule.name] = rule
        logger.info(f"Registered alert rule: {rule.name}")

    async def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert rules against current metrics."""
        for rule_name, rule in self.alert_rules.items():
            if await rule.evaluate(metrics):
                await self.trigger_alert(rule, metrics)

    async def trigger_alert(self, rule: "AlertRule", metrics: Dict[str, Any]):
        """Trigger an alert."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "rule_name": rule.name,
            "severity": rule.severity,
            "model_name": rule.model_name,
            "message": rule.message_template.format(**metrics),
            "metrics": metrics,
        }

        # Record in metrics
        alert_triggered.labels(
            alert_name=rule.name, severity=rule.severity, model_name=rule.model_name
        ).inc()

        # Store in history
        self.alert_history.append(alert)

        # Send notifications
        await self._send_notifications(alert)

        logger.warning(f"Alert triggered: {alert['message']}")

    async def _send_notifications(self, alert: Dict):
        """Send alert notifications."""
        # Slack webhook
        if self.config.alert_webhook_url:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                payload = {
                    "text": f"🚨 *{alert['severity'].upper()} Alert*",
                    "attachments": [
                        {
                            "color": "danger"
                            if alert["severity"] == "critical"
                            else "warning",
                            "fields": [
                                {
                                    "title": "Rule",
                                    "value": alert["rule_name"],
                                    "short": True,
                                },
                                {
                                    "title": "Model",
                                    "value": alert["model_name"],
                                    "short": True,
                                },
                                {
                                    "title": "Message",
                                    "value": alert["message"],
                                    "short": False,
                                },
                                {
                                    "title": "Time",
                                    "value": alert["timestamp"],
                                    "short": True,
                                },
                            ],
                        }
                    ],
                }
                await session.post(self.config.alert_webhook_url, json=payload)

        # Email notification (implement based on your email service)
        # PagerDuty integration (implement if needed)


@dataclass
class AlertRule:
    """Alert rule definition."""

    name: str
    model_name: str
    condition: str  # Expression to evaluate
    threshold: float
    severity: str  # info, warning, critical
    message_template: str
    cooldown_minutes: int = 5
    last_triggered: Optional[datetime] = None

    async def evaluate(self, metrics: Dict[str, Any]) -> bool:
        """Evaluate if alert should be triggered."""
        # Check cooldown
        if self.last_triggered:
            cooldown_delta = timedelta(minutes=self.cooldown_minutes)
            if datetime.now() - self.last_triggered < cooldown_delta:
                return False

        # Evaluate condition
        try:
            # Simple evaluation - in production use safe evaluation
            value = eval(self.condition, {"metrics": metrics})
            if value > self.threshold:
                self.last_triggered = datetime.now()
                return True
        except Exception as e:
            logger.error(f"Error evaluating alert rule {self.name}: {e}")

        return False


class PerformanceProfiler:
    """Performance profiling for model inference."""

    def __init__(self):
        self.profiles = {}

    def profile_inference(self, model_name: str):
        """Decorator for profiling model inference."""

        def decorator(func):
            async def wrapper(*args, **kwargs):
                import cProfile
                import pstats
                from io import StringIO

                profiler = cProfile.Profile()
                profiler.enable()

                try:
                    result = await func(*args, **kwargs)
                finally:
                    profiler.disable()

                    # Get profile stats
                    stream = StringIO()
                    stats = pstats.Stats(profiler, stream=stream)
                    stats.sort_stats("cumulative")
                    stats.print_stats(20)

                    # Store profile
                    self.profiles[f"{model_name}_{datetime.now().isoformat()}"] = {
                        "stats": stream.getvalue(),
                        "total_time": stats.total_tt,
                        "function_calls": stats.total_calls,
                    }

                return result

            return wrapper

        return decorator

    def get_profile_summary(self, model_name: str) -> Dict:
        """Get profiling summary for model."""
        model_profiles = [
            p for k, p in self.profiles.items() if k.startswith(model_name)
        ]

        if not model_profiles:
            return {}

        return {
            "average_time": np.mean([p["total_time"] for p in model_profiles]),
            "average_calls": np.mean([p["function_calls"] for p in model_profiles]),
            "profile_count": len(model_profiles),
        }


class LogAggregator:
    """Aggregate and analyze logs from multiple sources."""

    def __init__(self, es_client: Elasticsearch):
        self.es_client = es_client

    async def aggregate_logs(self, model_name: str, time_range: str = "1h") -> Dict:
        """Aggregate logs for analysis."""
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"model_name": model_name}},
                        {"range": {"timestamp": {"gte": f"now-{time_range}"}}},
                    ]
                }
            },
            "aggs": {
                "error_types": {"terms": {"field": "error_type.keyword"}},
                "latency_percentiles": {"percentiles": {"field": "latency_ms"}},
                "request_timeline": {
                    "date_histogram": {"field": "timestamp", "interval": "1m"}
                },
            },
        }

        result = self.es_client.search(index=f"ml-inference-*", body=query)

        return {
            "total_requests": result["hits"]["total"]["value"],
            "error_distribution": result["aggregations"]["error_types"]["buckets"],
            "latency_percentiles": result["aggregations"]["latency_percentiles"][
                "values"
            ],
            "request_timeline": result["aggregations"]["request_timeline"]["buckets"],
        }

    async def detect_anomalies(self, model_name: str) -> List[Dict]:
        """Detect anomalies in logs."""
        # Get recent logs
        logs = await self.aggregate_logs(model_name, "24h")

        anomalies = []

        # Check for unusual error patterns
        if logs["error_distribution"]:
            for error_type in logs["error_distribution"]:
                if error_type["doc_count"] > 100:  # Threshold
                    anomalies.append(
                        {
                            "type": "high_error_rate",
                            "error_type": error_type["key"],
                            "count": error_type["doc_count"],
                        }
                    )

        # Check for latency spikes
        if logs["latency_percentiles"]:
            p95 = logs["latency_percentiles"].get("95.0", 0)
            if p95 > 1000:  # 1 second
                anomalies.append({"type": "high_latency", "p95_latency": p95})

        return anomalies


# Monitoring utilities
def setup_monitoring(
    config: MonitoringConfig,
) -> Tuple[ModelMonitoringService, AlertingService]:
    """Setup monitoring infrastructure."""
    monitoring_service = ModelMonitoringService(config)
    alerting_service = AlertingService(config)

    # Register default alert rules
    default_rules = [
        AlertRule(
            name="high_error_rate",
            model_name="*",
            condition="metrics.get('error_rate', 0)",
            threshold=0.05,
            severity="critical",
            message_template="High error rate detected: {error_rate:.2%}",
        ),
        AlertRule(
            name="high_latency",
            model_name="*",
            condition="metrics.get('p95_latency', 0)",
            threshold=1.0,
            severity="warning",
            message_template="High latency detected: {p95_latency:.3f}s",
        ),
        AlertRule(
            name="data_drift",
            model_name="*",
            condition="metrics.get('max_drift_score', 0)",
            threshold=0.15,
            severity="warning",
            message_template="Data drift detected: {max_drift_score:.3f}",
        ),
        AlertRule(
            name="low_accuracy",
            model_name="*",
            condition="1 - metrics.get('accuracy', 1)",
            threshold=0.2,
            severity="critical",
            message_template="Low model accuracy: {accuracy:.2%}",
        ),
    ]

    for rule in default_rules:
        alerting_service.register_alert_rule(rule)

    return monitoring_service, alerting_service
