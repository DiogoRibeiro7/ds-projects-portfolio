"""
Observability Module
Comprehensive monitoring, logging, and tracing for ML systems
"""

import logging
import time
import json
import sys
import os
from typing import Dict, Any, Optional, List, Callable
from functools import wraps
from contextlib import contextmanager
from datetime import datetime, timedelta
import threading
import asyncio
from dataclasses import dataclass, asdict

# OpenTelemetry imports
from opentelemetry import trace, metrics, baggage
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

# Structured logging
import structlog
from pythonjsonlogger import jsonlogger

# Metrics
from prometheus_client import Counter, Histogram, Gauge, Summary, Info
from prometheus_client import generate_latest, REGISTRY

# Distributed tracing correlation
import uuid
import contextvars

# Performance profiling
import cProfile
import pstats
import io
import tracemalloc
import psutil


# Context variables for tracing
trace_id_var = contextvars.ContextVar("trace_id", default=None)
span_id_var = contextvars.ContextVar("span_id", default=None)


@dataclass
class MLMetrics:
    """ML-specific metrics"""

    model_name: str
    model_version: str
    prediction_count: int = 0
    prediction_latency_ms: float = 0.0
    feature_count: int = 0
    data_drift_score: float = 0.0
    model_accuracy: float = 0.0
    model_auc: float = 0.0
    cache_hit_rate: float = 0.0


class ObservabilityManager:
    """Centralized observability management"""

    def __init__(
        self, service_name: str = "ml-portfolio", environment: str = "production"
    ):
        self.service_name = service_name
        self.environment = environment
        self.logger = None
        self.tracer = None
        self.meter = None
        self.metrics_registry = {}

        # Initialize components
        self._setup_logging()
        self._setup_tracing()
        self._setup_metrics()
        self._setup_instrumentation()

    def _setup_logging(self):
        """Configure structured logging"""
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                self._add_trace_context,
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # Configure Python logging
        logHandler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        logHandler.setFormatter(formatter)

        logging.root.handlers = [logHandler]
        logging.root.setLevel(logging.INFO)

        self.logger = structlog.get_logger(self.service_name)

    def _add_trace_context(self, logger, method_name, event_dict):
        """Add trace context to logs"""
        trace_id = trace_id_var.get()
        span_id = span_id_var.get()

        if trace_id:
            event_dict["trace_id"] = trace_id
        if span_id:
            event_dict["span_id"] = span_id

        event_dict["service"] = self.service_name
        event_dict["environment"] = self.environment

        return event_dict

    def _setup_tracing(self):
        """Configure OpenTelemetry tracing"""
        resource = Resource.create(
            {
                "service.name": self.service_name,
                "service.version": "1.0.0",
                "deployment.environment": self.environment,
            }
        )

        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)

        # Add OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317"),
            insecure=True,
        )
        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)

        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)
        self.tracer = trace.get_tracer(__name__)

    def _setup_metrics(self):
        """Configure Prometheus metrics"""
        # Standard metrics
        self.request_counter = Counter(
            "requests_total",
            "Total number of requests",
            ["method", "endpoint", "status"],
        )

        self.request_duration = Histogram(
            "request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        # ML-specific metrics
        self.model_prediction_counter = Counter(
            "model_predictions_total",
            "Total model predictions",
            ["model_name", "model_version"],
        )

        self.model_prediction_duration = Histogram(
            "model_prediction_duration_seconds",
            "Model prediction duration",
            ["model_name", "model_version"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )

        self.data_drift_gauge = Gauge(
            "data_drift_score", "Data drift score", ["feature_name"]
        )

        self.model_accuracy_gauge = Gauge(
            "model_accuracy", "Model accuracy score", ["model_name", "model_version"]
        )

        self.cache_hit_rate = Gauge("cache_hit_rate", "Cache hit rate", ["cache_type"])

        # System metrics
        self.cpu_usage = Gauge("cpu_usage_percent", "CPU usage percentage")
        self.memory_usage = Gauge("memory_usage_bytes", "Memory usage in bytes")
        self.disk_usage = Gauge("disk_usage_percent", "Disk usage percentage")

        # Create meter for OpenTelemetry metrics
        meter_provider = MeterProvider(
            resource=resource, metric_readers=[PrometheusMetricReader()]
        )
        metrics.set_meter_provider(meter_provider)
        self.meter = metrics.get_meter(__name__)

    def _setup_instrumentation(self):
        """Setup automatic instrumentation"""
        # Instrument libraries
        RequestsInstrumentor().instrument()
        RedisInstrumentor().instrument()
        Psycopg2Instrumentor().instrument()
        # Note: FastAPI and SQLAlchemy instrumentors should be called with instances

    @contextmanager
    def trace_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a trace span"""
        with self.tracer.start_as_current_span(name) as span:
            # Set trace context
            trace_id = format(span.get_span_context().trace_id, "032x")
            span_id = format(span.get_span_context().span_id, "016x")
            trace_id_var.set(trace_id)
            span_id_var.set(span_id)

            # Add attributes
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)

            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
            finally:
                trace_id_var.set(None)
                span_id_var.set(None)

    def trace_function(self, func: Callable):
        """Decorator to trace function execution"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.trace_span(f"{func.__module__}.{func.__name__}"):
                return func(*args, **kwargs)

        return wrapper

    def trace_async_function(self, func: Callable):
        """Decorator to trace async function execution"""

        @wraps(func)
        async def wrapper(*args, **kwargs):
            with self.trace_span(f"{func.__module__}.{func.__name__}"):
                return await func(*args, **kwargs)

        return wrapper

    def log_ml_metrics(self, metrics: MLMetrics):
        """Log ML-specific metrics"""
        # Update Prometheus metrics
        self.model_prediction_counter.labels(
            model_name=metrics.model_name, model_version=metrics.model_version
        ).inc(metrics.prediction_count)

        self.model_prediction_duration.labels(
            model_name=metrics.model_name, model_version=metrics.model_version
        ).observe(metrics.prediction_latency_ms / 1000)

        self.model_accuracy_gauge.labels(
            model_name=metrics.model_name, model_version=metrics.model_version
        ).set(metrics.model_accuracy)

        # Log structured event
        self.logger.info("ml_metrics", **asdict(metrics))

    def log_performance_metrics(self):
        """Log system performance metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_usage.set(cpu_percent)

        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_usage.set(memory.used)

        # Disk usage
        disk = psutil.disk_usage("/")
        self.disk_usage.set(disk.percent)

        self.logger.info(
            "system_metrics",
            cpu_usage=cpu_percent,
            memory_used=memory.used,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
        )

    @contextmanager
    def profile_code(self, label: str = "profile"):
        """Profile code execution"""
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            yield
        finally:
            profiler.disable()

            # Get statistics
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
            ps.print_stats(20)

            self.logger.info("code_profile", label=label, profile_stats=s.getvalue())

    @contextmanager
    def trace_memory(self, label: str = "memory_trace"):
        """Trace memory allocations"""
        tracemalloc.start()

        try:
            yield
        finally:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            self.logger.info(
                "memory_trace",
                label=label,
                current_mb=current / 1024 / 1024,
                peak_mb=peak / 1024 / 1024,
            )


class MetricsCollector:
    """Collects and aggregates metrics"""

    def __init__(self, observability: ObservabilityManager):
        self.observability = observability
        self.metrics_buffer = []
        self.lock = threading.Lock()

    def collect_prediction_metrics(
        self,
        model_name: str,
        model_version: str,
        latency_ms: float,
        features: Dict[str, Any],
    ):
        """Collect prediction metrics"""
        metrics = MLMetrics(
            model_name=model_name,
            model_version=model_version,
            prediction_count=1,
            prediction_latency_ms=latency_ms,
            feature_count=len(features),
        )

        with self.lock:
            self.metrics_buffer.append(metrics)

        # Log immediately for real-time monitoring
        self.observability.log_ml_metrics(metrics)

    def flush_metrics(self):
        """Flush aggregated metrics"""
        with self.lock:
            if not self.metrics_buffer:
                return

            # Aggregate metrics
            aggregated = {}
            for metric in self.metrics_buffer:
                key = (metric.model_name, metric.model_version)
                if key not in aggregated:
                    aggregated[key] = {
                        "count": 0,
                        "total_latency": 0,
                        "feature_counts": [],
                    }

                aggregated[key]["count"] += metric.prediction_count
                aggregated[key]["total_latency"] += metric.prediction_latency_ms
                aggregated[key]["feature_counts"].append(metric.feature_count)

            # Log aggregated metrics
            for (model_name, model_version), data in aggregated.items():
                avg_latency = data["total_latency"] / data["count"]
                avg_features = sum(data["feature_counts"]) / len(data["feature_counts"])

                self.observability.logger.info(
                    "aggregated_metrics",
                    model_name=model_name,
                    model_version=model_version,
                    prediction_count=data["count"],
                    avg_latency_ms=avg_latency,
                    avg_feature_count=avg_features,
                )

            # Clear buffer
            self.metrics_buffer.clear()


class AlertManager:
    """Manages alerts and notifications"""

    def __init__(self, observability: ObservabilityManager):
        self.observability = observability
        self.alert_rules = []
        self.alert_history = []

    def add_rule(self, rule: Dict[str, Any]):
        """Add alert rule"""
        self.alert_rules.append(rule)

    def check_alerts(self, metrics: Dict[str, float]):
        """Check metrics against alert rules"""
        for rule in self.alert_rules:
            metric_value = metrics.get(rule["metric"])
            if metric_value is None:
                continue

            # Evaluate condition
            triggered = False
            if rule["condition"] == "gt" and metric_value > rule["threshold"]:
                triggered = True
            elif rule["condition"] == "lt" and metric_value < rule["threshold"]:
                triggered = True

            if triggered:
                self._trigger_alert(rule, metric_value)

    def _trigger_alert(self, rule: Dict[str, Any], value: float):
        """Trigger an alert"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "rule": rule["name"],
            "severity": rule["severity"],
            "metric": rule["metric"],
            "value": value,
            "threshold": rule["threshold"],
            "message": f"{rule['metric']} is {value}, threshold is {rule['threshold']}",
        }

        self.alert_history.append(alert)
        self.observability.logger.warning("alert_triggered", **alert)

        # Send notifications (implement based on requirements)
        # self._send_notification(alert)


# Global observability instance
observability = ObservabilityManager()

# Convenience decorators
trace = observability.trace_function
trace_async = observability.trace_async_function


# Example usage
if __name__ == "__main__":
    # Initialize observability
    obs = ObservabilityManager(service_name="ml-api")

    # Example: Trace a function
    @obs.trace_function
    def process_data(data: List[float]) -> float:
        """Example computation used to illustrate tracing behavior."""
        time.sleep(0.1)  # Simulate processing
        return sum(data) / len(data)

    # Example: Log ML metrics
    ml_metrics = MLMetrics(
        model_name="xgboost_classifier",
        model_version="v1.0",
        prediction_count=100,
        prediction_latency_ms=25.5,
        model_accuracy=0.92,
        model_auc=0.95,
    )
    obs.log_ml_metrics(ml_metrics)

    # Example: Profile code
    with obs.profile_code("data_processing"):
        result = process_data([1, 2, 3, 4, 5])

    # Example: Trace memory
    with obs.trace_memory("model_loading"):
        # Simulate model loading
        large_array = [0] * 10000000

    # Log system metrics
    obs.log_performance_metrics()
