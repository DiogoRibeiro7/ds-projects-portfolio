"""
Test suite for observability utilities.
"""

import pytest
import logging
import json
import time
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.utils.observability import (
    MetricsCollector,
    Logger,
    Tracer,
    AlertManager,
    HealthChecker,
    PerformanceMonitor,
    ErrorTracker
)

class TestLogger:
    """Test suite for logging functionality."""

    @pytest.fixture
    def logger(self):
        """Create logger instance."""
        return Logger(
            name="test_logger",
            level=logging.INFO,
            log_file="test.log"
        )

    def test_log_levels(self, logger):
        """Test different log levels."""
        with patch.object(logger._logger, 'debug') as mock_debug:
            logger.debug("Debug message")
            mock_debug.assert_called_once()

        with patch.object(logger._logger, 'info') as mock_info:
            logger.info("Info message")
            mock_info.assert_called_once()

        with patch.object(logger._logger, 'warning') as mock_warning:
            logger.warning("Warning message")
            mock_warning.assert_called_once()

        with patch.object(logger._logger, 'error') as mock_error:
            logger.error("Error message")
            mock_error.assert_called_once()

    def test_structured_logging(self, logger):
        """Test structured logging with metadata."""
        with patch.object(logger._logger, 'info') as mock_info:
            logger.info("User action", extra={
                'user_id': '123',
                'action': 'login',
                'ip_address': '192.168.1.1'
            })

            call_args = mock_info.call_args
            assert 'extra' in call_args.kwargs
            assert call_args.kwargs['extra']['user_id'] == '123'

    def test_log_rotation(self, logger):
        """Test log file rotation."""
        with patch('logging.handlers.RotatingFileHandler') as mock_handler:
            rotating_logger = Logger(
                name="rotating",
                log_file="rotating.log",
                max_bytes=1024*1024,  # 1MB
                backup_count=5
            )

            mock_handler.assert_called_with(
                'rotating.log',
                maxBytes=1024*1024,
                backupCount=5
            )

    def test_json_logging(self):
        """Test JSON formatted logging."""
        from src.utils.observability import JSONLogger

        json_logger = JSONLogger("json_test")

        with patch('sys.stdout.write') as mock_write:
            json_logger.info("Test message", extra={'key': 'value'})

            # Verify JSON output
            call_args = mock_write.call_args
            log_entry = json.loads(call_args[0][0])
            assert log_entry['message'] == "Test message"
            assert log_entry['key'] == 'value'

    def test_context_logging(self, logger):
        """Test logging with context manager."""
        with logger.context(request_id='req-123'):
            with patch.object(logger._logger, 'info') as mock_info:
                logger.info("Within context")

                call_args = mock_info.call_args
                assert 'request_id' in call_args.kwargs.get('extra', {})

class TestMetricsCollector:
    """Test suite for metrics collection."""

    @pytest.fixture
    def metrics(self):
        """Create metrics collector instance."""
        return MetricsCollector(
            namespace="test_app",
            flush_interval=1
        )

    def test_counter_metric(self, metrics):
        """Test counter metrics."""
        metrics.increment("api_calls", tags={"endpoint": "/users"})
        metrics.increment("api_calls", tags={"endpoint": "/users"})
        metrics.increment("api_calls", value=3, tags={"endpoint": "/posts"})

        stats = metrics.get_stats("api_calls")
        assert stats['total'] == 5
        assert stats['by_tags']['/users'] == 2
        assert stats['by_tags']['/posts'] == 3

    def test_gauge_metric(self, metrics):
        """Test gauge metrics."""
        metrics.gauge("memory_usage", 75.5, tags={"host": "server1"})
        metrics.gauge("memory_usage", 82.3, tags={"host": "server2"})

        current_value = metrics.get_gauge("memory_usage", {"host": "server1"})
        assert current_value == 75.5

    def test_histogram_metric(self, metrics):
        """Test histogram metrics."""
        for i in range(100):
            metrics.histogram("response_time", np.random.normal(200, 50))

        stats = metrics.get_histogram_stats("response_time")
        assert 'mean' in stats
        assert 'median' in stats
        assert 'p95' in stats
        assert 'p99' in stats

    def test_timer_metric(self, metrics):
        """Test timer metrics."""
        with metrics.timer("operation_duration"):
            time.sleep(0.1)

        duration = metrics.get_timer_stats("operation_duration")
        assert duration['last'] >= 0.1

    def test_metric_aggregation(self, metrics):
        """Test metric aggregation over time windows."""
        # Add metrics over time
        for i in range(10):
            metrics.increment("requests", timestamp=time.time() - i)

        # Get aggregated stats
        aggregated = metrics.aggregate(
            "requests",
            window="1m",
            aggregation="sum"
        )

        assert aggregated > 0

    def test_custom_metrics(self, metrics):
        """Test custom metric types."""
        # Rate metric
        metrics.rate("error_rate", errors=5, total=100)
        rate = metrics.get_rate("error_rate")
        assert rate == 0.05

        # Percentile metric
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for v in values:
            metrics.record("latency", v)

        p50 = metrics.get_percentile("latency", 50)
        p90 = metrics.get_percentile("latency", 90)
        assert p50 < p90

class TestTracer:
    """Test suite for distributed tracing."""

    @pytest.fixture
    def tracer(self):
        """Create tracer instance."""
        return Tracer(service_name="test_service")

    def test_span_creation(self, tracer):
        """Test span creation and attributes."""
        with tracer.start_span("test_operation") as span:
            span.set_attribute("user_id", "123")
            span.set_attribute("request_size", 1024)

            assert span.name == "test_operation"
            assert span.attributes["user_id"] == "123"

    def test_nested_spans(self, tracer):
        """Test nested span hierarchies."""
        with tracer.start_span("parent") as parent:
            with tracer.start_span("child", parent=parent) as child:
                assert child.parent_id == parent.span_id
                assert child.trace_id == parent.trace_id

    def test_span_events(self, tracer):
        """Test adding events to spans."""
        with tracer.start_span("operation") as span:
            span.add_event("cache_hit", attributes={"key": "user:123"})
            span.add_event("database_query", attributes={"table": "users"})

            assert len(span.events) == 2
            assert span.events[0]["name"] == "cache_hit"

    def test_span_status(self, tracer):
        """Test span status and error handling."""
        # Successful span
        with tracer.start_span("success") as span:
            span.set_status("OK")
            assert span.status == "OK"

        # Error span
        try:
            with tracer.start_span("error") as span:
                raise ValueError("Test error")
        except ValueError:
            assert span.status == "ERROR"
            assert "ValueError" in span.attributes.get("error.type", "")

    def test_distributed_context(self, tracer):
        """Test distributed tracing context propagation."""
        # Create parent span
        with tracer.start_span("service_a") as parent:
            context = tracer.inject_context(parent)

            # Simulate receiving context in another service
            with tracer.start_span_from_context("service_b", context) as child:
                assert child.trace_id == parent.trace_id
                assert child.parent_id == parent.span_id

    def test_sampling(self, tracer):
        """Test trace sampling strategies."""
        # Always sample
        sampler_always = tracer.create_sampler(rate=1.0)
        assert all(sampler_always.should_sample() for _ in range(100))

        # 50% sampling
        sampler_half = tracer.create_sampler(rate=0.5)
        samples = [sampler_half.should_sample() for _ in range(1000)]
        assert 0.4 < sum(samples) / len(samples) < 0.6

class TestAlertManager:
    """Test suite for alert management."""

    @pytest.fixture
    def alert_manager(self):
        """Create alert manager instance."""
        return AlertManager()

    def test_alert_creation(self, alert_manager):
        """Test alert creation and properties."""
        alert = alert_manager.create_alert(
            name="high_cpu",
            severity="warning",
            message="CPU usage above 80%",
            metadata={"host": "server1", "cpu_usage": 85}
        )

        assert alert['name'] == "high_cpu"
        assert alert['severity'] == "warning"
        assert alert['metadata']['cpu_usage'] == 85

    def test_alert_rules(self, alert_manager):
        """Test alert rule evaluation."""
        # Define rule
        alert_manager.add_rule(
            name="memory_alert",
            condition=lambda metrics: metrics.get('memory_usage', 0) > 90,
            severity="critical",
            message="Memory usage critical: {memory_usage}%"
        )

        # Trigger rule
        alerts = alert_manager.evaluate_rules({'memory_usage': 95})
        assert len(alerts) == 1
        assert alerts[0]['severity'] == "critical"
        assert "95%" in alerts[0]['message']

    def test_alert_deduplication(self, alert_manager):
        """Test alert deduplication."""
        # Send same alert multiple times
        for _ in range(5):
            alert_manager.send_alert("duplicate_alert", "warning", "Test alert")

        sent_alerts = alert_manager.get_sent_alerts()
        # Should only send once within dedup window
        assert sent_alerts.count("duplicate_alert") == 1

    def test_alert_throttling(self, alert_manager):
        """Test alert throttling."""
        alert_manager.set_throttle("test_alert", max_per_hour=2)

        # Try to send 5 alerts
        results = []
        for _ in range(5):
            result = alert_manager.send_alert("test_alert", "info", "Test")
            results.append(result)

        # Only 2 should be sent
        assert sum(results) == 2

    def test_alert_channels(self, alert_manager):
        """Test different alert channels."""
        with patch('smtplib.SMTP') as mock_smtp:
            alert_manager.configure_email_channel(
                smtp_host="smtp.example.com",
                from_email="alerts@example.com",
                to_emails=["admin@example.com"]
            )

            alert_manager.send_alert(
                "test",
                "critical",
                "Test alert",
                channels=["email"]
            )

            mock_smtp.assert_called()

    def test_alert_escalation(self, alert_manager):
        """Test alert escalation policies."""
        alert_manager.add_escalation_policy(
            name="critical_escalation",
            levels=[
                {"delay": 0, "contacts": ["oncall@example.com"]},
                {"delay": 300, "contacts": ["manager@example.com"]},
                {"delay": 900, "contacts": ["director@example.com"]}
            ]
        )

        escalation = alert_manager.get_escalation_contacts(
            "critical_escalation",
            alert_age=600  # 10 minutes old
        )

        assert "manager@example.com" in escalation

class TestHealthChecker:
    """Test suite for health checking."""

    @pytest.fixture
    def health_checker(self):
        """Create health checker instance."""
        return HealthChecker()

    def test_health_check_registration(self, health_checker):
        """Test registering health checks."""
        def database_check():
            return {"status": "healthy", "latency": 5}

        def cache_check():
            return {"status": "healthy", "latency": 1}

        health_checker.register_check("database", database_check)
        health_checker.register_check("cache", cache_check)

        assert len(health_checker.checks) == 2

    def test_run_health_checks(self, health_checker):
        """Test running all health checks."""
        health_checker.register_check(
            "service1",
            lambda: {"status": "healthy"}
        )
        health_checker.register_check(
            "service2",
            lambda: {"status": "unhealthy", "error": "Connection failed"}
        )

        results = health_checker.run_all_checks()

        assert results['overall_status'] == "unhealthy"
        assert results['checks']['service1']['status'] == "healthy"
        assert results['checks']['service2']['status'] == "unhealthy"

    def test_health_check_timeout(self, health_checker):
        """Test health check timeout handling."""
        def slow_check():
            time.sleep(5)
            return {"status": "healthy"}

        health_checker.register_check("slow", slow_check, timeout=1)
        results = health_checker.run_all_checks()

        assert results['checks']['slow']['status'] == "timeout"

    def test_liveness_probe(self, health_checker):
        """Test liveness probe for container orchestration."""
        liveness = health_checker.liveness_probe()

        assert 'status' in liveness
        assert 'timestamp' in liveness
        assert liveness['status'] in ["alive", "dead"]

    def test_readiness_probe(self, health_checker):
        """Test readiness probe."""
        # Add dependency checks
        health_checker.add_dependency("database", lambda: True)
        health_checker.add_dependency("cache", lambda: True)

        readiness = health_checker.readiness_probe()

        assert readiness['ready'] == True
        assert all(readiness['dependencies'].values())

class TestPerformanceMonitor:
    """Test suite for performance monitoring."""

    @pytest.fixture
    def perf_monitor(self):
        """Create performance monitor instance."""
        return PerformanceMonitor()

    def test_cpu_monitoring(self, perf_monitor):
        """Test CPU usage monitoring."""
        with patch('psutil.cpu_percent') as mock_cpu:
            mock_cpu.return_value = 65.5

            cpu_stats = perf_monitor.get_cpu_stats()

            assert cpu_stats['usage_percent'] == 65.5
            assert 'cores' in cpu_stats

    def test_memory_monitoring(self, perf_monitor):
        """Test memory usage monitoring."""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                percent=75.0,
                used=8000000000,
                total=16000000000
            )

            memory_stats = perf_monitor.get_memory_stats()

            assert memory_stats['usage_percent'] == 75.0
            assert memory_stats['used_gb'] == pytest.approx(7.45, rel=0.1)

    def test_disk_monitoring(self, perf_monitor):
        """Test disk I/O monitoring."""
        disk_stats = perf_monitor.get_disk_stats()

        assert 'read_bytes' in disk_stats
        assert 'write_bytes' in disk_stats
        assert 'usage_percent' in disk_stats

    def test_network_monitoring(self, perf_monitor):
        """Test network I/O monitoring."""
        network_stats = perf_monitor.get_network_stats()

        assert 'bytes_sent' in network_stats
        assert 'bytes_received' in network_stats
        assert 'packets_sent' in network_stats
        assert 'packets_received' in network_stats

    def test_performance_profiling(self, perf_monitor):
        """Test code performance profiling."""
        @perf_monitor.profile
        def slow_function():
            time.sleep(0.1)
            return sum(range(1000000))

        result = slow_function()
        profile = perf_monitor.get_profile("slow_function")

        assert profile['execution_time'] >= 0.1
        assert profile['call_count'] == 1

    def test_resource_limits(self, perf_monitor):
        """Test resource limit monitoring."""
        perf_monitor.set_limits({
            'cpu': 80,
            'memory': 90,
            'disk': 85
        })

        violations = perf_monitor.check_limits()

        # Mock current usage to test violations
        with patch.object(perf_monitor, 'get_cpu_stats') as mock_cpu:
            mock_cpu.return_value = {'usage_percent': 85}

            violations = perf_monitor.check_limits()
            assert 'cpu' in violations

class TestErrorTracker:
    """Test suite for error tracking."""

    @pytest.fixture
    def error_tracker(self):
        """Create error tracker instance."""
        return ErrorTracker()

    def test_error_capture(self, error_tracker):
        """Test error capture and storage."""
        try:
            raise ValueError("Test error")
        except ValueError as e:
            error_id = error_tracker.capture_exception(
                e,
                context={"user_id": "123", "endpoint": "/api/test"}
            )

        error = error_tracker.get_error(error_id)
        assert error['type'] == "ValueError"
        assert error['message'] == "Test error"
        assert error['context']['user_id'] == "123"

    def test_error_grouping(self, error_tracker):
        """Test error grouping by similarity."""
        # Capture similar errors
        for i in range(5):
            try:
                raise ValueError(f"Database connection failed: timeout {i}")
            except ValueError as e:
                error_tracker.capture_exception(e)

        groups = error_tracker.get_error_groups()
        # Should group similar errors together
        assert len(groups) == 1
        assert groups[0]['count'] == 5

    def test_error_rate_tracking(self, error_tracker):
        """Test error rate calculation."""
        # Simulate errors over time
        for i in range(10):
            error_tracker.capture_exception(Exception("Test"))

        error_rate = error_tracker.get_error_rate(window="1m")
        assert error_rate > 0

    def test_error_notifications(self, error_tracker):
        """Test error notification triggers."""
        with patch.object(error_tracker, 'send_notification') as mock_notify:
            error_tracker.set_notification_threshold(5)

            # Trigger threshold
            for i in range(6):
                error_tracker.capture_exception(Exception("Critical"))

            mock_notify.assert_called()

    def test_error_recovery_suggestions(self, error_tracker):
        """Test automatic recovery suggestions."""
        error_tracker.add_recovery_suggestion(
            error_pattern="connection.*timeout",
            suggestion="Check network connectivity and retry"
        )

        try:
            raise ConnectionError("connection timeout")
        except ConnectionError as e:
            error_id = error_tracker.capture_exception(e)

        suggestions = error_tracker.get_recovery_suggestions(error_id)
        assert len(suggestions) > 0
        assert "network connectivity" in suggestions[0]