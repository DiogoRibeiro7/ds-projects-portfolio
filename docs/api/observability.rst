Observability Module
====================

.. currentmodule:: src.utils

Observability utilities for monitoring, tracing, and performance tracking.

Observability Module
--------------------

.. automodule:: src.utils.observability
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Key Components
--------------

MetricsCollector
~~~~~~~~~~~~~~~~

.. autoclass:: src.utils.observability.MetricsCollector
   :members:
   :special-members: __init__
   :show-inheritance:

TracingManager
~~~~~~~~~~~~~~

.. autoclass:: src.utils.observability.TracingManager
   :members:
   :special-members: __init__
   :show-inheritance:

PerformanceMonitor
~~~~~~~~~~~~~~~~~~

.. autoclass:: src.utils.observability.PerformanceMonitor
   :members:
   :special-members: __init__
   :show-inheritance:

AlertingSystem
~~~~~~~~~~~~~~

.. autoclass:: src.utils.observability.AlertingSystem
   :members:
   :special-members: __init__
   :show-inheritance:

Usage Examples
--------------

Metrics Collection
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.utils.observability import MetricsCollector

    # Initialize metrics collector
    metrics = MetricsCollector(
        backend='prometheus',
        port=8000
    )

    # Record metrics
    metrics.increment('api_requests', tags={'endpoint': '/predict'})
    metrics.gauge('model_accuracy', 0.95)
    metrics.histogram('response_time', 0.150, tags={'service': 'ml_api'})

    # Create custom metric
    metrics.create_metric(
        name='experiment_conversions',
        type='counter',
        description='Number of successful conversions in experiments'
    )

Distributed Tracing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.utils.observability import TracingManager

    # Initialize tracing
    tracer = TracingManager(
        service_name='data_pipeline',
        backend='jaeger'
    )

    # Create trace
    with tracer.start_span('process_batch') as span:
        span.set_tag('batch_size', 1000)

        # Process data
        with tracer.start_span('load_data', parent=span):
            data = load_data()

        with tracer.start_span('transform_data', parent=span):
            transformed = transform_data(data)

        with tracer.start_span('save_results', parent=span):
            save_results(transformed)

        span.set_tag('status', 'success')

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.utils.observability import PerformanceMonitor

    # Initialize monitor
    monitor = PerformanceMonitor()

    # Monitor function performance
    @monitor.profile
    def expensive_operation(data):
        # Your code here
        return process(data)

    # Monitor code block
    with monitor.timer('data_processing'):
        result = process_large_dataset()

    # Get performance report
    report = monitor.get_report()
    print(f"Average execution time: {report['avg_time']}")
    print(f"P95 latency: {report['p95']}")

Alerting
~~~~~~~~

.. code-block:: python

    from src.utils.observability import AlertingSystem

    # Initialize alerting
    alerts = AlertingSystem(
        channels=['slack', 'email', 'pagerduty']
    )

    # Define alert rules
    alerts.add_rule(
        name='high_error_rate',
        condition='error_rate > 0.05',
        severity='critical',
        action='notify_oncall'
    )

    # Trigger alert manually
    alerts.trigger(
        alert='model_drift_detected',
        severity='warning',
        metadata={
            'model': 'churn_prediction',
            'drift_score': 0.15
        }
    )

Observability Best Practices
----------------------------

1. **Golden Signals**: Monitor latency, traffic, errors, and saturation
2. **Structured Logging**: Use consistent, structured log formats
3. **Context Propagation**: Pass trace context through all service calls
4. **Service Level Objectives**: Define and monitor SLOs for critical services
5. **Alert Fatigue**: Avoid noisy alerts; alert only on actionable issues
6. **Dashboards**: Create focused dashboards for different stakeholders
7. **Retention**: Balance data retention with storage costs