Data Quality Module
===================

.. currentmodule:: src.data_quality

Comprehensive data quality framework for validation, monitoring, and reporting.

Quality Framework
-----------------

.. automodule:: src.data_quality.quality_framework
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Quality Dashboard
-----------------

.. automodule:: src.data_quality.quality_dashboard
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
-----------

DataQualityChecker
~~~~~~~~~~~~~~~~~~

.. autoclass:: src.data_quality.quality_framework.DataQualityChecker
   :members:
   :special-members: __init__
   :show-inheritance:

QualityMetrics
~~~~~~~~~~~~~~

.. autoclass:: src.data_quality.quality_framework.QualityMetrics
   :members:
   :special-members: __init__
   :show-inheritance:

Usage Examples
--------------

Data Validation
~~~~~~~~~~~~~~~

.. code-block:: python

    from src.data_quality.quality_framework import DataQualityChecker
    import pandas as pd

    # Load data
    df = pd.read_csv('data.csv')

    # Initialize quality checker
    checker = DataQualityChecker()

    # Define validation rules
    rules = {
        'age': {'min': 0, 'max': 120, 'nullable': False},
        'email': {'pattern': r'^[\w\.-]+@[\w\.-]+\.\w+$'},
        'revenue': {'min': 0, 'nullable': True}
    }

    # Run validation
    results = checker.validate(df, rules)

    # Generate report
    report = checker.generate_report(results)
    print(f"Data quality score: {report['overall_score']}")
    print(f"Issues found: {report['total_issues']}")

Data Profiling
~~~~~~~~~~~~~~

.. code-block:: python

    from src.data_quality.quality_framework import DataProfiler

    # Initialize profiler
    profiler = DataProfiler()

    # Profile dataset
    profile = profiler.profile(df)

    # Get statistics
    print(f"Missing values: {profile['missing_percentage']}")
    print(f"Duplicate rows: {profile['duplicate_count']}")
    print(f"Unique values: {profile['unique_counts']}")

Quality Monitoring
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.data_quality.quality_dashboard import QualityMonitor

    # Initialize monitor
    monitor = QualityMonitor(
        check_frequency='hourly',
        alert_threshold=0.95
    )

    # Add data source
    monitor.add_source(
        name='customer_data',
        connection_string='postgresql://...',
        quality_rules=rules
    )

    # Start monitoring
    monitor.start()

    # Get dashboard URL
    dashboard_url = monitor.get_dashboard_url()
    print(f"Quality dashboard: {dashboard_url}")