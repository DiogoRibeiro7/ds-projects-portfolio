Data Profiling Module
=====================

.. currentmodule:: src.data_profiling

Advanced data profiling tools for exploratory data analysis and automated insights.

Profiling Tools
---------------

.. automodule:: src.data_profiling.profiling_tools
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Key Classes
-----------

DataProfiler
~~~~~~~~~~~~

.. autoclass:: src.data_profiling.profiling_tools.DataProfiler
   :members:
   :special-members: __init__
   :show-inheritance:

AutomatedEDA
~~~~~~~~~~~~

.. autoclass:: src.data_profiling.profiling_tools.AutomatedEDA
   :members:
   :special-members: __init__
   :show-inheritance:

Usage Examples
--------------

Comprehensive Profiling
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.data_profiling.profiling_tools import DataProfiler
    import pandas as pd

    # Load data
    df = pd.read_csv('dataset.csv')

    # Initialize profiler
    profiler = DataProfiler(
        include_correlations=True,
        include_distributions=True,
        sample_size=10000
    )

    # Generate profile
    profile = profiler.profile(df)

    # Export report
    profiler.export_report(
        profile,
        format='html',
        output_path='data_profile.html'
    )

Automated EDA
~~~~~~~~~~~~~

.. code-block:: python

    from src.data_profiling.profiling_tools import AutomatedEDA

    # Initialize EDA
    eda = AutomatedEDA()

    # Run automated analysis
    insights = eda.analyze(
        df,
        target_column='target',
        problem_type='classification'
    )

    # Get insights
    print("Key insights:")
    for insight in insights['key_findings']:
        print(f"- {insight}")

    # Get feature importance
    print("\nImportant features:")
    for feature, importance in insights['feature_importance'].items():
        print(f"- {feature}: {importance:.3f}")