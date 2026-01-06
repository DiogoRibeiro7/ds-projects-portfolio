Statistical Methods API
=======================

This module provides comprehensive statistical methods for A/B testing, power analysis,
and experimentation.

Core Statistical Functions
--------------------------

.. currentmodule:: src.statistics.core

.. automodule:: src.statistics.core
   :members:
   :undoc-members:
   :show-inheritance:

Two-Proportion Z-Test
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: two_prop_ztest

Example usage::

    from src.statistics.core import two_prop_ztest

    # Test conversion rates between control and treatment
    z_stat, p_value = two_prop_ztest(
        x1=450,  # Control conversions
        n1=5000,  # Control sample size
        x2=500,  # Treatment conversions
        n2=5000,  # Treatment sample size
        alternative='two-sided'
    )

    print(f"Z-statistic: {z_stat:.3f}")
    print(f"P-value: {p_value:.4f}")

Bootstrap Confidence Intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: bootstrap_ci_diff

Sample Size Calculation
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: calculate_sample_size

Power Calculation
~~~~~~~~~~~~~~~~~

.. autofunction:: calculate_power

ExperimentAnalyzer Class
------------------------

.. autoclass:: ExperimentAnalyzer
   :members:
   :special-members: __init__
   :show-inheritance:

   .. rubric:: Methods

   .. automethod:: check_srm
   .. automethod:: analyze_conversion
   .. automethod:: run_comprehensive_analysis

   Example usage::

       from src.statistics.core import ExperimentAnalyzer
       import pandas as pd

       # Initialize analyzer
       analyzer = ExperimentAnalyzer(alpha=0.05, power=0.8)

       # Load experiment data
       df = pd.read_csv('experiment_data.csv')

       # Check for Sample Ratio Mismatch
       srm_results = analyzer.check_srm(df)

       # Analyze conversion metrics
       conversion_results = analyzer.analyze_conversion(
           df,
           conversion_col='converted'
       )

       # Run comprehensive analysis
       full_results = analyzer.run_comprehensive_analysis(
           df,
           metrics=['converted', 'revenue'],
           group_col='experiment_group'
       )

Multiple Testing Correction
---------------------------

.. autofunction:: apply_multiple_testing_correction

Sequential Testing
------------------

.. autofunction:: sequential_testing_boundary

Bayesian Methods
----------------

.. currentmodule:: src.statistics.bayesian

.. automodule:: src.statistics.bayesian
   :members:
   :undoc-members:
   :show-inheritance:

Power Analysis
--------------

.. currentmodule:: src.statistics.power_analysis

.. automodule:: src.statistics.power_analysis
   :members:
   :undoc-members:
   :show-inheritance: