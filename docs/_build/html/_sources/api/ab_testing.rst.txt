A/B Testing Module
==================

.. currentmodule:: ab_testing

The A/B testing module provides tools for designing, running, and analyzing experiments.

Core Module
-----------

.. automodule:: ab_testing
   :members:
   :undoc-members:
   :show-inheritance:

Error Function Inverse Fix
--------------------------

.. automodule:: ab_testing.erfcinv_fix
   :members:
   :undoc-members:
   :show-inheritance:

Jupyter Notebooks
-----------------

The following Jupyter notebooks provide interactive examples and tutorials:

- **ab_testing_playbook_pro_enhanced.ipynb**: Comprehensive A/B testing playbook with advanced techniques
- **multi_armed_bandits_suite.ipynb**: Multi-armed bandits implementation and comparison

Key Functions
-------------

.. autofunction:: ab_testing.calculate_sample_size
.. autofunction:: ab_testing.run_ab_test
.. autofunction:: ab_testing.analyze_results
.. autofunction:: ab_testing.calculate_confidence_interval

Usage Example
-------------

.. code-block:: python

    from ab_testing import ABTest
    import pandas as pd

    # Load experiment data
    data = pd.read_csv('experiment_data.csv')

    # Initialize A/B test
    test = ABTest(
        control_data=data[data['group'] == 'control'],
        treatment_data=data[data['group'] == 'treatment']
    )

    # Run analysis
    results = test.analyze(
        metric='conversion_rate',
        confidence_level=0.95
    )

    # Print results
    print(f"Control: {results['control_mean']:.3f}")
    print(f"Treatment: {results['treatment_mean']:.3f}")
    print(f"Lift: {results['lift']:.2%}")
    print(f"P-value: {results['p_value']:.4f}")
    print(f"Statistical significance: {results['is_significant']}")