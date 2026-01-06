Statistical Methods
===================

.. currentmodule:: statistical_methods

Comprehensive statistical methods for experimentation, causal inference, and advanced testing.

Advanced Statistical Tests
--------------------------

.. automodule:: statistical_methods.advanced_statistical_tests
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Bayesian A/B Testing
--------------------

.. automodule:: statistical_methods.bayesian_ab_testing
   :members:
   :undoc-members:
   :show-inheritance:

Enhanced Bayesian Testing
------------------------

.. automodule:: statistical_methods.enhanced_bayesian_testing
   :members:
   :undoc-members:
   :show-inheritance:

Causal Inference
----------------

.. automodule:: statistical_methods.causal_inference
   :members:
   :undoc-members:
   :show-inheritance:

Multi-Armed Bandits
-------------------

.. automodule:: statistical_methods.multi_armed_bandits
   :members:
   :undoc-members:
   :show-inheritance:

Power Analysis Simulations
-------------------------

.. automodule:: statistical_methods.power_analysis_simulations
   :members:
   :undoc-members:
   :show-inheritance:

Statistical Validation Suite
---------------------------

.. automodule:: statistical_methods.statistical_validation_suite
   :members:
   :undoc-members:
   :show-inheritance:

Enhanced Statistical Validation
------------------------------

.. automodule:: statistical_methods.enhanced_statistical_validation
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Bayesian A/B Testing
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from statistical_methods.bayesian_ab_testing import BayesianABTest

    # Initialize test
    test = BayesianABTest(prior_alpha=1, prior_beta=1)

    # Add data
    test.update_control(successes=450, failures=4550)
    test.update_treatment(successes=500, failures=4500)

    # Calculate probability of treatment being better
    prob = test.probability_treatment_better()
    print(f"Probability treatment is better: {prob:.2%}")

Multi-Armed Bandits
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from statistical_methods.multi_armed_bandits import ThompsonSampling

    # Initialize bandit
    bandit = ThompsonSampling(n_arms=3)

    # Simulate rounds
    for _ in range(1000):
        # Select arm
        arm = bandit.select_arm()

        # Simulate reward (example)
        reward = simulate_reward(arm)

        # Update bandit
        bandit.update(arm, reward)

    # Get best arm
    best_arm = bandit.get_best_arm()
    print(f"Best performing arm: {best_arm}")