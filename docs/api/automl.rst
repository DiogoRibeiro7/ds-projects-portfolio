AutoML Module
=============

.. currentmodule:: src.automl

The AutoML module provides automated machine learning capabilities for model selection, hyperparameter tuning, and pipeline optimization.

AutoML Orchestrator
-------------------

.. automodule:: src.automl.automl_orchestrator
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Key Classes
-----------

.. autoclass:: src.automl.automl_orchestrator.AutoMLOrchestrator
   :members:
   :special-members: __init__
   :show-inheritance:

   .. rubric:: Methods

   .. automethod:: fit
   .. automethod:: predict
   .. automethod:: evaluate
   .. automethod:: get_best_model
   .. automethod:: get_feature_importance

Usage Example
-------------

.. code-block:: python

    from src.automl.automl_orchestrator import AutoMLOrchestrator
    import pandas as pd

    # Load your data
    df = pd.read_csv('data.csv')
    X = df.drop('target', axis=1)
    y = df['target']

    # Initialize AutoML
    automl = AutoMLOrchestrator(
        task='classification',
        time_limit=3600,  # 1 hour
        metric='accuracy'
    )

    # Fit the model
    automl.fit(X, y)

    # Make predictions
    predictions = automl.predict(X)

    # Get the best model
    best_model = automl.get_best_model()
    print(f"Best model: {best_model}")