Machine Learning API
====================

.. currentmodule:: src.api

The ML API module provides RESTful endpoints for machine learning model serving and inference.

ML API Module
-------------

.. automodule:: src.api.ml_api
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

API Endpoints
-------------

The following endpoints are available:

.. autofunction:: src.api.ml_api.predict
.. autofunction:: src.api.ml_api.batch_predict
.. autofunction:: src.api.ml_api.model_info
.. autofunction:: src.api.ml_api.health_check

Request/Response Models
-----------------------

.. autoclass:: src.api.ml_api.PredictionRequest
   :members:
   :undoc-members:

.. autoclass:: src.api.ml_api.PredictionResponse
   :members:
   :undoc-members:

.. autoclass:: src.api.ml_api.ModelInfo
   :members:
   :undoc-members:

Usage Example
-------------

.. code-block:: python

    import requests

    # Make a prediction request
    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "features": [1.0, 2.0, 3.0, 4.0],
            "model_id": "default"
        }
    )

    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")