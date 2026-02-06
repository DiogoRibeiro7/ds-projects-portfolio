Utilities Module
================

.. currentmodule:: src.utils

Common utilities and helper functions used throughout the project.

Constants
---------

.. automodule:: src.utils.constants
   :members:
   :undoc-members:
   :show-inheritance:

Exceptions
----------

.. automodule:: src.utils.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Observability
-------------

.. automodule:: src.utils.observability
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Using Constants
~~~~~~~~~~~~~~~

.. code-block:: python

    from src.utils.constants import (
        DEFAULT_CONFIDENCE_LEVEL,
        MAX_SAMPLE_SIZE,
        SUPPORTED_METRICS
    )

    # Use constants in your code
    def calculate_sample_size(confidence=DEFAULT_CONFIDENCE_LEVEL):
        if sample_size > MAX_SAMPLE_SIZE:
            raise ValueError(f"Sample size exceeds maximum: {MAX_SAMPLE_SIZE}")
        return sample_size

Custom Exceptions
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.utils.exceptions import (
        DataQualityError,
        ExperimentError,
        ValidationError
    )

    # Raise custom exceptions
    def validate_data(df):
        if df.empty:
            raise DataQualityError("Dataset is empty")

        if 'target' not in df.columns:
            raise ValidationError("Target column missing")

        if df.duplicated().any():
            raise DataQualityError("Duplicate rows found")

Error Handling
~~~~~~~~~~~~~~

.. code-block:: python

    from src.utils.exceptions import handle_errors

    @handle_errors(default_return=None)
    def risky_operation(data):
        # Operation that might fail
        result = process_data(data)
        return result

    # The decorator will catch and log errors
    result = risky_operation(data)
    if result is None:
        print("Operation failed, check logs")