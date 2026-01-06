Exceptions Module
=================

.. currentmodule:: src.utils

Custom exception classes for error handling across the project.

Exceptions Module
-----------------

.. automodule:: src.utils.exceptions
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Exception Hierarchy
-------------------

Base Exceptions
~~~~~~~~~~~~~~~

.. autoexception:: src.utils.exceptions.BaseProjectException
   :show-inheritance:

Data Exceptions
~~~~~~~~~~~~~~~

.. autoexception:: src.utils.exceptions.DataQualityError
   :show-inheritance:

.. autoexception:: src.utils.exceptions.DataValidationError
   :show-inheritance:

.. autoexception:: src.utils.exceptions.DataProcessingError
   :show-inheritance:

Experiment Exceptions
~~~~~~~~~~~~~~~~~~~~~

.. autoexception:: src.utils.exceptions.ExperimentError
   :show-inheritance:

.. autoexception:: src.utils.exceptions.SampleSizeError
   :show-inheritance:

.. autoexception:: src.utils.exceptions.StatisticalError
   :show-inheritance:

API Exceptions
~~~~~~~~~~~~~~

.. autoexception:: src.utils.exceptions.APIError
   :show-inheritance:

.. autoexception:: src.utils.exceptions.AuthenticationError
   :show-inheritance:

.. autoexception:: src.utils.exceptions.AuthorizationError
   :show-inheritance:

.. autoexception:: src.utils.exceptions.RateLimitError
   :show-inheritance:

Usage Examples
--------------

Custom Exception Handling
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.utils.exceptions import (
        DataQualityError,
        ExperimentError,
        handle_exception
    )

    try:
        # Data validation
        if missing_values > threshold:
            raise DataQualityError(
                f"Too many missing values: {missing_values}",
                data={'columns': missing_columns}
            )

        # Experiment validation
        if sample_size < min_sample:
            raise ExperimentError(
                "Insufficient sample size",
                required=min_sample,
                actual=sample_size
            )

    except DataQualityError as e:
        logger.error(f"Data quality issue: {e}")
        # Handle data quality issues
        clean_data()

    except ExperimentError as e:
        logger.error(f"Experiment error: {e}")
        # Handle experiment issues
        adjust_parameters()

Context Managers
~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.utils.exceptions import error_context

    with error_context("Processing batch", batch_id=123):
        # If an error occurs here, it will include
        # the context information
        process_batch(data)

Error Decorators
~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.utils.exceptions import retry_on_error

    @retry_on_error(
        max_retries=3,
        backoff=2.0,
        exceptions=(ConnectionError, TimeoutError)
    )
    def fetch_data():
        # Will retry up to 3 times with exponential backoff
        return api_client.get_data()

Error Reporting
~~~~~~~~~~~~~~~

.. code-block:: python

    from src.utils.exceptions import ErrorReporter

    # Initialize reporter
    reporter = ErrorReporter(
        sentry_dsn='your-sentry-dsn',
        environment='production'
    )

    # Report exception with context
    try:
        risky_operation()
    except Exception as e:
        reporter.report(
            exception=e,
            user_id='user123',
            tags={'module': 'data_processing'},
            extra={'input_size': 1000}
        )