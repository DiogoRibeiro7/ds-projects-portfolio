Data Preprocessing Module
=========================

.. currentmodule:: src.data_preprocessing

Preprocessing pipelines for data transformation, feature engineering, and preparation.

Preprocessing Pipelines
-----------------------

.. automodule:: src.data_preprocessing.preprocessing_pipelines
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Key Classes
-----------

PreprocessingPipeline
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: src.data_preprocessing.preprocessing_pipelines.PreprocessingPipeline
   :members:
   :special-members: __init__
   :show-inheritance:

FeatureEngineer
~~~~~~~~~~~~~~~

.. autoclass:: src.data_preprocessing.preprocessing_pipelines.FeatureEngineer
   :members:
   :special-members: __init__
   :show-inheritance:

Usage Examples
--------------

Building a Pipeline
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.data_preprocessing.preprocessing_pipelines import PreprocessingPipeline
    import pandas as pd

    # Initialize pipeline
    pipeline = PreprocessingPipeline()

    # Add steps
    pipeline.add_step('remove_duplicates')
    pipeline.add_step('handle_missing', strategy='median')
    pipeline.add_step('encode_categorical', method='one_hot')
    pipeline.add_step('scale_features', method='standard')

    # Fit and transform
    df_train = pd.read_csv('train.csv')
    pipeline.fit(df_train)

    df_transformed = pipeline.transform(df_train)

    # Apply to test data
    df_test = pd.read_csv('test.csv')
    df_test_transformed = pipeline.transform(df_test)

Feature Engineering
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.data_preprocessing.preprocessing_pipelines import FeatureEngineer

    # Initialize feature engineer
    engineer = FeatureEngineer()

    # Create polynomial features
    df_poly = engineer.create_polynomial_features(
        df,
        columns=['feature1', 'feature2'],
        degree=2
    )

    # Create interaction features
    df_interactions = engineer.create_interactions(
        df,
        columns=['feature1', 'feature2', 'feature3']
    )

    # Create time-based features
    df_time = engineer.create_time_features(
        df,
        date_column='timestamp',
        features=['hour', 'day_of_week', 'month', 'quarter']
    )