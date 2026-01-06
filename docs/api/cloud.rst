Cloud Integrations
==================

.. currentmodule:: src.cloud

Cloud integration module for AWS, Azure, and Google Cloud Platform.

Cloud Integrations Module
-------------------------

.. automodule:: src.cloud.cloud_integrations
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

AWS Integration
--------------

.. autoclass:: src.cloud.cloud_integrations.AWSIntegration
   :members:
   :special-members: __init__
   :show-inheritance:

Azure Integration
----------------

.. autoclass:: src.cloud.cloud_integrations.AzureIntegration
   :members:
   :special-members: __init__
   :show-inheritance:

GCP Integration
--------------

.. autoclass:: src.cloud.cloud_integrations.GCPIntegration
   :members:
   :special-members: __init__
   :show-inheritance:

Usage Examples
--------------

AWS S3 Example
~~~~~~~~~~~~~~

.. code-block:: python

    from src.cloud.cloud_integrations import AWSIntegration

    # Initialize AWS integration
    aws = AWSIntegration(
        region='us-west-2',
        access_key_id='YOUR_KEY',
        secret_access_key='YOUR_SECRET'
    )

    # Upload file to S3
    aws.upload_to_s3(
        file_path='data.csv',
        bucket='my-bucket',
        key='data/experiment.csv'
    )

    # Download from S3
    aws.download_from_s3(
        bucket='my-bucket',
        key='data/experiment.csv',
        local_path='downloaded.csv'
    )

Azure Blob Storage Example
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.cloud.cloud_integrations import AzureIntegration

    # Initialize Azure integration
    azure = AzureIntegration(
        connection_string='YOUR_CONNECTION_STRING'
    )

    # Upload to blob storage
    azure.upload_blob(
        file_path='model.pkl',
        container='models',
        blob_name='trained_model.pkl'
    )

GCP BigQuery Example
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.cloud.cloud_integrations import GCPIntegration

    # Initialize GCP integration
    gcp = GCPIntegration(
        project_id='your-project',
        credentials_path='credentials.json'
    )

    # Query BigQuery
    results = gcp.query_bigquery(
        query="SELECT * FROM dataset.table LIMIT 100"
    )