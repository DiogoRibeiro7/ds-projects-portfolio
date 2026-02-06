GraphQL API
===========

.. currentmodule:: dashboard_enhanced

GraphQL API for flexible data querying and mutations.

GraphQL API Module
------------------

.. automodule:: dashboard_enhanced.graphql_api
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Schema Definition
-----------------

.. autoclass:: dashboard_enhanced.graphql_api.Schema
   :members:
   :special-members: __init__
   :show-inheritance:

Query Types
-----------

.. autoclass:: dashboard_enhanced.graphql_api.Query
   :members:
   :show-inheritance:

Mutation Types
--------------

.. autoclass:: dashboard_enhanced.graphql_api.Mutation
   :members:
   :show-inheritance:

Usage Examples
--------------

GraphQL Queries
~~~~~~~~~~~~~~~

.. code-block:: python

    import requests

    # Define GraphQL query
    query = """
    query GetExperimentResults($experimentId: String!) {
        experiment(id: $experimentId) {
            id
            name
            status
            metrics {
                name
                control
                treatment
                lift
                pValue
            }
        }
    }
    """

    # Execute query
    response = requests.post(
        'http://localhost:5000/graphql',
        json={
            'query': query,
            'variables': {'experimentId': 'exp_123'}
        }
    )

    result = response.json()

GraphQL Mutations
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Define mutation
    mutation = """
    mutation CreateExperiment($input: ExperimentInput!) {
        createExperiment(input: $input) {
            experiment {
                id
                name
                createdAt
            }
            success
            message
        }
    }
    """

    # Execute mutation
    response = requests.post(
        'http://localhost:5000/graphql',
        json={
            'query': mutation,
            'variables': {
                'input': {
                    'name': 'New A/B Test',
                    'description': 'Testing new feature',
                    'controlSize': 5000,
                    'treatmentSize': 5000
                }
            }
        }
    )

Subscriptions
~~~~~~~~~~~~~

.. code-block:: python

    from graphql_ws_client import GraphQLWSClient

    # Connect to WebSocket
    client = GraphQLWSClient('ws://localhost:5000/graphql-ws')

    # Define subscription
    subscription = """
    subscription OnMetricUpdate($experimentId: String!) {
        metricUpdated(experimentId: $experimentId) {
            metric
            value
            timestamp
        }
    }
    """

    # Subscribe to updates
    def handle_update(data):
        print(f"Metric updated: {data}")

    client.subscribe(
        subscription,
        variables={'experimentId': 'exp_123'},
        callback=handle_update
    )