Distributed Computing Module
============================

.. currentmodule:: src.scalability

Distributed computing utilities for large-scale data processing and parallel computation.

Distributed Computing
--------------------

.. automodule:: src.scalability.distributed_computing
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Key Classes
-----------

DistributedProcessor
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: src.scalability.distributed_computing.DistributedProcessor
   :members:
   :special-members: __init__
   :show-inheritance:

ParallelExecutor
~~~~~~~~~~~~~~~~

.. autoclass:: src.scalability.distributed_computing.ParallelExecutor
   :members:
   :special-members: __init__
   :show-inheritance:

Usage Examples
--------------

Distributed Processing
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.scalability.distributed_computing import DistributedProcessor
    import pandas as pd

    # Initialize distributed processor
    processor = DistributedProcessor(
        backend='dask',
        n_workers=4,
        memory_per_worker='4GB'
    )

    # Process large dataset
    df = processor.read_parquet('s3://bucket/large_dataset/*.parquet')

    # Apply transformations
    result = processor.apply(
        df,
        func=process_chunk,
        partition_by='user_id'
    )

    # Save results
    processor.to_parquet(result, 's3://bucket/processed/')

Parallel Execution
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.scalability.distributed_computing import ParallelExecutor

    # Initialize executor
    executor = ParallelExecutor(n_jobs=8)

    # Define tasks
    tasks = [
        {'func': process_data, 'args': (chunk,)}
        for chunk in data_chunks
    ]

    # Execute in parallel
    results = executor.execute(tasks)

    # Combine results
    final_result = executor.reduce(results, combine_func)

MapReduce Operations
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.scalability.distributed_computing import MapReduceEngine

    # Initialize MapReduce engine
    engine = MapReduceEngine()

    # Define mapper and reducer
    def mapper(record):
        return (record['category'], record['value'])

    def reducer(key, values):
        return key, sum(values)

    # Run MapReduce
    result = engine.map_reduce(
        data=large_dataset,
        mapper=mapper,
        reducer=reducer,
        n_partitions=100
    )