Caching Module
==============

.. currentmodule:: src.utils

Advanced caching utilities for performance optimization.

Caching Module
--------------

.. automodule:: src.utils.caching
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Cache Integration
-----------------

.. automodule:: src.utils.cache_integration
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
-----------

CacheManager
~~~~~~~~~~~~

.. autoclass:: src.utils.caching.CacheManager
   :members:
   :special-members: __init__
   :show-inheritance:

DistributedCache
~~~~~~~~~~~~~~~~

.. autoclass:: src.utils.caching.DistributedCache
   :members:
   :special-members: __init__
   :show-inheritance:

Usage Examples
--------------

Basic Caching
~~~~~~~~~~~~~

.. code-block:: python

    from src.utils.caching import CacheManager

    # Initialize cache
    cache = CacheManager(
        backend='redis',
        ttl=3600,  # 1 hour
        max_size='1GB'
    )

    # Cache function results
    @cache.memoize(ttl=600)
    def expensive_computation(x, y):
        # Expensive operation
        return complex_calculation(x, y)

    # First call computes and caches
    result = expensive_computation(5, 10)

    # Second call retrieves from cache
    result = expensive_computation(5, 10)  # Fast!

Distributed Caching
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.utils.caching import DistributedCache

    # Initialize distributed cache
    cache = DistributedCache(
        nodes=['redis1:6379', 'redis2:6379', 'redis3:6379'],
        replication_factor=2
    )

    # Set with replication
    cache.set('key', 'value', replicate=True)

    # Get with fallback
    value = cache.get('key', fallback_to_replica=True)

Cache Invalidation
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.utils.cache_integration import CacheIntegration

    # Initialize integration
    integration = CacheIntegration()

    # Register cache dependencies
    integration.register_dependency(
        cache_key='user_stats',
        depends_on=['user_data', 'transaction_data']
    )

    # When dependency changes, invalidate
    integration.invalidate('user_data')
    # This also invalidates 'user_stats'

Cache Warming
~~~~~~~~~~~~~

.. code-block:: python

    from src.utils.caching import CacheWarmer

    # Initialize warmer
    warmer = CacheWarmer(cache_manager=cache)

    # Define warming strategy
    warmer.add_task(
        func=load_popular_items,
        schedule='0 */6 * * *',  # Every 6 hours
        priority='high'
    )

    # Start warming
    warmer.start()

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.utils.caching import CacheMetrics

    # Get cache metrics
    metrics = cache.get_metrics()

    print(f"Hit rate: {metrics['hit_rate']:.2%}")
    print(f"Miss rate: {metrics['miss_rate']:.2%}")
    print(f"Eviction rate: {metrics['eviction_rate']:.2%}")
    print(f"Average latency: {metrics['avg_latency_ms']}ms")