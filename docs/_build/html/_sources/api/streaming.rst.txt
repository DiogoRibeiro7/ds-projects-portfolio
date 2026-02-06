Streaming Processor Module
==========================

.. currentmodule:: src.scalability

Real-time streaming data processing for event-driven architectures.

Streaming Processor
-------------------

.. automodule:: src.scalability.streaming_processor
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Key Classes
-----------

StreamProcessor
~~~~~~~~~~~~~~~

.. autoclass:: src.scalability.streaming_processor.StreamProcessor
   :members:
   :special-members: __init__
   :show-inheritance:

EventHandler
~~~~~~~~~~~~

.. autoclass:: src.scalability.streaming_processor.EventHandler
   :members:
   :special-members: __init__
   :show-inheritance:

Usage Examples
--------------

Stream Processing
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.scalability.streaming_processor import StreamProcessor

    # Initialize stream processor
    processor = StreamProcessor(
        source='kafka',
        topic='events',
        consumer_group='analytics'
    )

    # Define processing function
    def process_event(event):
        # Transform event
        processed = transform(event)
        # Apply business logic
        result = apply_rules(processed)
        return result

    # Start processing
    processor.process(
        handler=process_event,
        output_topic='processed_events'
    )

Window Operations
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.scalability.streaming_processor import WindowedStream

    # Initialize windowed stream
    stream = WindowedStream(
        window_type='tumbling',
        window_size='5m'
    )

    # Define aggregation
    stream.aggregate(
        func='sum',
        group_by='user_id',
        metric='amount'
    )

    # Get results
    results = stream.get_window_results()

Event-Driven Processing
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.scalability.streaming_processor import EventHandler

    # Initialize event handler
    handler = EventHandler()

    # Register event processors
    @handler.on('user_signup')
    def handle_signup(event):
        send_welcome_email(event['user_id'])
        update_metrics(event)

    @handler.on('purchase_completed')
    def handle_purchase(event):
        update_inventory(event['items'])
        calculate_revenue(event['amount'])

    # Start event processing
    handler.start()