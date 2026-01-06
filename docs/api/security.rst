Security Module
===============

.. currentmodule:: src.security

Comprehensive security framework for authentication, authorization, and audit logging.

Security Framework
------------------

.. automodule:: src.security.security_framework
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Authentication
--------------

.. automodule:: src.security.authentication
   :members:
   :undoc-members:
   :show-inheritance:

Audit Logging
-------------

.. automodule:: src.security.audit_logging
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
-----------

SecurityManager
~~~~~~~~~~~~~~~

.. autoclass:: src.security.security_framework.SecurityManager
   :members:
   :special-members: __init__
   :show-inheritance:

Authenticator
~~~~~~~~~~~~~

.. autoclass:: src.security.authentication.Authenticator
   :members:
   :special-members: __init__
   :show-inheritance:

AuditLogger
~~~~~~~~~~~

.. autoclass:: src.security.audit_logging.AuditLogger
   :members:
   :special-members: __init__
   :show-inheritance:

Usage Examples
--------------

Authentication Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.security.authentication import Authenticator

    # Initialize authenticator
    auth = Authenticator(
        secret_key='your-secret-key',
        algorithm='HS256'
    )

    # Create JWT token
    token = auth.create_token(
        user_id='user123',
        roles=['admin', 'data_scientist']
    )

    # Verify token
    payload = auth.verify_token(token)
    print(f"User ID: {payload['user_id']}")
    print(f"Roles: {payload['roles']}")

Audit Logging Example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.security.audit_logging import AuditLogger

    # Initialize audit logger
    logger = AuditLogger(
        log_file='audit.log',
        max_size='100MB'
    )

    # Log an event
    logger.log_event(
        event_type='data_access',
        user_id='user123',
        resource='sensitive_data.csv',
        action='read',
        status='success',
        metadata={'ip_address': '192.168.1.1'}
    )

Security Best Practices
-----------------------

1. **Authentication**: Always use strong, unique API keys and rotate them regularly
2. **Authorization**: Implement role-based access control (RBAC) for all resources
3. **Encryption**: Encrypt sensitive data at rest and in transit
4. **Audit Logging**: Log all security-relevant events for compliance and forensics
5. **Input Validation**: Validate and sanitize all user inputs to prevent injection attacks