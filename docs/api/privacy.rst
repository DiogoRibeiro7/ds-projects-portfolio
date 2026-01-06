Privacy Protection Module
=========================

.. currentmodule:: src.privacy

Privacy protection utilities for GDPR compliance, PII handling, and data anonymization.

Privacy Protection
------------------

.. automodule:: src.privacy.privacy_protection
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Key Classes
-----------

DataAnonymizer
~~~~~~~~~~~~~~

.. autoclass:: src.privacy.privacy_protection.DataAnonymizer
   :members:
   :special-members: __init__
   :show-inheritance:

PIIDetector
~~~~~~~~~~~

.. autoclass:: src.privacy.privacy_protection.PIIDetector
   :members:
   :special-members: __init__
   :show-inheritance:

ConsentManager
~~~~~~~~~~~~~~

.. autoclass:: src.privacy.privacy_protection.ConsentManager
   :members:
   :special-members: __init__
   :show-inheritance:

Usage Examples
--------------

Data Anonymization
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.privacy.privacy_protection import DataAnonymizer
    import pandas as pd

    # Load data with PII
    df = pd.read_csv('customer_data.csv')

    # Initialize anonymizer
    anonymizer = DataAnonymizer(
        method='k-anonymity',
        k=5
    )

    # Anonymize data
    anonymized_df = anonymizer.anonymize(
        df,
        quasi_identifiers=['age', 'zip_code'],
        sensitive_attributes=['income', 'health_status']
    )

    # Save anonymized data
    anonymized_df.to_csv('anonymized_data.csv', index=False)

PII Detection
~~~~~~~~~~~~~

.. code-block:: python

    from src.privacy.privacy_protection import PIIDetector

    # Initialize PII detector
    detector = PIIDetector()

    # Detect PII in text
    text = "John Doe's email is john@example.com and SSN is 123-45-6789"
    pii_entities = detector.detect(text)

    for entity in pii_entities:
        print(f"Found {entity['type']}: {entity['value']}")

    # Redact PII
    redacted_text = detector.redact(text)
    print(f"Redacted: {redacted_text}")

GDPR Compliance
~~~~~~~~~~~~~~~

.. code-block:: python

    from src.privacy.privacy_protection import GDPRCompliance

    # Initialize GDPR compliance manager
    gdpr = GDPRCompliance()

    # Handle data subject request
    gdpr.handle_data_request(
        user_id='user123',
        request_type='deletion',
        data_categories=['profile', 'activity_logs']
    )

    # Generate compliance report
    report = gdpr.generate_compliance_report()
    print(report)

Privacy Best Practices
----------------------

1. **Data Minimization**: Collect only the data that is necessary for your purpose
2. **Purpose Limitation**: Use data only for the stated purpose
3. **Storage Limitation**: Delete data when it's no longer needed
4. **Consent Management**: Obtain and manage user consent properly
5. **Data Protection by Design**: Build privacy into your systems from the start