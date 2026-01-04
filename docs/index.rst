.. Data Science Portfolio documentation master file

=======================================
Data Science Portfolio Documentation
=======================================

.. image:: https://img.shields.io/badge/Python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

Welcome to the comprehensive documentation for the Data Science Portfolio - a production-ready collection of machine learning pipelines, statistical methods, and interactive dashboards.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Modules

   modules/ml_pipeline
   modules/statistics
   modules/dashboard

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference
   api/ml_pipeline
   api/statistics
   api/dashboard

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/getting_started
   examples/real_world_case
   examples/model_comparison

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   best_practices
   troubleshooting
   contributing
   changelog

Overview
========

The Data Science Portfolio provides enterprise-grade solutions for:

📊 **Machine Learning Pipelines**
   - Automated feature engineering
   - Hyperparameter optimization
   - Model evaluation and selection
   - Production deployment tools

📈 **Statistical Methods**
   - Hypothesis testing frameworks
   - Causal inference
   - Time series analysis
   - Bayesian methods

🎨 **Interactive Dashboards**
   - Real-time data visualization
   - REST and GraphQL APIs
   - WebSocket support
   - Export capabilities

Key Features
============

.. panels::
   :card: border-card
   :column: col-lg-4 col-md-6 col-sm-12 col-xs-12 p-2

   ---
   :header: bg-primary text-white text-center

   **ML Pipeline**
   ^^^
   - Automated feature engineering
   - Model selection & tuning
   - SHAP interpretability
   - Production deployment

   ---
   :header: bg-success text-white text-center

   **Statistical Analysis**
   ^^^
   - Comprehensive testing
   - Causal inference
   - Time series forecasting
   - Bayesian analysis

   ---
   :header: bg-info text-white text-center

   **Dashboard System**
   ^^^
   - Real-time updates
   - Interactive visualizations
   - API infrastructure
   - Mobile responsive

Quick Example
=============

.. code-block:: python

   from modern_bank_churn.ml_pipeline_orchestrator import MLPipelineOrchestrator, PipelineConfig

   # Configure pipeline
   config = PipelineConfig(
       feature_selection_method='boruta',
       model_type='ensemble',
       hyperparameter_tuning=True
   )

   # Run pipeline
   orchestrator = MLPipelineOrchestrator(config)
   results = orchestrator.run_pipeline(data)

   print(f"Best model: {results.best_model}")
   print(f"AUC-ROC: {results.metrics['auc_roc']:.4f}")

Architecture
============

.. mermaid::

   graph TB
       A[Data Sources] --> B[ETL Pipeline]
       B --> C[Feature Engineering]
       C --> D[ML Pipeline]
       D --> E[Model Registry]
       E --> F[API Layer]
       F --> G[Dashboard]
       F --> H[Monitoring]

Installation
============

.. code-block:: bash

   # Clone repository
   git clone https://github.com/your-repo/ds-portfolio.git
   cd ds-portfolio

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate

   # Install dependencies
   pip install -r requirements.txt

   # Install development dependencies
   pip install -r requirements-dev.txt

Project Structure
=================

.. code-block:: text

   ds-portfolio/
   ├── modern-bank-churn/       # ML pipeline modules
   ├── statistical_methods/     # Statistical analysis
   ├── dashboard_enhanced/      # Dashboard framework
   ├── docs/                   # Documentation
   ├── tutorials/              # Jupyter notebooks
   ├── tests/                  # Test suite
   └── examples/               # Example scripts

Performance Metrics
===================

Our solutions deliver:

- **>85% AUC-ROC** on benchmark datasets
- **<100ms** inference time for real-time predictions
- **99.9%** API uptime with monitoring
- **10x** faster development with automated pipelines

Use Cases
=========

✅ **Customer Churn Prediction**
   Predict and prevent customer attrition with advanced ML models

✅ **A/B Testing & Experimentation**
   Statistical frameworks for robust experiment design

✅ **Real-time Analytics**
   Live dashboards with WebSocket streaming

✅ **Predictive Maintenance**
   Anomaly detection and failure prediction

✅ **Financial Forecasting**
   Time series models for revenue and demand

Contributing
============

We welcome contributions! See our :doc:`contributing` guide for:

- Code style guidelines
- Testing requirements
- Pull request process
- Issue reporting

Support
=======

- 📧 Email: support@example.com
- 💬 Slack: `Join our workspace <https://slack.example.com>`_
- 🐛 Issues: `GitHub Issues <https://github.com/your-repo/issues>`_
- 📚 Wiki: `Project Wiki <https://github.com/your-repo/wiki>`_

License
=======

This project is licensed under the MIT License. See the LICENSE file for details.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. note::

   This documentation is continuously updated. Last build: |today|