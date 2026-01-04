# 🚀 Quick Start Guide

Get up and running with the Data Science Portfolio in under 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ds-projects-portfolio.git
cd ds-projects-portfolio
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Optional: Documentation dependencies
pip install -r docs/requirements-docs.txt
```

## Your First ML Pipeline

```python
import pandas as pd
from modern_bank_churn.ml_pipeline_orchestrator import MLPipelineOrchestrator, PipelineConfig

# Load your data
data = pd.read_csv('your_data.csv')

# Configure pipeline
config = PipelineConfig(
    feature_selection_method='mutual_info',
    model_type='ensemble',
    hyperparameter_tuning=True
)

# Run pipeline
orchestrator = MLPipelineOrchestrator(config)
results = orchestrator.run_pipeline(data)

# View results
print(f"Best model: {results.best_model}")
print(f"AUC-ROC: {results.metrics['auc_roc']:.4f}")
```

## Your First Statistical Analysis

```python
from statistical_methods.statistical_analyzer import StatisticalAnalyzer
from statistical_methods.hypothesis_tester import HypothesisTester

# Initialize analyzer
analyzer = StatisticalAnalyzer(data=data)

# Generate summary
summary = analyzer.generate_summary()
print(summary)

# Hypothesis testing
tester = HypothesisTester(alpha=0.05)
result = tester.t_test(group1, group2)
print(f"P-value: {result['p_value']:.4f}")
```

## Your First Dashboard

```python
from dashboard_framework import EnhancedDashboard, DashboardConfig

# Configure dashboard
config = DashboardConfig(
    app_name="My Dashboard",
    port=8050,
    enable_realtime=True
)

# Create dashboard
dashboard = EnhancedDashboard(config)

# Register data source
dashboard.register_data_source('my_data', lambda: data)

# Add charts
dashboard.add_chart('distribution', chart_type='histogram')
dashboard.add_chart('correlation', chart_type='heatmap')

# Run
dashboard.run()
# Open http://localhost:8050
```

## Common Commands

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific module tests
pytest tests/test_ml_pipeline.py

# Run with coverage
pytest --cov=. tests/
```

### Building Documentation
```bash
cd docs
make html
# Open _build/html/index.html
```

### Starting Jupyter Lab
```bash
jupyter lab tutorials/
```

### Checking Code Quality
```bash
# Format code
black .

# Check linting
flake8 .

# Type checking
mypy .
```

## Environment Variables

Create a `.env` file for configuration:

```env
# API Configuration
API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here

# Database
DATABASE_URL=postgresql://user:pass@localhost/db

# Redis (for real-time features)
REDIS_URL=redis://localhost:6379

# Model Storage
MODEL_PATH=./models
DATA_PATH=./data
```

## Troubleshooting

### Import Errors
```bash
# Ensure you're in the virtual environment
which python  # Should show venv path

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Memory Issues
```python
# Use chunking for large datasets
config = PipelineConfig(
    chunk_size=10000,
    use_dask=True
)
```

### Slow Performance
```python
# Enable parallel processing
config = PipelineConfig(
    n_jobs=-1,  # Use all cores
    use_gpu=True  # If available
)
```

## Next Steps

1. 📖 Read the [full documentation](index.rst)
2. 🎓 Try the [tutorials](tutorials/01_getting_started.ipynb)
3. 🔬 Explore [examples](examples/)
4. 🤝 [Contribute](contributing.md) to the project

## Quick Links

- [API Reference](api_reference.md)
- [ML Pipeline Guide](modules/ml_pipeline.md)
- [Statistical Methods](modules/statistics.md)
- [Dashboard Documentation](modules/dashboard.md)

## Getting Help

- 📧 Email: support@example.com
- 💬 Discord: [Join our server](https://discord.gg/example)
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)

---

Ready to build something amazing? Let's go! 🚀