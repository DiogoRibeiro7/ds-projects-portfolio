# Comprehensive API Documentation Guide

## Overview

This project now has comprehensive API documentation using Sphinx with autodoc capabilities. The documentation automatically generates API references from docstrings in your Python code, ensuring documentation stays synchronized with the codebase.

## Documentation Structure

### Core Components

The documentation system includes:

1. **Sphinx Configuration** (`docs/conf.py`):
   - Configured with multiple extensions for comprehensive documentation
   - Supports both RST and Markdown formats
   - Includes autodoc, Napoleon (for Google/NumPy docstrings), and MyST parser
   - Mocked imports for optional dependencies

2. **API Reference Structure** (`docs/api/`):
   - Organized into logical categories
   - Each module has its own documentation file
   - Automatic extraction of docstrings and type hints

3. **Build System** (`docs/make_docs.py`):
   - Automated documentation building
   - Notebook conversion support
   - HTML and PDF output generation

## API Documentation Categories

### Core Modules
- **Data Processing** (`api/data_processing.rst`): Data cleaning, optimization, and benchmarking
- **Statistics** (`api/statistics.rst`): Statistical methods and hypothesis testing
- **Visualization** (`api/visualization.rst`): Plotting and data visualization utilities

### Machine Learning
- **ML API** (`api/ml_api.rst`): RESTful API for model serving
- **AutoML** (`api/automl.rst`): Automated machine learning orchestration

### Statistical Methods
- **Statistical Methods** (`api/statistical_methods.rst`): Advanced statistical tests
- **A/B Testing** (`api/ab_testing.rst`): Experimentation framework

### Infrastructure
- **Cloud** (`api/cloud.rst`): AWS, Azure, and GCP integrations
- **Security** (`api/security.rst`): Authentication and audit logging
- **Privacy** (`api/privacy.rst`): GDPR compliance and PII protection
- **Compliance** (`api/compliance.rst`): Regulatory compliance tools
- **Observability** (`api/observability.rst`): Monitoring and tracing

### Data Quality
- **Data Quality** (`api/data_quality.rst`): Validation and monitoring
- **Data Profiling** (`api/data_profiling.rst`): Automated EDA
- **Data Preprocessing** (`api/data_preprocessing.rst`): Feature engineering pipelines

### Scalability
- **Distributed Computing** (`api/distributed_computing.rst`): Large-scale processing
- **Streaming** (`api/streaming.rst`): Real-time data processing

### User Interface
- **Dashboards** (`api/dashboards.md`): Interactive dashboard framework
- **Visualization Components** (`api/visualization_components.rst`): Reusable UI components
- **GraphQL** (`api/graphql.rst`): Flexible data querying

### Utilities
- **Utils** (`api/utils.rst`): Common utilities and constants
- **Caching** (`api/caching.rst`): Performance optimization
- **Exceptions** (`api/exceptions.rst`): Custom error handling

## Building Documentation

### Quick Build

```bash
# Build HTML documentation
cd docs
python -m sphinx -b html . _build/html
```

### Using the Build Script

```bash
# Full documentation build
python docs/make_docs.py

# HTML only
python docs/make_docs.py --html-only

# Check for broken links
python docs/make_docs.py --check-links
```

### Viewing Documentation

After building, open the documentation:
- **Windows**: `start docs/_build/html/index.html`
- **macOS**: `open docs/_build/html/index.html`
- **Linux**: `xdg-open docs/_build/html/index.html`

## Writing Documentation

### Docstring Format

The project supports both Google and NumPy style docstrings:

```python
def calculate_sample_size(alpha=0.05, power=0.8, effect_size=0.1):
    """Calculate required sample size for A/B test.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level (Type I error rate)
    power : float, default=0.8
        Statistical power (1 - Type II error rate)
    effect_size : float, default=0.1
        Minimum detectable effect size

    Returns
    -------
    int
        Required sample size per group

    Examples
    --------
    >>> calculate_sample_size(alpha=0.05, power=0.9, effect_size=0.05)
    8406
    """
```

### Adding New Modules

1. Create the module documentation file in `docs/api/`
2. Add it to the appropriate section in `docs/api/index.md`
3. Use autodoc directives to extract documentation:

```rst
.. automodule:: your_module
   :members:
   :undoc-members:
   :show-inheritance:
```

## Best Practices

1. **Keep Docstrings Updated**: Documentation is generated from code, so keep docstrings current
2. **Use Type Hints**: Add type hints for better API documentation
3. **Include Examples**: Add code examples in docstrings
4. **Cross-Reference**: Use Sphinx cross-references to link related documentation
5. **Version Control**: Track documentation changes alongside code changes

## Continuous Integration

Add documentation building to your CI/CD pipeline:

```yaml
# .github/workflows/docs.yml
name: Build Documentation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r docs/requirements-docs.txt
    - name: Build documentation
      run: |
        cd docs
        make html
    - name: Deploy to GitHub Pages
      if: github.event_name == 'push' && github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Add problematic imports to `autodoc_mock_imports` in `conf.py`
2. **Missing Documentation**: Ensure modules have `__init__.py` files
3. **Build Warnings**: Some warnings are expected for missing optional dependencies
4. **Duplicate Descriptions**: Use `:no-index:` directive to avoid duplicates

### Useful Commands

```bash
# Clean build artifacts
rm -rf docs/_build

# Build with verbose output
sphinx-build -v -b html docs docs/_build/html

# Check for syntax errors
python -m py_compile docs/conf.py
```

## Next Steps

1. **Add More Examples**: Create example scripts in `examples/` directory
2. **Interactive Notebooks**: Add Jupyter notebooks for tutorials
3. **API Versioning**: Implement documentation versioning
4. **Search Enhancement**: Configure better search indexing
5. **Theme Customization**: Customize the Furo theme for branding

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [MyST Parser](https://myst-parser.readthedocs.io/)
- [Napoleon Extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
- [Furo Theme](https://pradyunsg.me/furo/)

## Contributing

When contributing to the project:
1. Update docstrings for any new/modified functions
2. Run the documentation build to check for errors
3. Review the generated HTML to ensure correctness
4. Include documentation updates in your pull request