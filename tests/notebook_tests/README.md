# Jupyter Notebook Testing Framework

A comprehensive testing framework for validating and executing Jupyter notebooks with detailed reporting and CI/CD integration.

## Features

### 1. Notebook Validation
- **Hardcoded Path Detection**: Identifies absolute paths that should be parameterized
- **Data File Verification**: Checks that all referenced data files exist
- **Random Seed Validation**: Ensures reproducibility by checking for proper seed setting
- **Visualization Checks**: Verifies plots are properly displayed with show() calls
- **Memory Leak Detection**: Identifies potential memory issues and inefficient patterns
- **Code Quality Analysis**: Checks for long lines, excessive comments, bare exceptions
- **Import Organization**: Validates import structure and identifies unused imports
- **Documentation Quality**: Ensures adequate markdown documentation

### 2. Notebook Execution
- **End-to-End Testing**: Runs notebooks completely using papermill
- **Cell Timing Capture**: Records execution time for each cell
- **Error Handling**: Captures and reports execution errors with full tracebacks
- **Memory Monitoring**: Tracks memory usage throughout execution
- **Parallel Execution**: Supports concurrent notebook testing for faster CI/CD

### 3. Data Validation
- **Schema Validation**: Verifies DataFrames match expected schemas
- **Quality Checks**: Detects nulls, duplicates, and outliers
- **Metric Validation**: Ensures output metrics are within expected ranges
- **Cardinality Analysis**: Identifies high-cardinality categorical columns

### 4. Reporting
- **HTML Reports**: Beautiful, interactive HTML reports with charts
- **JSON Output**: Machine-readable results for CI/CD integration
- **Markdown Summaries**: Quick overview of test results
- **GitHub Integration**: Automatic PR comments with test results

## Installation

```bash
# Install required packages
pip install nbval pytest-notebook nbconvert nbformat papermill pytest-html pytest-xdist memory_profiler

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## Usage

### Command Line Interface

```bash
# Run validation only (fast)
python scripts/run_notebook_tests.py --validate-only

# Run full tests (validation + execution)
python scripts/run_notebook_tests.py --dirs ab_testing modern-bank-churn

# Run tests in parallel
python scripts/run_notebook_tests.py --parallel

# Generate report from existing results
python scripts/run_notebook_tests.py --report-only results.json

# Custom configuration
python scripts/run_notebook_tests.py --config my_config.json --timeout 300
```

### Python API

```python
from tests.notebook_tests.notebook_runner import NotebookTestRunner
from tests.notebook_tests.report_generator import generate_html_report

# Create runner
runner = NotebookTestRunner(
    notebook_dirs=["ab_testing", "modern-bank-churn"],
    output_dir="test_results",
    timeout=600,
    parallel=True
)

# Run tests
results = runner.run_all_tests()

# Generate HTML report
generate_html_report(results, "report.html")
```

### Pytest Integration

```bash
# Run notebook tests with pytest
pytest tests/notebook_tests/test_notebooks.py

# Run only validation tests (no execution)
pytest tests/notebook_tests/test_notebooks.py -m validation

# Run with specific markers
pytest tests/notebook_tests/test_notebooks.py -m "not slow"
```

## Configuration

Edit `tests/notebook_tests/test_config.json`:

```json
{
    "timeout": 600,
    "parallel": false,
    "expected_schemas": {
        "churn_data": {
            "CustomerId": "int",
            "Age": "int",
            "Balance": "float",
            "Exited": "int"
        }
    },
    "metric_ranges": {
        "accuracy": [0.7, 0.95],
        "auc": [0.7, 0.99]
    },
    "memory_limits": {
        "max_memory_mb": 4096
    },
    "validation_rules": {
        "require_random_seeds": true,
        "allow_hardcoded_paths": false
    }
}
```

## CI/CD Integration

### GitHub Actions

The framework includes a GitHub Actions workflow (`.github/workflows/notebook-tests.yml`) that:

1. Runs on push/PR for notebook changes
2. Tests with multiple Python versions
3. Validates notebooks first (quick check)
4. Executes notebooks if validation passes
5. Comments results on PRs
6. Uploads test artifacts

### Pre-commit Hooks

Configure in `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: notebook-validation
        name: Validate Jupyter Notebooks
        entry: python scripts/run_notebook_tests.py --validate-only
        language: system
        files: \.ipynb$
```

## Test Reports

### HTML Report Features
- **Summary Cards**: Total notebooks, pass rate, execution time
- **Interactive Charts**: Status distribution, execution times
- **Detailed Results Table**: Per-notebook metrics and scores
- **Issue Details**: Expandable sections for warnings and errors
- **Responsive Design**: Works on mobile and desktop

### Report Sections
1. **Executive Summary**: High-level metrics and pass/fail status
2. **Status Distribution**: Pie chart of test outcomes
3. **Execution Times**: Bar chart of notebook runtimes
4. **Validation Scores**: Quality scores for each notebook
5. **Detailed Issues**: Specific problems found in each notebook

## Validation Rules

### Critical (Test Failures)
- Missing required data files
- Execution errors or exceptions
- Validation score below 60%
- Memory usage exceeding limits

### Warnings (Non-blocking)
- Hardcoded paths (should use relative paths)
- Missing random seeds for reproducibility
- Plots without show() calls
- Excessive code complexity

### Info (Suggestions)
- Minimal documentation
- Potentially unused imports
- Long code lines
- Large memory allocations without cleanup

## Best Practices

### For Notebook Authors
1. **Clear outputs before committing**: Use nbstripout or clear manually
2. **Use relative paths**: Never hardcode absolute paths
3. **Set random seeds**: Ensure reproducibility
4. **Add documentation**: Include markdown cells explaining the analysis
5. **Handle errors gracefully**: Use try-except blocks appropriately
6. **Clean up resources**: Delete large variables when done

### For Test Configuration
1. **Set appropriate timeouts**: Balance thoroughness with CI/CD speed
2. **Define expected schemas**: Document data structure expectations
3. **Specify metric ranges**: Set reasonable bounds for model performance
4. **Use parallel execution**: Speed up tests when possible
5. **Archive test results**: Keep history for trend analysis

## Troubleshooting

### Common Issues

1. **Import errors during execution**
   - Ensure all dependencies are installed
   - Check for missing `__init__.py` files
   - Verify PYTHONPATH includes necessary directories

2. **Memory errors**
   - Increase memory limits in configuration
   - Add cleanup cells to notebooks
   - Use data sampling for tests

3. **Timeout errors**
   - Increase timeout in configuration
   - Optimize slow cells
   - Consider splitting large notebooks

4. **Path resolution issues**
   - Use relative paths from notebook location
   - Set working directory correctly
   - Check data file locations

## Architecture

```
tests/notebook_tests/
├── notebook_validator.py   # Validation logic
├── notebook_runner.py      # Execution engine
├── report_generator.py     # HTML report generation
├── test_notebooks.py       # Pytest integration
├── test_config.json        # Configuration
└── README.md              # This file

scripts/run_notebook_tests.py  # Main CLI entry point
.github/workflows/            # CI/CD workflows
.pre-commit-config.yaml       # Pre-commit hooks
```

## Contributing

1. Add new validation rules in `notebook_validator.py`
2. Extend test configuration schema
3. Add new report visualizations
4. Improve error messages and debugging
5. Add support for new notebook features

## License

This testing framework is part of the DS Projects Portfolio and follows the project's licensing terms.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review test output logs
3. Examine HTML reports for details
4. Open an issue with reproduction steps

---

*Comprehensive notebook testing for reliable data science workflows*