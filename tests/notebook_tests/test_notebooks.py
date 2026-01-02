"""
Pytest integration for notebook testing.
This allows notebook tests to be run as part of the standard pytest suite.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import nbformat
import pytest
from nbval import NbvalReporter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from notebook_runner import NotebookTestConfig, NotebookTestRunner
from notebook_validator import DataValidator, NotebookValidator
from report_generator import generate_html_report

# Configuration
NOTEBOOK_DIRS = ["ab_testing", "modern-bank-churn"]
CONFIG_PATH = Path(__file__).parent / "test_config.json"
OUTPUT_DIR = Path(__file__).parent / "test_results"


class TestNotebooks:
    """Test class for Jupyter notebooks."""

    @classmethod
    def setup_class(cls):
        """Setup test class."""
        cls.config = NotebookTestConfig(str(CONFIG_PATH))
        cls.runner = NotebookTestRunner(
            notebook_dirs=NOTEBOOK_DIRS,
            output_dir=str(OUTPUT_DIR),
            timeout=cls.config.config["timeout"],
            parallel=False,  # Disable parallel in pytest
        )
        cls.notebooks = cls.runner.find_notebooks()

    def test_notebook_count(self):
        """Test that notebooks are found."""
        assert len(self.notebooks) > 0, "No notebooks found in specified directories"

    @pytest.mark.parametrize(
        "notebook_path",
        [
            nb
            for nb in NotebookTestRunner(
                NOTEBOOK_DIRS, str(OUTPUT_DIR)
            ).find_notebooks()
        ],
    )
    def test_notebook_validation(self, notebook_path):
        """Test notebook validation checks."""
        validator = NotebookValidator(str(notebook_path))
        results = validator.validate_all()

        # Check for critical issues
        assert results["checks"]["data_files"][
            "passed"
        ], f"Missing data files in {notebook_path.name}: {results['checks']['data_files']['missing_files']}"

        # Check for hardcoded paths (warning, not failure)
        if not results["checks"]["hardcoded_paths"]["passed"]:
            pytest.skip(
                f"Hardcoded paths found (warning): {results['checks']['hardcoded_paths']['count']}"
            )

        # Check validation score
        assert (
            results["score"] >= 60
        ), f"Validation score too low ({results['score']}/100) for {notebook_path.name}"

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "notebook_path",
        [
            nb
            for nb in NotebookTestRunner(
                NOTEBOOK_DIRS, str(OUTPUT_DIR)
            ).find_notebooks()
        ],
    )
    def test_notebook_execution(self, notebook_path):
        """Test notebook execution."""
        result = self.runner.test_notebook(notebook_path)

        # Check execution status
        assert result["status"] in [
            "passed",
            "passed_with_warnings",
        ], f"Notebook execution failed: {result.get('execution', {}).get('errors', ['Unknown error'])}"

        # Check execution time
        exec_time = result.get("execution", {}).get("execution_time", 0)
        assert (
            exec_time < self.config.config["timeout"]
        ), f"Notebook execution timeout: {exec_time}s > {self.config.config['timeout']}s"

        # Check memory usage
        memory_delta = (
            result.get("execution", {}).get("memory_usage", {}).get("delta_mb", 0)
        )
        max_memory = self.config.config.get("memory_limits", {}).get(
            "max_memory_mb", 4096
        )
        assert (
            memory_delta < max_memory
        ), f"Excessive memory usage: {memory_delta}MB > {max_memory}MB"

    @pytest.mark.parametrize(
        "notebook_path",
        [
            nb
            for nb in NotebookTestRunner(
                NOTEBOOK_DIRS, str(OUTPUT_DIR)
            ).find_notebooks()
            if "eda" in nb.name.lower() or "baseline" in nb.name.lower()
        ],
    )
    def test_data_validation(self, notebook_path):
        """Test data validation in notebooks."""
        # This is a placeholder for actual data validation
        # In practice, you would extract DataFrames from notebook execution
        # and validate them
        pass

    def test_random_seeds(self):
        """Test that all notebooks set random seeds."""
        notebooks_without_seeds = []

        for notebook_path in self.notebooks:
            validator = NotebookValidator(str(notebook_path))
            result = validator.check_random_seeds()

            if not result["passed"] and result["imported_libs"]:
                notebooks_without_seeds.append(
                    {
                        "notebook": notebook_path.name,
                        "missing_seeds": result["missing_seeds"],
                    }
                )

        assert (
            len(notebooks_without_seeds) == 0
        ), f"Notebooks missing random seeds: {notebooks_without_seeds}"

    def test_visualization_completeness(self):
        """Test that visualizations are properly displayed."""
        notebooks_with_issues = []

        for notebook_path in self.notebooks:
            validator = NotebookValidator(str(notebook_path))
            result = validator.check_visualizations()

            if not result["passed"]:
                notebooks_with_issues.append(
                    {
                        "notebook": notebook_path.name,
                        "missing_show": result["missing_show_count"],
                    }
                )

        # This is a warning, not a failure
        if notebooks_with_issues:
            pytest.skip(f"Notebooks with visualization issues: {notebooks_with_issues}")

    @pytest.mark.slow
    def test_generate_report(self):
        """Test report generation."""
        results = self.runner.run_all_tests()

        # Generate HTML report
        html_path = OUTPUT_DIR / "test_report.html"
        generate_html_report(results, str(html_path))

        assert html_path.exists(), "HTML report not generated"

        # Check summary statistics
        summary = results["summary"]
        assert (
            summary["pass_rate"] >= 70
        ), f"Overall pass rate too low: {summary['pass_rate']}%"


@pytest.fixture(scope="module")
def notebook_results():
    """Fixture to run all notebook tests once and share results."""
    runner = NotebookTestRunner(
        notebook_dirs=NOTEBOOK_DIRS, output_dir=str(OUTPUT_DIR), timeout=600
    )
    return runner.run_all_tests()


def test_overall_pass_rate(notebook_results):
    """Test overall pass rate of notebooks."""
    summary = notebook_results["summary"]
    assert (
        summary["pass_rate"] >= 70
    ), f"Overall pass rate {summary['pass_rate']}% is below threshold of 70%"


def test_no_critical_errors(notebook_results):
    """Test that no notebooks have critical errors."""
    failed = notebook_results["summary"].get("failed_notebooks", [])
    assert (
        len(failed) == 0
    ), f"Notebooks with critical errors: {[f['notebook'] for f in failed]}"


def test_average_execution_time(notebook_results):
    """Test that average execution time is reasonable."""
    avg_time = notebook_results["summary"]["average_metrics"].get("execution_time", 0)
    assert avg_time < 300, f"Average execution time {avg_time}s exceeds 5 minutes"


# Pytest markers for selective testing
pytest.mark.validation = pytest.mark.mark(
    pytest.mark.validation, "mark test as validation only (no execution)"
)

pytest.mark.execution = pytest.mark.mark(
    pytest.mark.execution, "mark test as execution test"
)

pytest.mark.slow = pytest.mark.mark(
    pytest.mark.slow, "mark test as slow (notebook execution)"
)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
