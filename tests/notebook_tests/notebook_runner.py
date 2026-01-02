"""
Notebook test runner for executing and testing Jupyter notebooks.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nbformat
import pandas as pd
import papermill as pm
import psutil
from memory_profiler import memory_usage
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor
from notebook_validator import DataValidator, NotebookValidator


class NotebookTestRunner:
    """Runs notebook tests and collects execution metrics."""

    def __init__(
        self,
        notebook_dirs: List[str],
        output_dir: str = "test_results",
        timeout: int = 600,
        parallel: bool = False,
    ):
        """
        Initialize the test runner.

        Args:
            notebook_dirs: List of directories containing notebooks
            output_dir: Directory to save test results
            timeout: Maximum execution time per notebook in seconds
            parallel: Whether to run notebooks in parallel
        """
        self.notebook_dirs = [Path(d) for d in notebook_dirs]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.parallel = parallel
        self.results = []

    def find_notebooks(self) -> List[Path]:
        """Find all notebooks in the specified directories."""
        notebooks = []
        for directory in self.notebook_dirs:
            if directory.exists():
                notebooks.extend(directory.glob("**/*.ipynb"))

        # Filter out checkpoint notebooks
        notebooks = [nb for nb in notebooks if ".ipynb_checkpoints" not in str(nb)]
        return notebooks

    def run_all_tests(self) -> Dict[str, Any]:
        """Run tests on all notebooks."""
        notebooks = self.find_notebooks()
        print(f"Found {len(notebooks)} notebooks to test")

        start_time = time.time()

        if self.parallel and len(notebooks) > 1:
            results = self._run_parallel(notebooks)
        else:
            results = self._run_sequential(notebooks)

        total_time = time.time() - start_time

        # Generate summary
        summary = self._generate_summary(results, total_time)

        # Save results
        self._save_results(results, summary)

        return {"results": results, "summary": summary}

    def _run_sequential(self, notebooks: List[Path]) -> List[Dict[str, Any]]:
        """Run notebooks sequentially."""
        results = []
        for i, notebook_path in enumerate(notebooks, 1):
            print(f"\n[{i}/{len(notebooks)}] Testing: {notebook_path.name}")
            result = self.test_notebook(notebook_path)
            results.append(result)
            self._print_result(result)
        return results

    def _run_parallel(self, notebooks: List[Path]) -> List[Dict[str, Any]]:
        """Run notebooks in parallel."""
        results = []
        max_workers = min(os.cpu_count() or 1, len(notebooks))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.test_notebook, nb): nb for nb in notebooks}

            for i, future in enumerate(as_completed(futures), 1):
                notebook = futures[future]
                try:
                    result = future.result()
                    print(f"[{i}/{len(notebooks)}] Completed: {notebook.name}")
                    self._print_result(result)
                    results.append(result)
                except Exception as e:
                    print(f"[{i}/{len(notebooks)}] Failed: {notebook.name} - {e}")
                    results.append(
                        {"notebook": str(notebook), "status": "error", "error": str(e)}
                    )

        return results

    def test_notebook(self, notebook_path: Path) -> Dict[str, Any]:
        """
        Test a single notebook.

        Returns:
            Dictionary containing test results
        """
        result = {
            "notebook": str(notebook_path),
            "notebook_name": notebook_path.name,
            "timestamp": datetime.now().isoformat(),
            "status": "not_run",
            "execution": {},
            "validation": {},
            "metrics": {},
        }

        try:
            # Run validation checks
            print(f"  Validating {notebook_path.name}...")
            validator = NotebookValidator(str(notebook_path))
            result["validation"] = validator.validate_all()

            # Execute notebook
            print(f"  Executing {notebook_path.name}...")
            execution_result = self._execute_notebook(notebook_path)
            result["execution"] = execution_result

            # Determine overall status
            if execution_result["success"]:
                if result["validation"]["score"] >= 80:
                    result["status"] = "passed"
                else:
                    result["status"] = "passed_with_warnings"
            else:
                result["status"] = "failed"

            # Collect metrics
            result["metrics"] = self._collect_metrics(notebook_path, execution_result)

        except Exception as e:
            result["status"] = "error"
            result["error"] = {"message": str(e), "traceback": traceback.format_exc()}

        return result

    def _execute_notebook(self, notebook_path: Path) -> Dict[str, Any]:
        """Execute a notebook and capture metrics."""
        result = {
            "success": False,
            "execution_time": 0,
            "cell_timings": [],
            "errors": [],
            "warnings": [],
            "memory_usage": {},
        }

        # Create temporary output notebook
        temp_dir = tempfile.gettempdir()
        output_notebook = (
            Path(temp_dir) / f"test_{notebook_path.stem}_{int(time.time())}.ipynb"
        )

        try:
            # Monitor memory usage during execution
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            # Execute with papermill for better error handling
            start_time = time.time()

            # Use papermill to execute
            try:
                pm.execute_notebook(
                    str(notebook_path),
                    str(output_notebook),
                    kernel_name="python3",
                    timeout=self.timeout,
                    progress_bar=False,
                    report_mode=True,
                    cwd=str(notebook_path.parent),
                )
                result["success"] = True
            except pm.PapermillExecutionError as e:
                result["errors"].append(
                    {
                        "type": "execution_error",
                        "cell": e.exec_count,
                        "error": str(e.ename),
                        "message": str(e.evalue),
                        "traceback": e.traceback,
                    }
                )
            except Exception as e:
                result["errors"].append(
                    {
                        "type": "unexpected_error",
                        "error": str(type(e).__name__),
                        "message": str(e),
                    }
                )

            result["execution_time"] = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            result["memory_usage"] = {
                "start_mb": start_memory,
                "end_mb": end_memory,
                "delta_mb": end_memory - start_memory,
            }

            # Extract cell timings from executed notebook if it exists
            if output_notebook.exists():
                with open(output_notebook, "r", encoding="utf-8") as f:
                    executed_nb = nbformat.read(f, as_version=4)
                    result["cell_timings"] = self._extract_cell_timings(executed_nb)

        except Exception as e:
            result["errors"].append(
                {
                    "type": "runner_error",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            # Clean up temporary file
            if output_notebook.exists():
                try:
                    output_notebook.unlink()
                except:
                    pass

        return result

    def _extract_cell_timings(
        self, notebook: nbformat.NotebookNode
    ) -> List[Dict[str, Any]]:
        """Extract execution timings from notebook cells."""
        timings = []

        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                timing = {
                    "cell": i + 1,
                    "execution_count": cell.get("execution_count"),
                }

                # Try to get execution time from metadata
                metadata = cell.get("metadata", {})
                if "execution" in metadata:
                    execution_data = metadata["execution"]
                    if (
                        "iopub.execute_input" in execution_data
                        and "shell.execute_reply" in execution_data
                    ):
                        try:
                            start = pd.Timestamp(execution_data["iopub.execute_input"])
                            end = pd.Timestamp(execution_data["shell.execute_reply"])
                            timing["duration_ms"] = (end - start).total_seconds() * 1000
                        except:
                            pass

                timings.append(timing)

        return timings

    def _collect_metrics(
        self, notebook_path: Path, execution_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect various metrics about the notebook."""
        metrics = {
            "execution_time": execution_result["execution_time"],
            "memory_delta_mb": execution_result["memory_usage"].get("delta_mb", 0),
            "cell_count": 0,
            "code_cell_count": 0,
            "markdown_cell_count": 0,
            "total_lines": 0,
            "code_lines": 0,
        }

        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=4)

                metrics["cell_count"] = len(notebook.cells)

                for cell in notebook.cells:
                    if cell.cell_type == "code":
                        metrics["code_cell_count"] += 1
                        lines = cell.source.split("\n")
                        metrics["code_lines"] += len(lines)
                        metrics["total_lines"] += len(lines)
                    elif cell.cell_type == "markdown":
                        metrics["markdown_cell_count"] += 1
                        metrics["total_lines"] += len(cell.source.split("\n"))

                # Add performance metrics
                if execution_result["success"] and metrics["execution_time"] > 0:
                    metrics["cells_per_second"] = (
                        metrics["code_cell_count"] / metrics["execution_time"]
                    )
                    metrics["mb_per_cell"] = metrics["memory_delta_mb"] / max(
                        1, metrics["code_cell_count"]
                    )

        except Exception as e:
            metrics["error"] = str(e)

        return metrics

    def _print_result(self, result: Dict[str, Any]):
        """Print a summary of test result."""
        status = result["status"]
        name = result["notebook_name"]

        status_symbols = {
            "passed": "[PASS]",
            "passed_with_warnings": "[WARN]",
            "failed": "[FAIL]",
            "error": "[ERR]",
            "not_run": "[-]",
        }

        symbol = status_symbols.get(status, "?")

        # Color codes for terminal (optional)
        colors = {
            "passed": "\033[92m",  # Green
            "passed_with_warnings": "\033[93m",  # Yellow
            "failed": "\033[91m",  # Red
            "error": "\033[91m",  # Red
            "reset": "\033[0m",
        }

        color = colors.get(status, colors["reset"])

        execution_time = result.get("execution", {}).get("execution_time", 0)
        validation_score = result.get("validation", {}).get("score", 0)

        print(
            f"  {color}{symbol}{colors['reset']} {name:40} "
            f"Time: {execution_time:.2f}s  Score: {validation_score:.0f}/100"
        )

        # Print errors if any
        if status in ["failed", "error"]:
            errors = result.get("execution", {}).get("errors", [])
            for error in errors[:1]:  # Show first error
                print(f"    Error: {error.get('message', 'Unknown error')}")

    def _generate_summary(
        self, results: List[Dict[str, Any]], total_time: float
    ) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        summary = {
            "total_notebooks": len(results),
            "total_time": total_time,
            "timestamp": datetime.now().isoformat(),
            "status_counts": {},
            "average_metrics": {},
            "failed_notebooks": [],
            "warnings": [],
        }

        # Count statuses
        for result in results:
            status = result["status"]
            summary["status_counts"][status] = (
                summary["status_counts"].get(status, 0) + 1
            )

            if status in ["failed", "error"]:
                summary["failed_notebooks"].append(
                    {
                        "notebook": result["notebook_name"],
                        "errors": result.get("execution", {}).get("errors", []),
                    }
                )

            # Collect warnings
            validation_warnings = result.get("validation", {}).get("warnings", [])
            if validation_warnings:
                summary["warnings"].append(
                    {
                        "notebook": result["notebook_name"],
                        "warnings": validation_warnings,
                    }
                )

        # Calculate average metrics
        metric_sums = {}
        metric_counts = {}

        for result in results:
            metrics = result.get("metrics", {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_sums[key] = metric_sums.get(key, 0) + value
                    metric_counts[key] = metric_counts.get(key, 0) + 1

        for key in metric_sums:
            if metric_counts[key] > 0:
                summary["average_metrics"][key] = metric_sums[key] / metric_counts[key]

        # Calculate pass rate
        passed = summary["status_counts"].get("passed", 0) + summary[
            "status_counts"
        ].get("passed_with_warnings", 0)
        summary["pass_rate"] = (passed / len(results) * 100) if results else 0

        return summary

    def _save_results(self, results: List[Dict[str, Any]], summary: Dict[str, Any]):
        """Save test results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results as JSON
        results_file = self.output_dir / f"notebook_test_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(
                {"summary": summary, "results": results}, f, indent=2, default=str
            )

        print(f"\nResults saved to: {results_file}")

        # Save summary as markdown
        summary_file = self.output_dir / f"test_summary_{timestamp}.md"
        self._write_markdown_summary(summary_file, summary, results)
        print(f"Summary saved to: {summary_file}")

    def _write_markdown_summary(
        self, filepath: Path, summary: Dict[str, Any], results: List[Dict[str, Any]]
    ):
        """Write a markdown summary of test results."""
        with open(filepath, "w") as f:
            f.write("# Notebook Test Results\n\n")
            f.write(f"**Date:** {summary['timestamp']}\n")
            f.write(f"**Total Time:** {summary['total_time']:.2f} seconds\n")
            f.write(f"**Pass Rate:** {summary['pass_rate']:.1f}%\n\n")

            f.write("## Summary\n\n")
            f.write(f"- **Total Notebooks:** {summary['total_notebooks']}\n")
            for status, count in summary["status_counts"].items():
                f.write(f"- **{status.title()}:** {count}\n")
            f.write("\n")

            # Write failed notebooks
            if summary["failed_notebooks"]:
                f.write("## Failed Notebooks\n\n")
                for fail in summary["failed_notebooks"]:
                    f.write(f"### {fail['notebook']}\n")
                    for error in fail["errors"][:3]:  # Show up to 3 errors
                        f.write(f"- {error.get('message', 'Unknown error')}\n")
                    f.write("\n")

            # Write detailed results table
            f.write("## Detailed Results\n\n")
            f.write(
                "| Notebook | Status | Execution Time | Validation Score | Issues |\n"
            )
            f.write(
                "|----------|--------|----------------|------------------|--------|\n"
            )

            for result in results:
                name = result["notebook_name"]
                status = result["status"]
                exec_time = result.get("execution", {}).get("execution_time", 0)
                score = result.get("validation", {}).get("score", 0)
                issues = len(result.get("validation", {}).get("issues", []))

                f.write(
                    f"| {name} | {status} | {exec_time:.2f}s | {score:.0f}/100 | {issues} |\n"
                )

            # Write average metrics
            if summary["average_metrics"]:
                f.write("\n## Average Metrics\n\n")
                for metric, value in summary["average_metrics"].items():
                    f.write(f"- **{metric.replace('_', ' ').title()}:** {value:.2f}\n")


class NotebookTestConfig:
    """Configuration for notebook tests."""

    def __init__(self, config_path: Optional[str] = None):
        """Load test configuration."""
        self.config = {
            "timeout": 600,
            "parallel": False,
            "skip_patterns": [".ipynb_checkpoints", "__pycache__", "test_"],
            "expected_schemas": {},
            "metric_ranges": {},
            "data_files": [],
        }

        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                self.config.update(json.load(f))

    def get_expected_schema(self, data_name: str) -> Optional[Dict[str, str]]:
        """Get expected schema for a dataset."""
        return self.config["expected_schemas"].get(data_name)

    def get_metric_ranges(
        self, notebook_name: str
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        """Get expected metric ranges for a notebook."""
        return self.config["metric_ranges"].get(notebook_name)


def run_notebook_tests(
    notebook_dirs: List[str],
    output_dir: str = "test_results",
    config_path: Optional[str] = None,
    parallel: bool = False,
) -> Dict[str, Any]:
    """
    Main function to run notebook tests.

    Args:
        notebook_dirs: List of directories containing notebooks
        output_dir: Directory to save results
        config_path: Optional path to configuration file
        parallel: Whether to run tests in parallel

    Returns:
        Dictionary containing test results and summary
    """
    config = NotebookTestConfig(config_path)

    runner = NotebookTestRunner(
        notebook_dirs=notebook_dirs,
        output_dir=output_dir,
        timeout=config.config["timeout"],
        parallel=parallel,
    )

    return runner.run_all_tests()


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Run notebook tests")
    parser.add_argument(
        "directories", nargs="+", help="Directories containing notebooks"
    )
    parser.add_argument("--output", default="test_results", help="Output directory")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument(
        "--timeout", type=int, default=600, help="Timeout per notebook in seconds"
    )

    args = parser.parse_args()

    results = run_notebook_tests(
        notebook_dirs=args.directories,
        output_dir=args.output,
        config_path=args.config,
        parallel=args.parallel,
    )

    # Exit with appropriate code
    if results["summary"]["pass_rate"] == 100:
        sys.exit(0)
    else:
        sys.exit(1)
