#!/usr/bin/env python
"""Comprehensive test runner script for the DS Projects Portfolio.

This script provides a centralized way to run all tests with various configurations
and generate comprehensive reports.
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class TestRunner:
    """Comprehensive test runner with reporting capabilities."""

    def __init__(self):
        """Initialize the test runner."""
        self.project_root = Path(__file__).parent
        self.test_dir = self.project_root / "tests"
        self.reports_dir = self.project_root / "test_reports"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create reports directory if it doesn't exist
        self.reports_dir.mkdir(exist_ok=True)

    def run_unit_tests(self, verbose=False):
        """Run unit tests."""
        logger.info("Running unit tests...")
        cmd = ["pytest", "-m", "unit"]
        if verbose:
            cmd.append("-vv")
        return self._run_command(cmd)

    def run_integration_tests(self, verbose=False):
        """Run integration tests."""
        logger.info("Running integration tests...")
        cmd = ["pytest", "-m", "integration"]
        if verbose:
            cmd.append("-vv")
        return self._run_command(cmd)

    def run_performance_tests(self):
        """Run performance tests."""
        logger.info("Running performance tests...")
        cmd = [
            "pytest",
            "-m",
            "performance",
            "--benchmark-only",
            "--benchmark-json=benchmark.json",
        ]
        return self._run_command(cmd)

    def run_notebook_tests(self):
        """Run notebook tests."""
        logger.info("Running notebook tests...")
        cmd = [
            "pytest",
            "tests/notebook_tests",
            "--nbval-lax",
            "--nbval-cell-timeout=300",
        ]
        return self._run_command(cmd)

    def run_security_tests(self):
        """Run security tests."""
        logger.info("Running security tests...")

        # Run bandit for security analysis
        logger.info("Running bandit security scan...")
        bandit_cmd = ["bandit", "-r", "src", "-f", "json", "-o", "bandit_report.json"]
        self._run_command(bandit_cmd, check=False)

        # Run safety check for dependencies
        logger.info("Running safety check on dependencies...")
        safety_cmd = ["safety", "check", "--json"]
        self._run_command(safety_cmd, check=False)

        return True

    def run_all_tests(self, parallel=False, coverage=True):
        """Run all tests with optional parallelization."""
        logger.info("Running all tests...")
        cmd = ["pytest"]

        if parallel:
            cmd.extend(["-n", "auto"])

        if coverage:
            cmd.extend(
                [
                    "--cov=src",
                    "--cov=ab_testing",
                    "--cov=dashboard_enhanced",
                    "--cov=time_series",
                    "--cov-report=html:"
                    + str(self.reports_dir / f"coverage_{self.timestamp}"),
                    "--cov-report=term",
                    "--cov-report=xml",
                ]
            )

        # Add HTML report
        cmd.extend(
            [
                "--html=" + str(self.reports_dir / f"report_{self.timestamp}.html"),
                "--self-contained-html",
            ]
        )

        return self._run_command(cmd)

    def run_smoke_tests(self):
        """Run quick smoke tests."""
        logger.info("Running smoke tests...")
        cmd = ["pytest", "-m", "smoke", "--maxfail=1"]
        return self._run_command(cmd)

    def run_regression_tests(self):
        """Run regression tests."""
        logger.info("Running regression tests...")
        cmd = ["pytest", "-m", "regression"]
        return self._run_command(cmd)

    def run_specific_module(self, module_name):
        """Run tests for a specific module."""
        logger.info(f"Running tests for module: {module_name}")
        test_file = self.test_dir / f"test_{module_name}.py"

        if not test_file.exists():
            logger.error(f"Test file not found: {test_file}")
            return False

        cmd = ["pytest", str(test_file), "-v"]
        return self._run_command(cmd)

    def run_failed_tests(self):
        """Re-run only failed tests from last run."""
        logger.info("Re-running failed tests...")
        cmd = ["pytest", "--lf", "-v"]
        return self._run_command(cmd)

    def run_with_markers(self, markers):
        """Run tests with specific markers."""
        logger.info(f"Running tests with markers: {markers}")
        cmd = ["pytest", "-m", markers]
        return self._run_command(cmd)

    def generate_allure_report(self):
        """Generate Allure test report."""
        logger.info("Generating Allure report...")
        allure_dir = self.reports_dir / "allure"
        cmd = ["pytest", "--alluredir=" + str(allure_dir)]
        if self._run_command(cmd):
            # Generate HTML report
            allure_cmd = [
                "allure",
                "generate",
                str(allure_dir),
                "-o",
                str(allure_dir / "html"),
            ]
            self._run_command(allure_cmd, check=False)
            logger.info(f"Allure report generated at: {allure_dir / 'html'}")
        return True

    def run_mutation_tests(self):
        """Run mutation testing with mutmut."""
        logger.info("Running mutation tests...")
        cmd = ["mutmut", "run", "--paths-to-mutate=src/"]
        self._run_command(cmd, check=False)

        # Generate mutation report
        report_cmd = ["mutmut", "html"]
        self._run_command(report_cmd, check=False)
        return True

    def run_type_checks(self):
        """Run type checking with mypy."""
        logger.info("Running type checks with mypy...")
        cmd = [
            "mypy",
            "src",
            "--ignore-missing-imports",
            "--html-report",
            str(self.reports_dir / "mypy"),
        ]
        return self._run_command(cmd, check=False)

    def run_linting(self):
        """Run code linting."""
        logger.info("Running code linting...")

        # Run flake8
        logger.info("Running flake8...")
        flake8_cmd = [
            "flake8",
            "src",
            "tests",
            "--output-file=" + str(self.reports_dir / "flake8.txt"),
        ]
        self._run_command(flake8_cmd, check=False)

        # Run pylint
        logger.info("Running pylint...")
        pylint_cmd = ["pylint", "src", "--output-format=json"]
        self._run_command(pylint_cmd, check=False)

        # Run black check
        logger.info("Checking code formatting with black...")
        black_cmd = ["black", "--check", "src", "tests"]
        self._run_command(black_cmd, check=False)

        return True

    def create_test_summary(self):
        """Create a comprehensive test summary."""
        logger.info("Creating test summary...")

        summary = {
            "timestamp": self.timestamp,
            "reports_directory": str(self.reports_dir),
            "test_results": {},
            "coverage": {},
            "quality_metrics": {},
        }

        # Parse pytest results if available
        junit_file = Path("junit.xml")
        if junit_file.exists():
            import xml.etree.ElementTree as ET

            tree = ET.parse(junit_file)
            root = tree.getroot()
            summary["test_results"] = {
                "total": int(root.get("tests", 0)),
                "passed": int(root.get("tests", 0))
                - int(root.get("failures", 0))
                - int(root.get("errors", 0)),
                "failed": int(root.get("failures", 0)),
                "errors": int(root.get("errors", 0)),
                "time": float(root.get("time", 0)),
            }

        # Parse coverage results if available
        coverage_file = Path("coverage.xml")
        if coverage_file.exists():
            import xml.etree.ElementTree as ET

            tree = ET.parse(coverage_file)
            root = tree.getroot()
            summary["coverage"] = {
                "line_rate": float(root.get("line-rate", 0)) * 100,
                "branch_rate": float(root.get("branch-rate", 0)) * 100,
            }

        # Save summary
        summary_file = self.reports_dir / f"summary_{self.timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Test summary saved to: {summary_file}")
        return summary

    def _run_command(self, cmd, check=True):
        """Run a command and handle errors."""
        try:
            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)

            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            if e.stdout:
                print(e.stdout)
            if e.stderr:
                print(e.stderr, file=sys.stderr)
            return False
        except FileNotFoundError:
            logger.error(f"Command not found: {cmd[0]}")
            return False


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="DS Projects Portfolio Test Runner")

    parser.add_argument(
        "--type",
        choices=[
            "all",
            "unit",
            "integration",
            "performance",
            "notebook",
            "security",
            "smoke",
            "regression",
        ],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument("--module", help="Run tests for specific module")
    parser.add_argument("--markers", help="Run tests with specific markers")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument(
        "--coverage", action="store_true", default=True, help="Generate coverage report"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--failed", action="store_true", help="Re-run failed tests only"
    )
    parser.add_argument("--allure", action="store_true", help="Generate Allure report")
    parser.add_argument("--mutation", action="store_true", help="Run mutation tests")
    parser.add_argument("--type-check", action="store_true", help="Run type checking")
    parser.add_argument("--lint", action="store_true", help="Run linting")
    parser.add_argument(
        "--full-suite",
        action="store_true",
        help="Run complete test suite with all checks",
    )

    args = parser.parse_args()

    runner = TestRunner()
    success = True

    try:
        if args.full_suite:
            logger.info("Running full test suite...")
            success &= runner.run_linting()
            success &= runner.run_type_checks()
            success &= runner.run_security_tests()
            success &= runner.run_all_tests(parallel=args.parallel, coverage=True)
            success &= runner.run_notebook_tests()
            runner.generate_allure_report()
            runner.create_test_summary()

        elif args.failed:
            success = runner.run_failed_tests()

        elif args.module:
            success = runner.run_specific_module(args.module)

        elif args.markers:
            success = runner.run_with_markers(args.markers)

        elif args.allure:
            runner.generate_allure_report()

        elif args.mutation:
            success = runner.run_mutation_tests()

        elif args.type_check:
            success = runner.run_type_checks()

        elif args.lint:
            success = runner.run_linting()

        else:
            # Run specific test type
            test_methods = {
                "all": lambda: runner.run_all_tests(args.parallel, args.coverage),
                "unit": lambda: runner.run_unit_tests(args.verbose),
                "integration": lambda: runner.run_integration_tests(args.verbose),
                "performance": runner.run_performance_tests,
                "notebook": runner.run_notebook_tests,
                "security": runner.run_security_tests,
                "smoke": runner.run_smoke_tests,
                "regression": runner.run_regression_tests,
            }

            test_method = test_methods.get(args.type)
            if test_method:
                success = test_method()

        # Create summary if any tests were run
        if args.type != "all" or args.module or args.markers:
            runner.create_test_summary()

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
