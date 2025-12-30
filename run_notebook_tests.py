#!/usr/bin/env python
"""
Command-line interface for running notebook tests.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent / "tests" / "notebook_tests"))

from notebook_runner import NotebookTestRunner, NotebookTestConfig
from notebook_validator import NotebookValidator
from report_generator import generate_html_report


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Test Jupyter notebooks for quality and correctness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all notebooks in default directories
  python run_notebook_tests.py

  # Test specific directories
  python run_notebook_tests.py --dirs ab_testing modern-bank-churn

  # Run tests in parallel (faster but uses more resources)
  python run_notebook_tests.py --parallel

  # Only validate notebooks (no execution)
  python run_notebook_tests.py --validate-only

  # Generate only HTML report from existing results
  python run_notebook_tests.py --report-only results.json

  # Use custom configuration
  python run_notebook_tests.py --config my_config.json

  # Set custom timeout
  python run_notebook_tests.py --timeout 300
        """
    )

    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["ab_testing", "modern-bank-churn"],
        help="Directories containing notebooks to test"
    )

    parser.add_argument(
        "--output",
        default="test_results",
        help="Output directory for test results (default: test_results)"
    )

    parser.add_argument(
        "--config",
        default="tests/notebook_tests/test_config.json",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run notebook tests in parallel"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per notebook in seconds (default: 600)"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation checks, skip execution"
    )

    parser.add_argument(
        "--report-only",
        help="Generate HTML report from existing JSON results file"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop testing after first failure"
    )

    parser.add_argument(
        "--html",
        action="store_true",
        default=True,
        help="Generate HTML report (default: True)"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Save results as JSON (default: True)"
    )

    args = parser.parse_args()

    # Handle report-only mode
    if args.report_only:
        print(f"Generating report from {args.report_only}...")
        try:
            with open(args.report_only, 'r') as f:
                results = json.load(f)

            output_path = Path(args.output) / "notebook_test_report.html"
            generate_html_report(results, str(output_path))
            print(f"[SUCCESS] HTML report generated: {output_path}")
            return 0
        except Exception as e:
            print(f"[ERROR] Error generating report: {e}")
            return 1

    # Load configuration
    config = NotebookTestConfig(args.config)
    if args.timeout:
        config.config['timeout'] = args.timeout

    # Print test configuration
    print("=" * 70)
    print("NOTEBOOK TEST RUNNER")
    print("=" * 70)
    print(f"Directories: {', '.join(args.dirs)}")
    print(f"Output: {args.output}")
    print(f"Parallel: {args.parallel}")
    print(f"Timeout: {config.config['timeout']}s")
    print(f"Mode: {'Validation only' if args.validate_only else 'Full test'}")
    print("=" * 70)
    print()

    # Create test runner
    runner = NotebookTestRunner(
        notebook_dirs=args.dirs,
        output_dir=args.output,
        timeout=config.config['timeout'],
        parallel=args.parallel
    )

    # Find notebooks
    notebooks = runner.find_notebooks()
    if not notebooks:
        print("[ERROR] No notebooks found in specified directories")
        return 1

    print(f"Found {len(notebooks)} notebooks to test")
    print()

    # Run tests or validation only
    if args.validate_only:
        results = run_validation_only(notebooks, args)
    else:
        results = runner.run_all_tests()

    # Print summary
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    summary = results['summary']
    print(f"Total notebooks: {summary['total_notebooks']}")
    print(f"Pass rate: {summary['pass_rate']:.1f}%")
    print()

    # Print status counts
    print("Status breakdown:")
    for status, count in summary['status_counts'].items():
        symbol = get_status_symbol(status)
        status_display = status.replace('_', ' ').title() if isinstance(status, str) else str(status)
        print(f"  {symbol} {status_display}: {count}")
    print()

    # Print failed notebooks if any
    if summary['failed_notebooks']:
        print("Failed notebooks:")
        for failure in summary['failed_notebooks']:
            print(f"  [FAIL] {failure['notebook']}")
            if args.verbose and failure.get('errors'):
                for error in failure['errors'][:2]:
                    print(f"    - {error.get('message', 'Unknown error')}")
        print()

    # Print warnings if any
    if summary.get('warnings') and args.verbose:
        print("Notebooks with warnings:")
        for warning in summary['warnings'][:5]:
            print(f"  [WARN] {warning['notebook']}")
        print()

    # Print average metrics
    if summary.get('average_metrics'):
        print("Average metrics:")
        for metric, value in summary['average_metrics'].items():
            if metric != 'error':
                print(f"  {metric.replace('_', ' ').title()}: {value:.2f}")
        print()

    print(f"Total time: {summary['total_time']:.2f} seconds")
    print("=" * 70)

    # Generate reports
    if args.html:
        html_path = Path(args.output) / f"notebook_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        generate_html_report(results, str(html_path))
        print(f"\n[SUCCESS] HTML report generated: {html_path}")

    # Return appropriate exit code
    if summary['pass_rate'] == 100:
        print("\n[SUCCESS] All tests passed!")
        return 0
    elif summary['pass_rate'] >= 70:
        print(f"\n[WARNING] Tests passed with warnings (pass rate: {summary['pass_rate']:.1f}%)")
        return 0
    else:
        print(f"\n[FAILURE] Tests failed (pass rate: {summary['pass_rate']:.1f}%)")
        return 1


def run_validation_only(notebooks, args):
    """Run only validation checks without execution."""
    results = []
    summary = {
        'total_notebooks': len(notebooks),
        'total_time': 0,
        'timestamp': datetime.now().isoformat(),
        'status_counts': {},
        'failed_notebooks': [],
        'warnings': []
    }

    print("Running validation checks...")
    for i, notebook_path in enumerate(notebooks, 1):
        print(f"[{i}/{len(notebooks)}] Validating {notebook_path.name}...")

        try:
            validator = NotebookValidator(str(notebook_path))
            validation_results = validator.validate_all()

            result = {
                'notebook': str(notebook_path),
                'notebook_name': notebook_path.name,
                'validation': validation_results,
                'status': 'passed' if validation_results['score'] >= 80 else 'passed_with_warnings',
                'execution': {'success': True, 'execution_time': 0}
            }

            if validation_results['issues']:
                result['status'] = 'failed'
                summary['failed_notebooks'].append({
                    'notebook': notebook_path.name,
                    'errors': validation_results['issues']
                })

            if validation_results['warnings']:
                summary['warnings'].append({
                    'notebook': notebook_path.name,
                    'warnings': validation_results['warnings']
                })

            results.append(result)

            # Update status counts
            status = result['status']
            summary['status_counts'][status] = summary['status_counts'].get(status, 0) + 1

        except Exception as e:
            print(f"  [ERROR] Error validating {notebook_path.name}: {e}")
            results.append({
                'notebook': str(notebook_path),
                'notebook_name': notebook_path.name,
                'status': 'error',
                'error': str(e)
            })
            summary['status_counts']['error'] = summary['status_counts'].get('error', 0) + 1

        if args.fail_fast and result.get('status') in ['failed', 'error']:
            print("Stopping due to --fail-fast flag")
            break

    # Calculate pass rate
    passed = summary['status_counts'].get('passed', 0) + \
             summary['status_counts'].get('passed_with_warnings', 0)
    summary['pass_rate'] = (passed / len(results) * 100) if results else 0

    return {
        'results': results,
        'summary': summary
    }


def get_status_symbol(status):
    """Get symbol for status."""
    symbols = {
        'passed': '[PASS]',
        'passed_with_warnings': '[WARN]',
        'failed': '[FAIL]',
        'error': '[ERR]',
        'not_run': '[-]'
    }
    return symbols.get(status, '[?]')


if __name__ == "__main__":
    sys.exit(main())