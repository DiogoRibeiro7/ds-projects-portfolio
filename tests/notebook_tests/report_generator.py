"""
HTML report generator for notebook test results.
"""

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class HTMLReportGenerator:
    """Generates beautiful HTML reports for notebook test results."""

    def __init__(
        self, results: Dict[str, Any], output_path: str = "notebook_test_report.html"
    ):
        """
        Initialize the report generator.

        Args:
            results: Test results dictionary from NotebookTestRunner
            output_path: Path to save the HTML report
        """
        self.results = results
        self.output_path = Path(output_path)
        self.summary = results.get("summary", {})
        self.test_results = results.get("results", [])

    def generate(self):
        """Generate the HTML report."""
        html = self._generate_html()
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"HTML report generated: {self.output_path}")

    def _generate_html(self) -> str:
        """Generate the complete HTML report."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Notebook Test Report - {datetime.now().strftime("%Y-%m-%d %H:%M")}</title>
    {self._generate_styles()}
    {self._generate_scripts()}
</head>
<body>
    <div class="container">
        {self._generate_header()}
        {self._generate_summary_cards()}
        {self._generate_charts()}
        {self._generate_results_table()}
        {self._generate_detailed_results()}
        {self._generate_footer()}
    </div>
</body>
</html>
"""

    def _generate_styles(self) -> str:
        """Generate CSS styles."""
        return """
<style>
    :root {
        --primary: #2563eb;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --dark: #1f2937;
        --light: #f9fafb;
        --border: #e5e7eb;
    }

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 2rem;
    }

    .container {
        max-width: 1400px;
        margin: 0 auto;
        background: white;
        border-radius: 1rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        overflow: hidden;
    }

    .header {
        background: var(--dark);
        color: white;
        padding: 2rem;
        text-align: center;
    }

    .header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }

    .header .subtitle {
        opacity: 0.9;
        font-size: 1.1rem;
    }

    .summary-cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        padding: 2rem;
        background: var(--light);
    }

    .card {
        background: white;
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    .card-icon {
        width: 3rem;
        height: 3rem;
        border-radius: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
        font-size: 1.5rem;
    }

    .card-icon.success { background: rgba(16, 185, 129, 0.1); color: var(--success); }
    .card-icon.warning { background: rgba(245, 158, 11, 0.1); color: var(--warning); }
    .card-icon.danger { background: rgba(239, 68, 68, 0.1); color: var(--danger); }
    .card-icon.info { background: rgba(37, 99, 235, 0.1); color: var(--primary); }

    .card-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .card-label {
        color: #6b7280;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .charts-section {
        padding: 2rem;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
        gap: 2rem;
    }

    .chart-container {
        background: white;
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }

    .chart-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: var(--dark);
    }

    .results-section {
        padding: 2rem;
    }

    .results-table {
        width: 100%;
        background: white;
        border-radius: 0.75rem;
        overflow: hidden;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }

    table {
        width: 100%;
        border-collapse: collapse;
    }

    thead {
        background: var(--light);
    }

    th {
        padding: 1rem;
        text-align: left;
        font-weight: 600;
        color: var(--dark);
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    td {
        padding: 1rem;
        border-top: 1px solid var(--border);
    }

    tr:hover {
        background: rgba(37, 99, 235, 0.02);
    }

    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
    }

    .status-badge.passed {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success);
    }

    .status-badge.passed_with_warnings {
        background: rgba(245, 158, 11, 0.1);
        color: var(--warning);
    }

    .status-badge.failed, .status-badge.error {
        background: rgba(239, 68, 68, 0.1);
        color: var(--danger);
    }

    .progress-bar {
        height: 8px;
        background: var(--light);
        border-radius: 4px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }

    .details-section {
        padding: 2rem;
    }

    .detail-card {
        background: white;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border);
    }

    .detail-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border);
    }

    .detail-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--dark);
    }

    .collapsible {
        cursor: pointer;
        user-select: none;
    }

    .collapsible:after {
        content: '\\25BC';
        float: right;
        margin-left: 1rem;
        transition: transform 0.3s;
    }

    .collapsible.active:after {
        transform: rotate(-180deg);
    }

    .content {
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.3s ease-out;
        background: var(--light);
        border-radius: 0.5rem;
    }

    .content.active {
        max-height: 1000px;
        padding: 1rem;
        margin-top: 1rem;
    }

    .code-block {
        background: #1f2937;
        color: #f9fafb;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 0.875rem;
        overflow-x: auto;
        white-space: pre-wrap;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }

    .metric-item {
        background: var(--light);
        padding: 0.75rem;
        border-radius: 0.5rem;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .metric-value {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--dark);
    }

    .footer {
        background: var(--dark);
        color: white;
        padding: 1.5rem 2rem;
        text-align: center;
        font-size: 0.875rem;
        opacity: 0.9;
    }

    @media (max-width: 768px) {
        .summary-cards {
            grid-template-columns: 1fr;
        }

        .charts-section {
            grid-template-columns: 1fr;
        }

        .container {
            border-radius: 0;
        }

        body {
            padding: 0;
        }
    }
</style>
"""

    def _generate_scripts(self) -> str:
        """Generate JavaScript code."""
        return """
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
    // Toggle collapsible sections
    document.addEventListener('DOMContentLoaded', function() {
        const collapsibles = document.getElementsByClassName('collapsible');

        for (let i = 0; i < collapsibles.length; i++) {
            collapsibles[i].addEventListener('click', function() {
                this.classList.toggle('active');
                const content = this.nextElementSibling;
                content.classList.toggle('active');
            });
        }

        // Initialize charts
        initializeCharts();
    });

    function initializeCharts() {
        // Status Distribution Chart
        const statusCtx = document.getElementById('statusChart');
        if (statusCtx) {
            const statusChart = new Chart(statusCtx.getContext('2d'), {
                type: 'doughnut',
                data: statusChartData,
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'bottom',
                        }
                    }
                }
            });
        }

        // Execution Time Chart
        const timeCtx = document.getElementById('timeChart');
        if (timeCtx) {
            const timeChart = new Chart(timeCtx.getContext('2d'), {
                type: 'bar',
                data: timeChartData,
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Execution Time (seconds)'
                            }
                        }
                    }
                }
            });
        }
    }
</script>
"""

    def _generate_header(self) -> str:
        """Generate the header section."""
        return f"""
<div class="header">
    <h1>📊 Notebook Test Report</h1>
    <div class="subtitle">
        Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
    </div>
</div>
"""

    def _generate_summary_cards(self) -> str:
        """Generate summary cards."""
        total = self.summary.get("total_notebooks", 0)
        passed = self.summary.get("status_counts", {}).get(
            "passed", 0
        ) + self.summary.get("status_counts", {}).get("passed_with_warnings", 0)
        failed = self.summary.get("status_counts", {}).get(
            "failed", 0
        ) + self.summary.get("status_counts", {}).get("error", 0)
        pass_rate = self.summary.get("pass_rate", 0)
        total_time = self.summary.get("total_time", 0)

        return f"""
<div class="summary-cards">
    <div class="card">
        <div class="card-icon info">📝</div>
        <div class="card-value">{total}</div>
        <div class="card-label">Total Notebooks</div>
    </div>

    <div class="card">
        <div class="card-icon success">✅</div>
        <div class="card-value">{passed}</div>
        <div class="card-label">Passed</div>
    </div>

    <div class="card">
        <div class="card-icon danger">❌</div>
        <div class="card-value">{failed}</div>
        <div class="card-label">Failed</div>
    </div>

    <div class="card">
        <div class="card-icon info">📊</div>
        <div class="card-value">{pass_rate:.1f}%</div>
        <div class="card-label">Pass Rate</div>
    </div>

    <div class="card">
        <div class="card-icon warning">⏱️</div>
        <div class="card-value">{total_time:.1f}s</div>
        <div class="card-label">Total Time</div>
    </div>

    <div class="card">
        <div class="card-icon info">⚡</div>
        <div class="card-value">{total_time / max(1, total):.1f}s</div>
        <div class="card-label">Avg Time per Notebook</div>
    </div>
</div>
"""

    def _generate_charts(self) -> str:
        """Generate charts section."""
        # Prepare data for charts
        status_counts = self.summary.get("status_counts", {})
        status_data = {
            "labels": list(status_counts.keys()),
            "data": list(status_counts.values()),
            "colors": [],
        }

        # Map colors for statuses
        color_map = {
            "passed": "#10b981",
            "passed_with_warnings": "#f59e0b",
            "failed": "#ef4444",
            "error": "#ef4444",
            "not_run": "#6b7280",
        }

        for status in status_data["labels"]:
            status_data["colors"].append(color_map.get(status, "#6b7280"))

        # Get execution times for bar chart
        notebooks = []
        exec_times = []
        for result in self.test_results[:10]:  # Top 10 for readability
            notebooks.append(result.get("notebook_name", "Unknown"))
            exec_times.append(result.get("execution", {}).get("execution_time", 0))

        return f"""
<div class="charts-section">
    <div class="chart-container">
        <div class="chart-title">Test Status Distribution</div>
        <canvas id="statusChart"></canvas>
    </div>

    <div class="chart-container">
        <div class="chart-title">Execution Times (Top 10)</div>
        <canvas id="timeChart"></canvas>
    </div>
</div>

<script>
    const statusChartData = {{
        labels: {json.dumps(status_data["labels"])},
        datasets: [{{
            data: {json.dumps(status_data["data"])},
            backgroundColor: {json.dumps(status_data["colors"])},
            borderWidth: 0
        }}]
    }};

    const timeChartData = {{
        labels: {json.dumps([n[:20] + "..." if len(n) > 20 else n for n in notebooks])},
        datasets: [{{
            data: {json.dumps(exec_times)},
            backgroundColor: '#2563eb',
            borderRadius: 4
        }}]
    }};
</script>
"""

    def _generate_results_table(self) -> str:
        """Generate the results table."""
        rows = []
        for result in self.test_results:
            name = result.get("notebook_name", "Unknown")
            status = result.get("status", "not_run")
            exec_time = result.get("execution", {}).get("execution_time", 0)
            score = result.get("validation", {}).get("score", 0)
            issues = len(result.get("validation", {}).get("issues", []))
            warnings = len(result.get("validation", {}).get("warnings", []))

            # Create status badge
            status_class = status.replace("_", "_")
            status_display = status.replace("_", " ").title()

            # Create progress bar for score
            color = (
                "#10b981" if score >= 80 else "#f59e0b" if score >= 60 else "#ef4444"
            )
            progress = f"""
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {score}%; background: {color};"></div>
                </div>
                <small>{score:.0f}/100</small>
            """

            row = f"""
            <tr>
                <td>{name}</td>
                <td><span class="status-badge {status_class}">{status_display}</span></td>
                <td>{exec_time:.2f}s</td>
                <td>{progress}</td>
                <td>{issues}/{warnings}</td>
            </tr>
            """
            rows.append(row)

        return f"""
<div class="results-section">
    <h2 style="margin-bottom: 1rem;">Test Results</h2>
    <div class="results-table">
        <table>
            <thead>
                <tr>
                    <th>Notebook</th>
                    <th>Status</th>
                    <th>Execution Time</th>
                    <th>Validation Score</th>
                    <th>Issues/Warnings</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    </div>
</div>
"""

    def _generate_detailed_results(self) -> str:
        """Generate detailed results section."""
        details = []

        for result in self.test_results:
            if result.get("status") in ["failed", "error", "passed_with_warnings"]:
                name = result.get("notebook_name", "Unknown")
                status = result.get("status", "not_run")

                # Get errors
                errors = result.get("execution", {}).get("errors", [])
                warnings = result.get("validation", {}).get("warnings", [])
                issues = result.get("validation", {}).get("issues", [])

                error_html = ""
                if errors:
                    error_list = []
                    for error in errors:
                        error_list.append(
                            f"<li>{error.get('message', 'Unknown error')}</li>"
                        )
                    error_html = f"<h4>Errors:</h4><ul>{''.join(error_list)}</ul>"

                warning_html = ""
                if warnings:
                    warning_list = []
                    for warning in warnings:
                        warning_list.append(f"<li>{warning.get('message', '')}</li>")
                    warning_html = f"<h4>Warnings:</h4><ul>{''.join(warning_list)}</ul>"

                issue_html = ""
                if issues:
                    issue_list = []
                    for issue in issues:
                        issue_list.append(f"<li>{issue.get('message', '')}</li>")
                    issue_html = f"<h4>Issues:</h4><ul>{''.join(issue_list)}</ul>"

                detail = f"""
                <div class="detail-card">
                    <div class="detail-header">
                        <div class="detail-title">{name}</div>
                        <span class="status-badge {status}">{status.replace("_", " ").title()}</span>
                    </div>
                    <div class="collapsible">Click to view details</div>
                    <div class="content">
                        {error_html}
                        {warning_html}
                        {issue_html}
                    </div>
                </div>
                """
                details.append(detail)

        if details:
            return f"""
            <div class="details-section">
                <h2 style="margin-bottom: 1rem;">Detailed Issues</h2>
                {"".join(details)}
            </div>
            """
        return ""

    def _generate_footer(self) -> str:
        """Generate footer."""
        return """
<div class="footer">
    <p>Generated by Notebook Test Framework | © 2025 Data Science Portfolio</p>
</div>
"""


def generate_html_report(
    results: Dict[str, Any], output_path: str = "notebook_test_report.html"
):
    """
    Convenience function to generate HTML report.

    Args:
        results: Test results from NotebookTestRunner
        output_path: Path to save the HTML report
    """
    generator = HTMLReportGenerator(results, output_path)
    generator.generate()
    return output_path
