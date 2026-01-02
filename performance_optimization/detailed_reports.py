"""
Detailed performance reporting and analysis for optimization results.
"""

import json
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psutil
import seaborn as sns
from plotly.subplots import make_subplots


@dataclass
class DetailedMetrics:
    """Detailed performance metrics."""

    operation_name: str
    execution_time: float
    memory_peak_mb: float
    memory_allocated_mb: float
    cpu_percent: float
    cache_hits: int
    cache_misses: int
    data_processed: int
    throughput: float  # items/second
    timestamp: str
    optimization_applied: str
    speedup: Optional[float] = None
    memory_reduction: Optional[float] = None


class DetailedPerformanceReporter:
    """Generate detailed performance reports with advanced analytics."""

    def __init__(self, output_dir: str = "performance_reports"):
        """Initialize detailed reporter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics_history = []
        self.baselines = {}

    def track_operation(
        self, operation_name: str, func: callable, data: Any, optimization: str = "none"
    ) -> DetailedMetrics:
        """
        Track detailed metrics for an operation.

        Args:
            operation_name: Name of operation
            func: Function to execute
            data: Data to process
            optimization: Optimization technique applied

        Returns:
            DetailedMetrics object
        """
        # Start monitoring
        process = psutil.Process()
        tracemalloc.start()

        # Get initial state
        start_memory = process.memory_info().rss / 1024 / 1024
        cache_stats = getattr(
            func, "cache_info", lambda: type("obj", (), {"hits": 0, "misses": 0})
        )()

        # Execute operation
        start_time = time.perf_counter()
        result = func(data)
        execution_time = time.perf_counter() - start_time

        # Get final state
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        end_memory = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()

        # Calculate metrics
        data_size = len(data) if hasattr(data, "__len__") else 1
        throughput = data_size / execution_time if execution_time > 0 else 0

        # Cache statistics
        cache_stats_end = getattr(
            func, "cache_info", lambda: type("obj", (), {"hits": 0, "misses": 0})
        )()

        metrics = DetailedMetrics(
            operation_name=operation_name,
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            memory_allocated_mb=end_memory - start_memory,
            cpu_percent=cpu_percent,
            cache_hits=getattr(cache_stats_end, "hits", 0)
            - getattr(cache_stats, "hits", 0),
            cache_misses=getattr(cache_stats_end, "misses", 0)
            - getattr(cache_stats, "misses", 0),
            data_processed=data_size,
            throughput=throughput,
            timestamp=datetime.now().isoformat(),
            optimization_applied=optimization,
        )

        # Calculate speedup if baseline exists
        if operation_name in self.baselines:
            baseline = self.baselines[operation_name]
            metrics.speedup = baseline.execution_time / execution_time
            metrics.memory_reduction = (
                baseline.memory_peak_mb - metrics.memory_peak_mb
            ) / baseline.memory_peak_mb

        self.metrics_history.append(metrics)
        return metrics

    def set_baseline(self, operation_name: str, metrics: DetailedMetrics):
        """Set baseline metrics for comparison."""
        self.baselines[operation_name] = metrics

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive HTML performance report."""
        if not self.metrics_history:
            return "No metrics to report"

        # Convert metrics to DataFrame
        df = pd.DataFrame([asdict(m) for m in self.metrics_history])

        # Generate HTML report
        html_path = (
            self.output_dir
            / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Performance Optimization Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .improvement {{
            color: #27ae60;
        }}
        .regression {{
            color: #e74c3c;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #34495e;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Performance Optimization Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total Operations Tracked: {len(df)}</p>
    </div>

    <div class="summary-grid">
        <div class="metric-card">
            <div class="metric-value">{df['execution_time'].mean():.4f}s</div>
            <div class="metric-label">Average Execution Time</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{df['memory_peak_mb'].mean():.2f} MB</div>
            <div class="metric-label">Average Memory Peak</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{df['throughput'].mean():.0f}/s</div>
            <div class="metric-label">Average Throughput</div>
        </div>
        <div class="metric-card">
            <div class="metric-value improvement">{df['speedup'].mean():.2f}x</div>
            <div class="metric-label">Average Speedup</div>
        </div>
    </div>
"""

        # Add interactive charts
        html_content += self._generate_charts_html(df)

        # Add detailed table
        html_content += self._generate_table_html(df)

        html_content += """
</body>
</html>
"""

        with open(html_path, "w") as f:
            f.write(html_content)

        print(f"Comprehensive report saved to: {html_path}")
        return str(html_path)

    def _generate_charts_html(self, df: pd.DataFrame) -> str:
        """Generate HTML for interactive charts."""
        html = ""

        # Execution time comparison
        fig1 = go.Figure()
        for opt in df["optimization_applied"].unique():
            opt_df = df[df["optimization_applied"] == opt]
            fig1.add_trace(
                go.Bar(
                    x=opt_df["operation_name"],
                    y=opt_df["execution_time"],
                    name=opt,
                    text=[f"{t:.3f}s" for t in opt_df["execution_time"]],
                    textposition="auto",
                )
            )

        fig1.update_layout(
            title="Execution Time by Operation and Optimization",
            xaxis_title="Operation",
            yaxis_title="Time (seconds)",
            barmode="group",
            height=400,
        )

        html += f"""
    <div class="chart-container">
        <div id="execution-time-chart"></div>
        <script>
            Plotly.newPlot('execution-time-chart', {fig1.to_json()});
        </script>
    </div>
"""

        # Memory usage over time
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["memory_peak_mb"],
                mode="lines+markers",
                name="Memory Peak",
                line=dict(color="blue", width=2),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["memory_allocated_mb"],
                mode="lines+markers",
                name="Memory Allocated",
                line=dict(color="red", width=2),
            )
        )

        fig2.update_layout(
            title="Memory Usage Over Time",
            xaxis_title="Time",
            yaxis_title="Memory (MB)",
            height=400,
        )

        html += f"""
    <div class="chart-container">
        <div id="memory-chart"></div>
        <script>
            Plotly.newPlot('memory-chart', {fig2.to_json()});
        </script>
    </div>
"""

        # Speedup distribution
        speedup_data = df.dropna(subset=["speedup"])
        if not speedup_data.empty:
            fig3 = go.Figure(
                data=[
                    go.Box(
                        y=speedup_data["speedup"],
                        x=speedup_data["operation_name"],
                        name="Speedup Distribution",
                    )
                ]
            )

            fig3.update_layout(
                title="Speedup Distribution by Operation",
                yaxis_title="Speedup Factor",
                xaxis_title="Operation",
                height=400,
            )

            fig3.add_hline(
                y=1, line_dash="dash", line_color="red", annotation_text="Baseline"
            )

            html += f"""
    <div class="chart-container">
        <div id="speedup-chart"></div>
        <script>
            Plotly.newPlot('speedup-chart', {fig3.to_json()});
        </script>
    </div>
"""

        return html

    def _generate_table_html(self, df: pd.DataFrame) -> str:
        """Generate HTML table with detailed metrics."""
        html = """
    <div class="chart-container">
        <h2>Detailed Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Operation</th>
                    <th>Optimization</th>
                    <th>Time (s)</th>
                    <th>Memory (MB)</th>
                    <th>Throughput</th>
                    <th>Speedup</th>
                    <th>Cache Hit Rate</th>
                </tr>
            </thead>
            <tbody>
"""

        for _, row in df.iterrows():
            speedup_class = ""
            speedup_text = "-"
            if pd.notna(row["speedup"]):
                speedup_text = f"{row['speedup']:.2f}x"
                speedup_class = "improvement" if row["speedup"] > 1 else "regression"

            cache_total = row["cache_hits"] + row["cache_misses"]
            cache_rate = (
                f"{(row['cache_hits'] / cache_total * 100):.1f}%"
                if cache_total > 0
                else "-"
            )

            html += f"""
                <tr>
                    <td>{row['operation_name']}</td>
                    <td>{row['optimization_applied']}</td>
                    <td>{row['execution_time']:.4f}</td>
                    <td>{row['memory_peak_mb']:.2f}</td>
                    <td>{row['throughput']:.0f}/s</td>
                    <td class="{speedup_class}">{speedup_text}</td>
                    <td>{cache_rate}</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
    </div>
"""
        return html

    def generate_comparison_report(self, operation_name: str) -> pd.DataFrame:
        """Generate comparison report for specific operation."""
        operation_metrics = [
            m for m in self.metrics_history if m.operation_name == operation_name
        ]

        if not operation_metrics:
            return pd.DataFrame()

        df = pd.DataFrame([asdict(m) for m in operation_metrics])

        # Group by optimization
        comparison = (
            df.groupby("optimization_applied")
            .agg(
                {
                    "execution_time": ["mean", "std", "min", "max"],
                    "memory_peak_mb": ["mean", "std"],
                    "throughput": "mean",
                    "cache_hits": "sum",
                    "cache_misses": "sum",
                }
            )
            .round(4)
        )

        return comparison

    def export_metrics_json(self) -> str:
        """Export metrics to JSON for external analysis."""
        json_path = (
            self.output_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        metrics_dict = [asdict(m) for m in self.metrics_history]

        with open(json_path, "w") as f:
            json.dump(metrics_dict, f, indent=2, default=str)

        return str(json_path)


class OptimizationAnalyzer:
    """Analyze optimization effectiveness across different scenarios."""

    @staticmethod
    def analyze_optimization_impact(
        baseline: pd.DataFrame, optimized: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze impact of optimizations.

        Args:
            baseline: Baseline performance DataFrame
            optimized: Optimized performance DataFrame

        Returns:
            Analysis results
        """
        analysis = {
            "execution_time": {
                "baseline_mean": baseline["execution_time"].mean(),
                "optimized_mean": optimized["execution_time"].mean(),
                "improvement": 1
                - (
                    optimized["execution_time"].mean()
                    / baseline["execution_time"].mean()
                ),
                "speedup": baseline["execution_time"].mean()
                / optimized["execution_time"].mean(),
            },
            "memory": {
                "baseline_mean": baseline["memory_usage_mb"].mean(),
                "optimized_mean": optimized["memory_usage_mb"].mean(),
                "reduction": 1
                - (
                    optimized["memory_usage_mb"].mean()
                    / baseline["memory_usage_mb"].mean()
                ),
            },
            "statistical_significance": OptimizationAnalyzer._test_significance(
                baseline["execution_time"], optimized["execution_time"]
            ),
        }

        return analysis

    @staticmethod
    def _test_significance(
        baseline_times: pd.Series, optimized_times: pd.Series
    ) -> Dict[str, Any]:
        """Test statistical significance of performance improvement."""
        from scipy import stats

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(baseline_times, optimized_times)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            (baseline_times.std() ** 2 + optimized_times.std() ** 2) / 2
        )
        effect_size = (baseline_times.mean() - optimized_times.mean()) / pooled_std

        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "effect_size": effect_size,
            "effect_magnitude": (
                "large"
                if abs(effect_size) > 0.8
                else "medium" if abs(effect_size) > 0.5 else "small"
            ),
        }

    @staticmethod
    def identify_bottlenecks(metrics_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks from metrics."""
        bottlenecks = []

        # Find operations with high execution time
        time_threshold = metrics_df["execution_time"].quantile(0.9)
        slow_ops = metrics_df[metrics_df["execution_time"] > time_threshold]

        for _, op in slow_ops.iterrows():
            bottlenecks.append(
                {
                    "operation": op["operation_name"],
                    "type": "execution_time",
                    "value": op["execution_time"],
                    "severity": (
                        "high"
                        if op["execution_time"] > time_threshold * 1.5
                        else "medium"
                    ),
                }
            )

        # Find operations with high memory usage
        memory_threshold = metrics_df["memory_peak_mb"].quantile(0.9)
        memory_intensive = metrics_df[metrics_df["memory_peak_mb"] > memory_threshold]

        for _, op in memory_intensive.iterrows():
            bottlenecks.append(
                {
                    "operation": op["operation_name"],
                    "type": "memory",
                    "value": op["memory_peak_mb"],
                    "severity": (
                        "high"
                        if op["memory_peak_mb"] > memory_threshold * 1.5
                        else "medium"
                    ),
                }
            )

        return bottlenecks

    @staticmethod
    def recommend_optimizations(bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """Recommend optimizations based on identified bottlenecks."""
        recommendations = []

        for bottleneck in bottlenecks:
            if bottleneck["type"] == "execution_time":
                if bottleneck["severity"] == "high":
                    recommendations.append(
                        f"Consider GPU acceleration or distributed processing for '{bottleneck['operation']}'"
                    )
                else:
                    recommendations.append(
                        f"Apply vectorization or Numba JIT compilation to '{bottleneck['operation']}'"
                    )

            elif bottleneck["type"] == "memory":
                if bottleneck["severity"] == "high":
                    recommendations.append(
                        f"Implement chunked processing or data type optimization for '{bottleneck['operation']}'"
                    )
                else:
                    recommendations.append(
                        f"Consider caching or memory-mapped files for '{bottleneck['operation']}'"
                    )

        return recommendations


# Example usage
if __name__ == "__main__":
    print("Detailed Performance Reporting Module")
    print("=" * 60)

    # Initialize reporter
    reporter = DetailedPerformanceReporter()

    # Simulate tracking operations
    import numpy as np

    def baseline_operation(data):
        """Baseline operation without optimization."""
        result = []
        for item in data:
            result.append(item**2)
        return result

    def optimized_operation(data):
        """Optimized operation using vectorization."""
        return data**2

    # Track baseline
    data = np.random.randn(100000)
    baseline_metrics = reporter.track_operation(
        "square_operation", baseline_operation, data.tolist(), "baseline"
    )
    reporter.set_baseline("square_operation", baseline_metrics)

    # Track optimized
    optimized_metrics = reporter.track_operation(
        "square_operation", optimized_operation, data, "vectorized"
    )

    print(f"\nBaseline time: {baseline_metrics.execution_time:.4f}s")
    print(f"Optimized time: {optimized_metrics.execution_time:.4f}s")
    print(f"Speedup: {optimized_metrics.speedup:.2f}x")

    # Generate reports
    report_path = reporter.generate_comprehensive_report()
    print(f"\nReport generated: {report_path}")

    # Export metrics
    json_path = reporter.export_metrics_json()
    print(f"Metrics exported: {json_path}")
