"""
Performance benchmarking system for tracking optimization results.
"""

import functools
import json
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns


@dataclass
class BenchmarkResult:
    """Data class for benchmark results."""

    name: str
    execution_time: float
    memory_peak_mb: float
    memory_delta_mb: float
    cpu_percent: float
    iterations: int
    timestamp: str
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class PerformanceBenchmark:
    """Comprehensive performance benchmarking system."""

    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark system."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.baseline_results = {}

    def benchmark_function(
        self,
        func: Callable,
        *args,
        name: Optional[str] = None,
        iterations: int = 10,
        **kwargs,
    ) -> BenchmarkResult:
        """
        Benchmark a function with multiple metrics.

        Args:
            func: Function to benchmark
            name: Name for the benchmark
            iterations: Number of iterations
            *args, **kwargs: Arguments for the function

        Returns:
            BenchmarkResult object
        """
        if name is None:
            name = func.__name__

        print(f"Benchmarking: {name} ({iterations} iterations)")

        # Warm-up run
        func(*args, **kwargs)

        execution_times = []
        memory_peaks = []
        memory_deltas = []
        cpu_percents = []

        for i in range(iterations):
            # Memory tracking
            tracemalloc.start()
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB

            # CPU tracking
            process.cpu_percent()  # Initialize

            # Time tracking
            start_time = time.perf_counter()

            # Execute function
            result = func(*args, **kwargs)

            # End measurements
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            cpu_percent = process.cpu_percent()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Record metrics
            execution_times.append(end_time - start_time)
            memory_peaks.append(peak / 1024 / 1024)
            memory_deltas.append(end_memory - start_memory)
            cpu_percents.append(cpu_percent)

        # Calculate statistics
        benchmark_result = BenchmarkResult(
            name=name,
            execution_time=np.mean(execution_times),
            memory_peak_mb=np.mean(memory_peaks),
            memory_delta_mb=np.mean(memory_deltas),
            cpu_percent=np.mean(cpu_percents),
            iterations=iterations,
            timestamp=datetime.now().isoformat(),
            metadata={
                "time_std": np.std(execution_times),
                "time_min": np.min(execution_times),
                "time_max": np.max(execution_times),
                "memory_std": np.std(memory_peaks),
            },
        )

        self.results.append(benchmark_result)
        print(f"  Average time: {benchmark_result.execution_time:.4f}s")
        print(f"  Peak memory: {benchmark_result.memory_peak_mb:.2f} MB")

        return benchmark_result

    def set_baseline(
        self, func: Callable, *args, name: Optional[str] = None, **kwargs
    ) -> BenchmarkResult:
        """
        Set baseline performance for comparison.

        Args:
            func: Baseline function
            name: Name for the baseline
            *args, **kwargs: Function arguments

        Returns:
            Baseline benchmark result
        """
        if name is None:
            name = f"{func.__name__}_baseline"

        result = self.benchmark_function(func, *args, name=name, **kwargs)
        self.baseline_results[name] = result

        print(f"Baseline set: {name}")
        return result

    def compare_with_baseline(
        self, func: Callable, baseline_name: str, *args, **kwargs
    ) -> Dict[str, float]:
        """
        Compare function performance with baseline.

        Args:
            func: Function to compare
            baseline_name: Name of baseline
            *args, **kwargs: Function arguments

        Returns:
            Comparison metrics
        """
        if baseline_name not in self.baseline_results:
            raise ValueError(f"Baseline '{baseline_name}' not found")

        # Benchmark new function
        new_result = self.benchmark_function(func, *args, **kwargs)
        baseline = self.baseline_results[baseline_name]

        # Calculate improvements
        comparison = {
            "speedup": baseline.execution_time / new_result.execution_time,
            "memory_reduction": baseline.memory_peak_mb - new_result.memory_peak_mb,
            "memory_ratio": new_result.memory_peak_mb / baseline.memory_peak_mb,
            "cpu_reduction": baseline.cpu_percent - new_result.cpu_percent,
        }

        print(f"\nComparison with {baseline_name}:")
        print(f"  Speedup: {comparison['speedup']:.2f}x")
        print(f"  Memory reduction: {comparison['memory_reduction']:.2f} MB")

        return comparison

    def create_comparison_matrix(
        self, functions: Dict[str, Callable], test_data: Any
    ) -> pd.DataFrame:
        """
        Create comparison matrix for multiple implementations.

        Args:
            functions: Dictionary of {name: function}
            test_data: Test data for all functions

        Returns:
            Comparison DataFrame
        """
        results = []

        for name, func in functions.items():
            result = self.benchmark_function(func, test_data, name=name, iterations=5)
            results.append(
                {
                    "implementation": name,
                    "execution_time": result.execution_time,
                    "memory_peak_mb": result.memory_peak_mb,
                    "cpu_percent": result.cpu_percent,
                }
            )

        df = pd.DataFrame(results)

        # Add relative metrics
        df["time_ratio"] = df["execution_time"] / df["execution_time"].min()
        df["memory_ratio"] = df["memory_peak_mb"] / df["memory_peak_mb"].min()

        return df.sort_values("execution_time")

    def generate_report(self) -> str:
        """
        Generate comprehensive benchmark report.

        Returns:
            Path to generated report
        """
        if not self.results:
            print("No benchmark results to report")
            return None

        # Convert results to DataFrame
        df = pd.DataFrame([r.to_dict() for r in self.results])

        # Generate visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Execution time plot
        ax = axes[0, 0]
        df.plot(x="name", y="execution_time", kind="bar", ax=ax)
        ax.set_title("Execution Time Comparison")
        ax.set_ylabel("Time (seconds)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

        # Memory usage plot
        ax = axes[0, 1]
        df.plot(x="name", y="memory_peak_mb", kind="bar", ax=ax, color="orange")
        ax.set_title("Peak Memory Usage")
        ax.set_ylabel("Memory (MB)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

        # CPU usage plot
        ax = axes[1, 0]
        df.plot(x="name", y="cpu_percent", kind="bar", ax=ax, color="green")
        ax.set_title("Average CPU Usage")
        ax.set_ylabel("CPU %")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

        # Performance over time
        ax = axes[1, 1]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.plot(x="timestamp", y="execution_time", ax=ax, marker="o")
        ax.set_title("Performance Over Time")
        ax.set_ylabel("Time (seconds)")
        ax.set_xlabel("Timestamp")

        plt.tight_layout()

        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = self.output_dir / f"benchmark_plots_{timestamp}.png"
        plt.savefig(fig_path, dpi=100, bbox_inches="tight")
        plt.close()

        # Generate text report
        report_path = self.output_dir / f"benchmark_report_{timestamp}.txt"
        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("PERFORMANCE BENCHMARK REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total benchmarks: {len(self.results)}\n\n")

            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average execution time: {df['execution_time'].mean():.4f}s\n")
            f.write(f"Average memory usage: {df['memory_peak_mb'].mean():.2f} MB\n")
            f.write(f"Average CPU usage: {df['cpu_percent'].mean():.2f}%\n\n")

            # Detailed results
            f.write("DETAILED RESULTS\n")
            f.write("-" * 40 + "\n")
            for result in self.results:
                f.write(f"\n{result.name}:\n")
                f.write(f"  Execution time: {result.execution_time:.4f}s\n")
                f.write(f"  Memory peak: {result.memory_peak_mb:.2f} MB\n")
                f.write(f"  CPU usage: {result.cpu_percent:.2f}%\n")
                f.write(f"  Iterations: {result.iterations}\n")

            # Baselines
            if self.baseline_results:
                f.write("\nBASELINE COMPARISONS\n")
                f.write("-" * 40 + "\n")
                for name, baseline in self.baseline_results.items():
                    f.write(f"\n{name}:\n")
                    f.write(f"  Baseline time: {baseline.execution_time:.4f}s\n")
                    f.write(f"  Baseline memory: {baseline.memory_peak_mb:.2f} MB\n")

        # Save JSON data
        json_path = self.output_dir / f"benchmark_data_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2, default=str)

        print(f"Benchmark report saved to: {report_path}")
        print(f"Plots saved to: {fig_path}")
        print(f"Data saved to: {json_path}")

        return str(report_path)


class RegressionTester:
    """Performance regression testing."""

    def __init__(self, baseline_file: Optional[str] = None):
        """
        Initialize regression tester.

        Args:
            baseline_file: Path to baseline results JSON
        """
        self.baseline = {}
        if baseline_file and Path(baseline_file).exists():
            with open(baseline_file, "r") as f:
                baseline_data = json.load(f)
                self.baseline = {r["name"]: r for r in baseline_data}

    def test_regression(
        self, benchmark: BenchmarkResult, tolerance: float = 0.1
    ) -> bool:
        """
        Test for performance regression.

        Args:
            benchmark: Current benchmark result
            tolerance: Acceptable performance degradation (0.1 = 10%)

        Returns:
            True if no regression detected
        """
        if benchmark.name not in self.baseline:
            print(f"No baseline for {benchmark.name}")
            return True

        baseline = self.baseline[benchmark.name]

        # Check execution time regression
        time_increase = (
            benchmark.execution_time - baseline["execution_time"]
        ) / baseline["execution_time"]
        if time_increase > tolerance:
            print(
                f"REGRESSION: {benchmark.name} execution time increased by {time_increase:.1%}"
            )
            return False

        # Check memory regression
        memory_increase = (
            benchmark.memory_peak_mb - baseline["memory_peak_mb"]
        ) / baseline["memory_peak_mb"]
        if memory_increase > tolerance:
            print(
                f"REGRESSION: {benchmark.name} memory usage increased by {memory_increase:.1%}"
            )
            return False

        print(f"No regression detected for {benchmark.name}")
        return True


def create_optimization_report(
    original_func: Callable,
    optimized_func: Callable,
    test_data: Any,
    report_name: str = "optimization_report",
) -> None:
    """
    Create detailed optimization report comparing original and optimized versions.

    Args:
        original_func: Original implementation
        optimized_func: Optimized implementation
        test_data: Test data
        report_name: Name for the report
    """
    benchmark = PerformanceBenchmark()

    # Benchmark original
    original_result = benchmark.set_baseline(
        original_func, test_data, name=f"{report_name}_original"
    )

    # Benchmark optimized
    optimized_result = benchmark.benchmark_function(
        optimized_func, test_data, name=f"{report_name}_optimized"
    )

    # Calculate improvements
    speedup = original_result.execution_time / optimized_result.execution_time
    memory_reduction = original_result.memory_peak_mb - optimized_result.memory_peak_mb
    memory_ratio = optimized_result.memory_peak_mb / original_result.memory_peak_mb

    # Generate report
    report_path = (
        benchmark.output_dir
        / f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )

    with open(report_path, "w") as f:
        f.write(f"# Optimization Report: {report_name}\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Performance Summary\n\n")
        f.write("| Metric | Original | Optimized | Improvement |\n")
        f.write("|--------|----------|-----------|-------------|\n")
        f.write(
            f"| Execution Time | {original_result.execution_time:.4f}s | "
            f"{optimized_result.execution_time:.4f}s | **{speedup:.2f}x faster** |\n"
        )
        f.write(
            f"| Peak Memory | {original_result.memory_peak_mb:.2f} MB | "
            f"{optimized_result.memory_peak_mb:.2f} MB | "
            f"**{abs(memory_reduction):.2f} MB {'saved' if memory_reduction > 0 else 'increase'}** |\n"
        )
        f.write(
            f"| CPU Usage | {original_result.cpu_percent:.1f}% | "
            f"{optimized_result.cpu_percent:.1f}% | "
            f"{abs(original_result.cpu_percent - optimized_result.cpu_percent):.1f}% difference |\n"
        )

        f.write("\n## Detailed Metrics\n\n")
        f.write("### Original Implementation\n")
        f.write(f"- Average execution time: {original_result.execution_time:.4f}s\n")
        f.write(f"- Time std deviation: {original_result.metadata['time_std']:.4f}s\n")
        f.write(
            f"- Min/Max time: {original_result.metadata['time_min']:.4f}s / "
            f"{original_result.metadata['time_max']:.4f}s\n"
        )

        f.write("\n### Optimized Implementation\n")
        f.write(f"- Average execution time: {optimized_result.execution_time:.4f}s\n")
        f.write(f"- Time std deviation: {optimized_result.metadata['time_std']:.4f}s\n")
        f.write(
            f"- Min/Max time: {optimized_result.metadata['time_min']:.4f}s / "
            f"{optimized_result.metadata['time_max']:.4f}s\n"
        )

        f.write("\n## Conclusions\n\n")
        if speedup > 1.5:
            f.write(
                f"✅ **Significant performance improvement:** {speedup:.2f}x speedup achieved\n"
            )
        elif speedup > 1.1:
            f.write(
                f"✅ **Moderate performance improvement:** {speedup:.2f}x speedup achieved\n"
            )
        else:
            f.write(
                f"⚠️ **Minimal performance improvement:** Only {speedup:.2f}x speedup\n"
            )

        if memory_reduction > 0:
            f.write(f"✅ **Memory usage reduced:** {memory_reduction:.2f} MB saved\n")
        else:
            f.write(
                f"⚠️ **Memory usage increased:** {abs(memory_reduction):.2f} MB increase\n"
            )

    print(f"Optimization report saved to: {report_path}")
    benchmark.generate_report()
