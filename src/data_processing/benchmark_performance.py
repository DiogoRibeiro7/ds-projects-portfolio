"""Performance benchmarking utilities for data processing operations.

This module provides comprehensive benchmarking for comparing optimized
vs non-optimized data processing operations.
"""

import json
import logging
import time
import warnings
from collections.abc import Callable
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import memory_profiler  # type: ignore[import]

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    warnings.warn("memory_profiler not available - memory benchmarking disabled")

from .cleaning import OptimizedDataProcessor, _detect_outliers, apply_cuped

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BenchmarkResults = dict[str, Any]


class PerformanceBenchmark:
    """Comprehensive performance benchmarking for data processing operations."""

    def __init__(self, n_runs: int = 3, save_results: bool = True):
        """Initialize performance benchmark.

        Parameters
        ----------
        n_runs : int, default=3
            Number of benchmark runs for each test.
        save_results : bool, default=True
            Whether to save benchmark results to file.
        """
        self.n_runs = n_runs
        self.save_results = save_results
        self.results: BenchmarkResults = {}
        self.processor = OptimizedDataProcessor()

    def generate_test_data(
        self, n_rows: int, n_cols: int = 10, n_groups: int = 2
    ) -> pd.DataFrame:
        """Generate synthetic test data for benchmarking.

        Parameters
        ----------
        n_rows : int
            Number of rows in the dataset.
        n_cols : int, default=10
            Number of numeric columns.
        n_groups : int, default=2
            Number of experimental groups.

        Returns:
        -------
        df : pd.DataFrame
            Synthetic test dataset.
        """
        np.random.seed(42)

        data = {
            "user_id": range(n_rows),
            "group": np.random.choice([f"group_{i}" for i in range(n_groups)], n_rows),
            "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="h"),
        }

        # Add numeric columns
        for i in range(n_cols):
            # Mix of different distributions
            if i % 3 == 0:
                # Normal distribution
                data[f"metric_{i}"] = np.random.normal(100, 20, n_rows)
            elif i % 3 == 1:
                # Binary metric
                data[f"metric_{i}"] = np.random.binomial(1, 0.3, n_rows)
            else:
                # Exponential distribution
                data[f"metric_{i}"] = np.random.exponential(50, n_rows)

        # Add some outliers
        for i in range(0, n_cols, 2):
            outlier_idx = np.random.choice(n_rows, size=int(n_rows * 0.01))
            data[f"metric_{i}"][outlier_idx] *= 10

        # Add missing values
        for i in range(1, n_cols, 3):
            missing_idx = np.random.choice(n_rows, size=int(n_rows * 0.05))
            data[f"metric_{i}"][missing_idx] = np.nan

        return pd.DataFrame(data)

    def measure_time(self, func: Callable, *args, **kwargs) -> float:
        """Measure execution time of a function.

        Parameters
        ----------
        func : callable
            Function to benchmark.
        *args
            Positional arguments for the function.
        **kwargs
            Keyword arguments for the function.

        Returns:
        -------
        elapsed_time : float
            Average execution time in seconds.
        """
        times = []

        for _ in range(self.n_runs):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)

        return float(np.mean(times))

    def measure_memory(self, func: Callable, *args, **kwargs) -> dict[str, float]:
        """Measure memory usage of a function.

        Parameters
        ----------
        func : callable
            Function to benchmark.
        *args
            Positional arguments for the function.
        **kwargs
            Keyword arguments for the function.

        Returns:
        -------
        memory_stats : dict
            Memory usage statistics.
        """
        if not MEMORY_PROFILER_AVAILABLE:
            return {"peak_memory_mb": 0, "memory_increment_mb": 0}

        from memory_profiler import memory_usage  # type: ignore[import]

        # Baseline memory
        baseline = memory_usage()[0]

        # Memory during execution
        mem_usage = memory_usage((func, args, kwargs))

        return {
            "peak_memory_mb": max(mem_usage),
            "memory_increment_mb": max(mem_usage) - baseline,
        }

    def benchmark_outlier_detection(self, sizes: list[int] = None) -> dict[str, Any]:
        """Benchmark outlier detection methods.

        Parameters
        ----------
        sizes : list of int, optional
            Dataset sizes to test.

        Returns:
        -------
        results : dict
            Benchmark results.
        """
        if sizes is None:
            sizes = [1000, 10000, 50000, 100000]

        results: BenchmarkResults = {"sizes": sizes, "standard": [], "optimized": []}

        for size in sizes:
            logger.info(f"Benchmarking outlier detection for {size} rows...")

            # Generate test data
            df = self.generate_test_data(size)
            metric_cols = [col for col in df.columns if col.startswith("metric_")]

            # Standard method
            def standard_outlier():
                df_clean = df.copy()
                for col in metric_cols:
                    outliers = _detect_outliers(df_clean[col], method="zscore")
                    df_clean = df_clean[~outliers]
                return df_clean

            # Optimized method
            def optimized_outlier():
                return self.processor.parallel_outlier_detection(
                    df, metric_cols, method="zscore"
                )

            # Measure performance
            standard_time = self.measure_time(standard_outlier)
            optimized_time = self.measure_time(optimized_outlier)

            results["standard"].append(standard_time)
            results["optimized"].append(optimized_time)

            logger.info(
                f"  Standard: {standard_time:.3f}s, "
                f"Optimized: {optimized_time:.3f}s "
                f"(Speedup: {standard_time / optimized_time:.2f}x)"
            )

        return results

    def benchmark_cuped(self, sizes: list[int] = None) -> dict[str, Any]:
        """Benchmark CUPED variance reduction.

        Parameters
        ----------
        sizes : list of int, optional
            Dataset sizes to test.

        Returns:
        -------
        results : dict
            Benchmark results.
        """
        if sizes is None:
            sizes = [1000, 10000, 50000, 100000]

        results: BenchmarkResults = {"sizes": sizes, "standard": [], "optimized": []}

        for size in sizes:
            logger.info(f"Benchmarking CUPED for {size} rows...")

            # Generate test data
            df = self.generate_test_data(size)

            # Standard CUPED
            standard_time = self.measure_time(
                apply_cuped, df, "metric_0", "metric_1", "group"
            )

            # Optimized parallel CUPED
            optimized_time = self.measure_time(
                self.processor.parallel_apply_cuped, df, "metric_0", "metric_1", "group"
            )

            results["standard"].append(standard_time)
            results["optimized"].append(optimized_time)

            logger.info(
                f"  Standard: {standard_time:.3f}s, "
                f"Optimized: {optimized_time:.3f}s "
                f"(Speedup: {standard_time / optimized_time:.2f}x)"
            )

        return results

    def benchmark_groupby(self, sizes: list[int] = None) -> dict[str, Any]:
        """Benchmark groupby operations.

        Parameters
        ----------
        sizes : list of int, optional
            Dataset sizes to test.

        Returns:
        -------
        results : dict
            Benchmark results.
        """
        if sizes is None:
            sizes = [10000, 100000, 500000, 1000000]

        results: BenchmarkResults = {"sizes": sizes, "standard": [], "optimized": []}

        for size in sizes:
            logger.info(f"Benchmarking groupby for {size} rows...")

            # Generate test data
            df = self.generate_test_data(size, n_groups=100)

            agg_funcs = {
                "metric_0": ["mean", "std"],
                "metric_1": "sum",
                "metric_2": ["min", "max"],
            }

            # Standard groupby
            def standard_groupby():
                return df.groupby("group").agg(agg_funcs)

            # Optimized groupby
            def optimized_groupby():
                return self.processor.memory_efficient_groupby(df, "group", agg_funcs)

            standard_time = self.measure_time(standard_groupby)
            optimized_time = self.measure_time(optimized_groupby)

            results["standard"].append(standard_time)
            results["optimized"].append(optimized_time)

            logger.info(
                f"  Standard: {standard_time:.3f}s, "
                f"Optimized: {optimized_time:.3f}s "
                f"(Speedup: {standard_time / optimized_time:.2f}x)"
            )

        return results

    def benchmark_memory_optimization(self, sizes: list[int] = None) -> dict[str, Any]:
        """Benchmark memory optimization.

        Parameters
        ----------
        sizes : list of int, optional
            Dataset sizes to test.

        Returns:
        -------
        results : dict
            Benchmark results.
        """
        if sizes is None:
            sizes = [10000, 50000, 100000, 500000]

        results: BenchmarkResults = {
            "sizes": sizes,
            "before_mb": [],
            "after_mb": [],
            "reduction_pct": [],
        }

        for size in sizes:
            logger.info(f"Benchmarking memory optimization for {size} rows...")

            # Generate test data
            df = self.generate_test_data(size)

            # Measure memory before
            mem_before = df.memory_usage(deep=True).sum() / 1024 / 1024

            # Optimize
            df_optimized = self.processor.optimize_dataframe_dtypes(df)

            # Measure memory after
            mem_after = df_optimized.memory_usage(deep=True).sum() / 1024 / 1024

            reduction = (mem_before - mem_after) / mem_before * 100

            results["before_mb"].append(mem_before)
            results["after_mb"].append(mem_after)
            results["reduction_pct"].append(reduction)

            logger.info(
                f"  Before: {mem_before:.2f}MB, "
                f"After: {mem_after:.2f}MB "
                f"(Reduction: {reduction:.1f}%)"
            )

        return results

    def benchmark_feature_engineering(self, sizes: list[int] = None) -> dict[str, Any]:
        """Benchmark feature engineering operations.

        Parameters
        ----------
        sizes : list of int, optional
            Dataset sizes to test.

        Returns:
        -------
        results : dict
            Benchmark results.
        """
        if sizes is None:
            sizes = [1000, 10000, 50000, 100000]

        results: BenchmarkResults = {"sizes": sizes, "standard": [], "vectorized": []}

        for size in sizes:
            logger.info(f"Benchmarking feature engineering for {size} rows...")

            # Generate test data
            df = self.generate_test_data(size)

            # Feature engineering config
            config = {
                "ratio_feature": {
                    "operation": "ratio",
                    "numerator": "metric_0",
                    "denominator": "metric_1",
                },
                "interaction_feature": {
                    "operation": "interaction",
                    "columns": ["metric_0", "metric_2"],
                },
                "polynomial_feature": {
                    "operation": "polynomial",
                    "column": "metric_3",
                    "degree": 2,
                },
                "binned_feature": {
                    "operation": "binning",
                    "column": "metric_4",
                    "bins": 10,
                },
            }

            # Standard implementation using apply
            def standard_features():
                df_feat = df.copy()
                df_feat["ratio_feature"] = df_feat.apply(
                    lambda x: (
                        x["metric_0"] / x["metric_1"] if x["metric_1"] != 0 else 0
                    ),
                    axis=1,
                )
                df_feat["interaction_feature"] = df_feat.apply(
                    lambda x: x["metric_0"] * x["metric_2"], axis=1
                )
                df_feat["polynomial_feature"] = df_feat["metric_3"].apply(
                    lambda x: x**2
                )
                df_feat["binned_feature"] = pd.cut(
                    df_feat["metric_4"], bins=10, labels=False
                )
                return df_feat

            # Vectorized implementation
            def vectorized_features():
                return self.processor.vectorized_feature_engineering(df, config)

            standard_time = self.measure_time(standard_features)
            vectorized_time = self.measure_time(vectorized_features)

            results["standard"].append(standard_time)
            results["vectorized"].append(vectorized_time)

            logger.info(
                f"  Standard: {standard_time:.3f}s, "
                f"Vectorized: {vectorized_time:.3f}s "
                f"(Speedup: {standard_time / vectorized_time:.2f}x)"
            )

        return results

    def run_comprehensive_benchmark(self) -> dict[str, Any]:
        """Run comprehensive benchmark suite.

        Returns:
        -------
        all_results : dict
            Complete benchmark results.
        """
        logger.info("=" * 60)
        logger.info("Starting comprehensive performance benchmarks")
        logger.info("=" * 60)

        all_results = {"timestamp": datetime.now().isoformat(), "benchmarks": {}}

        # Run all benchmarks
        benchmark_functions = [
            ("outlier_detection", self.benchmark_outlier_detection),
            ("cuped", self.benchmark_cuped),
            ("groupby", self.benchmark_groupby),
            ("memory_optimization", self.benchmark_memory_optimization),
            ("feature_engineering", self.benchmark_feature_engineering),
        ]

        for name, func in benchmark_functions:
            logger.info(f"\nRunning {name} benchmark...")
            all_results["benchmarks"][name] = func()

        # Calculate summary statistics
        all_results["summary"] = self._calculate_summary(all_results["benchmarks"])

        # Save results
        if self.save_results:
            self._save_results(all_results)

        # Generate visualization
        self._plot_results(all_results["benchmarks"])

        logger.info("\n" + "=" * 60)
        logger.info("Benchmark complete!")
        logger.info("=" * 60)

        return all_results

    def _calculate_summary(self, benchmarks: dict[str, Any]) -> dict[str, Any]:
        """Calculate summary statistics from benchmark results.

        Parameters
        ----------
        benchmarks : dict
            Benchmark results.

        Returns:
        -------
        summary : dict
            Summary statistics.
        """
        summary = {"average_speedup": {}, "max_speedup": {}, "total_time_saved": {}}

        for name, results in benchmarks.items():
            if "standard" in results and "optimized" in results:
                standard = np.array(results["standard"])
                optimized = np.array(results["optimized"])

                speedups = standard / optimized
                summary["average_speedup"][name] = np.mean(speedups)
                summary["max_speedup"][name] = np.max(speedups)
                summary["total_time_saved"][name] = np.sum(standard - optimized)

        # Overall statistics
        all_speedups = list(summary["average_speedup"].values())
        summary["overall_average_speedup"] = np.mean(all_speedups)

        return summary

    def _save_results(self, results: dict[str, Any]) -> None:
        """Save benchmark results to file.

        Parameters
        ----------
        results : dict
            Benchmark results to save.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {filename}")

    def _plot_results(self, benchmarks: dict[str, Any]) -> None:
        """Generate visualization of benchmark results.

        Parameters
        ----------
        benchmarks : dict
            Benchmark results to plot.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        plot_idx = 0
        for name, results in benchmarks.items():
            if plot_idx >= len(axes):
                break

            ax = axes[plot_idx]

            if "standard" in results and "optimized" in results:
                sizes = results["sizes"]
                standard = results["standard"]
                optimized = results["optimized"]

                ax.plot(sizes, standard, "o-", label="Standard", linewidth=2)
                ax.plot(sizes, optimized, "s-", label="Optimized", linewidth=2)
                ax.set_xlabel("Dataset Size")
                ax.set_ylabel("Time (seconds)")
                ax.set_title(name.replace("_", " ").title())
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xscale("log")
                ax.set_yscale("log")

            elif "before_mb" in results and "after_mb" in results:
                sizes = results["sizes"]
                before = results["before_mb"]
                after = results["after_mb"]

                x = np.arange(len(sizes))
                width = 0.35

                ax.bar(x - width / 2, before, width, label="Before", alpha=0.8)
                ax.bar(x + width / 2, after, width, label="After", alpha=0.8)
                ax.set_xlabel("Dataset Size")
                ax.set_ylabel("Memory (MB)")
                ax.set_title("Memory Optimization")
                ax.set_xticks(x)
                ax.set_xticklabels([str(s) for s in sizes])
                ax.legend()
                ax.grid(True, alpha=0.3)

            plot_idx += 1

        # Remove unused subplots
        for idx in range(plot_idx, len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle("Performance Benchmark Results", fontsize=16, y=1.02)
        plt.tight_layout()

        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_plot_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.show()

        logger.info(f"Plot saved to {filename}")


def run_benchmark(dataset_sizes: list[int] = None) -> dict[str, Any]:
    """Run performance benchmarks.

    Parameters
    ----------
    dataset_sizes : list of int, optional
        Dataset sizes to benchmark.

    Returns:
    -------
    results : dict
        Benchmark results.
    """
    benchmark = PerformanceBenchmark()
    return benchmark.run_comprehensive_benchmark()


if __name__ == "__main__":
    # Run benchmarks
    results = run_benchmark()

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    summary = results["summary"]
    print(f"\nOverall Average Speedup: {summary['overall_average_speedup']:.2f}x")

    print("\nSpeedup by Operation:")
    for op, speedup in summary["average_speedup"].items():
        print(f"  {op}: {speedup:.2f}x")

    print("\nTime Saved by Operation:")
    for op, time_saved in summary["total_time_saved"].items():
        print(f"  {op}: {time_saved:.3f} seconds")
