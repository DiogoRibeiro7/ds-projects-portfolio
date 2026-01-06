"""
Performance profiling utilities for identifying bottlenecks and optimization opportunities.
"""

import cProfile
import functools
import io
import json
import pstats
import time
import tracemalloc
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
from memory_profiler import profile as memory_profile


class PerformanceProfiler:
    """Comprehensive performance profiling tool."""

    def __init__(self, output_dir: str = "performance_reports"):
        """Initialize the profiler."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    @contextmanager
    def profile_block(self, name: str, verbose: bool = True):
        """
        Context manager for profiling code blocks.

        Usage:
            with profiler.profile_block("data_processing"):
                # Your code here
                df = pd.read_csv("large_file.csv")
                df = df.groupby("category").sum()
        """
        # Start tracking
        tracemalloc.start()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.perf_counter()

        try:
            yield self
        finally:
            # End tracking
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Calculate metrics
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            peak_memory = peak / 1024 / 1024  # MB

            result = {
                "name": name,
                "execution_time": execution_time,
                "memory_delta_mb": memory_delta,
                "peak_memory_mb": peak_memory,
                "start_memory_mb": start_memory,
                "end_memory_mb": end_memory,
                "timestamp": datetime.now().isoformat(),
            }

            self.results.append(result)

            if verbose:
                print(f"\n[PROFILE] {name}")
                print(f"  Execution time: {execution_time:.3f} seconds")
                print(f"  Memory delta: {memory_delta:.2f} MB")
                print(f"  Peak memory: {peak_memory:.2f} MB")

    def time_function(self, func: Callable) -> Callable:
        """Decorator to time function execution."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()

            self.results.append(
                {
                    "name": func.__name__,
                    "execution_time": end - start,
                    "type": "function",
                    "timestamp": datetime.now().isoformat(),
                }
            )

            return result

        return wrapper

    def profile_dataframe_operation(
        self, df: pd.DataFrame, operation: Callable, name: str = "dataframe_operation"
    ) -> Tuple[Any, Dict]:
        """
        Profile a pandas DataFrame operation.

        Args:
            df: Input DataFrame
            operation: Function to apply to DataFrame
            name: Name for the operation

        Returns:
            Tuple of (result, metrics)
        """
        with self.profile_block(name) as prof:
            # Get initial DataFrame info
            initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            initial_shape = df.shape

            # Execute operation
            result = operation(df)

            # Get final info if result is a DataFrame
            if isinstance(result, pd.DataFrame):
                final_memory = result.memory_usage(deep=True).sum() / 1024 / 1024
                final_shape = result.shape
            else:
                final_memory = 0
                final_shape = None

        # Get the last recorded metrics
        metrics = self.results[-1].copy()
        metrics.update(
            {
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "initial_shape": initial_shape,
                "final_shape": final_shape,
                "memory_reduction": (
                    (initial_memory - final_memory) if final_memory > 0 else 0
                ),
            }
        )

        return result, metrics

    def compare_implementations(
        self, implementations: Dict[str, Callable], test_data: Any = None
    ) -> pd.DataFrame:
        """
        Compare multiple implementations of the same functionality.

        Args:
            implementations: Dictionary of {name: function}
            test_data: Data to pass to each function

        Returns:
            DataFrame comparing performance metrics
        """
        comparison_results = []

        for name, func in implementations.items():
            with self.profile_block(f"compare_{name}", verbose=False):
                if test_data is not None:
                    result = func(test_data)
                else:
                    result = func()

            metrics = self.results[-1]
            comparison_results.append(
                {
                    "implementation": name,
                    "execution_time": metrics["execution_time"],
                    "memory_delta_mb": metrics["memory_delta_mb"],
                    "peak_memory_mb": metrics["peak_memory_mb"],
                }
            )

        comparison_df = pd.DataFrame(comparison_results)
        comparison_df["time_ratio"] = (
            comparison_df["execution_time"] / comparison_df["execution_time"].min()
        )
        comparison_df["memory_ratio"] = (
            comparison_df["peak_memory_mb"] / comparison_df["peak_memory_mb"].min()
        )

        return comparison_df.sort_values("execution_time")

    def profile_function_detailed(self, func: Callable, *args, **kwargs) -> Dict:
        """
        Detailed profiling of a function using cProfile.

        Returns:
            Dictionary with detailed profiling information
        """
        profiler = cProfile.Profile()
        profiler.enable()

        # Execute function
        result = func(*args, **kwargs)

        profiler.disable()

        # Get stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(20)  # Top 20 functions

        # Parse stats
        stats_str = s.getvalue()

        return {
            "result": result,
            "stats": stats_str,
            "top_functions": self._parse_profile_stats(ps),
        }

    def _parse_profile_stats(self, stats: pstats.Stats) -> List[Dict]:
        """Parse profile stats into structured format."""
        top_functions = []

        for func, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:10]:
            top_functions.append(
                {
                    "function": f"{func[0]}:{func[1]}:{func[2]}",
                    "calls": nc,
                    "total_time": tt,
                    "cumulative_time": ct,
                    "time_per_call": tt / nc if nc > 0 else 0,
                }
            )

        return top_functions

    def save_report(self, filename: Optional[str] = None) -> str:
        """Save profiling results to a report."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"profile_report_{timestamp}.json"

        filepath = self.output_dir / filename

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_operations": len(self.results),
            "results": self.results,
            "summary": self._generate_summary(),
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Also create a markdown summary
        md_filepath = filepath.with_suffix(".md")
        self._save_markdown_report(md_filepath, report)

        print(f"Report saved to: {filepath}")
        return str(filepath)

    def _generate_summary(self) -> Dict:
        """Generate summary statistics from results."""
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)

        return {
            "total_execution_time": (
                df["execution_time"].sum() if "execution_time" in df else 0
            ),
            "average_execution_time": (
                df["execution_time"].mean() if "execution_time" in df else 0
            ),
            "total_memory_delta": (
                df["memory_delta_mb"].sum() if "memory_delta_mb" in df else 0
            ),
            "peak_memory_usage": (
                df["peak_memory_mb"].max() if "peak_memory_mb" in df else 0
            ),
            "operations_by_time": (
                df.nlargest(5, "execution_time")[["name", "execution_time"]].to_dict(
                    "records"
                )
                if "execution_time" in df
                else []
            ),
            "operations_by_memory": (
                df.nlargest(5, "memory_delta_mb")[["name", "memory_delta_mb"]].to_dict(
                    "records"
                )
                if "memory_delta_mb" in df
                else []
            ),
        }

    def _save_markdown_report(self, filepath: Path, report: Dict):
        """Save a markdown version of the report."""
        with open(filepath, "w") as f:
            f.write("# Performance Profile Report\n\n")
            f.write(f"**Generated:** {report['timestamp']}\n")
            f.write(f"**Total Operations:** {report['total_operations']}\n\n")

            summary = report["summary"]
            if summary:
                f.write("## Summary\n\n")
                f.write(
                    f"- **Total Execution Time:** {summary['total_execution_time']:.3f} seconds\n"
                )
                f.write(
                    f"- **Average Execution Time:** {summary['average_execution_time']:.3f} seconds\n"
                )
                f.write(
                    f"- **Total Memory Delta:** {summary['total_memory_delta']:.2f} MB\n"
                )
                f.write(
                    f"- **Peak Memory Usage:** {summary['peak_memory_usage']:.2f} MB\n\n"
                )

                if summary["operations_by_time"]:
                    f.write("### Top Operations by Time\n\n")
                    f.write("| Operation | Time (s) |\n")
                    f.write("|-----------|----------|\n")
                    for op in summary["operations_by_time"]:
                        f.write(f"| {op['name']} | {op['execution_time']:.3f} |\n")
                    f.write("\n")

                if summary["operations_by_memory"]:
                    f.write("### Top Operations by Memory\n\n")
                    f.write("| Operation | Memory Delta (MB) |\n")
                    f.write("|-----------|-------------------|\n")
                    for op in summary["operations_by_memory"]:
                        f.write(f"| {op['name']} | {op['memory_delta_mb']:.2f} |\n")
                    f.write("\n")

            f.write("## Detailed Results\n\n")
            for i, result in enumerate(report["results"], 1):
                f.write(f"### {i}. {result['name']}\n")
                f.write(f"- Execution Time: {result.get('execution_time', 0):.3f}s\n")
                if "memory_delta_mb" in result:
                    f.write(f"- Memory Delta: {result['memory_delta_mb']:.2f} MB\n")
                if "peak_memory_mb" in result:
                    f.write(f"- Peak Memory: {result['peak_memory_mb']:.2f} MB\n")
                f.write("\n")


class MemoryOptimizer:
    """Utilities for optimizing memory usage in data processing."""

    @staticmethod
    def optimize_dataframe_dtypes(
        df: pd.DataFrame, verbose: bool = True
    ) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by converting to more efficient dtypes.

        Args:
            df: Input DataFrame
            verbose: Print optimization details

        Returns:
            Optimized DataFrame
        """
        initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        df_optimized = df.copy()

        # Optimize numeric columns
        for col in df_optimized.select_dtypes(include=["int64"]).columns:
            col_min = df_optimized[col].min()
            col_max = df_optimized[col].max()

            if col_min >= -128 and col_max <= 127:
                df_optimized[col] = df_optimized[col].astype("int8")
            elif col_min >= -32768 and col_max <= 32767:
                df_optimized[col] = df_optimized[col].astype("int16")
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df_optimized[col] = df_optimized[col].astype("int32")

        for col in df_optimized.select_dtypes(include=["float64"]).columns:
            df_optimized[col] = df_optimized[col].astype("float32")

        # Convert object columns to category if appropriate
        for col in df_optimized.select_dtypes(include=["object"]).columns:
            num_unique = df_optimized[col].nunique()
            num_total = len(df_optimized[col])
            if num_unique / num_total < 0.5:  # Less than 50% unique values
                df_optimized[col] = df_optimized[col].astype("category")

        final_memory = df_optimized.memory_usage(deep=True).sum() / 1024 / 1024
        reduction_pct = (initial_memory - final_memory) / initial_memory * 100

        if verbose:
            print(f"Memory optimization results:")
            print(f"  Initial memory: {initial_memory:.2f} MB")
            print(f"  Final memory: {final_memory:.2f} MB")
            print(f"  Reduction: {reduction_pct:.1f}%")

        return df_optimized

    @staticmethod
    def estimate_dataframe_memory(
        shape: Tuple[int, int], dtypes: Dict[str, str]
    ) -> float:
        """
        Estimate memory usage for a DataFrame with given shape and dtypes.

        Args:
            shape: (rows, columns)
            dtypes: Dictionary of column names to dtype strings

        Returns:
            Estimated memory in MB
        """
        dtype_sizes = {
            "int8": 1,
            "int16": 2,
            "int32": 4,
            "int64": 8,
            "uint8": 1,
            "uint16": 2,
            "uint32": 4,
            "uint64": 8,
            "float16": 2,
            "float32": 4,
            "float64": 8,
            "bool": 1,
            "category": 2,  # Approximate for category
            "object": 50,  # Approximate average for object
        }

        rows, _ = shape
        total_bytes = 0

        for dtype in dtypes.values():
            bytes_per_element = dtype_sizes.get(dtype, 50)  # Default to object size
            total_bytes += rows * bytes_per_element

        return total_bytes / 1024 / 1024  # Convert to MB


def benchmark_function(
    func: Callable, n_runs: int = 10, *args, **kwargs
) -> Dict[str, float]:
    """
    Benchmark a function over multiple runs.

    Args:
        func: Function to benchmark
        n_runs: Number of times to run the function
        *args, **kwargs: Arguments to pass to the function

    Returns:
        Dictionary with timing statistics
    """
    times = []

    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "median": np.median(times),
        "total": np.sum(times),
        "runs": n_runs,
    }


def profile_memory(func: Callable) -> Callable:
    """
    Decorator to profile memory usage of a function.

    Usage:
        @profile_memory
        def process_data(df):
            return df.groupby('category').sum()
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()

        result = func(*args, **kwargs)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"[MEMORY] {func.__name__}:")
        print(f"  Current: {current / 1024 / 1024:.2f} MB")
        print(f"  Peak: {peak / 1024 / 1024:.2f} MB")

        return result

    return wrapper
