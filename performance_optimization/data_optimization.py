"""
Data processing optimization utilities for improved performance.
"""

import functools
import hashlib
import json
import multiprocessing as mp
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import dask.dataframe as dd
import numba
import numpy as np
import pandas as pd
from numba import jit, prange

warnings.filterwarnings("ignore", category=numba.NumbaWarning)


class DataProcessingOptimizer:
    """Optimizations for data processing operations."""

    @staticmethod
    def vectorize_operation(
        df: pd.DataFrame, operation: str, columns: List[str]
    ) -> pd.Series:
        """
        Vectorize operations instead of using loops.

        Args:
            df: Input DataFrame
            operation: Operation to perform (e.g., 'sum', 'mean', 'custom')
            columns: Columns to operate on

        Returns:
            Result as Series
        """
        if operation == "sum":
            return df[columns].sum(axis=1)
        elif operation == "mean":
            return df[columns].mean(axis=1)
        elif operation == "product":
            return df[columns].prod(axis=1)
        elif operation == "max":
            return df[columns].max(axis=1)
        elif operation == "min":
            return df[columns].min(axis=1)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    @staticmethod
    def process_in_chunks(
        filepath: str, chunk_size: int = 10000, processor: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        Process large CSV files in chunks to reduce memory usage.

        Args:
            filepath: Path to CSV file
            chunk_size: Number of rows per chunk
            processor: Function to apply to each chunk

        Returns:
            Processed DataFrame
        """
        chunks = []

        for chunk in pd.read_csv(filepath, chunksize=chunk_size):
            if processor:
                chunk = processor(chunk)
            chunks.append(chunk)

        return pd.concat(chunks, ignore_index=True)

    @staticmethod
    def parallel_apply(
        df: pd.DataFrame, func: Callable, n_workers: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Apply function to DataFrame in parallel.

        Args:
            df: Input DataFrame
            func: Function to apply
            n_workers: Number of workers (default: CPU count)

        Returns:
            Processed DataFrame
        """
        if n_workers is None:
            n_workers = mp.cpu_count()

        # Split DataFrame into chunks
        chunks = np.array_split(df, n_workers)

        # Process in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(func, chunks))

        return pd.concat(results, ignore_index=True)

    @staticmethod
    def optimize_groupby(
        df: pd.DataFrame,
        groupby_cols: List[str],
        agg_funcs: Dict[str, Union[str, List[str]]],
    ) -> pd.DataFrame:
        """
        Optimized groupby operation using categorical dtypes.

        Args:
            df: Input DataFrame
            groupby_cols: Columns to group by
            agg_funcs: Aggregation functions

        Returns:
            Aggregated DataFrame
        """
        # Convert groupby columns to category for faster grouping
        df_optimized = df.copy()
        for col in groupby_cols:
            if df_optimized[col].dtype == "object":
                df_optimized[col] = df_optimized[col].astype("category")

        # Perform groupby
        return df_optimized.groupby(groupby_cols).agg(agg_funcs)

    @staticmethod
    def use_query_optimization(df: pd.DataFrame, conditions: str) -> pd.DataFrame:
        """
        Use query() method for faster filtering.

        Args:
            df: Input DataFrame
            conditions: Query string

        Returns:
            Filtered DataFrame
        """
        # Use numexpr backend for faster evaluation
        return df.query(conditions, engine="numexpr")

    @staticmethod
    def optimize_merge(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        on: Union[str, List[str]],
        how: str = "inner",
    ) -> pd.DataFrame:
        """
        Optimized merge operation.

        Args:
            df1, df2: DataFrames to merge
            on: Columns to merge on
            how: Merge type

        Returns:
            Merged DataFrame
        """
        # Sort DataFrames for faster merge
        if isinstance(on, str):
            on = [on]

        df1_sorted = df1.sort_values(on)
        df2_sorted = df2.sort_values(on)

        return pd.merge(df1_sorted, df2_sorted, on=on, how=how)


class NumbaOptimizations:
    """Numba JIT compiled functions for performance critical operations."""

    @staticmethod
    @jit(nopython=True, parallel=True)
    def fast_sum_2d(arr: np.ndarray) -> np.ndarray:
        """Fast 2D array sum using Numba."""
        n_rows, n_cols = arr.shape
        result = np.zeros(n_rows)

        for i in prange(n_rows):
            for j in range(n_cols):
                result[i] += arr[i, j]

        return result

    @staticmethod
    @jit(nopython=True)
    def fast_moving_average(arr: np.ndarray, window: int) -> np.ndarray:
        """Fast moving average calculation."""
        n = len(arr)
        result = np.zeros(n)

        for i in range(window - 1, n):
            result[i] = np.mean(arr[i - window + 1 : i + 1])

        return result

    @staticmethod
    @jit(nopython=True, parallel=True)
    def fast_correlation_matrix(data: np.ndarray) -> np.ndarray:
        """Fast correlation matrix calculation."""
        n_features = data.shape[1]
        corr_matrix = np.zeros((n_features, n_features))

        # Standardize data
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)
        standardized = (data - means) / stds

        # Calculate correlations
        for i in prange(n_features):
            for j in range(i, n_features):
                corr = np.mean(standardized[:, i] * standardized[:, j])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        return corr_matrix

    @staticmethod
    @jit(nopython=True)
    def fast_euclidean_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Fast euclidean distance calculation between two arrays."""
        n_x, n_y = X.shape[0], Y.shape[0]
        distances = np.zeros((n_x, n_y))

        for i in range(n_x):
            for j in range(n_y):
                dist = 0.0
                for k in range(X.shape[1]):
                    dist += (X[i, k] - Y[j, k]) ** 2
                distances[i, j] = np.sqrt(dist)

        return distances

    @staticmethod
    @jit(nopython=True, parallel=True)
    def fast_binning(arr: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """Fast binning operation."""
        n = len(arr)
        n_bins = len(bins) - 1
        result = np.zeros(n, dtype=np.int32)

        for i in prange(n):
            for j in range(n_bins):
                if bins[j] <= arr[i] < bins[j + 1]:
                    result[i] = j
                    break

        return result


class CachingOptimizer:
    """Caching utilities for expensive computations."""

    def __init__(self, cache_dir: str = ".cache"):
        """Initialize caching optimizer."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}

    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from function arguments."""
        # Create a string representation of arguments
        key_data = {"args": str(args), "kwargs": str(kwargs)}
        key_str = json.dumps(key_data, sort_keys=True)

        # Generate hash
        return hashlib.md5(key_str.encode()).hexdigest()

    def cache_result(self, func: Callable) -> Callable:
        """
        Decorator to cache function results.

        Usage:
            @cache_optimizer.cache_result
            def expensive_computation(data):
                # Expensive operation
                return result
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}_{self._get_cache_key(args, kwargs)}"

            # Check memory cache first
            if cache_key in self.memory_cache:
                print(f"[CACHE HIT - Memory] {func.__name__}")
                return self.memory_cache[cache_key]

            # Check disk cache
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                print(f"[CACHE HIT - Disk] {func.__name__}")
                with open(cache_file, "rb") as f:
                    result = pickle.load(f)
                self.memory_cache[cache_key] = result
                return result

            # Compute result
            print(f"[CACHE MISS] Computing {func.__name__}")
            result = func(*args, **kwargs)

            # Store in cache
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
            self.memory_cache[cache_key] = result

            return result

        return wrapper

    def clear_cache(self):
        """Clear all cached results."""
        # Clear memory cache
        self.memory_cache.clear()

        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

        print("Cache cleared")


class BatchProcessor:
    """Batch processing utilities for large datasets."""

    @staticmethod
    def process_dataframe_batches(
        df: pd.DataFrame, batch_size: int, processor: Callable
    ) -> pd.DataFrame:
        """
        Process DataFrame in batches.

        Args:
            df: Input DataFrame
            batch_size: Size of each batch
            processor: Function to apply to each batch

        Returns:
            Processed DataFrame
        """
        results = []
        n_batches = (len(df) - 1) // batch_size + 1

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            batch = df.iloc[start_idx:end_idx]

            result = processor(batch)
            results.append(result)

            print(f"Processed batch {i + 1}/{n_batches}")

        return pd.concat(results, ignore_index=True)

    @staticmethod
    def parallel_batch_process(
        data_list: List[Any],
        processor: Callable,
        batch_size: int = 1000,
        n_workers: Optional[int] = None,
    ) -> List[Any]:
        """
        Process list of items in parallel batches.

        Args:
            data_list: List of items to process
            processor: Function to apply to each item
            batch_size: Items per batch
            n_workers: Number of parallel workers

        Returns:
            List of processed items
        """
        if n_workers is None:
            n_workers = mp.cpu_count()

        # Create batches
        batches = [
            data_list[i : i + batch_size] for i in range(0, len(data_list), batch_size)
        ]

        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            batch_results = list(
                executor.map(lambda batch: [processor(item) for item in batch], batches)
            )

        # Flatten results
        return [item for batch in batch_results for item in batch]


class VectorizedOperations:
    """Vectorized implementations of common operations."""

    @staticmethod
    def vectorized_string_contains(series: pd.Series, pattern: str) -> pd.Series:
        """Vectorized string contains operation."""
        return series.str.contains(pattern, na=False)

    @staticmethod
    def vectorized_date_operations(dates: pd.Series) -> pd.DataFrame:
        """Extract multiple date features in a vectorized manner."""
        return pd.DataFrame(
            {
                "year": dates.dt.year,
                "month": dates.dt.month,
                "day": dates.dt.day,
                "dayofweek": dates.dt.dayofweek,
                "quarter": dates.dt.quarter,
                "is_weekend": dates.dt.dayofweek.isin([5, 6]),
            }
        )

    @staticmethod
    def vectorized_conditional_operation(
        df: pd.DataFrame, conditions: List[Tuple], values: List[Any], default: Any
    ) -> pd.Series:
        """
        Vectorized conditional operation (replacement for nested if-else).

        Args:
            df: Input DataFrame
            conditions: List of (column, operator, value) tuples
            values: Values to assign for each condition
            default: Default value

        Returns:
            Series with conditional values
        """
        # Create condition masks
        masks = []
        for col, op, val in conditions:
            if op == "==":
                masks.append(df[col] == val)
            elif op == ">":
                masks.append(df[col] > val)
            elif op == "<":
                masks.append(df[col] < val)
            elif op == ">=":
                masks.append(df[col] >= val)
            elif op == "<=":
                masks.append(df[col] <= val)
            elif op == "!=":
                masks.append(df[col] != val)
            elif op == "in":
                masks.append(df[col].isin(val))

        # Apply conditions using numpy.select
        return pd.Series(np.select(masks, values, default=default))

    @staticmethod
    def vectorized_cumulative_operations(
        df: pd.DataFrame, column: str, operations: List[str]
    ) -> pd.DataFrame:
        """Perform multiple cumulative operations efficiently."""
        result = pd.DataFrame(index=df.index)

        if "cumsum" in operations:
            result[f"{column}_cumsum"] = df[column].cumsum()
        if "cummax" in operations:
            result[f"{column}_cummax"] = df[column].cummax()
        if "cummin" in operations:
            result[f"{column}_cummin"] = df[column].cummin()
        if "cumprod" in operations:
            result[f"{column}_cumprod"] = df[column].cumprod()

        return result


def optimize_loop_to_vectorized(
    df: pd.DataFrame, loop_func: Callable, vectorized_func: Callable
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compare loop-based vs vectorized implementation.

    Args:
        df: Input DataFrame
        loop_func: Loop-based implementation
        vectorized_func: Vectorized implementation

    Returns:
        Tuple of (result, performance_metrics)
    """
    import time

    # Time loop implementation
    start = time.perf_counter()
    loop_result = loop_func(df)
    loop_time = time.perf_counter() - start

    # Time vectorized implementation
    start = time.perf_counter()
    vec_result = vectorized_func(df)
    vec_time = time.perf_counter() - start

    metrics = {
        "loop_time": loop_time,
        "vectorized_time": vec_time,
        "speedup": loop_time / vec_time if vec_time > 0 else float("inf"),
        "results_equal": (
            loop_result.equals(vec_result)
            if hasattr(loop_result, "equals")
            else loop_result == vec_result
        ),
    }

    print(f"Performance Comparison:")
    print(f"  Loop time: {loop_time:.4f}s")
    print(f"  Vectorized time: {vec_time:.4f}s")
    print(f"  Speedup: {metrics['speedup']:.2f}x")

    return vec_result, metrics
