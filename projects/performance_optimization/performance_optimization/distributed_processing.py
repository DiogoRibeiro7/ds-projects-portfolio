"""Distributed and parallel processing utilities for large-scale data operations.
"""

import multiprocessing as mp
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client

# Check for distributed processing libraries
DASK_AVAILABLE = True  # Already imported
RAY_AVAILABLE = False
SPARK_AVAILABLE = False

try:
    import ray

    ray.init(ignore_reinit_error=True, num_cpus=mp.cpu_count())
    RAY_AVAILABLE = True
except ImportError:
    ray = None
    warnings.warn("Ray not installed. Distributed actor model disabled.")
except:
    warnings.warn("Ray not initialized. Distributed actor model disabled.")

try:
    from pyspark.sql import SparkSession

    SPARK_AVAILABLE = True
except ImportError:
    warnings.warn("PySpark not installed. Spark processing disabled.")


class DistributedConfig:
    """Configuration for distributed processing."""

    def __init__(
        self,
        n_workers: int | None = None,
        memory_limit: str = "2GB",
        threads_per_worker: int = 1,
        scheduler: str = "threads",
    ):
        """Initialize distributed configuration.

        Args:
            n_workers: Number of workers (default: CPU count)
            memory_limit: Memory limit per worker
            threads_per_worker: Threads per worker
            scheduler: Scheduler type ('threads', 'processes', 'distributed')
        """
        self.n_workers = n_workers or mp.cpu_count()
        self.memory_limit = memory_limit
        self.threads_per_worker = threads_per_worker
        self.scheduler = scheduler


class DaskOperations:
    """Distributed operations using Dask."""

    def __init__(self, config: DistributedConfig | None = None):
        """Initialize Dask operations with configuration."""
        self.config = config or DistributedConfig()
        self.client = None

    def start_cluster(self, dashboard: bool = True) -> Client:
        """Start a Dask distributed cluster."""
        if self.client is None:
            self.client = Client(
                n_workers=self.config.n_workers,
                threads_per_worker=self.config.threads_per_worker,
                memory_limit=self.config.memory_limit,
                dashboard_address=":8787" if dashboard else None,
            )
        return self.client

    def shutdown_cluster(self):
        """Shutdown the Dask cluster."""
        if self.client:
            self.client.close()
            self.client = None

    @staticmethod
    def parallelize_dataframe(
        df: pd.DataFrame, n_partitions: int | None = None
    ) -> dd.DataFrame:
        """Convert pandas DataFrame to Dask DataFrame for parallel processing."""
        if n_partitions is None:
            n_partitions = mp.cpu_count() * 2
        return dd.from_pandas(df, npartitions=n_partitions)

    @staticmethod
    def parallelize_array(arr: np.ndarray, chunks: tuple | None = None) -> da.Array:
        """Convert numpy array to Dask array for parallel processing."""
        if chunks is None:
            # Auto-determine chunk size
            chunks = "auto"
        return da.from_array(arr, chunks=chunks)

    def parallel_apply(
        self, df: dd.DataFrame, func: Callable, meta: Any | None = None
    ) -> dd.DataFrame:
        """Apply function in parallel across DataFrame partitions."""
        return df.map_partitions(func, meta=meta)

    def parallel_groupby_agg(
        self,
        df: pd.DataFrame,
        groupby_cols: list[str],
        agg_dict: dict[str, str | list[str]],
    ) -> pd.DataFrame:
        """Parallel groupby aggregation using Dask."""
        # Convert to Dask DataFrame
        ddf = self.parallelize_dataframe(df)

        # Perform groupby
        result = ddf.groupby(groupby_cols).agg(agg_dict)

        # Compute and return result
        return result.compute()

    def parallel_merge(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        on: str | list[str],
        how: str = "inner",
    ) -> pd.DataFrame:
        """Parallel merge operation using Dask."""
        ddf1 = self.parallelize_dataframe(df1)
        ddf2 = self.parallelize_dataframe(df2)

        result = ddf1.merge(ddf2, on=on, how=how)
        return result.compute()

    def parallel_read_csv(self, pattern: str, **kwargs) -> pd.DataFrame:
        """Read multiple CSV files in parallel."""
        ddf = dd.read_csv(pattern, **kwargs)
        return ddf.compute()

    def parallel_to_parquet(
        self, df: pd.DataFrame, path: str, compression: str = "snappy"
    ) -> None:
        """Write DataFrame to Parquet in parallel."""
        ddf = self.parallelize_dataframe(df)
        ddf.to_parquet(path, compression=compression)


class RayOperations:
    """Distributed operations using Ray."""

    @staticmethod
    def is_available() -> bool:
        """Check if Ray is available and initialized."""
        return RAY_AVAILABLE

    @staticmethod
    def process_partition(func: Callable, data: Any) -> Any:
        """Ray remote function to process data partition."""
        return func(data)

    @staticmethod
    def parallel_map(
        func: Callable, data_list: list[Any], n_workers: int | None = None
    ) -> list[Any]:
        """Parallel map operation using Ray."""
        if not RAY_AVAILABLE or ray is None:
            return [func(d) for d in data_list]

        # Put function in shared memory
        func_id = ray.put(func)

        # Create remote function dynamically
        remote_func = ray.remote(RayOperations.process_partition)

        # Create futures
        futures = [remote_func.remote(func_id, data) for data in data_list]

        # Get results
        return ray.get(futures)

    class DataProcessor:
        """Ray actor for stateful data processing."""

        def __init__(self):
            self.cache = {}

        def process(self, key: str, data: Any, func: Callable) -> Any:
            """Process data with caching."""
            if key in self.cache:
                return self.cache[key]

            result = func(data)
            self.cache[key] = result
            return result

        def clear_cache(self):
            """Clear the cache."""
            self.cache.clear()


class ParallelProcessor:
    """High-level parallel processing interface."""

    def __init__(
        self, backend: str = "multiprocessing", n_workers: int | None = None
    ):
        """Initialize parallel processor.

        Args:
            backend: Processing backend ('multiprocessing', 'threading', 'dask', 'ray')
            n_workers: Number of workers
        """
        self.backend = backend
        self.n_workers = n_workers or mp.cpu_count()

    def parallel_execute(
        self, func: Callable, data_list: list[Any], show_progress: bool = True
    ) -> list[Any]:
        """Execute function in parallel across data list.

        Args:
            func: Function to apply to each element
            data_list: List of data to process
            show_progress: Show progress bar

        Returns:
            List of results
        """
        if self.backend == "threading":
            return self._thread_execute(func, data_list, show_progress)
        elif self.backend == "multiprocessing":
            return self._process_execute(func, data_list, show_progress)
        elif self.backend == "dask":
            return self._dask_execute(func, data_list, show_progress)
        elif self.backend == "ray" and RAY_AVAILABLE:
            return RayOperations.parallel_map(func, data_list, self.n_workers)
        else:
            # Fallback to sequential
            return [func(d) for d in data_list]

    def _thread_execute(
        self, func: Callable, data_list: list[Any], show_progress: bool
    ) -> list[Any]:
        """Execute using ThreadPoolExecutor."""
        results = []
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(func, data): i for i, data in enumerate(data_list)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                    if show_progress:
                        print(f"Completed {len(results)}/{len(data_list)}", end="\r")
                except Exception as e:
                    print(f"Error processing item {idx}: {e}")

        # Sort by original order
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def _process_execute(
        self, func: Callable, data_list: list[Any], show_progress: bool
    ) -> list[Any]:
        """Execute using ProcessPoolExecutor."""
        results = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(func, data): i for i, data in enumerate(data_list)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                    if show_progress:
                        print(f"Completed {len(results)}/{len(data_list)}", end="\r")
                except Exception as e:
                    print(f"Error processing item {idx}: {e}")

        # Sort by original order
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def _dask_execute(
        self, func: Callable, data_list: list[Any], show_progress: bool
    ) -> list[Any]:
        """Execute using Dask."""
        from dask import compute, delayed

        # Create delayed tasks
        tasks = [delayed(func)(data) for data in data_list]

        # Compute results
        results = compute(*tasks, scheduler=self.backend)
        return list(results)


class MapReduce:
    """MapReduce implementation for large-scale data processing."""

    def __init__(self, n_workers: int | None = None):
        """Initialize MapReduce with specified workers."""
        self.n_workers = n_workers or mp.cpu_count()

    def map_reduce(
        self,
        data: list[Any],
        mapper: Callable[[Any], list[tuple[Any, Any]]],
        reducer: Callable[[Any, list[Any]], Any],
    ) -> dict[Any, Any]:
        """Execute MapReduce operation.

        Args:
            data: Input data
            mapper: Function that maps input to (key, value) pairs
            reducer: Function that reduces values for each key

        Returns:
            Dictionary of reduced results
        """
        # Map phase
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            map_results = list(executor.map(mapper, data))

        # Flatten and group by key
        from collections import defaultdict

        grouped = defaultdict(list)
        for kvpairs in map_results:
            for key, value in kvpairs:
                grouped[key].append(value)

        # Reduce phase
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(reducer, key, values): key
                for key, values in grouped.items()
            }

            results = {}
            for future in as_completed(futures):
                key = futures[future]
                results[key] = future.result()

        return results


class DataParallelizer:
    """Utilities for data parallelization strategies."""

    @staticmethod
    def partition_dataframe(df: pd.DataFrame, n_partitions: int) -> list[pd.DataFrame]:
        """Partition DataFrame into chunks for parallel processing."""
        chunk_size = len(df) // n_partitions + 1
        return [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

    @staticmethod
    def partition_array(arr: np.ndarray, n_partitions: int) -> list[np.ndarray]:
        """Partition numpy array into chunks."""
        return np.array_split(arr, n_partitions)

    @staticmethod
    def scatter_gather(
        func: Callable,
        data: pd.DataFrame | np.ndarray,
        n_workers: int | None = None,
    ) -> pd.DataFrame | np.ndarray:
        """Scatter data to workers, apply function, and gather results.

        Args:
            func: Function to apply to each partition
            data: Data to process
            n_workers: Number of workers

        Returns:
            Combined results
        """
        n_workers = n_workers or mp.cpu_count()

        # Partition data
        if isinstance(data, pd.DataFrame):
            partitions = DataParallelizer.partition_dataframe(data, n_workers)
        else:
            partitions = DataParallelizer.partition_array(data, n_workers)

        # Process in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(func, partitions))

        # Combine results
        if isinstance(data, pd.DataFrame):
            return pd.concat(results, ignore_index=True)
        else:
            return np.concatenate(results)


def benchmark_distributed(
    func: Callable,
    data: Any,
    backends: list[str],
    n_workers: int = 4,
    iterations: int = 3,
) -> pd.DataFrame:
    """Benchmark different distributed processing backends.

    Args:
        func: Function to benchmark
        data: Data to process
        backends: List of backends to test
        n_workers: Number of workers
        iterations: Number of iterations

    Returns:
        DataFrame with benchmark results
    """
    results = []

    for backend in backends:
        processor = ParallelProcessor(backend=backend, n_workers=n_workers)

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            try:
                processor.parallel_execute(func, data, show_progress=False)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            except Exception as e:
                print(f"Error with {backend}: {e}")
                times.append(None)

        if times and all(t is not None for t in times):
            results.append(
                {
                    "backend": backend,
                    "mean_time": np.mean(times),
                    "std_time": np.std(times),
                    "min_time": np.min(times),
                    "max_time": np.max(times),
                    "n_workers": n_workers,
                }
            )

    return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    print("Distributed Processing Module")
    print("=" * 60)

    # Test data
    n_items = 1000
    data = [np.random.randn(100, 100) for _ in range(n_items)]

    # Define test function
    def process_matrix(matrix):
        """Example processing function."""
        return np.linalg.svd(matrix, full_matrices=False)[1].sum()

    # Benchmark different backends
    print("\nBenchmarking distributed backends...")
    backends = ["threading", "multiprocessing"]
    if DASK_AVAILABLE:
        backends.append("dask")
    if RAY_AVAILABLE:
        backends.append("ray")

    results = benchmark_distributed(
        process_matrix,
        data[:10],  # Use subset for demo
        backends=backends,
        n_workers=4,
        iterations=2,
    )

    print("\nBenchmark Results:")
    print(results.to_string())

    # Find best backend
    if not results.empty:
        best = results.loc[results["mean_time"].idxmin()]
        print(f"\nBest backend: {best['backend']} ({best['mean_time']:.3f}s)")
