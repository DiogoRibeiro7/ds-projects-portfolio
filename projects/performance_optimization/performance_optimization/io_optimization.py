"""I/O and caching optimization utilities for efficient data handling.
"""

import hashlib
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiofiles
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class IOOptimizer:
    """Optimized I/O operations for various data formats."""

    @staticmethod
    def read_csv_optimized(filepath: str, **kwargs) -> pd.DataFrame:
        """Optimized CSV reading with automatic dtype inference.

        Args:
            filepath: Path to CSV file
            **kwargs: Additional pandas read_csv arguments

        Returns:
            DataFrame with optimized dtypes
        """
        # First pass: sample to infer dtypes
        sample_df = pd.read_csv(filepath, nrows=1000)

        # Infer optimal dtypes
        dtype_dict = {}
        for col in sample_df.columns:
            if sample_df[col].dtype == "object":
                try:
                    # Try to convert to numeric
                    pd.to_numeric(sample_df[col])
                    dtype_dict[col] = "float32"
                except:
                    # Check if categorical
                    if sample_df[col].nunique() / len(sample_df) < 0.5:
                        dtype_dict[col] = "category"
                    else:
                        dtype_dict[col] = "object"
            elif sample_df[col].dtype == "int64":
                dtype_dict[col] = "int32"
            elif sample_df[col].dtype == "float64":
                dtype_dict[col] = "float32"

        # Second pass: read with optimized dtypes
        df = pd.read_csv(filepath, dtype=dtype_dict, **kwargs)

        return df

    @staticmethod
    def to_parquet_compressed(
        df: pd.DataFrame, filepath: str, compression: str = "snappy"
    ) -> None:
        """Save DataFrame to Parquet with compression.

        Args:
            df: DataFrame to save
            filepath: Output file path
            compression: Compression algorithm ('snappy', 'gzip', 'brotli')
        """
        table = pa.Table.from_pandas(df)
        pq.write_table(table, filepath, compression=compression)

        # Print compression stats
        original_size = df.memory_usage(deep=True).sum() / 1024 / 1024
        compressed_size = Path(filepath).stat().st_size / 1024 / 1024
        compression_ratio = original_size / compressed_size

        print("Parquet compression stats:")
        print(f"  Original size: {original_size:.2f} MB")
        print(f"  Compressed size: {compressed_size:.2f} MB")
        print(f"  Compression ratio: {compression_ratio:.2f}x")

    @staticmethod
    def read_parquet_columns(filepath: str, columns: list[str]) -> pd.DataFrame:
        """Read specific columns from Parquet file.

        Args:
            filepath: Path to Parquet file
            columns: List of columns to read

        Returns:
            DataFrame with specified columns
        """
        return pd.read_parquet(filepath, columns=columns)

    @staticmethod
    def chunked_csv_writer(
        df: pd.DataFrame, filepath: str, chunk_size: int = 10000
    ) -> None:
        """Write large DataFrame to CSV in chunks.

        Args:
            df: DataFrame to write
            filepath: Output file path
            chunk_size: Rows per chunk
        """
        n_chunks = (len(df) - 1) // chunk_size + 1

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx]

            # Write header only for first chunk
            mode = "w" if i == 0 else "a"
            header = i == 0

            chunk.to_csv(filepath, mode=mode, header=header, index=False)

            print(f"Written chunk {i + 1}/{n_chunks}")

    @staticmethod
    async def async_read_file(filepath: str) -> str:
        """Asynchronously read file contents.

        Args:
            filepath: Path to file

        Returns:
            File contents
        """
        async with aiofiles.open(filepath) as f:
            contents = await f.read()
        return contents

    @staticmethod
    async def async_write_file(filepath: str, content: str) -> None:
        """Asynchronously write content to file.

        Args:
            filepath: Path to file
            content: Content to write
        """
        async with aiofiles.open(filepath, mode="w") as f:
            await f.write(content)

    @staticmethod
    def parallel_file_reader(
        filepaths: list[str], reader_func: callable, n_workers: int | None = None
    ) -> list[pd.DataFrame]:
        """Read multiple files in parallel.

        Args:
            filepaths: List of file paths
            reader_func: Function to read each file
            n_workers: Number of parallel workers

        Returns:
            List of DataFrames
        """
        if n_workers is None:
            n_workers = min(len(filepaths), 4)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(reader_func, filepaths))

        return results


class DataCache:
    """Advanced caching system for data operations."""

    def __init__(self, cache_dir: str = ".cache", max_cache_size_mb: int = 1000):
        """Initialize data cache.

        Args:
            cache_dir: Directory for cache files
            max_cache_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # Convert to bytes
        self.cache_metadata = self._load_metadata()

    def _load_metadata(self) -> dict[str, Any]:
        """Load cache metadata."""
        metadata_file = self.cache_dir / "cache_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save cache metadata."""
        metadata_file = self.cache_dir / "cache_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self.cache_metadata, f)

    def _get_cache_key(self, key_data: Any) -> str:
        """Generate cache key from data."""
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()

    def cache_dataframe(
        self, df: pd.DataFrame, key: str, ttl_hours: int | None = None
    ) -> None:
        """Cache DataFrame with optional TTL.

        Args:
            df: DataFrame to cache
            key: Cache key
            ttl_hours: Time to live in hours
        """
        cache_key = self._get_cache_key(key)
        cache_file = self.cache_dir / f"{cache_key}.parquet"

        # Save DataFrame
        df.to_parquet(cache_file, compression="snappy")

        # Update metadata
        self.cache_metadata[cache_key] = {
            "key": key,
            "timestamp": datetime.now().isoformat(),
            "ttl_hours": ttl_hours,
            "size_bytes": cache_file.stat().st_size,
            "shape": df.shape,
            "columns": list(df.columns),
        }
        self._save_metadata()

        # Check cache size and evict if necessary
        self._evict_if_needed()

    def get_cached_dataframe(self, key: str) -> pd.DataFrame | None:
        """Retrieve cached DataFrame.

        Args:
            key: Cache key

        Returns:
            Cached DataFrame or None if not found/expired
        """
        cache_key = self._get_cache_key(key)

        # Check if exists and not expired
        if cache_key in self.cache_metadata:
            metadata = self.cache_metadata[cache_key]

            # Check TTL
            if metadata.get("ttl_hours"):
                cached_time = datetime.fromisoformat(metadata["timestamp"])
                if datetime.now() - cached_time > timedelta(
                    hours=metadata["ttl_hours"]
                ):
                    print(f"Cache expired for key: {key}")
                    self._remove_cache_entry(cache_key)
                    return None

            # Load DataFrame
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            if cache_file.exists():
                print(f"Cache hit for key: {key}")
                return pd.read_parquet(cache_file)

        print(f"Cache miss for key: {key}")
        return None

    def _evict_if_needed(self):
        """Evict oldest cache entries if size exceeds limit."""
        total_size = sum(m["size_bytes"] for m in self.cache_metadata.values())

        if total_size > self.max_cache_size:
            # Sort by timestamp and remove oldest
            sorted_entries = sorted(
                self.cache_metadata.items(), key=lambda x: x[1]["timestamp"]
            )

            while total_size > self.max_cache_size and sorted_entries:
                oldest_key, oldest_meta = sorted_entries.pop(0)
                self._remove_cache_entry(oldest_key)
                total_size -= oldest_meta["size_bytes"]
                print(f"Evicted cache entry: {oldest_meta['key']}")

    def _remove_cache_entry(self, cache_key: str):
        """Remove cache entry."""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        if cache_file.exists():
            cache_file.unlink()
        if cache_key in self.cache_metadata:
            del self.cache_metadata[cache_key]
        self._save_metadata()

    def clear_cache(self):
        """Clear all cache entries."""
        for cache_file in self.cache_dir.glob("*.parquet"):
            cache_file.unlink()
        self.cache_metadata = {}
        self._save_metadata()
        print("Cache cleared")


class DatabaseOptimizer:
    """Optimization for database operations."""

    def __init__(self, db_path: str = "data_cache.db"):
        """Initialize database optimizer."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

    def create_indexed_table(
        self, df: pd.DataFrame, table_name: str, index_columns: list[str]
    ) -> None:
        """Create database table with indexes for fast queries.

        Args:
            df: DataFrame to store
            table_name: Name of the table
            index_columns: Columns to create indexes on
        """
        # Write DataFrame to SQLite
        df.to_sql(table_name, self.conn, if_exists="replace", index=False)

        # Create indexes
        cursor = self.conn.cursor()
        for col in index_columns:
            index_name = f"idx_{table_name}_{col}"
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({col})"
            )

        self.conn.commit()
        print(f"Created table '{table_name}' with indexes on {index_columns}")

    def batch_insert(
        self, data: list[dict], table_name: str, batch_size: int = 1000
    ) -> None:
        """Insert data in batches for better performance.

        Args:
            data: List of dictionaries to insert
            table_name: Target table name
            batch_size: Records per batch
        """
        df = pd.DataFrame(data)
        n_batches = (len(df) - 1) // batch_size + 1

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            batch = df.iloc[start_idx:end_idx]

            batch.to_sql(table_name, self.conn, if_exists="append", index=False)

            print(f"Inserted batch {i + 1}/{n_batches}")

        self.conn.commit()

    def cached_query(self, query: str, cache_key: str) -> pd.DataFrame:
        """Execute query with caching.

        Args:
            query: SQL query
            cache_key: Cache key for results

        Returns:
            Query results as DataFrame
        """
        # Check cache
        cache_table = f"cache_{cache_key}"
        try:
            df = pd.read_sql_query(f"SELECT * FROM {cache_table}", self.conn)
            print(f"Query cache hit: {cache_key}")
            return df
        except:
            pass

        # Execute query
        print(f"Executing query: {cache_key}")
        df = pd.read_sql_query(query, self.conn)

        # Cache results
        df.to_sql(cache_table, self.conn, if_exists="replace", index=False)

        return df

    def close(self):
        """Close database connection."""
        self.conn.close()


class HDF5Storage:
    """Optimized HDF5 storage for large datasets."""

    def __init__(self, filepath: str):
        """Initialize HDF5 storage."""
        self.filepath = filepath
        self.store = pd.HDFStore(filepath, mode="a")

    def store_dataframe(
        self, df: pd.DataFrame, key: str, compress: bool = True
    ) -> None:
        """Store DataFrame in HDF5 format.

        Args:
            df: DataFrame to store
            key: Storage key
            compress: Whether to compress data
        """
        if compress:
            self.store.put(key, df, format="table", complib="blosc", complevel=9)
        else:
            self.store.put(key, df, format="table")

        print(f"Stored DataFrame with key '{key}' ({df.shape})")

    def read_dataframe(
        self, key: str, columns: list[str] | None = None, where: str | None = None
    ) -> pd.DataFrame:
        """Read DataFrame from HDF5 storage.

        Args:
            key: Storage key
            columns: Specific columns to read
            where: Query condition

        Returns:
            DataFrame
        """
        if columns:
            return self.store.select(key, columns=columns, where=where)
        else:
            return self.store.select(key, where=where)

    def append_dataframe(self, df: pd.DataFrame, key: str) -> None:
        """Append DataFrame to existing HDF5 table.

        Args:
            df: DataFrame to append
            key: Storage key
        """
        self.store.append(key, df, format="table")
        print(f"Appended {len(df)} rows to key '{key}'")

    def list_keys(self) -> list[str]:
        """List all keys in HDF5 store."""
        return list(self.store.keys())

    def get_info(self, key: str) -> dict[str, Any]:
        """Get information about stored data."""
        storer = self.store.get_storer(key)
        return {
            "nrows": storer.nrows,
            "columns": list(storer.ncols),
            "is_table": storer.is_table,
        }

    def close(self):
        """Close HDF5 store."""
        self.store.close()


def create_data_pipeline(
    input_file: str,
    output_file: str,
    transformations: list[callable],
    chunk_size: int = 10000,
) -> None:
    """Create efficient data pipeline with chunking.

    Args:
        input_file: Input file path
        output_file: Output file path
        transformations: List of transformation functions
        chunk_size: Size of chunks to process
    """
    first_chunk = True

    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        # Apply transformations
        for transform in transformations:
            chunk = transform(chunk)

        # Write to output
        mode = "w" if first_chunk else "a"
        header = first_chunk

        chunk.to_csv(output_file, mode=mode, header=header, index=False)
        first_chunk = False

    print(f"Pipeline completed: {input_file} -> {output_file}")


def benchmark_io_methods(df: pd.DataFrame, filepath_base: str) -> pd.DataFrame:
    """Benchmark different I/O methods.

    Args:
        df: DataFrame to test
        filepath_base: Base path for test files

    Returns:
        Benchmark results
    """
    results = []

    # CSV
    start = time.perf_counter()
    df.to_csv(f"{filepath_base}.csv", index=False)
    csv_write_time = time.perf_counter() - start

    start = time.perf_counter()
    pd.read_csv(f"{filepath_base}.csv")
    csv_read_time = time.perf_counter() - start

    csv_size = Path(f"{filepath_base}.csv").stat().st_size / 1024 / 1024

    results.append(
        {
            "format": "CSV",
            "write_time": csv_write_time,
            "read_time": csv_read_time,
            "file_size_mb": csv_size,
        }
    )

    # Parquet
    start = time.perf_counter()
    df.to_parquet(f"{filepath_base}.parquet")
    parquet_write_time = time.perf_counter() - start

    start = time.perf_counter()
    pd.read_parquet(f"{filepath_base}.parquet")
    parquet_read_time = time.perf_counter() - start

    parquet_size = Path(f"{filepath_base}.parquet").stat().st_size / 1024 / 1024

    results.append(
        {
            "format": "Parquet",
            "write_time": parquet_write_time,
            "read_time": parquet_read_time,
            "file_size_mb": parquet_size,
        }
    )

    # Feather
    start = time.perf_counter()
    df.to_feather(f"{filepath_base}.feather")
    feather_write_time = time.perf_counter() - start

    start = time.perf_counter()
    pd.read_feather(f"{filepath_base}.feather")
    feather_read_time = time.perf_counter() - start

    feather_size = Path(f"{filepath_base}.feather").stat().st_size / 1024 / 1024

    results.append(
        {
            "format": "Feather",
            "write_time": feather_write_time,
            "read_time": feather_read_time,
            "file_size_mb": feather_size,
        }
    )

    # Pickle
    start = time.perf_counter()
    df.to_pickle(f"{filepath_base}.pkl")
    pickle_write_time = time.perf_counter() - start

    start = time.perf_counter()
    pd.read_pickle(f"{filepath_base}.pkl")
    pickle_read_time = time.perf_counter() - start

    pickle_size = Path(f"{filepath_base}.pkl").stat().st_size / 1024 / 1024

    results.append(
        {
            "format": "Pickle",
            "write_time": pickle_write_time,
            "read_time": pickle_read_time,
            "file_size_mb": pickle_size,
        }
    )

    # Clean up
    for ext in [".csv", ".parquet", ".feather", ".pkl"]:
        Path(f"{filepath_base}{ext}").unlink(missing_ok=True)

    return pd.DataFrame(results)
