"""Intelligent caching system for expensive computations.

This module provides a comprehensive caching solution with multiple backends,
automatic storage selection, and intelligent cache key generation.
"""

import functools
import hashlib
import json
import logging
import pickle
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Optional imports with fallbacks
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    warnings.warn("Redis not available - using disk cache only")

try:
    from diskcache import Cache

    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False
    warnings.warn("DiskCache not available - using simple file cache")

try:
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmartCache:
    """Intelligent caching system with memory/disk/Redis backends.

    The cache automatically picks the fastest backend available, keeps an
    in-memory LRU bounded by ``max_memory_size`` bytes, optionally compresses
    payloads, and exposes decorators for functions that return pandas objects.

    Examples:
    --------
    >>> cache = SmartCache(use_redis=False, max_memory_size=1_000_000)
    >>> @cache.cache_dataframe(key_prefix="agg", ttl=60)
    ... def build_df(limit: int):
    ...     return pd.DataFrame({"x": range(limit)})
    >>> len(build_df(3)), len(build_df(3))  # second call hits cache
    (3, 3)
    """

    def __init__(
        self,
        cache_dir: str = ".cache",
        use_redis: bool = False,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        max_memory_size: int = 100_000_000,  # 100MB
        enable_compression: bool = True,
        enable_stats: bool = True,
    ):
        """Initialize the smart caching system.

        Parameters
        ----------
        cache_dir : str, default='.cache'
            Directory for disk cache storage.
        use_redis : bool, default=False
            Whether to use Redis for fast caching.
        redis_host : str, default='localhost'
            Redis server hostname.
        redis_port : int, default=6379
            Redis server port.
        max_memory_size : int, default=100_000_000
            Maximum size for in-memory cache (bytes).
        enable_compression : bool, default=True
            Whether to compress large cached objects.
        enable_stats : bool, default=True
            Whether to track cache statistics.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_memory_size = max_memory_size
        self.enable_compression = enable_compression
        self.enable_stats = enable_stats

        # Initialize caching backends
        self._init_disk_cache()
        self._init_redis_cache(use_redis, redis_host, redis_port)
        self._init_memory_cache()

        # Cache statistics
        if enable_stats:
            self.stats = {
                "hits": 0,
                "misses": 0,
                "memory_hits": 0,
                "redis_hits": 0,
                "disk_hits": 0,
                "total_compute_time_saved": 0.0,
                "cache_writes": 0,
            }

        logger.info(
            f"SmartCache initialized with backends: "
            f"memory={True}, disk={DISKCACHE_AVAILABLE}, redis={self.redis_available}"
        )

    def _init_disk_cache(self):
        """Initialize disk cache backend."""
        if DISKCACHE_AVAILABLE:
            self.disk_cache = Cache(
                str(self.cache_dir / "disk"),
                size_limit=10 * 1024**3,  # 10GB limit
                cull_limit=0,  # No automatic culling
            )
        else:
            # Fallback to simple file cache
            self.disk_cache = SimpleFileCache(self.cache_dir / "disk")

    def _init_redis_cache(self, use_redis: bool, host: str, port: int):
        """Initialize Redis cache backend."""
        self.redis_available = False
        self.redis_client = None

        if use_redis and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=host,
                    port=port,
                    db=0,
                    decode_responses=False,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
                # Test connection
                self.redis_client.ping()
                self.redis_available = True
                logger.info("Redis cache connected successfully")
            except (redis.ConnectionError, redis.TimeoutError) as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None

    def _init_memory_cache(self):
        """Initialize memory cache with LRU eviction."""
        from collections import OrderedDict

        class LRUCache(OrderedDict):
            def __init__(self, max_size: int):
                super().__init__()
                self.max_size = max_size
                self.current_size = 0

            def __setitem__(self, key, value):
                value_size = len(pickle.dumps(value, protocol=4))
                if key in self:
                    self.move_to_end(key)
                    old_size = len(pickle.dumps(self[key], protocol=4))
                    self.current_size -= old_size
                else:
                    while (
                        self.current_size + value_size > self.max_size and len(self) > 0
                    ):
                        oldest = next(iter(self))
                        old_size = len(pickle.dumps(self[oldest], protocol=4))
                        del self[oldest]
                        self.current_size -= old_size

                super().__setitem__(key, value)
                self.current_size += value_size

            def __getitem__(self, key):
                self.move_to_end(key)
                return super().__getitem__(key)

        self.memory_cache = LRUCache(self.max_memory_size)

    def cache_dataframe(
        self, key_prefix: str = "df", ttl: int = 3600, force_refresh: bool = False
    ):
        """Decorator for caching DataFrame operations.

        Parameters
        ----------
        key_prefix : str, default='df'
            Prefix for cache keys.
        ttl : int, default=3600
            Time-to-live in seconds.
        force_refresh : bool, default=False
            Whether to force recomputation.

        Returns:
        -------
        decorator : Callable
            Caching decorator.
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_key(key_prefix, func.__name__, args, kwargs)

                # Check if force refresh
                if not force_refresh:
                    # Try to get from cache
                    start_time = time.time()
                    cached = self.get(cache_key)
                    if cached is not None:
                        retrieval_time = time.time() - start_time
                        logger.debug(
                            f"Cache hit: {cache_key} (retrieved in {retrieval_time:.3f}s)"
                        )
                        if self.enable_stats:
                            self.stats["hits"] += 1
                        return cached

                # Cache miss - compute result
                logger.debug(f"Cache miss: {cache_key}")
                if self.enable_stats:
                    self.stats["misses"] += 1

                compute_start = time.time()
                result = func(*args, **kwargs)
                compute_time = time.time() - compute_start

                # Store in cache
                self.set(cache_key, result, ttl=ttl, compute_time=compute_time)

                return result

            return wrapper

        return decorator

    def cache_computation(
        self,
        key_prefix: str = "compute",
        ttl: int = 3600,
        serialize_method: str = "auto",
    ):
        """Decorator for caching general computations.

        Parameters
        ----------
        key_prefix : str, default='compute'
            Prefix for cache keys.
        ttl : int, default=3600
            Time-to-live in seconds.
        serialize_method : str, default='auto'
            Serialization method: 'auto', 'pickle', 'joblib', 'json'.

        Returns:
        -------
        decorator : Callable
            Caching decorator.
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_key(key_prefix, func.__name__, args, kwargs)

                # Try to get from cache
                cached = self.get(cache_key, serialize_method=serialize_method)
                if cached is not None:
                    logger.debug(f"Cache hit: {cache_key}")
                    return cached

                # Compute and cache
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl=ttl, serialize_method=serialize_method)
                return result

            return wrapper

        return decorator

    def _generate_key(
        self, prefix: str, func_name: str, args: tuple, kwargs: dict
    ) -> str:
        """Generate unique cache key based on function inputs.

        Parameters
        ----------
        prefix : str
            Key prefix.
        func_name : str
            Function name.
        args : tuple
            Positional arguments.
        kwargs : dict
            Keyword arguments.

        Returns:
        -------
        key : str
            Unique cache key.
        """
        key_parts = [prefix, func_name]

        # Process arguments
        for i, arg in enumerate(args):
            if isinstance(arg, pd.DataFrame):
                # For DataFrames, use shape, columns, and sample hash
                df_hash = hashlib.md5(
                    f"{arg.shape}_{list(arg.columns)}_{arg.iloc[: min(100, len(arg))].values.tobytes()}".encode()
                ).hexdigest()[:8]
                key_parts.append(f"df{i}_{df_hash}")
            elif isinstance(arg, np.ndarray):
                # For arrays, use shape, dtype, and sample hash
                arr_hash = hashlib.md5(
                    arg.flat[: min(1000, arg.size)].tobytes()
                ).hexdigest()[:8]
                key_parts.append(f"arr{i}_{arg.shape}_{arg.dtype}_{arr_hash}")
            elif isinstance(arg, (list, tuple, dict)):
                # For collections, use JSON hash
                try:
                    json_str = json.dumps(arg, sort_keys=True, default=str)
                    coll_hash = hashlib.md5(json_str.encode()).hexdigest()[:8]
                    key_parts.append(f"coll{i}_{coll_hash}")
                except:
                    key_parts.append(f"arg{i}_{hash(str(arg))}")
            else:
                key_parts.append(f"arg{i}_{str(arg)}")

        # Process keyword arguments
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (pd.DataFrame, np.ndarray)):
                key_parts.append(f"{k}=complex_{hash(str(type(v)))}")
            else:
                key_parts.append(f"{k}={v}")

        # Generate final key
        key_string = "_".join(str(part) for part in key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, key: str, default: Any = None, serialize_method: str = "auto") -> Any:
        """Get value from cache.

        Parameters
        ----------
        key : str
            Cache key.
        default : Any, optional
            Default value if not found.
        serialize_method : str, default='auto'
            Deserialization method.

        Returns:
        -------
        value : Any
            Cached value or default.
        """
        # Try memory cache first
        if key in self.memory_cache:
            cached = self.memory_cache[key]
            value = cached[0] if isinstance(cached, tuple) and len(cached) == 2 else cached
            expires_at = cached[1] if isinstance(cached, tuple) and len(cached) == 2 else None

            if expires_at is not None and time.time() > expires_at:
                del self.memory_cache[key]
            else:
                if self.enable_stats:
                    self.stats["memory_hits"] += 1
                logger.debug(f"Memory cache hit: {key}")
                return value

        # Try Redis if available
        if self.redis_available:
            try:
                value = self.redis_client.get(key)
                if value:
                    deserialized = self._deserialize(value, serialize_method)
                    # Promote to memory cache
                    self._add_to_memory_cache(key, deserialized)
                    if self.enable_stats:
                        self.stats["redis_hits"] += 1
                    logger.debug(f"Redis cache hit: {key}")
                    return deserialized
            except Exception as e:
                logger.warning(f"Redis get error: {e}")

        # Try disk cache
        try:
            if DISKCACHE_AVAILABLE:
                value = self.disk_cache.get(key, default)
            else:
                value = self.disk_cache.get(key)

            if value is not None and value != default:
                # Promote to faster caches
                self._add_to_memory_cache(key, value)
                self._add_to_redis_cache(key, value)
                if self.enable_stats:
                    self.stats["disk_hits"] += 1
                logger.debug(f"Disk cache hit: {key}")
                return value
        except Exception as e:
            logger.warning(f"Disk cache get error: {e}")

        return default

    def set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600,
        serialize_method: str = "auto",
        compute_time: float = 0.0,
    ):
        """Set value in cache.

        Parameters
        ----------
        key : str
            Cache key.
        value : Any
            Value to cache.
        ttl : int, default=3600
            Time-to-live in seconds.
        serialize_method : str, default='auto'
            Serialization method.
        compute_time : float, default=0.0
            Time taken to compute the value.
        """
        if self.enable_stats:
            self.stats["cache_writes"] += 1
            self.stats["total_compute_time_saved"] += compute_time

        expire_at = time.time() + ttl if ttl is not None else None

        # Determine size
        size = self._get_size(value)

        # Store in appropriate caches based on size
        if size < 1_000_000:  # < 1MB: all caches
            self._add_to_memory_cache(key, value, expires_at=expire_at)
            self._add_to_redis_cache(key, value, ttl)
            self._add_to_disk_cache(key, value, ttl)
        elif size < 10_000_000:  # < 10MB: Redis + disk
            self._add_to_redis_cache(key, value, ttl)
            self._add_to_disk_cache(key, value, ttl)
        else:  # Large: disk only
            self._add_to_disk_cache(key, value, ttl)

        logger.debug(f"Cached {key} (size: {size / 1024:.1f}KB)")

    def _add_to_memory_cache(self, key: str, value: Any, expires_at: float | None = None):
        """Add value to memory cache."""
        try:
            self.memory_cache[key] = (value, expires_at)
        except Exception as e:
            logger.warning(f"Memory cache set error: {e}")

    def _add_to_redis_cache(self, key: str, value: Any, ttl: int = 3600):
        """Add value to Redis cache."""
        if self.redis_available:
            try:
                serialized = self._serialize(value)
                self.redis_client.setex(key, ttl, serialized)
            except Exception as e:
                logger.warning(f"Redis set error: {e}")

    def _add_to_disk_cache(self, key: str, value: Any, ttl: int = 3600):
        """Add value to disk cache."""
        try:
            if DISKCACHE_AVAILABLE:
                self.disk_cache.set(key, value, expire=ttl)
            else:
                self.disk_cache.set(key, value, ttl)
        except Exception as e:
            logger.warning(f"Disk cache set error: {e}")

    def _serialize(self, obj: Any, method: str = "auto") -> bytes:
        """Serialize object for storage.

        Parameters
        ----------
        obj : Any
            Object to serialize.
        method : str
            Serialization method.

        Returns:
        -------
        serialized : bytes
            Serialized object.
        """
        if method == "auto":
            # Choose best method based on object type
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                method = "pickle"
            elif isinstance(obj, np.ndarray):
                method = "pickle"
            elif isinstance(obj, (dict, list)) and self._is_json_serializable(obj):
                method = "json"
            else:
                method = "pickle"

        if method == "json":
            return json.dumps(obj, default=str).encode("utf-8")
        elif method == "joblib" and JOBLIB_AVAILABLE:
            import io

            buffer = io.BytesIO()
            joblib.dump(obj, buffer)
            return buffer.getvalue()
        else:  # pickle
            serialized = pickle.dumps(obj, protocol=4)
            if self.enable_compression and len(serialized) > 10000:
                import gzip

                serialized = gzip.compress(serialized)
            return serialized

    def _deserialize(self, data: bytes, method: str = "auto") -> Any:
        """Deserialize object from storage.

        Parameters
        ----------
        data : bytes
            Serialized data.
        method : str
            Deserialization method.

        Returns:
        -------
        obj : Any
            Deserialized object.
        """
        if method == "json":
            return json.loads(data.decode("utf-8"))
        elif method == "joblib" and JOBLIB_AVAILABLE:
            import io

            buffer = io.BytesIO(data)
            return joblib.load(buffer)
        else:  # pickle
            try:
                return pickle.loads(data)
            except:
                # Try with decompression
                import gzip

                decompressed = gzip.decompress(data)
                return pickle.loads(decompressed)

    def _is_json_serializable(self, obj: Any) -> bool:
        """Check if object is JSON serializable."""
        try:
            json.dumps(obj, default=str)
            return True
        except:
            return False

    def _get_size(self, obj: Any) -> int:
        """Get approximate size of object in bytes.

        Parameters
        ----------
        obj : Any
            Object to measure.

        Returns:
        -------
        size : int
            Size in bytes.
        """
        try:
            return len(pickle.dumps(obj, protocol=4))
        except:
            return 0

    def clear(self, backend: str | None = None):
        """Clear cache.

        Parameters
        ----------
        backend : str, optional
            Specific backend to clear: 'memory', 'redis', 'disk', or None for all.
        """
        if backend in [None, "memory"]:
            self.memory_cache.clear()
            logger.info("Memory cache cleared")

        if backend in [None, "redis"] and self.redis_available:
            try:
                self.redis_client.flushdb()
                logger.info("Redis cache cleared")
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")

        if backend in [None, "disk"]:
            try:
                if DISKCACHE_AVAILABLE:
                    self.disk_cache.clear()
                else:
                    self.disk_cache.clear()
                logger.info("Disk cache cleared")
            except Exception as e:
                logger.warning(f"Disk clear error: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
        -------
        stats : dict
            Cache statistics.
        """
        if not self.enable_stats:
            return {}

        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0

        return {
            "total_requests": total_requests,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate,
            "memory_hits": self.stats["memory_hits"],
            "redis_hits": self.stats["redis_hits"],
            "disk_hits": self.stats["disk_hits"],
            "cache_writes": self.stats["cache_writes"],
            "time_saved_seconds": self.stats["total_compute_time_saved"],
            "memory_cache_size": len(self.memory_cache),
            "memory_cache_bytes": self.memory_cache.current_size
            if hasattr(self.memory_cache, "current_size")
            else 0,
        }

    def print_stats(self):
        """Print cache statistics."""
        stats = self.get_stats()
        if not stats:
            print("Statistics not enabled")
            return

        print("\n" + "=" * 50)
        print("CACHE STATISTICS")
        print("=" * 50)
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Hits: {stats['hits']} | Misses: {stats['misses']}")
        print(f"Hit Rate: {stats['hit_rate']:.1%}")
        print("\nHits by Cache Level:")
        print(f"  Memory: {stats['memory_hits']}")
        print(f"  Redis: {stats['redis_hits']}")
        print(f"  Disk: {stats['disk_hits']}")
        print(f"\nCache Writes: {stats['cache_writes']}")
        print(f"Time Saved: {stats['time_saved_seconds']:.2f} seconds")
        print(
            f"Memory Cache: {stats['memory_cache_size']} items "
            f"({stats['memory_cache_bytes'] / 1024 / 1024:.1f} MB)"
        )
        print("=" * 50)


class SimpleFileCache:
    """Simple file-based cache fallback when DiskCache is not available."""

    def __init__(self, cache_dir: Path):
        """Initialize simple file cache.

        Parameters
        ----------
        cache_dir : Path
            Directory for cache files.
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> dict:
        """Load cache index."""
        if self.index_file.exists():
            with open(self.index_file) as f:
                return json.load(f)
        return {}

    def _save_index(self):
        """Save cache index."""
        with open(self.index_file, "w") as f:
            json.dump(self.index, f)

    def get(self, key: str) -> Any:
        """Get value from cache."""
        if key in self.index:
            file_path = self.cache_dir / f"{key}.pkl"
            if file_path.exists():
                # Check expiration
                if "expire" in self.index[key]:
                    if time.time() > self.index[key]["expire"]:
                        # Expired
                        self.delete(key)
                        return None

                with open(file_path, "rb") as f:
                    return pickle.load(f)
        return None

    def set(self, key: str, value: Any, ttl: int | None = None):
        """Set value in cache."""
        file_path = self.cache_dir / f"{key}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(value, f, protocol=4)

        entry: dict[str, Any] = {"created": time.time(), "file": str(file_path)}
        if ttl is not None:
            entry["expire"] = time.time() + ttl
        self.index[key] = entry
        self._save_index()

    def delete(self, key: str):
        """Delete key from cache."""
        if key in self.index:
            file_path = Path(self.index[key]["file"])
            if file_path.exists():
                file_path.unlink()
            del self.index[key]
            self._save_index()

    def clear(self):
        """Clear all cache files."""
        for key in list(self.index.keys()):
            self.delete(key)


# Convenience functions
_global_cache = None


def get_cache(
    cache_dir: str = ".cache", use_redis: bool = False, **kwargs
) -> SmartCache:
    """Get or create global cache instance.

    Parameters
    ----------
    cache_dir : str
        Cache directory.
    use_redis : bool
        Whether to use Redis.
    **kwargs
        Additional arguments for SmartCache.

    Returns:
    -------
    cache : SmartCache
        Cache instance.
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = SmartCache(cache_dir, use_redis, **kwargs)
    return _global_cache


def auto_cache(
    cache: SmartCache,
    prefix: str = "auto",
    ttl: int = 3600,
    exclude_args: list[int] | None = None,
):
    """Decorator to cache function results automatically.

    Parameters
    ----------
    cache : SmartCache
        Cache instance to use.
    prefix : str, default="auto"
        Cache key prefix.
    ttl : int, default=3600
        Time-to-live in seconds.
    exclude_args : list[int] | None, default=None
        Positional argument indices to exclude from cache key generation.
    """
    exclude_args = exclude_args or []

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            filtered_args = tuple(
                arg for idx, arg in enumerate(args) if idx not in exclude_args
            )
            cache_key = cache._generate_key(
                prefix, func.__name__, filtered_args, kwargs
            )

            cached = cache.get(cache_key)
            if cached is not None:
                return cached

            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            return result

        return wrapper

    return decorator


# Usage examples
if __name__ == "__main__":
    # Example 1: Basic caching
    cache = SmartCache(use_redis=False)

    @cache.cache_dataframe("analysis")
    def expensive_analysis(df: pd.DataFrame, method: str = "full") -> pd.DataFrame:
        """Expensive analysis that benefits from caching."""
        import time

        print(f"Computing expensive analysis with method={method}...")
        time.sleep(2)  # Simulate expensive computation
        return df.describe()

    # Example 2: Caching with custom TTL
    @cache.cache_computation("model", ttl=7200)
    def train_model(data: np.ndarray, params: dict) -> dict:
        """Train a model with caching."""
        import time

        print("Training model...")
        time.sleep(3)
        return {"accuracy": 0.95, "params": params}

    # Test caching
    print("\nTesting DataFrame caching:")
    test_df = pd.DataFrame(np.random.randn(100, 5), columns=list("ABCDE"))

    # First call - cache miss
    result1 = expensive_analysis(test_df, method="full")
    print("First call completed (cache miss)")

    # Second call - cache hit
    result2 = expensive_analysis(test_df, method="full")
    print("Second call completed (cache hit)")

    # Print statistics
    cache.print_stats()

    # Clear cache
    cache.clear()
    print("\nCache cleared")
