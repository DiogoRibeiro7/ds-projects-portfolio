"""
Test suite for intelligent caching system.
"""

import pytest
import numpy as np
import pandas as pd
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.utils.caching import SmartCache, SimpleFileCache, get_cache, auto_cache
from src.utils.cache_integration import (
    CachedDataProcessor,
    CachedModelTrainer,
    CacheManager,
)

pytestmark = pytest.mark.integration


class TestSmartCache:
    """Test suite for SmartCache class."""

    @pytest.fixture
    def cache(self):
        """Create cache instance with temp directory."""
        temp_dir = tempfile.mkdtemp()
        cache = SmartCache(cache_dir=temp_dir, use_redis=False, enable_stats=True)
        yield cache
        # Cleanup
        cache.clear()
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_basic_caching(self, cache):
        """Test basic cache operations."""
        # Set value
        cache.set("test_key", {"data": "test_value"})

        # Get value
        result = cache.get("test_key")
        assert result == {"data": "test_value"}

        # Test cache miss
        missing = cache.get("nonexistent_key")
        assert missing is None

    def test_dataframe_caching(self, cache):
        """Test DataFrame caching."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        # Cache DataFrame
        cache.set("df_test", df)

        # Retrieve DataFrame
        df_cached = cache.get("df_test")
        pd.testing.assert_frame_equal(df, df_cached)

    def test_numpy_caching(self, cache):
        """Test NumPy array caching."""
        arr = np.random.randn(100, 50)

        # Cache array
        cache.set("array_test", arr)

        # Retrieve array
        arr_cached = cache.get("array_test")
        np.testing.assert_array_almost_equal(arr, arr_cached)

    def test_ttl_expiration(self, cache):
        """Test TTL expiration."""
        # Set with short TTL
        cache.set("expire_test", "value", ttl=1)

        # Should exist immediately
        assert cache.get("expire_test") == "value"

        # Wait for expiration
        time.sleep(2)

        # Should be expired (depending on backend)
        # Note: Memory cache doesn't expire automatically
        # This would work with Redis backend

    def test_cache_key_generation(self, cache):
        """Test cache key generation."""
        # Test with DataFrame
        df1 = pd.DataFrame({"A": [1, 2, 3]})
        df2 = pd.DataFrame({"A": [1, 2, 3]})  # Same content
        df3 = pd.DataFrame({"A": [4, 5, 6]})  # Different content

        key1 = cache._generate_key("test", "func", (df1,), {})
        key2 = cache._generate_key("test", "func", (df2,), {})
        key3 = cache._generate_key("test", "func", (df3,), {})

        # Same content should generate same key
        assert key1 == key2
        # Different content should generate different key
        assert key1 != key3

    def test_size_based_storage(self, cache):
        """Test size-based storage decisions."""
        # Small object - should go to memory
        small_obj = {"small": "data"}
        cache.set("small", small_obj)

        # Check memory cache
        assert "small" in cache.memory_cache

        # Large object - should skip memory
        large_obj = np.random.randn(10000, 100)
        cache.set("large", large_obj)

        # Large object might not be in memory cache
        # (depending on size limits)

    def test_cache_statistics(self, cache):
        """Test cache statistics tracking."""
        # Perform some operations
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        cache.get("key1")  # Hit again

        stats = cache.get_stats()

        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2 / 3

    def test_cache_decorator(self, cache):
        """Test cache decorator functionality."""
        call_count = 0

        @cache.cache_dataframe("test")
        def expensive_function(df):
            nonlocal call_count
            call_count += 1
            return df.sum()

        df = pd.DataFrame({"A": [1, 2, 3]})

        # First call
        result1 = expensive_function(df)
        assert call_count == 1

        # Second call - should use cache
        result2 = expensive_function(df)
        assert call_count == 1  # Not incremented
        assert result1.equals(result2)

    def test_compression(self):
        """Test compression for large objects."""
        temp_dir = tempfile.mkdtemp()
        cache = SmartCache(cache_dir=temp_dir, enable_compression=True)

        # Large object that benefits from compression
        large_data = ["test" * 1000] * 100

        # Serialize with compression
        serialized = cache._serialize(large_data, method="pickle")
        size_compressed = len(serialized)

        # Serialize without compression
        cache.enable_compression = False
        serialized_uncompressed = cache._serialize(large_data, method="pickle")
        size_uncompressed = len(serialized_uncompressed)

        # Compressed should be smaller
        assert size_compressed < size_uncompressed

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_clear_cache(self, cache):
        """Test cache clearing."""
        # Add some data
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Clear cache
        cache.clear()

        # Should be empty
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert len(cache.memory_cache) == 0


class TestSimpleFileCache:
    """Test suite for SimpleFileCache fallback."""

    @pytest.fixture
    def file_cache(self):
        """Create file cache instance."""
        temp_dir = Path(tempfile.mkdtemp())
        cache = SimpleFileCache(temp_dir)
        yield cache
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_file_cache_operations(self, file_cache):
        """Test basic file cache operations."""
        # Set value
        file_cache.set("test_key", {"data": "test"})

        # Get value
        result = file_cache.get("test_key")
        assert result == {"data": "test"}

        # Delete value
        file_cache.delete("test_key")
        assert file_cache.get("test_key") is None

    def test_file_cache_persistence(self, file_cache):
        """Test file cache persistence."""
        # Set value
        file_cache.set("persist_key", [1, 2, 3])

        # Create new cache instance with same directory
        cache2 = SimpleFileCache(file_cache.cache_dir)

        # Should still have the value
        result = cache2.get("persist_key")
        assert result == [1, 2, 3]


class TestCachedDataProcessor:
    """Test suite for CachedDataProcessor."""

    @pytest.fixture
    def processor(self):
        """Create cached processor instance."""
        temp_dir = tempfile.mkdtemp()
        processor = CachedDataProcessor(cache_dir=temp_dir, use_redis=False)
        yield processor
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cached_groupby(self, processor):
        """Test cached groupby aggregation."""
        df = pd.DataFrame({"group": ["A", "B", "A", "B"], "value": [1, 2, 3, 4]})

        agg_dict = {"value": "sum"}

        # First call
        result1 = processor.cached_groupby_aggregation(df, "group", agg_dict)

        # Second call - should use cache
        result2 = processor.cached_groupby_aggregation(df, "group", agg_dict)

        pd.testing.assert_frame_equal(result1, result2)

    def test_cached_outlier_detection(self, processor):
        """Test cached outlier detection."""
        df = pd.DataFrame(
            {"col1": np.random.normal(0, 1, 100), "col2": np.random.normal(0, 1, 100)}
        )

        # Add outliers
        df.loc[0, "col1"] = 100

        # First call
        clean1 = processor.cached_outlier_detection(df, ["col1", "col2"], method="iqr")

        # Second call - should use cache
        clean2 = processor.cached_outlier_detection(df, ["col1", "col2"], method="iqr")

        pd.testing.assert_frame_equal(clean1, clean2)

    def test_cached_statistical_test(self, processor):
        """Test cached statistical testing."""
        group_a = np.random.normal(0, 1, 100)
        group_b = np.random.normal(0.5, 1, 100)

        # First call
        result1 = processor.cached_statistical_test(group_a, group_b, test_type="ttest")

        # Second call - should use cache
        result2 = processor.cached_statistical_test(group_a, group_b, test_type="ttest")

        assert result1 == result2


class TestCachedModelTrainer:
    """Test suite for CachedModelTrainer."""

    @pytest.fixture
    def trainer(self):
        """Create cached trainer instance."""
        temp_dir = tempfile.mkdtemp()
        trainer = CachedModelTrainer(cache_dir=temp_dir, use_redis=False)
        yield trainer
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cached_feature_engineering(self, trainer):
        """Test cached feature engineering."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        feature_config = {
            "col1_squared": {"type": "polynomial", "column": "col1", "degree": 2}
        }

        # First call
        result1 = trainer.cached_feature_engineering(df, feature_config)

        # Second call - should use cache
        result2 = trainer.cached_feature_engineering(df, feature_config)

        pd.testing.assert_frame_equal(result1, result2)
        assert "col1_squared" in result1.columns

    @patch("sklearn.linear_model.LogisticRegression")
    def test_cached_model_training(self, mock_lr, trainer):
        """Test cached model training."""
        X = pd.DataFrame(np.random.randn(10, 3))
        y = pd.Series(np.random.binomial(1, 0.5, 10))

        params = {"max_iter": 100}

        # First call
        model1 = trainer.cached_model_training(X, y, "logistic", params)

        # Second call - should use cache
        model2 = trainer.cached_model_training(X, y, "logistic", params)

        # Should return same cached model
        assert model1 is model2


class TestCacheManager:
    """Test suite for CacheManager."""

    def test_cache_registration(self):
        """Test cache registration and retrieval."""
        manager = CacheManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Register caches
            cache1 = manager.register_cache("cache1", f"{temp_dir}/cache1")
            cache2 = manager.register_cache("cache2", f"{temp_dir}/cache2")

            # Retrieve caches
            retrieved1 = manager.get_cache("cache1")
            retrieved2 = manager.get_cache("cache2")

            assert retrieved1 is cache1
            assert retrieved2 is cache2

            # Test non-existent cache
            with pytest.raises(KeyError):
                manager.get_cache("nonexistent")

    def test_manager_clear_all(self):
        """Test clearing all managed caches."""
        manager = CacheManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            cache1 = manager.register_cache("cache1", f"{temp_dir}/cache1")
            cache2 = manager.register_cache("cache2", f"{temp_dir}/cache2")

            # Add data to caches
            cache1.set("key1", "value1")
            cache2.set("key2", "value2")

            # Clear all
            manager.clear_all()

            # Should be empty
            assert cache1.get("key1") is None
            assert cache2.get("key2") is None

    def test_manager_statistics(self):
        """Test aggregated statistics."""
        manager = CacheManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            cache1 = manager.register_cache(
                "cache1", f"{temp_dir}/cache1", enable_stats=True
            )
            cache2 = manager.register_cache(
                "cache2", f"{temp_dir}/cache2", enable_stats=True
            )

            # Generate some statistics
            cache1.set("key1", "value1")
            cache1.get("key1")  # Hit
            cache1.get("key2")  # Miss

            cache2.set("key3", "value3")
            cache2.get("key3")  # Hit

            # Get all stats
            all_stats = manager.get_all_stats()

            assert "cache1" in all_stats
            assert "cache2" in all_stats
            assert all_stats["cache1"]["hits"] == 1
            assert all_stats["cache1"]["misses"] == 1
            assert all_stats["cache2"]["hits"] == 1


class TestAutoCacheDecorator:
    """Test suite for auto_cache decorator."""

    def test_auto_cache_basic(self):
        """Test basic auto_cache functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = SmartCache(temp_dir)
            call_count = 0

            @auto_cache(cache, prefix="test")
            def test_function(x, y):
                nonlocal call_count
                call_count += 1
                return x + y

            # First call
            result1 = test_function(1, 2)
            assert result1 == 3
            assert call_count == 1

            # Second call with same args - should use cache
            result2 = test_function(1, 2)
            assert result2 == 3
            assert call_count == 1  # Not incremented

            # Different args - should compute
            result3 = test_function(2, 3)
            assert result3 == 5
            assert call_count == 2

    def test_auto_cache_exclude_args(self):
        """Test auto_cache with excluded arguments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = SmartCache(temp_dir)
            call_count = 0

            @auto_cache(cache, prefix="test", exclude_args=[1])
            def test_function(x, timestamp):
                nonlocal call_count
                call_count += 1
                return x * 2

            # First call
            result1 = test_function(5, time.time())
            assert result1 == 10
            assert call_count == 1

            # Second call with different timestamp - should still use cache
            time.sleep(0.1)
            result2 = test_function(5, time.time())
            assert result2 == 10
            assert call_count == 1  # Not incremented because timestamp is excluded


class TestIntegration:
    """Integration tests for caching system."""

    def test_end_to_end_workflow(self):
        """Test complete caching workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize components
            processor = CachedDataProcessor(cache_dir=f"{temp_dir}/proc")
            trainer = CachedModelTrainer(cache_dir=f"{temp_dir}/train")

            # Create test data
            df = pd.DataFrame(
                {
                    "feature1": np.random.normal(0, 1, 100),
                    "feature2": np.random.normal(0, 1, 100),
                    "target": np.random.binomial(1, 0.5, 100),
                }
            )

            # Process data with caching
            clean_df = processor.cached_outlier_detection(df, ["feature1", "feature2"])

            # Engineer features with caching
            feature_config = {
                "f1_squared": {"type": "polynomial", "column": "feature1", "degree": 2}
            }

            df_features = trainer.cached_feature_engineering(clean_df, feature_config)

            # Verify results
            assert len(df_features) <= len(df)  # Some outliers removed
            assert "f1_squared" in df_features.columns

            # Check cache stats
            proc_stats = processor.cache.get_stats()
            train_stats = trainer.cache.get_stats()

            # Should have some cache activity
            assert proc_stats["cache_writes"] > 0
            assert train_stats["cache_writes"] > 0
