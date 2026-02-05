"""
Integration utilities for intelligent caching with data processing operations.

This module provides seamless integration of the caching system with
expensive data processing and analysis operations.
"""

import numpy as np
import pandas as pd
import functools
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path

from .caching import SmartCache, get_cache

# Configure logging
logger = logging.getLogger(__name__)


class CachedDataProcessor:
    """Data processor with integrated caching for expensive operations."""

    def __init__(
        self,
        cache_dir: str = ".cache/processing",
        use_redis: bool = False,
        cache_ttl: int = 3600,
    ):
        """Initialize cached data processor.

        Parameters
        ----------
        cache_dir : str
            Directory for cache storage.
        use_redis : bool
            Whether to use Redis backend.
        cache_ttl : int
            Default cache TTL in seconds.
        """
        self.cache = SmartCache(
            cache_dir=cache_dir, use_redis=use_redis, enable_stats=True
        )
        self.cache_ttl = cache_ttl

    @functools.lru_cache(maxsize=128)
    def _get_dataframe_hash(self, df: pd.DataFrame) -> str:
        """Get hash for DataFrame for cache key generation."""
        import hashlib

        # Combine shape, columns, and sample data
        shape_str = str(df.shape)
        cols_str = str(list(df.columns))
        sample_str = str(df.head(100).values.tobytes()[:1000])

        combined = f"{shape_str}_{cols_str}_{sample_str}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def cached_groupby_aggregation(
        self,
        df: pd.DataFrame,
        group_cols: Union[str, List[str]],
        agg_dict: Dict[str, Union[str, List[str]]],
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Perform cached groupby aggregation.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        group_cols : str or list of str
            Columns to group by.
        agg_dict : dict
            Aggregation specifications.
        use_cache : bool
            Whether to use caching.

        Returns
        -------
        result : pd.DataFrame
            Aggregated results.
        """
        if not use_cache:
            return df.groupby(group_cols).agg(agg_dict)

        @self.cache.cache_dataframe("groupby", ttl=self.cache_ttl)
        def _perform_aggregation(df_hash: str) -> pd.DataFrame:
            logger.info(f"Computing groupby aggregation...")
            return df.groupby(group_cols).agg(agg_dict)

        # Use DataFrame hash as cache key component
        df_hash = self._get_dataframe_hash(df)
        return _perform_aggregation(df_hash)

    def cached_outlier_detection(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "iqr",
        threshold: float = 1.5,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Perform cached outlier detection.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        columns : list of str
            Columns to check for outliers.
        method : str
            Outlier detection method.
        threshold : float
            Detection threshold.
        use_cache : bool
            Whether to use caching.

        Returns
        -------
        df_clean : pd.DataFrame
            DataFrame with outliers removed.
        """
        if not use_cache:
            return self._detect_outliers(df, columns, method, threshold)

        @self.cache.cache_dataframe("outliers", ttl=self.cache_ttl)
        def _cached_outlier_detection(df_hash: str) -> pd.DataFrame:
            logger.info(f"Computing outlier detection...")
            return self._detect_outliers(df, columns, method, threshold)

        df_hash = self._get_dataframe_hash(df)
        return _cached_outlier_detection(df_hash)

    def _detect_outliers(
        self, df: pd.DataFrame, columns: List[str], method: str, threshold: float
    ) -> pd.DataFrame:
        """Perform actual outlier detection."""
        df_clean = df.copy()

        for col in columns:
            if col not in df_clean.columns:
                continue

            if method == "iqr":
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
            elif method == "zscore":
                from scipy import stats

                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                df_clean = df_clean[z_scores < threshold]

        return df_clean

    def cached_statistical_test(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        test_type: str = "ttest",
        use_cache: bool = True,
    ) -> Dict[str, float]:
        """Perform cached statistical testing.

        Parameters
        ----------
        group_a : np.ndarray
            First group data.
        group_b : np.ndarray
            Second group data.
        test_type : str
            Type of test to perform.
        use_cache : bool
            Whether to use caching.

        Returns
        -------
        results : dict
            Test results.
        """
        if not use_cache:
            return self._perform_statistical_test(group_a, group_b, test_type)

        @self.cache.cache_computation("stats", ttl=self.cache_ttl * 2)
        def _cached_test(a_hash: str, b_hash: str, test: str) -> Dict[str, float]:
            logger.info(f"Computing statistical test: {test}")
            return self._perform_statistical_test(group_a, group_b, test)

        # Generate hashes for arrays
        a_hash = hashlib.md5(group_a.tobytes()).hexdigest()[:8]
        b_hash = hashlib.md5(group_b.tobytes()).hexdigest()[:8]

        return _cached_test(a_hash, b_hash, test_type)

    def _perform_statistical_test(
        self, group_a: np.ndarray, group_b: np.ndarray, test_type: str
    ) -> Dict[str, float]:
        """Perform actual statistical test."""
        from scipy import stats

        if test_type == "ttest":
            statistic, pvalue = stats.ttest_ind(group_a, group_b)
        elif test_type == "mannwhitney":
            statistic, pvalue = stats.mannwhitneyu(group_a, group_b)
        elif test_type == "ks":
            statistic, pvalue = stats.ks_2samp(group_a, group_b)
        else:
            raise ValueError(f"Unknown test type: {test_type}")

        return {
            "statistic": statistic,
            "pvalue": pvalue,
            "test_type": test_type,
            "sample_size_a": len(group_a),
            "sample_size_b": len(group_b),
        }


class CachedModelTrainer:
    """Model training with intelligent caching of results and intermediate steps."""

    def __init__(self, cache_dir: str = ".cache/models", use_redis: bool = False):
        """Initialize cached model trainer.

        Parameters
        ----------
        cache_dir : str
            Directory for cache storage.
        use_redis : bool
            Whether to use Redis backend.
        """
        self.cache = SmartCache(
            cache_dir=cache_dir,
            use_redis=use_redis,
            enable_compression=True,
            enable_stats=True,
        )

    def cached_feature_engineering(
        self, df: pd.DataFrame, feature_config: Dict[str, Any], use_cache: bool = True
    ) -> pd.DataFrame:
        """Perform cached feature engineering.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        feature_config : dict
            Feature engineering configuration.
        use_cache : bool
            Whether to use caching.

        Returns
        -------
        df_features : pd.DataFrame
            DataFrame with engineered features.
        """

        @self.cache.cache_dataframe("features", ttl=7200)
        def _engineer_features(df_hash: str, config_hash: str) -> pd.DataFrame:
            logger.info("Engineering features...")
            df_feat = df.copy()

            for feature_name, config in feature_config.items():
                if config["type"] == "polynomial":
                    df_feat[feature_name] = (
                        df_feat[config["column"]] ** config["degree"]
                    )
                elif config["type"] == "interaction":
                    df_feat[feature_name] = (
                        df_feat[config["col1"]] * df_feat[config["col2"]]
                    )
                elif config["type"] == "ratio":
                    df_feat[feature_name] = df_feat[config["numerator"]] / df_feat[
                        config["denominator"]
                    ].replace(0, np.nan)
                elif config["type"] == "log":
                    df_feat[feature_name] = np.log1p(df_feat[config["column"]])

            return df_feat

        if not use_cache:
            return _engineer_features("no_cache", "no_cache")

        # Generate cache keys
        import hashlib
        import json

        df_hash = hashlib.md5(f"{df.shape}_{list(df.columns)}".encode()).hexdigest()[
            :16
        ]

        config_hash = hashlib.md5(
            json.dumps(feature_config, sort_keys=True).encode()
        ).hexdigest()[:16]

        return _engineer_features(df_hash, config_hash)

    def cached_model_training(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str,
        params: Dict[str, Any],
        use_cache: bool = True,
    ) -> Any:
        """Train model with caching.

        Parameters
        ----------
        X : pd.DataFrame
            Features.
        y : pd.Series
            Target.
        model_type : str
            Type of model to train.
        params : dict
            Model parameters.
        use_cache : bool
            Whether to use caching.

        Returns
        -------
        model : Any
            Trained model.
        """

        @self.cache.cache_computation("model", ttl=86400, serialize_method="joblib")
        def _train_model(
            X_hash: str, y_hash: str, model_type: str, params_str: str
        ) -> Any:
            logger.info(f"Training {model_type} model...")

            if model_type == "logistic":
                from sklearn.linear_model import LogisticRegression

                model = LogisticRegression(**params)
            elif model_type == "rf":
                from sklearn.ensemble import RandomForestClassifier

                model = RandomForestClassifier(**params)
            elif model_type == "xgb":
                import xgboost as xgb

                model = xgb.XGBClassifier(**params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Simulate training time
            time.sleep(2)
            model.fit(X, y)
            return model

        if not use_cache:
            return _train_model("no_cache", "no_cache", model_type, str(params))

        # Generate cache keys
        import hashlib
        import json

        X_hash = hashlib.md5(
            f"{X.shape}_{X.values.tobytes()[:1000]}".encode()
        ).hexdigest()[:16]

        y_hash = hashlib.md5(y.values.tobytes()[:1000]).hexdigest()[:16]

        params_str = json.dumps(params, sort_keys=True)

        return _train_model(X_hash, y_hash, model_type, params_str)

    def cached_cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model,
        cv_folds: int = 5,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Perform cached cross-validation.

        Parameters
        ----------
        X : pd.DataFrame
            Features.
        y : pd.Series
            Target.
        model : Any
            Model to evaluate.
        cv_folds : int
            Number of CV folds.
        use_cache : bool
            Whether to use caching.

        Returns
        -------
        results : dict
            CV results.
        """

        @self.cache.cache_computation("cv", ttl=7200)
        def _perform_cv(
            X_hash: str, y_hash: str, model_str: str, folds: int
        ) -> Dict[str, Any]:
            logger.info(f"Performing {folds}-fold cross-validation...")
            from sklearn.model_selection import cross_val_score

            scores = cross_val_score(model, X, y, cv=folds, scoring="accuracy")

            return {
                "scores": scores.tolist(),
                "mean_score": scores.mean(),
                "std_score": scores.std(),
                "cv_folds": folds,
            }

        if not use_cache:
            return _perform_cv("no_cache", "no_cache", str(model), cv_folds)

        # Generate cache keys
        import hashlib

        X_hash = hashlib.md5(
            f"{X.shape}_{X.values.tobytes()[:1000]}".encode()
        ).hexdigest()[:16]

        y_hash = hashlib.md5(y.values.tobytes()[:1000]).hexdigest()[:16]

        model_str = str(model.__class__.__name__)

        return _perform_cv(X_hash, y_hash, model_str, cv_folds)


class CacheManager:
    """Centralized cache management for multiple caches."""

    def __init__(self):
        """Initialize cache manager."""
        self.caches = {}

    def register_cache(
        self, name: str, cache_dir: str, use_redis: bool = False, **kwargs
    ) -> SmartCache:
        """Register a new cache.

        Parameters
        ----------
        name : str
            Cache name.
        cache_dir : str
            Cache directory.
        use_redis : bool
            Whether to use Redis.
        **kwargs
            Additional cache parameters.

        Returns
        -------
        cache : SmartCache
            Registered cache instance.
        """
        cache = SmartCache(cache_dir, use_redis, **kwargs)
        self.caches[name] = cache
        logger.info(f"Registered cache: {name}")
        return cache

    def get_cache(self, name: str) -> SmartCache:
        """Get registered cache by name.

        Parameters
        ----------
        name : str
            Cache name.

        Returns
        -------
        cache : SmartCache
            Cache instance.
        """
        if name not in self.caches:
            raise KeyError(f"Cache '{name}' not found")
        return self.caches[name]

    def clear_all(self):
        """Clear all registered caches."""
        for name, cache in self.caches.items():
            cache.clear()
            logger.info(f"Cleared cache: {name}")

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all caches.

        Returns
        -------
        stats : dict
            Statistics for all caches.
        """
        stats = {}
        for name, cache in self.caches.items():
            stats[name] = cache.get_stats()
        return stats

    def print_all_stats(self):
        """Print statistics for all caches."""
        print("\n" + "=" * 60)
        print("CACHE MANAGER STATISTICS")
        print("=" * 60)

        all_stats = self.get_all_stats()

        for name, stats in all_stats.items():
            if stats:
                print(f"\n[{name}]")
                print(f"  Hit Rate: {stats.get('hit_rate', 0):.1%}")
                print(
                    f"  Hits: {stats.get('hits', 0)} | Misses: {stats.get('misses', 0)}"
                )
                print(f"  Time Saved: {stats.get('time_saved_seconds', 0):.2f}s")

        # Calculate totals
        total_hits = sum(s.get("hits", 0) for s in all_stats.values())
        total_misses = sum(s.get("misses", 0) for s in all_stats.values())
        total_time_saved = sum(
            s.get("time_saved_seconds", 0) for s in all_stats.values()
        )

        print(f"\nTOTALS:")
        print(f"  Total Hits: {total_hits}")
        print(f"  Total Misses: {total_misses}")
        print(f"  Total Time Saved: {total_time_saved:.2f}s")
        print("=" * 60)


# Decorator for automatic cache key generation
def auto_cache(
    cache: SmartCache,
    prefix: str = "auto",
    ttl: int = 3600,
    exclude_args: List[int] = None,
):
    """Decorator with automatic cache key generation.

    Parameters
    ----------
    cache : SmartCache
        Cache instance.
    prefix : str
        Cache key prefix.
    ttl : int
        Time-to-live.
    exclude_args : list of int
        Argument indices to exclude from cache key.

    Returns
    -------
    decorator : Callable
        Caching decorator.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Filter args for cache key
            cache_args = args
            if exclude_args:
                cache_args = tuple(
                    arg for i, arg in enumerate(args) if i not in exclude_args
                )

            # Generate cache key
            import hashlib
            import json

            key_data = {
                "func": func.__name__,
                "args": str(cache_args)[:100],
                "kwargs": json.dumps(kwargs, sort_keys=True, default=str)[:100],
            }

            cache_key = f"{prefix}_{hashlib.md5(str(key_data).encode()).hexdigest()}"

            # Try cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            return result

        return wrapper

    return decorator


if __name__ == "__main__":
    import hashlib

    # Example usage
    print("Testing Cached Data Processing")
    print("=" * 50)

    # Initialize cached processor
    processor = CachedDataProcessor(use_redis=False)

    # Create test data
    test_df = pd.DataFrame(
        {
            "group": np.random.choice(["A", "B", "C"], 1000),
            "value1": np.random.normal(100, 20, 1000),
            "value2": np.random.exponential(50, 1000),
            "value3": np.random.binomial(10, 0.3, 1000),
        }
    )

    # Test cached aggregation
    print("\n1. Testing Cached Aggregation:")
    agg_dict = {"value1": ["mean", "std"], "value2": "sum", "value3": ["min", "max"]}

    # First call - miss
    start = time.time()
    result1 = processor.cached_groupby_aggregation(test_df, "group", agg_dict)
    print(f"   First call: {time.time() - start:.3f}s (cache miss)")

    # Second call - hit
    start = time.time()
    result2 = processor.cached_groupby_aggregation(test_df, "group", agg_dict)
    print(f"   Second call: {time.time() - start:.3f}s (cache hit)")

    # Test cached outlier detection
    print("\n2. Testing Cached Outlier Detection:")
    columns = ["value1", "value2", "value3"]

    # First call - miss
    start = time.time()
    clean_df1 = processor.cached_outlier_detection(test_df, columns)
    print(f"   First call: {time.time() - start:.3f}s (cache miss)")
    print(f"   Original: {len(test_df)} rows, Clean: {len(clean_df1)} rows")

    # Second call - hit
    start = time.time()
    clean_df2 = processor.cached_outlier_detection(test_df, columns)
    print(f"   Second call: {time.time() - start:.3f}s (cache hit)")

    # Print cache statistics
    print("\n3. Cache Statistics:")
    processor.cache.print_stats()

    # Test model caching
    print("\n4. Testing Cached Model Training:")
    trainer = CachedModelTrainer(use_redis=False)

    # Create training data
    X = pd.DataFrame(np.random.randn(100, 5), columns=["f1", "f2", "f3", "f4", "f5"])
    y = pd.Series(np.random.binomial(1, 0.5, 100))

    # Test feature engineering caching
    feature_config = {
        "f1_squared": {"type": "polynomial", "column": "f1", "degree": 2},
        "f1_f2_interaction": {"type": "interaction", "col1": "f1", "col2": "f2"},
        "f3_log": {"type": "log", "column": "f3"},
    }

    X_feat1 = trainer.cached_feature_engineering(X, feature_config)
    print(f"   Features added: {list(feature_config.keys())}")

    # Test cache manager
    print("\n5. Testing Cache Manager:")
    manager = CacheManager()

    # Register multiple caches
    manager.register_cache("processing", ".cache/processing")
    manager.register_cache("models", ".cache/models")
    manager.register_cache("analysis", ".cache/analysis")

    # Print all stats
    manager.print_all_stats()

    print("\n✅ Cache integration testing complete!")
