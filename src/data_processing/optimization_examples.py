"""
Examples demonstrating the usage of optimized data processing operations.

This module shows how to leverage the high-performance data processing
utilities for maximum speed and efficiency.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

from .cleaning import OptimizedDataProcessor, get_optimized_processor
from .benchmark_performance import PerformanceBenchmark

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_parallel_outlier_detection():
    """Example: Parallel outlier detection across multiple columns."""
    logger.info("Running parallel outlier detection example...")

    # Create sample data
    np.random.seed(42)
    n_rows = 100000
    df = pd.DataFrame({
        'user_id': range(n_rows),
        'group': np.random.choice(['control', 'treatment'], n_rows),
        'metric_1': np.random.normal(100, 20, n_rows),
        'metric_2': np.random.exponential(50, n_rows),
        'metric_3': np.random.gamma(2, 2, n_rows),
    })

    # Add outliers
    outlier_idx = np.random.choice(n_rows, size=1000, replace=False)
    df.loc[outlier_idx, 'metric_1'] *= 10

    # Get optimized processor
    processor = get_optimized_processor(n_jobs=4)

    # Parallel outlier detection
    metric_cols = ['metric_1', 'metric_2', 'metric_3']
    df_clean = processor.parallel_outlier_detection(
        df,
        columns=metric_cols,
        method='zscore',
        threshold=3.0
    )

    logger.info(f"Original rows: {len(df)}, Clean rows: {len(df_clean)}")
    logger.info(f"Outliers removed: {len(df) - len(df_clean)}")

    return df_clean


def example_optimized_cuped():
    """Example: Parallel CUPED variance reduction."""
    logger.info("Running optimized CUPED example...")

    # Create sample data with covariate
    np.random.seed(42)
    n_users = 50000

    df = pd.DataFrame({
        'user_id': range(n_users),
        'group': np.random.choice(['control', 'treatment'], n_users),
        'pre_experiment_value': np.random.normal(100, 30, n_users),
    })

    # Post-experiment value correlated with pre-experiment
    df['post_experiment_value'] = (
        df['pre_experiment_value'] * 0.7 +
        np.random.normal(30, 10, n_users)
    )

    # Add treatment effect
    treatment_mask = df['group'] == 'treatment'
    df.loc[treatment_mask, 'post_experiment_value'] += 5

    # Apply parallel CUPED
    processor = OptimizedDataProcessor(n_jobs=4)

    df_cuped = processor.parallel_apply_cuped(
        df,
        metric_col='post_experiment_value',
        covariate_col='pre_experiment_value',
        group_col='group'
    )

    # Compare variance
    var_before = df['post_experiment_value'].var()
    var_after = df_cuped['post_experiment_value_cuped'].var()
    variance_reduction = (var_before - var_after) / var_before

    logger.info(f"Variance before CUPED: {var_before:.2f}")
    logger.info(f"Variance after CUPED: {var_after:.2f}")
    logger.info(f"Variance reduction: {variance_reduction:.1%}")

    return df_cuped


def example_memory_efficient_aggregation():
    """Example: Memory-efficient aggregation for large datasets."""
    logger.info("Running memory-efficient aggregation example...")

    # Create large dataset
    n_rows = 1_000_000
    n_groups = 10000

    df = pd.DataFrame({
        'group_id': np.random.choice(range(n_groups), n_rows),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'value_1': np.random.normal(100, 20, n_rows),
        'value_2': np.random.exponential(50, n_rows),
        'value_3': np.random.binomial(100, 0.3, n_rows),
    })

    # Define aggregations
    agg_funcs = {
        'value_1': ['mean', 'std', 'min', 'max'],
        'value_2': ['sum', 'median'],
        'value_3': ['mean', 'sum']
    }

    # Use memory-efficient groupby
    processor = OptimizedDataProcessor(n_jobs=-1)

    result = processor.memory_efficient_groupby(
        df,
        group_col='group_id',
        agg_funcs=agg_funcs
    )

    logger.info(f"Aggregated {n_rows} rows into {len(result)} groups")
    logger.info(f"Result shape: {result.shape}")

    return result


def example_vectorized_feature_engineering():
    """Example: Vectorized feature engineering for speed."""
    logger.info("Running vectorized feature engineering example...")

    # Create sample data
    n_rows = 100000
    df = pd.DataFrame({
        'user_id': range(n_rows),
        'sessions': np.random.poisson(5, n_rows),
        'page_views': np.random.poisson(20, n_rows),
        'time_on_site': np.random.exponential(300, n_rows),
        'purchases': np.random.binomial(3, 0.2, n_rows),
        'revenue': np.random.exponential(50, n_rows),
    })

    # Feature engineering configuration
    feature_config = {
        'pages_per_session': {
            'operation': 'ratio',
            'numerator': 'page_views',
            'denominator': 'sessions'
        },
        'avg_time_per_page': {
            'operation': 'ratio',
            'numerator': 'time_on_site',
            'denominator': 'page_views'
        },
        'conversion_rate': {
            'operation': 'ratio',
            'numerator': 'purchases',
            'denominator': 'sessions'
        },
        'avg_order_value': {
            'operation': 'ratio',
            'numerator': 'revenue',
            'denominator': 'purchases'
        },
        'engagement_score': {
            'operation': 'complex',
            'expression': 'log(page_views + 1) * sqrt(time_on_site / 60)',
            'columns': ['page_views', 'time_on_site']
        },
        'revenue_squared': {
            'operation': 'polynomial',
            'column': 'revenue',
            'degree': 2
        },
        'session_bucket': {
            'operation': 'binning',
            'column': 'sessions',
            'bins': [0, 2, 5, 10, 20, 100]
        },
        'session_pageview_interaction': {
            'operation': 'interaction',
            'columns': ['sessions', 'page_views']
        }
    }

    # Apply vectorized feature engineering
    processor = OptimizedDataProcessor()
    df_features = processor.vectorized_feature_engineering(df, feature_config)

    logger.info(f"Created {len(feature_config)} new features")
    logger.info(f"New columns: {list(feature_config.keys())}")

    return df_features


def example_batch_experiment_processing():
    """Example: Batch processing multiple experiments in parallel."""
    logger.info("Running batch experiment processing example...")

    # Create multiple experiment datasets
    experiments = []
    for exp_id in range(10):
        np.random.seed(exp_id)
        n_users = np.random.randint(5000, 20000)

        df = pd.DataFrame({
            'user_id': range(n_users),
            'group': np.random.choice(['control', 'treatment'], n_users),
            'metric_1': np.random.normal(100 + exp_id, 20, n_users),
            'metric_2': np.random.exponential(50, n_users),
            'pre_value': np.random.normal(90, 25, n_users),
        })

        # Add treatment effect
        treatment_mask = df['group'] == 'treatment'
        df.loc[treatment_mask, 'metric_1'] += np.random.uniform(2, 10)

        experiments.append(df)

    # Processing configuration
    processing_config = {
        'remove_outliers': True,
        'metric_columns': ['metric_1', 'metric_2'],
        'outlier_method': 'zscore',
        'outlier_threshold': 3.0,
        'apply_cuped': True,
        'metric_column': 'metric_1',
        'covariate_column': 'pre_value',
        'group_column': 'group',
        'engineer_features': True,
        'feature_config': {
            'metric_ratio': {
                'operation': 'ratio',
                'numerator': 'metric_1',
                'denominator': 'metric_2'
            }
        }
    }

    # Batch process experiments
    processor = OptimizedDataProcessor(n_jobs=-1)
    processed_experiments = processor.batch_process_experiments(
        experiments,
        processing_config
    )

    logger.info(f"Processed {len(experiments)} experiments in parallel")
    for i, exp in enumerate(processed_experiments):
        logger.info(f"  Experiment {i}: {len(exp)} rows after processing")

    return processed_experiments


def example_memory_optimization():
    """Example: Memory optimization for DataFrames."""
    logger.info("Running memory optimization example...")

    # Create a large DataFrame with inefficient dtypes
    n_rows = 500000
    df = pd.DataFrame({
        'id': range(n_rows),  # int64 by default
        'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows),  # object
        'small_int': np.random.randint(0, 100, n_rows),  # int64
        'flag': np.random.randint(0, 2, n_rows),  # int64
        'score': np.random.uniform(0, 100, n_rows),  # float64
        'amount': np.random.uniform(0, 1000, n_rows),  # float64
        'text_id': [f'ID_{i:06d}' for i in range(n_rows)],  # object
    })

    # Check memory before
    mem_before = df.memory_usage(deep=True).sum() / 1024 / 1024

    # Optimize memory
    processor = OptimizedDataProcessor()
    df_optimized = processor.optimize_dataframe_dtypes(df)

    # Check memory after
    mem_after = df_optimized.memory_usage(deep=True).sum() / 1024 / 1024

    logger.info(f"Memory before: {mem_before:.2f} MB")
    logger.info(f"Memory after: {mem_after:.2f} MB")
    logger.info(f"Reduction: {(mem_before - mem_after) / mem_before:.1%}")

    # Show dtype changes
    logger.info("\nDtype changes:")
    for col in df.columns:
        if df[col].dtype != df_optimized[col].dtype:
            logger.info(f"  {col}: {df[col].dtype} → {df_optimized[col].dtype}")

    return df_optimized


def example_complete_pipeline():
    """Example: Complete optimized data processing pipeline."""
    logger.info("Running complete optimized pipeline example...")

    # Create realistic experiment data
    np.random.seed(42)
    n_users = 100000

    df = pd.DataFrame({
        'user_id': range(n_users),
        'group': np.random.choice(['control', 'treatment'], n_users, p=[0.5, 0.5]),
        'signup_date': pd.date_range('2024-01-01', periods=n_users, freq='min'),
        'pre_experiment_purchases': np.random.poisson(3, n_users),
        'pre_experiment_revenue': np.random.exponential(100, n_users),
        'experiment_purchases': np.random.poisson(3.5, n_users),
        'experiment_revenue': np.random.exponential(110, n_users),
        'sessions': np.random.poisson(10, n_users),
        'page_views': np.random.poisson(50, n_users),
    })

    # Add treatment effect
    treatment_mask = df['group'] == 'treatment'
    df.loc[treatment_mask, 'experiment_purchases'] = np.random.poisson(4, treatment_mask.sum())
    df.loc[treatment_mask, 'experiment_revenue'] *= 1.15

    # Initialize processor
    processor = OptimizedDataProcessor(n_jobs=-1)

    logger.info(f"Starting with {len(df)} users")

    # Step 1: Memory optimization
    df = processor.optimize_dataframe_dtypes(df)
    logger.info("✓ Memory optimized")

    # Step 2: Outlier removal
    metric_cols = ['experiment_purchases', 'experiment_revenue', 'sessions', 'page_views']
    df = processor.parallel_outlier_detection(df, metric_cols, threshold=4.0)
    logger.info(f"✓ Outliers removed, {len(df)} users remaining")

    # Step 3: CUPED variance reduction for purchases
    df = processor.parallel_apply_cuped(
        df,
        metric_col='experiment_purchases',
        covariate_col='pre_experiment_purchases',
        group_col='group'
    )
    logger.info("✓ CUPED applied to purchases")

    # Step 4: Feature engineering
    feature_config = {
        'conversion_rate': {
            'operation': 'ratio',
            'numerator': 'experiment_purchases',
            'denominator': 'sessions'
        },
        'avg_purchase_value': {
            'operation': 'ratio',
            'numerator': 'experiment_revenue',
            'denominator': 'experiment_purchases'
        },
        'engagement_score': {
            'operation': 'ratio',
            'numerator': 'page_views',
            'denominator': 'sessions'
        },
        'revenue_lift': {
            'operation': 'ratio',
            'numerator': 'experiment_revenue',
            'denominator': 'pre_experiment_revenue'
        }
    }

    df = processor.vectorized_feature_engineering(df, feature_config)
    logger.info("✓ Features engineered")

    # Step 5: Aggregated statistics
    agg_funcs = {
        'experiment_purchases_cuped': ['mean', 'std'],
        'experiment_revenue': ['mean', 'std'],
        'conversion_rate': 'mean',
        'avg_purchase_value': 'mean',
        'engagement_score': 'mean',
        'revenue_lift': 'mean'
    }

    results = processor.memory_efficient_groupby(df, 'group', agg_funcs)
    logger.info("✓ Results aggregated")

    # Display results
    logger.info("\nExperiment Results:")
    logger.info(results.round(3))

    # Calculate lift
    control_purchases = results.loc['control', ('experiment_purchases_cuped', 'mean')]
    treatment_purchases = results.loc['treatment', ('experiment_purchases_cuped', 'mean')]
    lift = (treatment_purchases - control_purchases) / control_purchases

    logger.info(f"\nPurchase Lift: {lift:.1%}")

    return df, results


def run_all_examples():
    """Run all optimization examples."""
    logger.info("=" * 60)
    logger.info("OPTIMIZED DATA PROCESSING EXAMPLES")
    logger.info("=" * 60)

    examples = [
        ("Parallel Outlier Detection", example_parallel_outlier_detection),
        ("Optimized CUPED", example_optimized_cuped),
        ("Memory-Efficient Aggregation", example_memory_efficient_aggregation),
        ("Vectorized Feature Engineering", example_vectorized_feature_engineering),
        ("Batch Experiment Processing", example_batch_experiment_processing),
        ("Memory Optimization", example_memory_optimization),
        ("Complete Pipeline", example_complete_pipeline),
    ]

    for name, func in examples:
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Example: {name}")
        logger.info('=' * 40)
        try:
            func()
        except Exception as e:
            logger.error(f"Error in {name}: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("All examples completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_all_examples()