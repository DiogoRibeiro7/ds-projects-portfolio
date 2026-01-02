"""
Optimization templates and code generators for creating optimized notebooks.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


class OptimizationTemplates:
    """Templates for common optimization patterns."""

    @staticmethod
    def get_import_cell() -> str:
        """Get optimized import statements."""
        return """# Optimized imports
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Performance monitoring
from performance_optimization.profiling_utils import PerformanceProfiler, MemoryOptimizer
from performance_optimization.data_optimization import (
    DataProcessingOptimizer, NumbaOptimizations, CachingOptimizer, VectorizedOperations
)
from performance_optimization.visualization_optimization import VisualizationOptimizer
from performance_optimization.io_optimization import IOOptimizer

# Initialize optimizers
profiler = PerformanceProfiler()
cache = CachingOptimizer()
io_optimizer = IOOptimizer()
viz_optimizer = VisualizationOptimizer()

# Set pandas options for better performance
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

# Enable parallel processing
import multiprocessing as mp
N_CORES = mp.cpu_count()
print(f"Available CPU cores: {N_CORES}")

# Check GPU availability
try:
    from performance_optimization.gpu_acceleration import get_gpu_info
    gpu_info = get_gpu_info()
    if gpu_info['gpu_available']:
        print(f"GPU available: {gpu_info['devices']}")
except:
    pass"""

    @staticmethod
    def get_data_loading_template() -> str:
        """Get optimized data loading template."""
        return """# Optimized data loading
def load_data_optimized(filepath: str, optimize_memory: bool = True) -> pd.DataFrame:
    \"\"\"
    Load data with automatic optimization.

    Args:
        filepath: Path to data file
        optimize_memory: Whether to optimize memory usage

    Returns:
        Optimized DataFrame
    \"\"\"
    print(f"Loading data from: {filepath}")

    # Detect file format and use appropriate loader
    file_ext = Path(filepath).suffix.lower()

    if file_ext == '.parquet':
        df = pd.read_parquet(filepath)
    elif file_ext == '.feather':
        df = pd.read_feather(filepath)
    elif file_ext == '.csv':
        # Use optimized CSV reading
        df = io_optimizer.read_csv_optimized(filepath)
    elif file_ext == '.pkl':
        df = pd.read_pickle(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

    # Optimize memory if requested
    if optimize_memory:
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        df = MemoryOptimizer.optimize_dataframe_dtypes(df, verbose=False)
        optimized_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"Memory optimization: {original_memory:.2f} MB → {optimized_memory:.2f} MB")
        print(f"Reduction: {(1 - optimized_memory/original_memory)*100:.1f}%")

    print(f"Data shape: {df.shape}")
    return df

# Usage example
# df = load_data_optimized('data/file.csv')"""

    @staticmethod
    def get_eda_template() -> str:
        """Get optimized EDA template."""
        return """# Optimized Exploratory Data Analysis
@cache.cache_result
def generate_eda_report(df: pd.DataFrame) -> Dict[str, Any]:
    \"\"\"
    Generate comprehensive EDA report with caching.
    \"\"\"
    report = {}

    # Basic statistics (vectorized)
    report['shape'] = df.shape
    report['dtypes'] = df.dtypes.value_counts().to_dict()
    report['missing'] = df.isnull().sum().to_dict()
    report['memory_mb'] = df.memory_usage(deep=True).sum() / 1024 / 1024

    # Numeric statistics (use describe for efficiency)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        report['numeric_stats'] = df[numeric_cols].describe().to_dict()

        # Correlation matrix (use Numba for large datasets)
        if len(numeric_cols) > 1 and len(df) > 1000:
            corr_matrix = NumbaOptimizations.fast_correlation_matrix(df[numeric_cols].values)
            report['correlation'] = pd.DataFrame(corr_matrix,
                                                index=numeric_cols,
                                                columns=numeric_cols)

    # Categorical statistics (vectorized)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        report['categorical_stats'] = {
            col: {'unique': df[col].nunique(),
                  'top_values': df[col].value_counts().head(10).to_dict()}
            for col in cat_cols
        }

    return report

# Generate and cache report
# eda_report = generate_eda_report(df)"""

    @staticmethod
    def get_feature_engineering_template() -> str:
        """Get optimized feature engineering template."""
        return """# Optimized Feature Engineering
def create_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"
    Create features using vectorized operations.
    \"\"\"
    df = df.copy()

    # Date features (vectorized extraction)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

    # Numeric features (vectorized operations)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Avoid creating features from ID columns
        if 'id' not in col.lower():
            # Log transformation for skewed features
            if df[col].min() > 0 and df[col].skew() > 1:
                df[f'{col}_log'] = np.log1p(df[col])

            # Binning (vectorized)
            df[f'{col}_quintile'] = pd.qcut(df[col], q=5, labels=False, duplicates='drop')

    # Categorical encoding (use category dtype for efficiency)
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].nunique() < 100:  # Only for low cardinality
            df[col] = df[col].astype('category')

    return df

# Apply feature engineering
# df_features = create_features_vectorized(df)"""

    @staticmethod
    def get_model_training_template() -> str:
        """Get optimized model training template."""
        return """# Optimized Model Training
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

def train_model_optimized(X: pd.DataFrame, y: pd.Series,
                          model_type: str = 'lightgbm',
                          use_gpu: bool = False) -> Dict[str, Any]:
    \"\"\"
    Train model with optimizations.

    Args:
        X: Features
        y: Target
        model_type: Type of model to use
        use_gpu: Whether to use GPU acceleration

    Returns:
        Dictionary with model and metrics
    \"\"\"
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() < 100 else None
    )

    # Optimize data types for training
    X_train = MemoryOptimizer.optimize_dataframe_dtypes(X_train, verbose=False)
    X_test = MemoryOptimizer.optimize_dataframe_dtypes(X_test, verbose=False)

    # Model selection with GPU support
    if model_type == 'lightgbm':
        import lightgbm as lgb
        params = {
            'objective': 'binary' if y.nunique() == 2 else 'multiclass',
            'metric': 'auc' if y.nunique() == 2 else 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'device': 'gpu' if use_gpu else 'cpu',
            'gpu_platform_id': 0 if use_gpu else -1,
            'gpu_device_id': 0 if use_gpu else -1
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test)

        with profiler.profile_block("model_training"):
            model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=100,
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(25)]
            )

        # Predictions
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        if y.nunique() == 2:
            y_pred_class = (y_pred > 0.5).astype(int)
        else:
            y_pred_class = y_pred.argmax(axis=1)

    elif model_type == 'xgboost' and use_gpu:
        import xgboost as xgb
        model = xgb.XGBClassifier(
            tree_method='gpu_hist' if use_gpu else 'hist',
            gpu_id=0 if use_gpu else -1,
            n_estimators=100,
            early_stopping_rounds=10
        )

        with profiler.profile_block("model_training"):
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred_class = model.predict(X_test)
        y_pred = model.predict_proba(X_test)[:, 1] if y.nunique() == 2 else model.predict_proba(X_test)

    else:
        # Fallback to sklearn
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

        with profiler.profile_block("model_training"):
            model.fit(X_train, y_train)

        y_pred_class = model.predict(X_test)
        y_pred = model.predict_proba(X_test)[:, 1] if y.nunique() == 2 else model.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_class)
    if y.nunique() == 2:
        auc = roc_auc_score(y_test, y_pred)
    else:
        auc = None

    return {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'auc': auc,
        'training_time': profiler.get_profile_stats()
    }

# Train model
# results = train_model_optimized(X, y, use_gpu=True)"""

    @staticmethod
    def get_visualization_template() -> str:
        """Get optimized visualization template."""
        return """# Optimized Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def create_optimized_visualizations(df: pd.DataFrame, sample_size: int = 10000):
    \"\"\"
    Create visualizations with optimization for large datasets.
    \"\"\"
    # Sample data if too large
    if len(df) > sample_size:
        df_viz = viz_optimizer.sample_for_visualization(df, sample_size=sample_size)
        print(f"Sampled {len(df_viz)} from {len(df)} records for visualization")
    else:
        df_viz = df

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))

    # 1. Distribution plots (use histograms for large data)
    numeric_cols = df_viz.select_dtypes(include=[np.number]).columns[:4]
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(2, 2, i)
        if len(df_viz) > 5000:
            # Use histogram for large datasets (faster)
            plt.hist(df_viz[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
        else:
            # Use KDE for smaller datasets (more detail)
            df_viz[col].plot(kind='density')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency' if len(df_viz) > 5000 else 'Density')

    plt.tight_layout()
    plt.show()

    # 2. Interactive scatter plot with sampling
    if len(numeric_cols) >= 2:
        # Use adaptive sampling for scatter plots
        df_scatter = viz_optimizer.adaptive_sampling(
            df, numeric_cols[0], numeric_cols[1], max_points=5000
        )

        fig_scatter = px.scatter(
            df_scatter,
            x=numeric_cols[0],
            y=numeric_cols[1],
            title=f'Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}',
            opacity=0.6
        )
        fig_scatter.update_layout(
            title=f'Scatter Plot ({len(df_scatter)} points sampled from {len(df)})'
        )
        fig_scatter.show()

    # 3. Correlation heatmap (use clustering for large number of features)
    if len(numeric_cols) > 1:
        # Calculate correlation efficiently
        if len(numeric_cols) > 50:
            # Sample features for very wide datasets
            selected_cols = numeric_cols[:30]
        else:
            selected_cols = numeric_cols

        corr_matrix = df_viz[selected_cols].corr()

        # Create heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        fig_heatmap.update_layout(
            title='Correlation Heatmap',
            width=800,
            height=800
        )
        fig_heatmap.show()

    # 4. Time series with decimation
    if 'date' in df.columns:
        df_time = df.set_index('date').sort_index()

        # Decimate for plotting if too many points
        if len(df_time) > 10000:
            # Use resampling for time series
            df_time_plot = df_time.resample('D').mean()  # Daily averages
        else:
            df_time_plot = df_time

        fig_ts = go.Figure()
        for col in numeric_cols[:3]:  # Plot first 3 numeric columns
            if col in df_time_plot.columns:
                fig_ts.add_trace(go.Scatter(
                    x=df_time_plot.index,
                    y=df_time_plot[col],
                    mode='lines',
                    name=col
                ))

        fig_ts.update_layout(
            title='Time Series Plot',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x'
        )
        fig_ts.show()

# Create visualizations
# create_optimized_visualizations(df)"""

    @staticmethod
    def get_batch_processing_template() -> str:
        """Get batch processing template."""
        return """# Batch Processing Template
from performance_optimization.distributed_processing import ParallelProcessor, DataParallelizer

def process_large_dataset_in_batches(df: pd.DataFrame, batch_size: int = 10000) -> pd.DataFrame:
    \"\"\"
    Process large dataset in batches to manage memory.

    Args:
        df: Large DataFrame
        batch_size: Size of each batch

    Returns:
        Processed DataFrame
    \"\"\"
    n_batches = (len(df) - 1) // batch_size + 1
    results = []

    print(f"Processing {len(df)} rows in {n_batches} batches...")

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        batch = df.iloc[start_idx:end_idx].copy()

        # Process batch (your custom logic here)
        with profiler.profile_block(f"batch_{i}"):
            # Example: Apply transformations
            batch = create_features_vectorized(batch)

            # Example: Calculate aggregations
            batch_stats = batch.describe()

            results.append(batch)

        print(f"Processed batch {i+1}/{n_batches} ({len(batch)} rows)")

        # Clear memory periodically
        if i % 10 == 0:
            import gc
            gc.collect()

    # Combine results
    final_df = pd.concat(results, ignore_index=True)
    print(f"Batch processing complete. Final shape: {final_df.shape}")

    return final_df

# Parallel batch processing
def parallel_batch_process(df: pd.DataFrame, process_func: Callable) -> pd.DataFrame:
    \"\"\"
    Process batches in parallel using multiple cores.
    \"\"\"
    processor = ParallelProcessor(backend='multiprocessing', n_workers=N_CORES)

    # Split into partitions
    partitions = DataParallelizer.partition_dataframe(df, N_CORES)

    # Process in parallel
    results = processor.parallel_execute(process_func, partitions, show_progress=True)

    # Combine results
    return pd.concat(results, ignore_index=True)"""

    @staticmethod
    def get_caching_template() -> str:
        """Get caching template."""
        return """# Caching Template
import hashlib
import pickle
from pathlib import Path

class DataCache:
    \"\"\"Simple caching system for expensive computations.\"\"\"

    def __init__(self, cache_dir: str = '.cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_cache_key(self, *args, **kwargs) -> str:
        \"\"\"Generate cache key from arguments.\"\"\"
        key_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        \"\"\"Retrieve from cache.\"\"\"
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def set(self, key: str, value: Any):
        \"\"\"Store in cache.\"\"\"
        cache_file = self.cache_dir / f"{key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(value, f)

# Initialize cache
data_cache = DataCache()

@cache.cache_result
def expensive_computation(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"
    Example of cached expensive computation.
    \"\"\"
    # Check cache first
    cache_key = data_cache.get_cache_key('expensive', df.shape)
    result = data_cache.get(cache_key)

    if result is not None:
        print("Cache hit!")
        return result

    print("Cache miss - computing...")
    # Expensive operation
    result = df.groupby(df.columns.tolist()[0]).agg('mean')

    # Store in cache
    data_cache.set(cache_key, result)

    return result"""


class NotebookGenerator:
    """Generate optimized Jupyter notebooks from templates."""

    def __init__(self, output_dir: str = "optimized_notebooks"):
        """Initialize notebook generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.templates = OptimizationTemplates()

    def create_optimized_notebook(
        self, notebook_type: str, title: str = "Optimized Notebook"
    ) -> str:
        """
        Create an optimized notebook for specific use case.

        Args:
            notebook_type: Type of notebook ('eda', 'ml', 'timeseries', 'ab_testing')
            title: Notebook title

        Returns:
            Path to created notebook
        """
        nb = new_notebook()

        # Title
        nb.cells.append(
            new_markdown_cell(
                f"# {title}\n\nCreated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
        )

        # Common imports
        nb.cells.append(new_markdown_cell("## Setup and Imports"))
        nb.cells.append(new_code_cell(self.templates.get_import_cell()))

        # Data loading
        nb.cells.append(new_markdown_cell("## Data Loading"))
        nb.cells.append(new_code_cell(self.templates.get_data_loading_template()))

        # Type-specific cells
        if notebook_type == "eda":
            nb.cells.append(new_markdown_cell("## Exploratory Data Analysis"))
            nb.cells.append(new_code_cell(self.templates.get_eda_template()))
            nb.cells.append(new_markdown_cell("## Visualizations"))
            nb.cells.append(new_code_cell(self.templates.get_visualization_template()))

        elif notebook_type == "ml":
            nb.cells.append(new_markdown_cell("## Feature Engineering"))
            nb.cells.append(
                new_code_cell(self.templates.get_feature_engineering_template())
            )
            nb.cells.append(new_markdown_cell("## Model Training"))
            nb.cells.append(new_code_cell(self.templates.get_model_training_template()))

        elif notebook_type == "batch":
            nb.cells.append(new_markdown_cell("## Batch Processing"))
            nb.cells.append(
                new_code_cell(self.templates.get_batch_processing_template())
            )

        # Caching
        nb.cells.append(new_markdown_cell("## Caching System"))
        nb.cells.append(new_code_cell(self.templates.get_caching_template()))

        # Performance monitoring
        nb.cells.append(new_markdown_cell("## Performance Monitoring"))
        nb.cells.append(
            new_code_cell(
                """# Display performance statistics
profiler.print_stats()

# Memory usage
import psutil
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")"""
            )
        )

        # Save notebook
        filename = f"{notebook_type}_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            nbformat.write(nb, f)

        print(f"Created optimized notebook: {filepath}")
        return str(filepath)

    def create_all_templates(self) -> List[str]:
        """Create all template notebooks."""
        templates = ["eda", "ml", "timeseries", "batch"]
        created = []

        for template in templates:
            path = self.create_optimized_notebook(
                template, title=f"Optimized {template.upper()} Notebook"
            )
            created.append(path)

        return created


# Quick-start code snippets
OPTIMIZATION_SNIPPETS = {
    "memory_optimization": """
# Optimize DataFrame memory usage
df = MemoryOptimizer.optimize_dataframe_dtypes(df, verbose=True)
    """,
    "parallel_groupby": """
# Parallel groupby operation
from performance_optimization.distributed_processing import DaskOperations
dask_ops = DaskOperations()
result = dask_ops.parallel_groupby_agg(df, ['category'], {'value': 'mean'})
    """,
    "gpu_correlation": """
# GPU-accelerated correlation matrix
from performance_optimization.gpu_acceleration import GPUArrayOperations
corr = GPUArrayOperations.correlation_matrix_gpu(df.select_dtypes(include=[np.number]).values)
    """,
    "cached_function": """
# Add caching to expensive function
@cache.cache_result
def expensive_function(data):
    # Your expensive computation here
    return result
    """,
    "batch_process": """
# Process large file in chunks
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    # Process chunk
    result = process(chunk)
    """,
    "vectorized_operation": """
# Replace loop with vectorized operation
# Instead of: for i in range(len(df)): df.loc[i, 'new'] = df.loc[i, 'old'] * 2
df['new'] = df['old'] * 2  # Vectorized version
    """,
    "profile_code": """
# Profile code block
with profiler.profile_block("operation_name"):
    # Your code here
    result = expensive_operation()
    """,
    "sample_visualization": """
# Sample data for visualization
df_viz = viz_optimizer.sample_for_visualization(df, sample_size=10000)
    """,
}


if __name__ == "__main__":
    print("Optimization Templates Module")
    print("=" * 60)

    # Create notebook generator
    generator = NotebookGenerator()

    # Create all templates
    print("\nCreating optimization template notebooks...")
    created_notebooks = generator.create_all_templates()

    print("\nCreated notebooks:")
    for nb in created_notebooks:
        print(f"  - {nb}")

    print("\nAvailable optimization snippets:")
    for name in OPTIMIZATION_SNIPPETS.keys():
        print(f"  - {name}")

    print("\nTemplates created successfully!")
