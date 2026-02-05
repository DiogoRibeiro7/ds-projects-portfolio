"""Distributed Computing Module
Implements Dask and Ray for scalable ML processing
"""

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Dask imports
import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

# Ray imports
import ray
from dask.delayed import delayed
from dask.distributed import Client
from dask_ml.preprocessing import StandardScaler
from ray import tune
from ray.data import Dataset
from ray.train import ScalingConfig
from ray.train.xgboost import XGBoostTrainer
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

# GPU acceleration
try:
    import cudf
    import cuml
    import cupy as cp
    from cuml.dask.ensemble import RandomForestClassifier as CumlRFC
    from cuml.dask.linear_model import LogisticRegression as CumlLR
    from cuml.dask.preprocessing import StandardScaler as CumlStandardScaler

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logging.warning("GPU libraries not available, using CPU implementations")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComputeConfig:
    """Configuration for distributed computing"""

    backend: str = "ray"  # "ray", "dask", "spark"
    n_workers: int = 4
    threads_per_worker: int = 2
    memory_per_worker: str = "4GB"
    use_gpu: bool = False
    gpu_per_worker: float = 0.0
    dashboard_address: str | None = None
    scheduler_address: str | None = None
    adaptive_scaling: bool = True
    min_workers: int = 1
    max_workers: int = 10
    chunk_size: str = "128MB"


class DistributedComputer:
    """Base class for distributed computing"""

    def __init__(self, config: ComputeConfig):
        self.config = config
        self.client = None
        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize computing backend"""
        if self.config.backend == "ray":
            self._initialize_ray()
        elif self.config.backend == "dask":
            self._initialize_dask()
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

    def _initialize_ray(self):
        """Initialize Ray cluster"""
        if not ray.is_initialized():
            ray.init(
                address=self.config.scheduler_address or "auto",
                num_cpus=self.config.n_workers * self.config.threads_per_worker,
                num_gpus=self.config.n_workers * self.config.gpu_per_worker
                if self.config.use_gpu
                else 0,
                dashboard_host="0.0.0.0" if self.config.dashboard_address else None,
                _memory=self._parse_memory(self.config.memory_per_worker)
                * self.config.n_workers,
                object_store_memory=self._parse_memory(self.config.memory_per_worker)
                * self.config.n_workers
                // 2,
            )
        logger.info(f"Ray initialized with {ray.cluster_resources()}")

    def _initialize_dask(self):
        """Initialize Dask cluster"""
        if self.config.use_gpu and GPU_AVAILABLE:
            from dask_cuda import LocalCUDACluster

            cluster = LocalCUDACluster(
                n_workers=self.config.n_workers,
                threads_per_worker=self.config.threads_per_worker,
                memory_limit=self.config.memory_per_worker,
                dashboard_address=self.config.dashboard_address or ":8787",
            )
        else:
            from dask.distributed import LocalCluster

            cluster = LocalCluster(
                n_workers=self.config.n_workers,
                threads_per_worker=self.config.threads_per_worker,
                memory_limit=self.config.memory_per_worker,
                dashboard_address=self.config.dashboard_address or ":8787",
            )

        self.client = Client(cluster)

        # Setup adaptive scaling if enabled
        if self.config.adaptive_scaling:
            self.client.cluster.adapt(
                minimum=self.config.min_workers, maximum=self.config.max_workers
            )

        logger.info(f"Dask client initialized: {self.client}")
        logger.info(f"Dashboard available at: {self.client.dashboard_link}")

    def _parse_memory(self, memory_str: str) -> int:
        """Parse memory string to bytes"""
        units = {"KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
        for unit, multiplier in units.items():
            if unit in memory_str.upper():
                return int(float(memory_str.upper().replace(unit, "")) * multiplier)
        return int(memory_str)

    def shutdown(self):
        """Shutdown computing backend"""
        if self.config.backend == "ray":
            ray.shutdown()
        elif self.config.backend == "dask" and self.client:
            self.client.close()


class DaskMLPipeline:
    """Dask-based ML pipeline for distributed processing"""

    def __init__(self, client: Client = None):
        self.client = client or Client.current()
        self.scaler = None
        self.model = None

    def load_data(self, file_path: str, **kwargs) -> dd.DataFrame:
        """Load data as Dask DataFrame"""
        if file_path.endswith(".parquet"):
            df = dd.read_parquet(file_path, **kwargs)
        elif file_path.endswith(".csv"):
            df = dd.read_csv(file_path, **kwargs)
        else:
            # Load with pandas and convert
            pdf = pd.read_pickle(file_path)
            df = dd.from_pandas(pdf, npartitions=self.client.ncores() * 2)

        logger.info(f"Loaded data with {df.npartitions} partitions")
        return df

    def preprocess_data(
        self, X: dd.DataFrame, y: dd.Series | None = None, use_gpu: bool = False
    ) -> tuple[dd.DataFrame, dd.Series | None]:
        """Preprocess data with scaling"""
        if use_gpu and GPU_AVAILABLE:
            self.scaler = CumlStandardScaler()
        else:
            self.scaler = StandardScaler()

        # Persist data in memory
        X = X.persist()
        if y is not None:
            y = y.persist()

        # Fit and transform
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    @delayed
    def train_model_partition(self, X_partition, y_partition, model_class, **kwargs):
        """Train model on a single partition"""
        model = model_class(**kwargs)
        model.fit(X_partition, y_partition)
        return model

    def train_distributed(
        self, X: dd.DataFrame, y: dd.Series, model_class: Any, **model_kwargs
    ) -> list:
        """Train models on distributed partitions"""
        # Get partition pairs
        X_parts = X.to_delayed()
        y_parts = y.to_delayed()

        # Train on each partition
        models = []
        for X_part, y_part in zip(X_parts, y_parts, strict=False):
            model = self.train_model_partition(
                X_part, y_part, model_class, **model_kwargs
            )
            models.append(model)

        # Compute all models
        models = dask.compute(*models)
        return models

    def ensemble_predict(self, models: list, X: dd.DataFrame) -> dd.Series:
        """Make ensemble predictions from multiple models"""
        predictions = []

        for model in models:
            pred = dd.from_pandas(
                pd.Series(model.predict(X.compute())), npartitions=X.npartitions
            )
            predictions.append(pred)

        # Average predictions (for regression) or majority vote (for classification)
        ensemble_pred = dd.concat(predictions, axis=1).mean(axis=1)
        return ensemble_pred

    def hyperparameter_search(
        self, X: dd.DataFrame, y: dd.Series, model_class: Any, param_grid: dict
    ) -> dict:
        """Distributed hyperparameter search"""
        from dask_ml.model_selection import GridSearchCV

        model = model_class()
        search = GridSearchCV(model, param_grid, cv=5, scoring="roc_auc")

        search.fit(X, y)
        return {
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "cv_results": search.cv_results_,
        }


class RayMLPipeline:
    """Ray-based ML pipeline for distributed processing"""

    def __init__(self):
        if not ray.is_initialized():
            ray.init()
        self.dataset = None
        self.trainer = None

    def load_data(self, file_path: str, **kwargs) -> Dataset:
        """Load data as Ray Dataset"""
        if file_path.endswith(".parquet"):
            dataset = ray.data.read_parquet(file_path)
        elif file_path.endswith(".csv"):
            dataset = ray.data.read_csv(file_path)
        else:
            # Load with pandas and convert
            pdf = pd.read_pickle(file_path)
            dataset = ray.data.from_pandas(pdf)

        self.dataset = dataset
        logger.info(f"Loaded Ray dataset with {dataset.count()} rows")
        return dataset

    @ray.remote
    def preprocess_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Preprocess a batch of data"""
        from sklearn.preprocessing import StandardScaler

        # Numeric columns
        numeric_cols = batch.select_dtypes(include=[np.number]).columns

        # Scale numeric features
        scaler = StandardScaler()
        batch[numeric_cols] = scaler.fit_transform(batch[numeric_cols])

        return batch

    def preprocess_data(self, dataset: Dataset) -> Dataset:
        """Preprocess data using Ray"""
        # Map preprocessing to batches
        preprocessed = dataset.map_batches(
            lambda batch: self.preprocess_batch.remote(batch), batch_size=1000
        )
        return preprocessed

    def train_xgboost(
        self, dataset: Dataset, label_column: str, num_boost_rounds: int = 100
    ) -> XGBoostTrainer:
        """Train XGBoost model using Ray Train"""
        trainer = XGBoostTrainer(
            scaling_config=ScalingConfig(
                num_workers=4,
                use_gpu=GPU_AVAILABLE,
                resources_per_worker={"CPU": 2, "GPU": 0.25 if GPU_AVAILABLE else 0},
            ),
            label_column=label_column,
            params={
                "objective": "binary:logistic",
                "eval_metric": ["logloss", "auc"],
                "tree_method": "gpu_hist" if GPU_AVAILABLE else "hist",
                "max_depth": 6,
                "eta": 0.3,
            },
            datasets={"train": dataset},
            num_boost_rounds=num_boost_rounds,
        )

        result = trainer.fit()
        self.trainer = trainer
        return result

    def hyperparameter_tuning(
        self, dataset: Dataset, label_column: str
    ) -> tune.ResultGrid:
        """Hyperparameter tuning using Ray Tune"""

        def train_model(config):
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import train_test_split

            # Load data (in real scenario, would use dataset)
            X, y = dataset.to_pandas()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train model with config
            model = RandomForestClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                min_samples_split=config["min_samples_split"],
                random_state=42,
            )
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred)

            return {"auc": auc}

        # Define search space
        search_space = {
            "n_estimators": tune.choice([50, 100, 200]),
            "max_depth": tune.choice([5, 10, 15, None]),
            "min_samples_split": tune.choice([2, 5, 10]),
        }

        # Use Optuna for search
        optuna_search = OptunaSearch(metric="auc", mode="max")

        # Use ASHA scheduler for early stopping
        scheduler = ASHAScheduler(metric="auc", mode="max", max_t=100, grace_period=10)

        # Run tuning
        tuner = tune.Tuner(
            tune.with_resources(
                train_model, {"cpu": 2, "gpu": 0.25 if GPU_AVAILABLE else 0}
            ),
            param_space=search_space,
            tune_config=tune.TuneConfig(
                search_alg=optuna_search,
                scheduler=scheduler,
                num_samples=20,
                max_concurrent_trials=4,
            ),
        )

        results = tuner.fit()
        return results

    @ray.remote
    def distributed_inference(self, model, data_chunk: pd.DataFrame) -> np.ndarray:
        """Perform distributed inference"""
        predictions = model.predict(data_chunk)
        return predictions

    def batch_inference(
        self, model: Any, dataset: Dataset, batch_size: int = 1000
    ) -> list[np.ndarray]:
        """Perform batch inference using Ray"""
        # Serialize model
        model_ref = ray.put(model)

        # Process in batches
        predictions = []
        for batch in dataset.iter_batches(batch_size=batch_size):
            pred_future = self.distributed_inference.remote(model_ref, batch)
            predictions.append(pred_future)

        # Gather results
        predictions = ray.get(predictions)
        return np.concatenate(predictions)


class GPUAccelerator:
    """GPU acceleration for ML workloads"""

    def __init__(self):
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU libraries not available")
        self.device_count = cp.cuda.runtime.getDeviceCount()
        logger.info(f"GPU Accelerator initialized with {self.device_count} devices")

    def to_gpu(
        self, data: pd.DataFrame | np.ndarray
    ) -> cudf.DataFrame | cp.ndarray:
        """Transfer data to GPU"""
        if isinstance(data, pd.DataFrame):
            return cudf.from_pandas(data)
        elif isinstance(data, np.ndarray):
            return cp.asarray(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def to_cpu(
        self, data: cudf.DataFrame | cp.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Transfer data from GPU to CPU"""
        if isinstance(data, cudf.DataFrame):
            return data.to_pandas()
        elif isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def train_gpu_model(
        self, X: cudf.DataFrame, y: cudf.Series, model_type: str = "xgboost"
    ) -> Any:
        """Train model on GPU"""
        if model_type == "xgboost":
            import xgboost as xgb

            dtrain = xgb.DMatrix(X, label=y)
            params = {
                "objective": "binary:logistic",
                "tree_method": "gpu_hist",
                "predictor": "gpu_predictor",
                "eval_metric": "auc",
                "max_depth": 6,
                "learning_rate": 0.3,
            }
            model = xgb.train(params, dtrain, num_boost_round=100)

        elif model_type == "random_forest":
            from cuml.ensemble import RandomForestClassifier

            model = RandomForestClassifier(n_estimators=100, max_depth=10, n_streams=4)
            model.fit(X, y)

        elif model_type == "logistic_regression":
            from cuml.linear_model import LogisticRegression

            model = LogisticRegression()
            model.fit(X, y)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return model

    def gpu_preprocessing(self, X: cudf.DataFrame) -> cudf.DataFrame:
        """Preprocess data on GPU"""
        from cuml.preprocessing import StandardScaler

        # Numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        # Scale on GPU
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        return X

    @staticmethod
    def benchmark_gpu_vs_cpu(
        func_gpu: Callable, func_cpu: Callable, data: Any, iterations: int = 10
    ) -> dict:
        """Benchmark GPU vs CPU performance"""
        # GPU benchmark
        gpu_times = []
        for _ in range(iterations):
            start = time.time()
            func_gpu(data)
            gpu_times.append(time.time() - start)

        # CPU benchmark
        cpu_times = []
        for _ in range(iterations):
            start = time.time()
            func_cpu(data)
            cpu_times.append(time.time() - start)

        return {
            "gpu_mean": np.mean(gpu_times),
            "gpu_std": np.std(gpu_times),
            "cpu_mean": np.mean(cpu_times),
            "cpu_std": np.std(cpu_times),
            "speedup": np.mean(cpu_times) / np.mean(gpu_times),
        }


class ParallelProcessor:
    """Parallel processing utilities"""

    def __init__(self, n_workers: int = None):
        self.n_workers = n_workers or os.cpu_count()

    @ray.remote
    def process_chunk(self, func: Callable, chunk: Any, **kwargs) -> Any:
        """Process a single chunk"""
        return func(chunk, **kwargs)

    def parallel_map(self, func: Callable, data: list, **kwargs) -> list:
        """Parallel map operation"""
        # Split data into chunks
        chunk_size = len(data) // self.n_workers
        chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

        # Process in parallel
        futures = [self.process_chunk.remote(func, chunk, **kwargs) for chunk in chunks]

        # Gather results
        results = ray.get(futures)
        return [item for sublist in results for item in sublist]

    def parallel_reduce(self, func: Callable, data: list, initial: Any = None) -> Any:
        """Parallel reduce operation"""
        if len(data) == 0:
            return initial
        if len(data) == 1:
            return data[0]

        # Parallel reduction tree
        while len(data) > 1:
            # Pair up elements
            pairs = []
            for i in range(0, len(data), 2):
                if i + 1 < len(data):
                    pairs.append((data[i], data[i + 1]))
                else:
                    pairs.append((data[i], initial))

            # Reduce pairs in parallel
            futures = [
                self.process_chunk.remote(lambda p: func(p[0], p[1]), pair)
                for pair in pairs
            ]

            data = ray.get(futures)

        return data[0]


# Example usage
if __name__ == "__main__":
    # Initialize distributed computing
    config = ComputeConfig(
        backend="ray", n_workers=4, use_gpu=GPU_AVAILABLE, adaptive_scaling=True
    )

    computer = DistributedComputer(config)

    # Ray ML Pipeline example
    ray_pipeline = RayMLPipeline()

    # Create sample data
    sample_data = pd.DataFrame(
        {
            "feature1": np.random.randn(10000),
            "feature2": np.random.randn(10000),
            "feature3": np.random.randn(10000),
            "target": np.random.randint(0, 2, 10000),
        }
    )

    # Convert to Ray Dataset
    dataset = ray.data.from_pandas(sample_data)

    # Preprocess data
    processed_dataset = ray_pipeline.preprocess_data(dataset)

    # Train model
    result = ray_pipeline.train_xgboost(processed_dataset, label_column="target")

    logger.info(f"Training completed: {result}")

    # Cleanup
    computer.shutdown()
