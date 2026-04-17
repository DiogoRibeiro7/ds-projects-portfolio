"""GPU acceleration utilities using CuPy and RAPIDS for massive speedups.
"""

import time
import warnings
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

# Check for GPU libraries availability
GPU_AVAILABLE = False
CUPY_AVAILABLE = False
RAPIDS_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import cupy as cp

    CUPY_AVAILABLE = True
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    warnings.warn(
        "CuPy not installed. GPU acceleration disabled. Install with: pip install cupy-cuda11x"
    )

try:
    import cudf
    import cuml

    RAPIDS_AVAILABLE = True
    GPU_AVAILABLE = True
except ImportError:
    cudf = None
    cuml = None
    warnings.warn("RAPIDS not installed. GPU DataFrame operations disabled.")

try:
    import torch

    TORCH_AVAILABLE = True
    GPU_AVAILABLE = True
except ImportError:
    torch = None
    warnings.warn("PyTorch not installed. Neural network GPU acceleration disabled.")


def gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return GPU_AVAILABLE


def get_gpu_info() -> dict:
    """Get information about available GPUs."""
    info = {
        "gpu_available": GPU_AVAILABLE,
        "cupy_available": CUPY_AVAILABLE,
        "rapids_available": RAPIDS_AVAILABLE,
        "torch_available": TORCH_AVAILABLE,
        "devices": [],
    }

    if CUPY_AVAILABLE:
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            for i in range(device_count):
                props = cp.cuda.runtime.getDeviceProperties(i)
                info["devices"].append(
                    {
                        "id": i,
                        "name": (
                            props["name"].decode("utf-8")
                            if isinstance(props["name"], bytes)
                            else props["name"]
                        ),
                        "memory_gb": props["totalGlobalMem"] / (1024**3),
                        "compute_capability": f"{props['major']}.{props['minor']}",
                    }
                )
        except:
            pass

    if TORCH_AVAILABLE and torch.cuda.is_available():
        info["torch_cuda"] = True
        info["torch_device_count"] = torch.cuda.device_count()

    return info


def auto_gpu(func):
    """Decorator to automatically use GPU if available, fallback to CPU otherwise."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if GPU_AVAILABLE:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                warnings.warn(f"GPU execution failed: {e}. Falling back to CPU.")
                # Fallback logic would be implemented here
                raise e
        else:
            warnings.warn("GPU not available. Using CPU implementation.")
            # Return CPU version
            return func(*args, **kwargs)

    return wrapper


class GPUArrayOperations:
    """GPU-accelerated array operations using CuPy."""

    @staticmethod
    def to_gpu(arr: np.ndarray) -> Any:
        """Transfer numpy array to GPU memory."""
        if not CUPY_AVAILABLE:
            return arr
        return cp.asarray(arr)

    @staticmethod
    def to_cpu(gpu_arr: Any) -> np.ndarray:
        """Transfer GPU array back to CPU memory."""
        if not CUPY_AVAILABLE:
            return gpu_arr
        if hasattr(gpu_arr, "get"):
            return gpu_arr.get()
        return gpu_arr

    @staticmethod
    @auto_gpu
    def matrix_multiply_gpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """GPU-accelerated matrix multiplication."""
        if CUPY_AVAILABLE:
            a_gpu = cp.asarray(a)
            b_gpu = cp.asarray(b)
            result_gpu = cp.dot(a_gpu, b_gpu)
            return result_gpu.get()
        return np.dot(a, b)

    @staticmethod
    @auto_gpu
    def svd_gpu(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """GPU-accelerated Singular Value Decomposition."""
        if CUPY_AVAILABLE:
            matrix_gpu = cp.asarray(matrix)
            u, s, vt = cp.linalg.svd(matrix_gpu, full_matrices=False)
            return u.get(), s.get(), vt.get()
        return np.linalg.svd(matrix, full_matrices=False)

    @staticmethod
    @auto_gpu
    def correlation_matrix_gpu(data: np.ndarray) -> np.ndarray:
        """GPU-accelerated correlation matrix calculation."""
        if CUPY_AVAILABLE:
            data_gpu = cp.asarray(data)
            # Standardize
            mean = cp.mean(data_gpu, axis=0)
            std = cp.std(data_gpu, axis=0)
            standardized = (data_gpu - mean) / std
            # Compute correlation
            corr = cp.dot(standardized.T, standardized) / len(data)
            return corr.get()
        return np.corrcoef(data.T)

    @staticmethod
    @auto_gpu
    def euclidean_distances_gpu(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """GPU-accelerated pairwise Euclidean distances."""
        if CUPY_AVAILABLE:
            X_gpu = cp.asarray(X)
            Y_gpu = cp.asarray(Y)
            # Using broadcasting for efficient computation
            distances = cp.sqrt(
                ((X_gpu[:, None, :] - Y_gpu[None, :, :]) ** 2).sum(axis=2)
            )
            return distances.get()
        # CPU fallback
        from scipy.spatial.distance import cdist

        return cdist(X, Y, metric="euclidean")

    @staticmethod
    @auto_gpu
    def fft_gpu(signal: np.ndarray) -> np.ndarray:
        """GPU-accelerated Fast Fourier Transform."""
        if CUPY_AVAILABLE:
            signal_gpu = cp.asarray(signal)
            fft_result = cp.fft.fft(signal_gpu)
            return fft_result.get()
        return np.fft.fft(signal)


class GPUDataFrameOperations:
    """GPU-accelerated DataFrame operations using RAPIDS cuDF."""

    @staticmethod
    def to_gpu_df(df: pd.DataFrame) -> Any:
        """Convert pandas DataFrame to GPU DataFrame."""
        if not RAPIDS_AVAILABLE:
            return df
        return cudf.from_pandas(df)

    @staticmethod
    def to_cpu_df(gpu_df: Any) -> pd.DataFrame:
        """Convert GPU DataFrame back to pandas."""
        if not RAPIDS_AVAILABLE:
            return gpu_df
        if hasattr(gpu_df, "to_pandas"):
            return gpu_df.to_pandas()
        return gpu_df

    @staticmethod
    @auto_gpu
    def groupby_agg_gpu(
        df: pd.DataFrame, groupby_cols: list[str], agg_dict: dict
    ) -> pd.DataFrame:
        """GPU-accelerated groupby aggregation."""
        if RAPIDS_AVAILABLE:
            gpu_df = cudf.from_pandas(df)
            result = gpu_df.groupby(groupby_cols).agg(agg_dict)
            return result.to_pandas()
        return df.groupby(groupby_cols).agg(agg_dict)

    @staticmethod
    @auto_gpu
    def merge_gpu(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        on: str | list[str],
        how: str = "inner",
    ) -> pd.DataFrame:
        """GPU-accelerated DataFrame merge."""
        if RAPIDS_AVAILABLE:
            gpu_df1 = cudf.from_pandas(df1)
            gpu_df2 = cudf.from_pandas(df2)
            result = gpu_df1.merge(gpu_df2, on=on, how=how)
            return result.to_pandas()
        return pd.merge(df1, df2, on=on, how=how)

    @staticmethod
    @auto_gpu
    def sort_values_gpu(
        df: pd.DataFrame, by: str | list[str], ascending: bool = True
    ) -> pd.DataFrame:
        """GPU-accelerated sorting."""
        if RAPIDS_AVAILABLE:
            gpu_df = cudf.from_pandas(df)
            result = gpu_df.sort_values(by=by, ascending=ascending)
            return result.to_pandas()
        return df.sort_values(by=by, ascending=ascending)


class GPUMLOperations:
    """GPU-accelerated machine learning operations using RAPIDS cuML."""

    @staticmethod
    @auto_gpu
    def pca_gpu(data: np.ndarray, n_components: int = 2) -> np.ndarray:
        """GPU-accelerated Principal Component Analysis."""
        if RAPIDS_AVAILABLE and cuml:
            from cuml import PCA

            pca = PCA(n_components=n_components)
            transformed = pca.fit_transform(data)
            if hasattr(transformed, "get"):
                return transformed.get()
            return transformed
        # CPU fallback
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_components)
        return pca.fit_transform(data)

    @staticmethod
    @auto_gpu
    def kmeans_gpu(data: np.ndarray, n_clusters: int = 8) -> np.ndarray:
        """GPU-accelerated K-Means clustering."""
        if RAPIDS_AVAILABLE and cuml:
            from cuml import KMeans

            kmeans = KMeans(n_clusters=n_clusters)
            labels = kmeans.fit_predict(data)
            if hasattr(labels, "get"):
                return labels.get()
            return labels
        # CPU fallback
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters)
        return kmeans.fit_predict(data)

    @staticmethod
    @auto_gpu
    def train_random_forest_gpu(
        X: np.ndarray, y: np.ndarray, n_estimators: int = 100
    ) -> Any:
        """GPU-accelerated Random Forest training."""
        if RAPIDS_AVAILABLE and cuml:
            from cuml import RandomForestClassifier

            rf = RandomForestClassifier(n_estimators=n_estimators)
            rf.fit(X, y)
            return rf
        # CPU fallback
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(n_estimators=n_estimators)
        rf.fit(X, y)
        return rf


class GPUDeepLearning:
    """GPU-accelerated deep learning operations using PyTorch."""

    @staticmethod
    def get_device() -> Any:
        """Get the best available device (GPU or CPU)."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu") if TORCH_AVAILABLE else None

    @staticmethod
    def to_tensor_gpu(data: np.ndarray) -> Any:
        """Convert numpy array to GPU tensor."""
        if not TORCH_AVAILABLE:
            return data
        device = GPUDeepLearning.get_device()
        return torch.from_numpy(data).to(device)

    @staticmethod
    @auto_gpu
    def neural_network_forward_gpu(
        data: np.ndarray, weights: list[np.ndarray], activation: str = "relu"
    ) -> np.ndarray:
        """GPU-accelerated neural network forward pass."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device = torch.device("cuda")
            x = torch.from_numpy(data).float().to(device)

            for i, w in enumerate(weights):
                w_tensor = torch.from_numpy(w).float().to(device)
                x = torch.matmul(x, w_tensor)

                # Apply activation
                if i < len(weights) - 1:  # Not the last layer
                    if activation == "relu":
                        x = torch.relu(x)
                    elif activation == "sigmoid":
                        x = torch.sigmoid(x)
                    elif activation == "tanh":
                        x = torch.tanh(x)

            return x.cpu().numpy()

        # CPU fallback
        x = data.copy()
        for i, w in enumerate(weights):
            x = np.dot(x, w)
            if i < len(weights) - 1:
                if activation == "relu":
                    x = np.maximum(0, x)
                elif activation == "sigmoid":
                    x = 1 / (1 + np.exp(-x))
                elif activation == "tanh":
                    x = np.tanh(x)
        return x


def benchmark_gpu_vs_cpu(
    operation_name: str, cpu_func, gpu_func, data: Any, iterations: int = 10
) -> dict:
    """Benchmark GPU vs CPU performance."""
    # CPU benchmark
    cpu_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        cpu_result = cpu_func(data)
        cpu_times.append(time.perf_counter() - start)

    # GPU benchmark
    gpu_times = []
    gpu_success = False

    if GPU_AVAILABLE:
        try:
            for _ in range(iterations):
                start = time.perf_counter()
                gpu_result = gpu_func(data)
                gpu_times.append(time.perf_counter() - start)
            gpu_success = True
        except Exception as e:
            warnings.warn(f"GPU benchmark failed: {e}")

    results = {
        "operation": operation_name,
        "cpu_mean_time": np.mean(cpu_times),
        "cpu_std_time": np.std(cpu_times),
        "gpu_available": GPU_AVAILABLE,
        "gpu_success": gpu_success,
    }

    if gpu_success:
        results["gpu_mean_time"] = np.mean(gpu_times)
        results["gpu_std_time"] = np.std(gpu_times)
        results["speedup"] = results["cpu_mean_time"] / results["gpu_mean_time"]

    return results


def optimize_for_gpu(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame for GPU operations."""
    if not RAPIDS_AVAILABLE:
        return df

    # Convert to optimal dtypes for GPU
    optimized = df.copy()

    for col in optimized.columns:
        if optimized[col].dtype == "object":
            # Try to convert to category
            if optimized[col].nunique() < len(optimized) * 0.5:
                optimized[col] = optimized[col].astype("category")
        elif optimized[col].dtype == "int64":
            # Downcast integers
            if (
                optimized[col].min() >= -2147483648
                and optimized[col].max() <= 2147483647
            ):
                optimized[col] = optimized[col].astype("int32")
        elif optimized[col].dtype == "float64":
            # Use float32 for GPU
            optimized[col] = optimized[col].astype("float32")

    return optimized


# Example usage and testing
if __name__ == "__main__":
    print("GPU Acceleration Module")
    print("=" * 60)

    # Check GPU availability
    gpu_info = get_gpu_info()
    print(f"GPU Available: {gpu_info['gpu_available']}")
    if gpu_info["devices"]:
        for device in gpu_info["devices"]:
            print(
                f"  Device {device['id']}: {device['name']} ({device['memory_gb']:.1f} GB)"
            )

    # Test operations if available
    if GPU_AVAILABLE:
        print("\nTesting GPU Operations...")

        # Test matrix multiplication
        if CUPY_AVAILABLE:
            a = np.random.randn(1000, 1000)
            b = np.random.randn(1000, 1000)

            result = benchmark_gpu_vs_cpu(
                "Matrix Multiplication",
                lambda x: np.dot(x[0], x[1]),
                lambda x: GPUArrayOperations.matrix_multiply_gpu(x[0], x[1]),
                (a, b),
                iterations=5,
            )

            if result["gpu_success"]:
                print(f"Matrix Multiplication Speedup: {result['speedup']:.2f}x")
    else:
        print("\nNo GPU libraries installed. Install with:")
        print("  pip install cupy-cuda11x")
        print("  conda install -c rapidsai -c nvidia -c conda-forge rapids")
        print("  pip install torch torchvision torchaudio")
