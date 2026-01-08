"""
Advanced Feature Transformation Pipeline

Sophisticated feature transformation techniques for various data types.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer, KBinsDiscretizer
)
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, FactorAnalysis, NMF
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')


class OutlierTransformer(BaseEstimator, TransformerMixin):
    """Handle outliers using various strategies."""

    def __init__(self, method: str = 'iqr', threshold: float = 1.5):
        """
        Initialize outlier transformer.

        Args:
            method: 'iqr', 'zscore', 'isolation_forest', 'lof'
            threshold: Threshold for outlier detection
        """
        self.method = method
        self.threshold = threshold
        self.bounds_ = {}

    def fit(self, X, y=None):
        """Fit outlier detector."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        if self.method == 'iqr':
            for col in X_df.columns:
                Q1 = X_df[col].quantile(0.25)
                Q3 = X_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.threshold * IQR
                upper_bound = Q3 + self.threshold * IQR
                self.bounds_[col] = (lower_bound, upper_bound)

        elif self.method == 'zscore':
            for col in X_df.columns:
                mean = X_df[col].mean()
                std = X_df[col].std()
                lower_bound = mean - self.threshold * std
                upper_bound = mean + self.threshold * std
                self.bounds_[col] = (lower_bound, upper_bound)

        elif self.method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            self.detector = IsolationForest(contamination=0.1, random_state=42)
            self.detector.fit(X_df)

        elif self.method == 'lof':
            from sklearn.neighbors import LocalOutlierFactor
            self.detector = LocalOutlierFactor(contamination=0.1)
            self.detector.fit(X_df)

        return self

    def transform(self, X):
        """Transform data by handling outliers."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        X_transformed = X_df.copy()

        if self.method in ['iqr', 'zscore']:
            for col in X_df.columns:
                if col in self.bounds_:
                    lower, upper = self.bounds_[col]
                    X_transformed[col] = X_transformed[col].clip(lower, upper)

        elif self.method in ['isolation_forest', 'lof']:
            outlier_mask = self.detector.predict(X_df) == -1
            # Replace outliers with median
            for col in X_df.columns:
                median_val = X_df.loc[~outlier_mask, col].median()
                X_transformed.loc[outlier_mask, col] = median_val

        return X_transformed


class SkewnessTransformer(BaseEstimator, TransformerMixin):
    """Transform features to reduce skewness."""

    def __init__(self, threshold: float = 0.5, method: str = 'boxcox'):
        """
        Initialize skewness transformer.

        Args:
            threshold: Skewness threshold for transformation
            method: 'boxcox', 'log', 'sqrt', 'reciprocal'
        """
        self.threshold = threshold
        self.method = method
        self.lambdas_ = {}
        self.skewed_features_ = []

    def fit(self, X, y=None):
        """Fit skewness transformer."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        # Identify skewed features
        for col in X_df.columns:
            col_skew = skew(X_df[col].dropna())
            if abs(col_skew) > self.threshold:
                self.skewed_features_.append(col)

                if self.method == 'boxcox':
                    # Find optimal lambda for Box-Cox
                    if (X_df[col] > 0).all():
                        _, lambda_param = stats.boxcox(X_df[col])
                        self.lambdas_[col] = lambda_param

        return self

    def transform(self, X):
        """Transform skewed features."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        X_transformed = X_df.copy()

        for col in self.skewed_features_:
            if col not in X_df.columns:
                continue

            if self.method == 'boxcox' and col in self.lambdas_:
                if (X_df[col] > 0).all():
                    X_transformed[col] = boxcox1p(X_df[col], self.lambdas_[col])

            elif self.method == 'log':
                X_transformed[col] = np.log1p(np.abs(X_df[col]))

            elif self.method == 'sqrt':
                X_transformed[col] = np.sqrt(np.abs(X_df[col]))

            elif self.method == 'reciprocal':
                X_transformed[col] = 1 / (X_df[col] + 1e-8)

        return X_transformed


class DimensionalityReducer(BaseEstimator, TransformerMixin):
    """Reduce dimensionality using various techniques."""

    def __init__(self, method: str = 'pca', n_components: int = 10, **kwargs):
        """
        Initialize dimensionality reducer.

        Args:
            method: 'pca', 'svd', 'ica', 'nmf', 'factor', 'tsne', 'autoencoder'
            n_components: Number of components to keep
            **kwargs: Additional arguments for specific methods
        """
        self.method = method
        self.n_components = n_components
        self.kwargs = kwargs
        self.reducer = None

    def fit(self, X, y=None):
        """Fit dimensionality reducer."""
        if self.method == 'pca':
            self.reducer = PCA(n_components=self.n_components, **self.kwargs)

        elif self.method == 'svd':
            self.reducer = TruncatedSVD(n_components=self.n_components, **self.kwargs)

        elif self.method == 'ica':
            self.reducer = FastICA(n_components=self.n_components, **self.kwargs)

        elif self.method == 'nmf':
            # NMF requires non-negative values
            self.reducer = NMF(n_components=self.n_components, **self.kwargs)

        elif self.method == 'factor':
            self.reducer = FactorAnalysis(n_components=self.n_components, **self.kwargs)

        elif self.method == 'tsne':
            self.reducer = TSNE(n_components=min(3, self.n_components), **self.kwargs)

        elif self.method == 'autoencoder':
            self.reducer = self._create_autoencoder(X.shape[1])
            self._train_autoencoder(X)
            return self

        if self.reducer and self.method != 'tsne':
            self.reducer.fit(X)

        return self

    def transform(self, X):
        """Transform data to lower dimensions."""
        if self.method == 'nmf':
            # Ensure non-negative values
            X_positive = X - X.min() + 1e-8
            return self.reducer.transform(X_positive)

        elif self.method == 'tsne':
            # TSNE doesn't have separate fit/transform
            return self.reducer.fit_transform(X)

        elif self.method == 'autoencoder':
            return self._encode(X)

        else:
            return self.reducer.transform(X)

    def _create_autoencoder(self, input_dim: int):
        """Create autoencoder neural network."""
        try:
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
            from tensorflow.keras import regularizers

            # Encoder
            input_layer = Input(shape=(input_dim,))
            encoded = Dense(
                max(input_dim // 2, self.n_components * 4),
                activation='relu',
                kernel_regularizer=regularizers.l2(0.01)
            )(input_layer)
            encoded = BatchNormalization()(encoded)
            encoded = Dropout(0.2)(encoded)

            encoded = Dense(
                max(input_dim // 4, self.n_components * 2),
                activation='relu'
            )(encoded)
            encoded = BatchNormalization()(encoded)

            encoded = Dense(self.n_components, activation='linear')(encoded)

            # Decoder
            decoded = Dense(
                max(input_dim // 4, self.n_components * 2),
                activation='relu'
            )(encoded)
            decoded = BatchNormalization()(decoded)

            decoded = Dense(
                max(input_dim // 2, self.n_components * 4),
                activation='relu'
            )(decoded)
            decoded = BatchNormalization()(decoded)

            decoded = Dense(input_dim, activation='linear')(decoded)

            # Models
            self.autoencoder = Model(input_layer, decoded)
            self.encoder = Model(input_layer, encoded)

            self.autoencoder.compile(optimizer='adam', loss='mse')

            return self.autoencoder

        except ImportError:
            print("TensorFlow not available. Using PCA instead.")
            self.method = 'pca'
            self.reducer = PCA(n_components=self.n_components)
            return self.reducer

    def _train_autoencoder(self, X, epochs: int = 50, batch_size: int = 32):
        """Train autoencoder model."""
        if hasattr(self, 'autoencoder'):
            self.autoencoder.fit(
                X, X,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=0
            )

    def _encode(self, X):
        """Encode data using trained autoencoder."""
        if hasattr(self, 'encoder'):
            return self.encoder.predict(X, verbose=0)
        else:
            return X


class ClusteringTransformer(BaseEstimator, TransformerMixin):
    """Create features based on clustering."""

    def __init__(self, method: str = 'kmeans', n_clusters: int = 5, **kwargs):
        """
        Initialize clustering transformer.

        Args:
            method: 'kmeans', 'dbscan', 'gmm', 'hierarchical'
            n_clusters: Number of clusters
            **kwargs: Additional arguments for specific methods
        """
        self.method = method
        self.n_clusters = n_clusters
        self.kwargs = kwargs
        self.clusterer = None

    def fit(self, X, y=None):
        """Fit clustering model."""
        if self.method == 'kmeans':
            self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, **self.kwargs)

        elif self.method == 'dbscan':
            self.clusterer = DBSCAN(**self.kwargs)

        elif self.method == 'gmm':
            self.clusterer = GaussianMixture(n_components=self.n_clusters, random_state=42, **self.kwargs)

        elif self.method == 'hierarchical':
            from sklearn.cluster import AgglomerativeClustering
            self.clusterer = AgglomerativeClustering(n_clusters=self.n_clusters, **self.kwargs)

        self.clusterer.fit(X)
        return self

    def transform(self, X):
        """Transform data by adding cluster features."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        # Get cluster labels
        if self.method == 'gmm':
            labels = self.clusterer.predict(X)
            probabilities = self.clusterer.predict_proba(X)

            # Add cluster probabilities as features
            for i in range(probabilities.shape[1]):
                X_df[f'cluster_{i}_prob'] = probabilities[:, i]

        else:
            labels = self.clusterer.predict(X) if hasattr(self.clusterer, 'predict') else self.clusterer.labels_

        X_df['cluster_label'] = labels

        # Add distance to cluster centers for kmeans
        if self.method == 'kmeans':
            distances = self.clusterer.transform(X)
            for i in range(distances.shape[1]):
                X_df[f'dist_to_cluster_{i}'] = distances[:, i]

            # Add distance to nearest cluster center
            X_df['min_cluster_dist'] = distances.min(axis=1)
            X_df['max_cluster_dist'] = distances.max(axis=1)

        return X_df


class InteractionTransformer(BaseEstimator, TransformerMixin):
    """Create interaction features between variables."""

    def __init__(self, interaction_type: str = 'multiply',
                 max_features: int = 10,
                 interaction_only: bool = True):
        """
        Initialize interaction transformer.

        Args:
            interaction_type: 'multiply', 'divide', 'add', 'subtract', 'all'
            max_features: Maximum number of features to create interactions for
            interaction_only: If True, only return interaction features
        """
        self.interaction_type = interaction_type
        self.max_features = max_features
        self.interaction_only = interaction_only
        self.selected_features = None

    def fit(self, X, y=None):
        """Fit interaction transformer."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        # Select top features based on variance
        variances = X_df.var()
        self.selected_features = variances.nlargest(self.max_features).index.tolist()

        return self

    def transform(self, X):
        """Create interaction features."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        result = X_df.copy() if not self.interaction_only else pd.DataFrame(index=X_df.index)

        # Create interactions for selected features
        for i, col1 in enumerate(self.selected_features):
            if col1 not in X_df.columns:
                continue

            for col2 in self.selected_features[i+1:]:
                if col2 not in X_df.columns:
                    continue

                if self.interaction_type in ['multiply', 'all']:
                    result[f'{col1}_times_{col2}'] = X_df[col1] * X_df[col2]

                if self.interaction_type in ['divide', 'all']:
                    result[f'{col1}_div_{col2}'] = X_df[col1] / (X_df[col2] + 1e-8)
                    result[f'{col2}_div_{col1}'] = X_df[col2] / (X_df[col1] + 1e-8)

                if self.interaction_type in ['add', 'all']:
                    result[f'{col1}_plus_{col2}'] = X_df[col1] + X_df[col2]

                if self.interaction_type in ['subtract', 'all']:
                    result[f'{col1}_minus_{col2}'] = X_df[col1] - X_df[col2]
                    result[f'{col2}_minus_{col1}'] = X_df[col2] - X_df[col1]

        return result


class BinningTransformer(BaseEstimator, TransformerMixin):
    """Create binned features using various strategies."""

    def __init__(self, strategy: str = 'quantile', n_bins: int = 5,
                 encode: str = 'ordinal'):
        """
        Initialize binning transformer.

        Args:
            strategy: 'uniform', 'quantile', 'kmeans'
            n_bins: Number of bins
            encode: 'ordinal', 'onehot', 'onehot-dense'
        """
        self.strategy = strategy
        self.n_bins = n_bins
        self.encode = encode
        self.discretizers = {}

    def fit(self, X, y=None):
        """Fit binning transformer."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        for col in X_df.columns:
            discretizer = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode=self.encode,
                strategy=self.strategy
            )
            discretizer.fit(X_df[[col]])
            self.discretizers[col] = discretizer

        return self

    def transform(self, X):
        """Transform features by binning."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        result_dfs = []

        for col in X_df.columns:
            if col in self.discretizers:
                binned = self.discretizers[col].transform(X_df[[col]])

                if self.encode == 'onehot':
                    # Convert sparse to dense
                    binned = binned.toarray()

                # Create column names
                if self.encode in ['onehot', 'onehot-dense'] or binned.shape[1] > 1:
                    col_names = [f'{col}_bin_{i}' for i in range(binned.shape[1])]
                    binned_df = pd.DataFrame(binned, columns=col_names, index=X_df.index)
                else:
                    binned_df = pd.DataFrame(binned, columns=[f'{col}_binned'], index=X_df.index)

                result_dfs.append(binned_df)

        result = pd.concat(result_dfs, axis=1)

        # Add original features if not encoding as one-hot
        if self.encode == 'ordinal':
            result = pd.concat([X_df, result], axis=1)

        return result


class TargetTransformer(BaseEstimator, TransformerMixin):
    """Transform target variable for better model performance."""

    def __init__(self, method: str = 'auto'):
        """
        Initialize target transformer.

        Args:
            method: 'auto', 'log', 'sqrt', 'boxcox', 'quantile'
        """
        self.method = method
        self.transformer = None
        self.lambda_param = None

    def fit(self, y):
        """Fit target transformer."""
        y = np.array(y).ravel()

        if self.method == 'auto':
            # Automatically select best transformation
            skewness = skew(y)
            if abs(skewness) > 1:
                if (y > 0).all():
                    self.method = 'boxcox'
                else:
                    self.method = 'quantile'
            else:
                self.method = None  # No transformation needed

        if self.method == 'log':
            if (y <= 0).any():
                print("Warning: Negative values found. Using log1p with shifted values.")

        elif self.method == 'boxcox':
            if (y > 0).all():
                _, self.lambda_param = stats.boxcox(y)
            else:
                print("Warning: Non-positive values found. Using Yeo-Johnson instead.")
                self.method = 'yeo-johnson'

        elif self.method == 'quantile':
            self.transformer = QuantileTransformer(output_distribution='normal')
            self.transformer.fit(y.reshape(-1, 1))

        elif self.method == 'yeo-johnson':
            self.transformer = PowerTransformer(method='yeo-johnson')
            self.transformer.fit(y.reshape(-1, 1))

        return self

    def transform(self, y):
        """Transform target variable."""
        y = np.array(y).ravel()

        if self.method is None:
            return y

        elif self.method == 'log':
            return np.log1p(np.maximum(0, y))

        elif self.method == 'sqrt':
            return np.sqrt(np.abs(y))

        elif self.method == 'boxcox':
            if self.lambda_param is not None and (y > 0).all():
                return boxcox1p(y, self.lambda_param)
            else:
                return y

        elif self.method in ['quantile', 'yeo-johnson']:
            return self.transformer.transform(y.reshape(-1, 1)).ravel()

        else:
            return y

    def inverse_transform(self, y):
        """Inverse transform target variable."""
        y = np.array(y).ravel()

        if self.method is None:
            return y

        elif self.method == 'log':
            return np.expm1(y)

        elif self.method == 'sqrt':
            return y ** 2

        elif self.method == 'boxcox':
            if self.lambda_param is not None:
                # Inverse Box-Cox
                if self.lambda_param == 0:
                    return np.exp(y) - 1
                else:
                    return ((y * self.lambda_param + 1) ** (1 / self.lambda_param)) - 1
            else:
                return y

        elif self.method in ['quantile', 'yeo-johnson']:
            return self.transformer.inverse_transform(y.reshape(-1, 1)).ravel()

        else:
            return y


class FeatureAugmenter(BaseEstimator, TransformerMixin):
    """Augment features using various techniques."""

    def __init__(self, augmentation_types: List[str] = None):
        """
        Initialize feature augmenter.

        Args:
            augmentation_types: List of augmentation types to apply
        """
        if augmentation_types is None:
            augmentation_types = ['noise', 'rolling', 'lag', 'diff']
        self.augmentation_types = augmentation_types

    def fit(self, X, y=None):
        """Fit is not needed for augmentation."""
        return self

    def transform(self, X):
        """Augment features."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        augmented_dfs = [X_df]

        if 'noise' in self.augmentation_types:
            # Add Gaussian noise
            noise_df = X_df + np.random.normal(0, X_df.std() * 0.01, X_df.shape)
            noise_df.columns = [f'{col}_noise' for col in X_df.columns]
            augmented_dfs.append(noise_df)

        if 'rolling' in self.augmentation_types:
            # Rolling statistics
            for window in [3, 5, 7]:
                rolling_mean = X_df.rolling(window, min_periods=1).mean()
                rolling_mean.columns = [f'{col}_roll_mean_{window}' for col in X_df.columns]
                augmented_dfs.append(rolling_mean)

                rolling_std = X_df.rolling(window, min_periods=1).std()
                rolling_std.columns = [f'{col}_roll_std_{window}' for col in X_df.columns]
                augmented_dfs.append(rolling_std)

        if 'lag' in self.augmentation_types:
            # Lag features
            for lag in [1, 2, 3]:
                lag_df = X_df.shift(lag)
                lag_df.columns = [f'{col}_lag_{lag}' for col in X_df.columns]
                augmented_dfs.append(lag_df)

        if 'diff' in self.augmentation_types:
            # Difference features
            diff_df = X_df.diff()
            diff_df.columns = [f'{col}_diff' for col in X_df.columns]
            augmented_dfs.append(diff_df)

            # Percentage change
            pct_change = X_df.pct_change()
            pct_change.columns = [f'{col}_pct_change' for col in X_df.columns]
            augmented_dfs.append(pct_change)

        result = pd.concat(augmented_dfs, axis=1)
        result = result.fillna(method='ffill').fillna(method='bfill').fillna(0)

        return result


def create_transformation_pipeline(transformations: List[str], **kwargs) -> List:
    """
    Create a transformation pipeline from a list of transformation names.

    Args:
        transformations: List of transformation names
        **kwargs: Additional arguments for transformers

    Returns:
        List of transformer instances
    """
    pipeline = []

    for trans in transformations:
        if trans == 'outlier':
            pipeline.append(OutlierTransformer(**kwargs.get('outlier', {})))

        elif trans == 'skewness':
            pipeline.append(SkewnessTransformer(**kwargs.get('skewness', {})))

        elif trans == 'dimensionality':
            pipeline.append(DimensionalityReducer(**kwargs.get('dimensionality', {})))

        elif trans == 'clustering':
            pipeline.append(ClusteringTransformer(**kwargs.get('clustering', {})))

        elif trans == 'interaction':
            pipeline.append(InteractionTransformer(**kwargs.get('interaction', {})))

        elif trans == 'binning':
            pipeline.append(BinningTransformer(**kwargs.get('binning', {})))

        elif trans == 'augmentation':
            pipeline.append(FeatureAugmenter(**kwargs.get('augmentation', {})))

    return pipeline