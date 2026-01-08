"""
Utility functions for recommendation systems

This module provides comprehensive utilities for building and evaluating recommendation systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import hashlib
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocessing utilities for recommendation data"""

    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.stats = {}

    def preprocess_interactions(self, df: pd.DataFrame,
                                min_user_interactions: int = 5,
                                min_item_interactions: int = 5) -> pd.DataFrame:
        """
        Preprocess interaction data

        Args:
            df: DataFrame with user_id, item_id, rating/interaction columns
            min_user_interactions: Minimum interactions per user
            min_item_interactions: Minimum interactions per item

        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Initial shape: {df.shape}")

        # Remove duplicates
        df = df.drop_duplicates(subset=['user_id', 'item_id'])

        # Filter users and items by minimum interactions
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()

        valid_users = user_counts[user_counts >= min_user_interactions].index
        valid_items = item_counts[item_counts >= min_item_interactions].index

        df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]

        # Encode user and item IDs
        df['user_idx'] = self.user_encoder.fit_transform(df['user_id'])
        df['item_idx'] = self.item_encoder.fit_transform(df['item_id'])

        # Store statistics
        self.stats = {
            'n_users': df['user_idx'].nunique(),
            'n_items': df['item_idx'].nunique(),
            'n_interactions': len(df),
            'density': len(df) / (df['user_idx'].nunique() * df['item_idx'].nunique()),
            'avg_user_interactions': df.groupby('user_idx').size().mean(),
            'avg_item_interactions': df.groupby('item_idx').size().mean()
        }

        logger.info(f"Preprocessed shape: {df.shape}")
        logger.info(f"Statistics: {self.stats}")

        return df

    def create_interaction_matrix(self, df: pd.DataFrame,
                                  value_col: str = 'rating',
                                  sparse: bool = True) -> Union[np.ndarray, sp.csr_matrix]:
        """
        Create user-item interaction matrix

        Args:
            df: DataFrame with user_idx, item_idx, and value columns
            value_col: Column to use for matrix values
            sparse: Whether to return sparse matrix

        Returns:
            Interaction matrix
        """
        n_users = df['user_idx'].max() + 1
        n_items = df['item_idx'].max() + 1

        if sparse:
            matrix = sp.csr_matrix(
                (df[value_col].values, (df['user_idx'].values, df['item_idx'].values)),
                shape=(n_users, n_items)
            )
        else:
            matrix = np.zeros((n_users, n_items))
            matrix[df['user_idx'].values, df['item_idx'].values] = df[value_col].values

        return matrix

    def train_test_split_temporal(self, df: pd.DataFrame,
                                  test_size: float = 0.2,
                                  timestamp_col: str = 'timestamp') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally for time-aware evaluation

        Args:
            df: DataFrame with interactions
            test_size: Proportion of data for testing
            timestamp_col: Column containing timestamps

        Returns:
            Train and test DataFrames
        """
        df = df.sort_values(timestamp_col)
        split_idx = int(len(df) * (1 - test_size))

        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        # Ensure test users/items are in train
        test_users = set(test_df['user_id'].unique())
        test_items = set(test_df['item_id'].unique())
        train_users = set(train_df['user_id'].unique())
        train_items = set(train_df['item_id'].unique())

        test_df = test_df[
            test_df['user_id'].isin(train_users) &
            test_df['item_id'].isin(train_items)
        ]

        logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

        return train_df, test_df

    def create_negative_samples(self, df: pd.DataFrame,
                               n_negative: int = 4) -> pd.DataFrame:
        """
        Create negative samples for implicit feedback

        Args:
            df: DataFrame with positive interactions
            n_negative: Number of negative samples per positive

        Returns:
            DataFrame with positive and negative samples
        """
        all_items = set(df['item_id'].unique())
        negative_samples = []

        for user_id in df['user_id'].unique():
            user_items = set(df[df['user_id'] == user_id]['item_id'].values)
            non_interacted = list(all_items - user_items)

            if len(non_interacted) >= n_negative:
                sampled_items = np.random.choice(non_interacted, n_negative, replace=False)
                for item_id in sampled_items:
                    negative_samples.append({
                        'user_id': user_id,
                        'item_id': item_id,
                        'label': 0
                    })

        # Add positive samples
        positive_samples = df[['user_id', 'item_id']].copy()
        positive_samples['label'] = 1

        # Combine
        all_samples = pd.concat([
            positive_samples,
            pd.DataFrame(negative_samples)
        ], ignore_index=True)

        return all_samples.sample(frac=1).reset_index(drop=True)


class FeatureEngineering:
    """Feature engineering for recommendation systems"""

    def __init__(self):
        self.feature_stats = {}

    def create_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create user features from interaction data

        Args:
            df: Interaction DataFrame

        Returns:
            User feature DataFrame
        """
        user_features = []

        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id]

            features = {
                'user_id': user_id,
                'n_interactions': len(user_data),
                'avg_rating': user_data['rating'].mean() if 'rating' in df.columns else 0,
                'std_rating': user_data['rating'].std() if 'rating' in df.columns else 0,
                'n_unique_items': user_data['item_id'].nunique(),
                'interaction_recency': (datetime.now() - pd.to_datetime(user_data['timestamp'].max())).days
                    if 'timestamp' in df.columns else 0,
                'interaction_frequency': len(user_data) / max(1, (
                    pd.to_datetime(user_data['timestamp'].max()) -
                    pd.to_datetime(user_data['timestamp'].min())
                ).days) if 'timestamp' in df.columns else 0
            }

            # Add category preferences if available
            if 'category' in df.columns:
                category_dist = user_data['category'].value_counts(normalize=True)
                for cat, prop in category_dist.head(5).items():
                    features[f'pref_{cat}'] = prop

            user_features.append(features)

        return pd.DataFrame(user_features)

    def create_item_features(self, df: pd.DataFrame,
                            item_metadata: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create item features from interaction data and metadata

        Args:
            df: Interaction DataFrame
            item_metadata: Optional item metadata

        Returns:
            Item feature DataFrame
        """
        item_features = []

        for item_id in df['item_id'].unique():
            item_data = df[df['item_id'] == item_id]

            features = {
                'item_id': item_id,
                'n_interactions': len(item_data),
                'avg_rating': item_data['rating'].mean() if 'rating' in df.columns else 0,
                'std_rating': item_data['rating'].std() if 'rating' in df.columns else 0,
                'n_unique_users': item_data['user_id'].nunique(),
                'popularity_score': len(item_data) / len(df),
                'recency': (datetime.now() - pd.to_datetime(item_data['timestamp'].max())).days
                    if 'timestamp' in df.columns else 0
            }

            # Add metadata features if available
            if item_metadata is not None and item_id in item_metadata.index:
                metadata = item_metadata.loc[item_id]
                for col in ['category', 'price', 'brand']:
                    if col in metadata:
                        features[col] = metadata[col]

            item_features.append(features)

        return pd.DataFrame(item_features)

    def create_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create contextual features from interaction data

        Args:
            df: Interaction DataFrame with timestamp

        Returns:
            DataFrame with context features added
        """
        if 'timestamp' not in df.columns:
            logger.warning("No timestamp column found, skipping context features")
            return df

        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Time of day categories
        df['time_of_day'] = pd.cut(df['hour'],
                                   bins=[0, 6, 12, 18, 24],
                                   labels=['night', 'morning', 'afternoon', 'evening'])

        # Season
        df['season'] = df['month'].apply(lambda x:
            'winter' if x in [12, 1, 2] else
            'spring' if x in [3, 4, 5] else
            'summer' if x in [6, 7, 8] else
            'fall')

        return df

    def create_sequence_features(self, df: pd.DataFrame,
                                 sequence_length: int = 5) -> Dict[int, List]:
        """
        Create sequence features for sequential recommendation

        Args:
            df: Interaction DataFrame
            sequence_length: Length of sequences to create

        Returns:
            Dictionary of user sequences
        """
        sequences = {}

        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id].sort_values('timestamp')
            items = user_data['item_id'].values

            # Create sliding window sequences
            user_sequences = []
            for i in range(len(items) - sequence_length + 1):
                seq = items[i:i + sequence_length]
                user_sequences.append(seq.tolist())

            if user_sequences:
                sequences[user_id] = user_sequences

        return sequences


class EvaluationMetrics:
    """Comprehensive evaluation metrics for recommendation systems"""

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def precision_at_k(recommendations: List[int], relevant: List[int], k: int) -> float:
        """Precision@K"""
        if not recommendations or not relevant:
            return 0.0

        recs_at_k = recommendations[:k]
        relevant_set = set(relevant)

        return sum(1 for item in recs_at_k if item in relevant_set) / k

    @staticmethod
    def recall_at_k(recommendations: List[int], relevant: List[int], k: int) -> float:
        """Recall@K"""
        if not recommendations or not relevant:
            return 0.0

        recs_at_k = set(recommendations[:k])
        relevant_set = set(relevant)

        if not relevant_set:
            return 0.0

        return len(recs_at_k & relevant_set) / len(relevant_set)

    @staticmethod
    def f1_at_k(recommendations: List[int], relevant: List[int], k: int) -> float:
        """F1@K"""
        prec = EvaluationMetrics.precision_at_k(recommendations, relevant, k)
        rec = EvaluationMetrics.recall_at_k(recommendations, relevant, k)

        if prec + rec == 0:
            return 0.0

        return 2 * (prec * rec) / (prec + rec)

    @staticmethod
    def ndcg_at_k(recommendations: List[int], relevant: List[int],
                  relevance_scores: Optional[Dict[int, float]] = None, k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain@K

        Args:
            recommendations: List of recommended items
            relevant: List of relevant items
            relevance_scores: Optional relevance scores for items
            k: Cutoff

        Returns:
            NDCG@K score
        """
        if not recommendations or not relevant:
            return 0.0

        # Create relevance scores if not provided
        if relevance_scores is None:
            relevance_scores = {item: 1.0 for item in relevant}

        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommendations[:k]):
            if item in relevance_scores:
                dcg += relevance_scores[item] / np.log2(i + 2)

        # Calculate IDCG (ideal DCG)
        ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def map_at_k(user_recommendations: Dict[int, List[int]],
                 user_relevant: Dict[int, List[int]], k: int = 10) -> float:
        """
        Mean Average Precision@K

        Args:
            user_recommendations: Dictionary of user recommendations
            user_relevant: Dictionary of relevant items per user
            k: Cutoff

        Returns:
            MAP@K score
        """
        ap_scores = []

        for user_id in user_recommendations:
            if user_id not in user_relevant:
                continue

            recs = user_recommendations[user_id][:k]
            rel = set(user_relevant[user_id])

            if not rel:
                continue

            # Calculate average precision
            ap = 0.0
            n_relevant = 0

            for i, item in enumerate(recs):
                if item in rel:
                    n_relevant += 1
                    ap += n_relevant / (i + 1)

            if n_relevant > 0:
                ap /= min(len(rel), k)

            ap_scores.append(ap)

        return np.mean(ap_scores) if ap_scores else 0.0

    @staticmethod
    def coverage(recommendations: List[List[int]], catalog_size: int) -> float:
        """
        Catalog coverage

        Args:
            recommendations: List of recommendation lists
            catalog_size: Total number of items

        Returns:
            Coverage ratio
        """
        recommended_items = set()
        for recs in recommendations:
            recommended_items.update(recs)

        return len(recommended_items) / catalog_size

    @staticmethod
    def diversity(recommendations: List[int], item_similarity: np.ndarray) -> float:
        """
        Intra-list diversity

        Args:
            recommendations: List of recommended items
            item_similarity: Item similarity matrix

        Returns:
            Diversity score
        """
        if len(recommendations) < 2:
            return 0.0

        diversity = 0.0
        n_pairs = 0

        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                diversity += 1 - item_similarity[recommendations[i], recommendations[j]]
                n_pairs += 1

        return diversity / n_pairs if n_pairs > 0 else 0.0

    @staticmethod
    def novelty(recommendations: List[int], item_popularity: Dict[int, float]) -> float:
        """
        Novelty of recommendations

        Args:
            recommendations: List of recommended items
            item_popularity: Dictionary of item popularity scores

        Returns:
            Novelty score
        """
        if not recommendations:
            return 0.0

        novelty_scores = []
        for item in recommendations:
            pop = item_popularity.get(item, 0.0)
            if pop > 0:
                novelty_scores.append(-np.log2(pop))

        return np.mean(novelty_scores) if novelty_scores else 0.0

    @staticmethod
    def serendipity(recommendations: List[int], expected: List[int],
                    relevant: List[int]) -> float:
        """
        Serendipity of recommendations

        Args:
            recommendations: List of recommended items
            expected: List of expected/obvious recommendations
            relevant: List of relevant items

        Returns:
            Serendipity score
        """
        unexpected_relevant = set(recommendations) & set(relevant) - set(expected)

        if not recommendations:
            return 0.0

        return len(unexpected_relevant) / len(recommendations)


class ColdStartHandler:
    """Handle cold start problems in recommendation systems"""

    def __init__(self):
        self.popular_items = []
        self.default_embeddings = {}

    def fit(self, interactions: pd.DataFrame):
        """
        Fit cold start handler

        Args:
            interactions: Interaction data
        """
        # Get popular items
        item_counts = interactions['item_id'].value_counts()
        self.popular_items = item_counts.head(100).index.tolist()

        # Calculate default embeddings
        self._calculate_defaults(interactions)

    def _calculate_defaults(self, interactions: pd.DataFrame):
        """Calculate default embeddings for cold start"""
        # User default: average of all users
        user_groups = interactions.groupby('user_id')['rating'].mean()
        self.default_embeddings['user'] = user_groups.mean()

        # Item default: average of all items
        item_groups = interactions.groupby('item_id')['rating'].mean()
        self.default_embeddings['item'] = item_groups.mean()

    def handle_cold_user(self, user_features: Optional[Dict] = None,
                         n_recommendations: int = 10) -> List[int]:
        """
        Handle cold start for new users

        Args:
            user_features: Optional user features
            n_recommendations: Number of recommendations

        Returns:
            List of recommended items
        """
        if user_features:
            # Use content-based approach if features available
            return self._content_based_cold_start(user_features, n_recommendations)
        else:
            # Return popular items
            return self.popular_items[:n_recommendations]

    def handle_cold_item(self, item_features: Dict,
                        similar_items: List[int]) -> Dict:
        """
        Handle cold start for new items

        Args:
            item_features: Item features
            similar_items: List of similar existing items

        Returns:
            Cold start strategy for item
        """
        strategy = {
            'method': 'similarity_based',
            'similar_items': similar_items[:10],
            'predicted_rating': self.default_embeddings.get('item', 3.0),
            'confidence': 0.5
        }

        return strategy

    def _content_based_cold_start(self, user_features: Dict,
                                  n_recommendations: int) -> List[int]:
        """Content-based recommendations for cold start users"""
        # Simplified: return items matching user preferences
        # In practice, would use more sophisticated matching
        return self.popular_items[:n_recommendations]


class RecommendationCache:
    """Caching system for recommendations"""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.access_times = {}

    def _create_key(self, user_id: int, context: Optional[Dict] = None) -> str:
        """Create cache key"""
        key_data = {'user_id': user_id}
        if context:
            key_data.update(context)

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, user_id: int, context: Optional[Dict] = None) -> Optional[List]:
        """Get cached recommendations"""
        key = self._create_key(user_id, context)

        if key in self.cache:
            # Check TTL
            if datetime.now() - self.access_times[key] < timedelta(seconds=self.ttl_seconds):
                self.access_times[key] = datetime.now()  # Update access time
                return self.cache[key]
            else:
                # Expired
                del self.cache[key]
                del self.access_times[key]

        return None

    def set(self, user_id: int, recommendations: List,
            context: Optional[Dict] = None):
        """Cache recommendations"""
        key = self._create_key(user_id, context)

        # Check cache size
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        self.cache[key] = recommendations
        self.access_times[key] = datetime.now()

    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_times.clear()

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds,
            'oldest_entry': min(self.access_times.values()) if self.access_times else None,
            'newest_entry': max(self.access_times.values()) if self.access_times else None
        }


class ABTestFramework:
    """A/B testing framework for recommendation systems"""

    def __init__(self):
        self.experiments = {}
        self.results = defaultdict(lambda: {'control': [], 'treatment': []})

    def create_experiment(self, name: str, control_model: Any,
                         treatment_model: Any, split_ratio: float = 0.5):
        """
        Create new A/B test experiment

        Args:
            name: Experiment name
            control_model: Control recommendation model
            treatment_model: Treatment recommendation model
            split_ratio: Proportion of users for treatment
        """
        self.experiments[name] = {
            'control': control_model,
            'treatment': treatment_model,
            'split_ratio': split_ratio,
            'start_time': datetime.now(),
            'metrics': defaultdict(list)
        }

    def assign_variant(self, user_id: int, experiment_name: str) -> str:
        """
        Assign user to variant

        Args:
            user_id: User ID
            experiment_name: Experiment name

        Returns:
            Variant assignment ('control' or 'treatment')
        """
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")

        # Simple hash-based assignment
        hash_val = hash(f"{user_id}_{experiment_name}") % 100
        split_threshold = self.experiments[experiment_name]['split_ratio'] * 100

        return 'treatment' if hash_val < split_threshold else 'control'

    def get_recommendations(self, user_id: int, experiment_name: str,
                           n_recommendations: int = 10) -> Tuple[List, str]:
        """
        Get recommendations for user in experiment

        Args:
            user_id: User ID
            experiment_name: Experiment name
            n_recommendations: Number of recommendations

        Returns:
            Recommendations and variant
        """
        variant = self.assign_variant(user_id, experiment_name)
        model = self.experiments[experiment_name][variant]

        # Get recommendations from appropriate model
        recommendations = model.recommend(user_id, n_recommendations)

        return recommendations, variant

    def log_interaction(self, user_id: int, item_id: int,
                       action: str, experiment_name: str):
        """Log user interaction for analysis"""
        variant = self.assign_variant(user_id, experiment_name)

        self.results[experiment_name][variant].append({
            'user_id': user_id,
            'item_id': item_id,
            'action': action,
            'timestamp': datetime.now()
        })

    def analyze_experiment(self, experiment_name: str) -> Dict:
        """
        Analyze experiment results

        Args:
            experiment_name: Experiment name

        Returns:
            Analysis results
        """
        if experiment_name not in self.results:
            return {}

        control_data = pd.DataFrame(self.results[experiment_name]['control'])
        treatment_data = pd.DataFrame(self.results[experiment_name]['treatment'])

        analysis = {
            'control_size': len(control_data),
            'treatment_size': len(treatment_data),
            'control_ctr': self._calculate_ctr(control_data),
            'treatment_ctr': self._calculate_ctr(treatment_data),
            'lift': None,
            'p_value': None
        }

        # Calculate lift
        if analysis['control_ctr'] > 0:
            analysis['lift'] = (
                (analysis['treatment_ctr'] - analysis['control_ctr']) /
                analysis['control_ctr'] * 100
            )

        # Statistical significance (simplified)
        from scipy.stats import chi2_contingency
        if len(control_data) > 0 and len(treatment_data) > 0:
            # Create contingency table
            control_clicks = len(control_data[control_data['action'] == 'click'])
            treatment_clicks = len(treatment_data[treatment_data['action'] == 'click'])
            control_no_clicks = len(control_data) - control_clicks
            treatment_no_clicks = len(treatment_data) - treatment_clicks

            contingency = [
                [control_clicks, control_no_clicks],
                [treatment_clicks, treatment_no_clicks]
            ]

            if all(sum(row) > 0 for row in contingency):
                _, p_value, _, _ = chi2_contingency(contingency)
                analysis['p_value'] = p_value

        return analysis

    def _calculate_ctr(self, data: pd.DataFrame) -> float:
        """Calculate click-through rate"""
        if len(data) == 0:
            return 0.0

        clicks = len(data[data['action'] == 'click'])
        return clicks / len(data)


# Example usage functions
def create_sample_data(n_users: int = 1000, n_items: int = 500,
                      n_interactions: int = 10000) -> pd.DataFrame:
    """Create sample interaction data for testing"""
    np.random.seed(42)

    data = {
        'user_id': np.random.randint(0, n_users, n_interactions),
        'item_id': np.random.randint(0, n_items, n_interactions),
        'rating': np.random.uniform(1, 5, n_interactions),
        'timestamp': pd.date_range('2024-01-01', periods=n_interactions, freq='1H')
    }

    df = pd.DataFrame(data)

    # Add some categories
    categories = ['electronics', 'books', 'clothing', 'food', 'sports']
    df['category'] = np.random.choice(categories, n_interactions)

    return df


def evaluate_recommender(recommender, test_data: pd.DataFrame, k: int = 10) -> Dict:
    """
    Evaluate a recommender system

    Args:
        recommender: Recommender model with recommend method
        test_data: Test interaction data
        k: Number of recommendations

    Returns:
        Evaluation metrics
    """
    metrics = EvaluationMetrics()
    results = defaultdict(list)

    # Group test data by user
    user_items = test_data.groupby('user_id')['item_id'].apply(list).to_dict()

    for user_id, relevant_items in user_items.items():
        # Get recommendations
        try:
            recommendations = recommender.recommend(user_id, k)
            if isinstance(recommendations[0], tuple):
                recommendations = [item for item, _ in recommendations]
        except:
            continue

        # Calculate metrics
        results['precision'].append(
            metrics.precision_at_k(recommendations, relevant_items, k)
        )
        results['recall'].append(
            metrics.recall_at_k(recommendations, relevant_items, k)
        )
        results['ndcg'].append(
            metrics.ndcg_at_k(recommendations, relevant_items, k=k)
        )

    # Average metrics
    return {
        metric: np.mean(values) if values else 0.0
        for metric, values in results.items()
    }


if __name__ == "__main__":
    # Example usage
    print("Creating sample data...")
    df = create_sample_data()

    print("\nPreprocessing data...")
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.preprocess_interactions(df, min_user_interactions=3)

    print("\nCreating features...")
    feature_eng = FeatureEngineering()
    user_features = feature_eng.create_user_features(processed_df)
    item_features = feature_eng.create_item_features(processed_df)

    print("\nUser feature sample:")
    print(user_features.head())

    print("\nItem feature sample:")
    print(item_features.head())

    print("\nDemonstration complete!")