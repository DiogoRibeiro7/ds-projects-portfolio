"""NLP Utilities
=============

Utility functions for natural language processing tasks including:
- Text preprocessing
- Feature extraction
- Metrics and evaluation
- Visualization
- Data handling
"""

import json
import re
import string
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

# Visualization
import matplotlib.pyplot as plt

# NLP libraries
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

# Download required NLTK data
nltk_resources = [
    "punkt",
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger",
    "maxent_ne_chunker",
    "words",
    "vader_lexicon",
    "omw-1.4",
]
for resource in nltk_resources:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass


# ============================================================================
# Text Preprocessing
# ============================================================================


class TextPreprocessor:
    """Comprehensive text preprocessing utilities."""

    def __init__(self, language: str = "english"):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.snowball_stemmer = SnowballStemmer(language)

    def clean_text(
        self,
        text: str,
        lowercase: bool = True,
        remove_numbers: bool = False,
        remove_punctuation: bool = True,
        remove_whitespace: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_special_chars: bool = True,
    ) -> str:
        """Clean text with various options."""
        # Remove URLs
        if remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Remove emails
        if remove_emails:
            text = re.sub(r"\S+@\S+", "", text)

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Lowercase
        if lowercase:
            text = text.lower()

        # Remove numbers
        if remove_numbers:
            text = re.sub(r"\d+", "", text)

        # Remove punctuation
        if remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove special characters
        if remove_special_chars:
            text = re.sub(r"[^\w\s]", " ", text)

        # Remove extra whitespace
        if remove_whitespace:
            text = " ".join(text.split())

        return text

    def tokenize(
        self, text: str, method: str = "word", preserve_case: bool = False
    ) -> list[str]:
        """Tokenize text."""
        if method == "word":
            tokens = word_tokenize(text)
        elif method == "sentence":
            tokens = sent_tokenize(text)
        elif method == "whitespace":
            tokens = text.split()
        else:
            raise ValueError(f"Unknown tokenization method: {method}")

        if not preserve_case and method != "sentence":
            tokens = [t.lower() for t in tokens]

        return tokens

    def remove_stopwords(
        self, tokens: list[str], custom_stopwords: list[str] | None = None
    ) -> list[str]:
        """Remove stopwords from tokens."""
        stop_words = self.stop_words.copy()
        if custom_stopwords:
            stop_words.update(custom_stopwords)

        return [token for token in tokens if token.lower() not in stop_words]

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """Lemmatize tokens."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def stem(self, tokens: list[str], method: str = "porter") -> list[str]:
        """Stem tokens."""
        if method == "porter":
            return [self.stemmer.stem(token) for token in tokens]
        elif method == "snowball":
            return [self.snowball_stemmer.stem(token) for token in tokens]
        else:
            raise ValueError(f"Unknown stemming method: {method}")

    def ngrams(self, tokens: list[str], n: int = 2) -> list[tuple[str, ...]]:
        """Generate n-grams from tokens."""
        if n < 1:
            raise ValueError("n must be at least 1")

        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i : i + n]))

        return ngrams

    def preprocess_pipeline(
        self, text: str, steps: list[str] = None
    ) -> str | list[str]:
        """Apply preprocessing pipeline."""
        if steps is None:
            steps = ["clean", "tokenize", "stopwords", "lemmatize"]

        result = text

        for step in steps:
            if step == "clean":
                result = self.clean_text(result)
            elif step == "tokenize":
                result = self.tokenize(result)
            elif step == "stopwords" and isinstance(result, list):
                result = self.remove_stopwords(result)
            elif step == "lemmatize" and isinstance(result, list):
                result = self.lemmatize(result)
            elif step == "stem" and isinstance(result, list):
                result = self.stem(result)
            elif step == "join" and isinstance(result, list):
                result = " ".join(result)

        return result


# ============================================================================
# Feature Extraction
# ============================================================================


class FeatureExtractor:
    """Extract features from text."""

    @staticmethod
    def extract_basic_features(text: str) -> dict[str, Any]:
        """Extract basic text features."""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())

        features = {
            "char_count": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": np.mean([len(word) for word in words]) if words else 0,
            "avg_sentence_length": np.mean(
                [len(word_tokenize(sent)) for sent in sentences]
            )
            if sentences
            else 0,
            "unique_words": len(set(words)),
            "vocabulary_richness": len(set(words)) / len(words) if words else 0,
        }

        return features

    @staticmethod
    def extract_pos_features(text: str) -> dict[str, int]:
        """Extract part-of-speech features."""
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)

        pos_counts = Counter([tag for _, tag in pos_tags])

        return dict(pos_counts)

    @staticmethod
    def extract_readability_features(text: str) -> dict[str, float]:
        """Extract readability features."""
        try:
            import textstat

            features = {
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                "gunning_fog": textstat.gunning_fog(text),
                "smog_index": textstat.smog_index(text),
                "automated_readability_index": textstat.automated_readability_index(
                    text
                ),
                "coleman_liau_index": textstat.coleman_liau_index(text),
            }
        except ImportError:
            features = {}

        return features

    @staticmethod
    def extract_sentiment_features(text: str) -> dict[str, float]:
        """Extract sentiment features."""
        from nltk.sentiment import SentimentIntensityAnalyzer

        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)

        return scores

    @staticmethod
    def create_tfidf_features(
        texts: list[str],
        max_features: int = 1000,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> tuple[np.ndarray, list[str]]:
        """Create TF-IDF features."""
        vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range, stop_words="english"
        )

        features = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        return features, feature_names


# ============================================================================
# Text Similarity
# ============================================================================


def calculate_similarity(text1: str, text2: str, method: str = "jaccard") -> float:
    """Calculate similarity between two texts."""
    # Tokenize
    tokens1 = set(word_tokenize(text1.lower()))
    tokens2 = set(word_tokenize(text2.lower()))

    if method == "jaccard":
        # Jaccard similarity
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        return len(intersection) / len(union) if union else 0

    elif method == "dice":
        # Dice coefficient
        intersection = tokens1.intersection(tokens2)
        return (
            2 * len(intersection) / (len(tokens1) + len(tokens2))
            if (tokens1 or tokens2)
            else 0
        )

    elif method == "overlap":
        # Overlap coefficient
        intersection = tokens1.intersection(tokens2)
        return (
            len(intersection) / min(len(tokens1), len(tokens2))
            if min(len(tokens1), len(tokens2)) > 0
            else 0
        )

    else:
        raise ValueError(f"Unknown similarity method: {method}")


def edit_distance(str1: str, str2: str) -> int:
    """Calculate Levenshtein edit distance."""
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


# ============================================================================
# Evaluation Metrics
# ============================================================================


class NLPMetrics:
    """NLP-specific evaluation metrics."""

    @staticmethod
    def bleu_score(reference: str, hypothesis: str, n: int = 4) -> float:
        """Calculate BLEU score."""
        from nltk.translate.bleu_score import sentence_bleu

        reference_tokens = word_tokenize(reference)
        hypothesis_tokens = word_tokenize(hypothesis)

        # Calculate weights for n-grams
        weights = tuple([1 / n] * n)

        score = sentence_bleu([reference_tokens], hypothesis_tokens, weights=weights)

        return score

    @staticmethod
    def rouge_scores(reference: str, hypothesis: str) -> dict[str, float]:
        """Calculate ROUGE scores."""
        try:
            from rouge import Rouge

            rouge = Rouge()
            scores = rouge.get_scores(hypothesis, reference)[0]

            return {
                "rouge-1": scores["rouge-1"]["f"],
                "rouge-2": scores["rouge-2"]["f"],
                "rouge-l": scores["rouge-l"]["f"],
            }
        except ImportError:
            return {}

    @staticmethod
    def perplexity(probabilities: np.ndarray) -> float:
        """Calculate perplexity."""
        log_prob = np.log2(probabilities + 1e-10)
        avg_log_prob = np.mean(log_prob)
        perplexity = 2 ** (-avg_log_prob)

        return perplexity

    @staticmethod
    def coherence_score(topics: list[list[str]], texts: list[str]) -> float:
        """Calculate topic coherence score."""
        # Simplified coherence calculation
        # In practice, use gensim's CoherenceModel

        coherence_scores = []

        for topic in topics:
            topic_coherence = 0
            count = 0

            for text in texts:
                text_words = set(word_tokenize(text.lower()))
                topic_words = set(topic)

                intersection = len(text_words.intersection(topic_words))
                if intersection > 0:
                    topic_coherence += intersection / len(topic_words)
                    count += 1

            if count > 0:
                coherence_scores.append(topic_coherence / count)

        return np.mean(coherence_scores) if coherence_scores else 0


# ============================================================================
# Visualization
# ============================================================================


class NLPVisualizer:
    """Visualization utilities for NLP."""

    @staticmethod
    def plot_word_cloud(
        text: str,
        max_words: int = 100,
        background_color: str = "white",
        colormap: str = "viridis",
        save_path: str | None = None,
    ) -> None:
        """Generate and plot word cloud."""
        wordcloud = WordCloud(
            width=800,
            height=400,
            max_words=max_words,
            background_color=background_color,
            colormap=colormap,
            relative_scaling=0.5,
        ).generate(text)

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)

        plt.show()

    @staticmethod
    def plot_pos_distribution(text: str) -> None:
        """Plot part-of-speech distribution."""
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        pos_counts = Counter([tag for _, tag in pos_tags])

        # Sort by frequency
        sorted_pos = sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        labels, values = zip(*sorted_pos, strict=False)

        plt.figure(figsize=(10, 6))
        plt.bar(labels, values, color="skyblue", edgecolor="navy")
        plt.xlabel("Part of Speech")
        plt.ylabel("Frequency")
        plt.title("Part-of-Speech Distribution")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_topic_distribution(
        topic_weights: np.ndarray, topic_labels: list[str] | None = None
    ) -> None:
        """Plot topic distribution."""
        n_topics = (
            topic_weights.shape[1]
            if len(topic_weights.shape) > 1
            else len(topic_weights)
        )

        if topic_labels is None:
            topic_labels = [f"Topic {i}" for i in range(n_topics)]

        if len(topic_weights.shape) > 1:
            # Multiple documents
            avg_weights = np.mean(topic_weights, axis=0)
        else:
            avg_weights = topic_weights

        plt.figure(figsize=(10, 6))
        plt.bar(topic_labels, avg_weights, color="coral", edgecolor="darkred")
        plt.xlabel("Topics")
        plt.ylabel("Average Weight")
        plt.title("Topic Distribution")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(
        y_true: list, y_pred: list, labels: list[str] | None = None
    ) -> None:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.show()


# ============================================================================
# Data Handling
# ============================================================================


class NLPDataHandler:
    """Handle NLP data loading and saving."""

    @staticmethod
    def load_text_file(file_path: str, encoding: str = "utf-8") -> str:
        """Load text from file."""
        with open(file_path, encoding=encoding) as f:
            text = f.read()

        return text

    @staticmethod
    def load_json_lines(file_path: str) -> list[dict]:
        """Load JSON lines file."""
        data = []
        with open(file_path) as f:
            for line in f:
                data.append(json.loads(line))

        return data

    @staticmethod
    def save_json_lines(data: list[dict], file_path: str) -> None:
        """Save data as JSON lines."""
        with open(file_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    @staticmethod
    def load_corpus(directory: str, pattern: str = "*.txt") -> list[tuple[str, str]]:
        """Load corpus from directory."""
        corpus = []
        path = Path(directory)

        for file_path in path.glob(pattern):
            text = NLPDataHandler.load_text_file(file_path)
            corpus.append((file_path.name, text))

        return corpus

    @staticmethod
    def split_dataset(
        texts: list[str],
        labels: list[Any],
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> tuple:
        """Split dataset into train, validation, and test sets."""
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels if len(set(labels)) > 1 else None,
        )

        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp if len(set(y_temp)) > 1 else None,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# Text Generation Helpers
# ============================================================================


def generate_text_statistics_report(texts: list[str]) -> pd.DataFrame:
    """Generate comprehensive statistics report for texts."""
    extractor = FeatureExtractor()
    stats = []

    for i, text in enumerate(texts):
        text_stats = extractor.extract_basic_features(text)
        text_stats["text_id"] = i
        text_stats["text_preview"] = text[:100] + "..." if len(text) > 100 else text
        stats.append(text_stats)

    df = pd.DataFrame(stats)

    # Add summary statistics
    summary = pd.DataFrame(
        {
            "mean": df.select_dtypes(include=[np.number]).mean(),
            "std": df.select_dtypes(include=[np.number]).std(),
            "min": df.select_dtypes(include=[np.number]).min(),
            "max": df.select_dtypes(include=[np.number]).max(),
        }
    ).T

    return df, summary


def extract_keywords(
    text: str, method: str = "tfidf", n_keywords: int = 10
) -> list[tuple[str, float]]:
    """Extract keywords from text."""
    if method == "tfidf":
        # Use TF-IDF for single document (treating sentences as documents)
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            sentences = [text]

        vectorizer = TfidfVectorizer(max_features=n_keywords, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Get feature names and scores
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.sum(axis=0).A1

        # Sort by score
        keyword_scores = [
            (feature_names[i], scores[i]) for i in scores.argsort()[-n_keywords:][::-1]
        ]

        return keyword_scores

    elif method == "frequency":
        # Simple frequency-based extraction
        preprocessor = TextPreprocessor()
        tokens = preprocessor.tokenize(text)
        tokens = preprocessor.remove_stopwords(tokens)

        freq_dist = Counter(tokens)
        return freq_dist.most_common(n_keywords)

    else:
        raise ValueError(f"Unknown keyword extraction method: {method}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example text
    sample_text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science,
    and artificial intelligence concerned with the interactions between computers and
    human language, in particular how to program computers to process and analyze
    large amounts of natural language data.
    """

    # Preprocessing
    preprocessor = TextPreprocessor()
    clean_text = preprocessor.clean_text(sample_text)
    tokens = preprocessor.tokenize(clean_text)
    tokens_no_stop = preprocessor.remove_stopwords(tokens)
    lemmatized = preprocessor.lemmatize(tokens_no_stop)

    print("Original text:", sample_text[:100])
    print("Clean text:", clean_text[:100])
    print("Tokens:", tokens[:10])
    print("After removing stopwords:", tokens_no_stop[:10])
    print("After lemmatization:", lemmatized[:10])

    # Feature extraction
    extractor = FeatureExtractor()
    features = extractor.extract_basic_features(sample_text)
    print("\nBasic features:", features)

    # Keywords
    keywords = extract_keywords(sample_text, method="tfidf", n_keywords=5)
    print("\nKeywords:", keywords)

    print("\nNLP utilities loaded successfully!")
