"""Embedding-model wrapper.

Defaults to `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ~90MB, CPU-friendly).
The model is loaded lazily on first `embed` call and cached at module scope via
`get_embedder`. Notebooks should use `get_embedder(...)` rather than constructing
their own instances, to avoid reloading the model across cells.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int
    model_name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Thin wrapper around sentence-transformers with batching and L2-normalisation."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
        batch_size: int = 32,
    ):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        self._normalize = normalize
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        arr = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
        )
        return arr.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


@lru_cache(maxsize=4)
def get_embedder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> SentenceTransformerEmbedder:
    """Return a process-wide cached embedder. Safe to call from any notebook cell."""
    return SentenceTransformerEmbedder(model_name=model_name)
