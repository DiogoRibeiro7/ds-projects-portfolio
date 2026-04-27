"""Retrieval primitives:

- `DenseRetriever` — FAISS flat-IP over sentence-transformer embeddings.
- `BM25Retriever` — rank_bm25 Okapi over tokenised chunk text.
- `HybridRetriever` — reciprocal-rank-fusion of dense + BM25.
- `CrossEncoderReranker` — `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranker.
- `hyde_transform` / `multi_query_rewrite` — query transforms.
"""
from __future__ import annotations

import json
import re
from typing import Sequence

import numpy as np

from .embeddings import Embedder
from .llm import LLMClient
from .schemas import Chunk, LLMMessage, RetrievedChunk

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


# ----------------------------------------------------------------------------
# Dense retrieval
# ----------------------------------------------------------------------------

class DenseRetriever:
    """Cosine-similarity retrieval over an embedded chunk corpus, backed by FAISS flat-IP."""

    def __init__(self, embedder: Embedder, chunks: list[Chunk]):
        import faiss
        self._embedder = embedder
        self._chunks = chunks
        self._by_id = {c.chunk_id: c for c in chunks}
        vecs = embedder.embed([c.text for c in chunks])
        self._index = faiss.IndexFlatIP(embedder.dim)
        self._index.add(vecs)
        self._vecs = vecs

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    def retrieve(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        qv = self._embedder.embed_query(query).reshape(1, -1)
        return self._search_vec(qv, k)

    def retrieve_by_vector(self, qv: np.ndarray, k: int = 10) -> list[RetrievedChunk]:
        if qv.ndim == 1:
            qv = qv.reshape(1, -1)
        return self._search_vec(qv.astype(np.float32), k)

    def _search_vec(self, qv: np.ndarray, k: int) -> list[RetrievedChunk]:
        if not self._chunks:
            return []
        D, I = self._index.search(qv, min(k, len(self._chunks)))
        out: list[RetrievedChunk] = []
        for rank, (score, idx) in enumerate(zip(D[0], I[0])):
            if idx < 0:
                continue
            c = self._chunks[int(idx)]
            out.append(RetrievedChunk(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                text=c.text,
                score=float(score),
                retriever="dense",
                rank=rank,
            ))
        return out


# ----------------------------------------------------------------------------
# BM25 retrieval
# ----------------------------------------------------------------------------

class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi
        self._chunks = chunks
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    def retrieve(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        if not self._chunks:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        order = np.argsort(-scores)[:k]
        return [
            RetrievedChunk(
                chunk_id=self._chunks[int(i)].chunk_id,
                doc_id=self._chunks[int(i)].doc_id,
                text=self._chunks[int(i)].text,
                score=float(scores[int(i)]),
                retriever="bm25",
                rank=rank,
            )
            for rank, i in enumerate(order)
        ]


# ----------------------------------------------------------------------------
# Hybrid = RRF(dense, bm25)
# ----------------------------------------------------------------------------

def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[RetrievedChunk]],
    *,
    k_rrf: int = 60,
    k: int = 10,
) -> list[RetrievedChunk]:
    """Cormack et al. 2009 reciprocal rank fusion."""
    scores: dict[str, float] = {}
    source: dict[str, RetrievedChunk] = {}
    for rlist in result_lists:
        for r in rlist:
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (k_rrf + r.rank + 1)
            source[r.chunk_id] = r
    ordered = sorted(scores.keys(), key=lambda cid: -scores[cid])[:k]
    return [
        RetrievedChunk(
            chunk_id=cid,
            doc_id=source[cid].doc_id,
            text=source[cid].text,
            score=scores[cid],
            retriever="hybrid",
            rank=rank,
        )
        for rank, cid in enumerate(ordered)
    ]


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, k_rrf: int = 60):
        self.dense = dense
        self.bm25 = bm25
        self.k_rrf = k_rrf

    def retrieve(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        dres = self.dense.retrieve(query, k=k * 2)
        bres = self.bm25.retrieve(query, k=k * 2)
        return reciprocal_rank_fusion([dres, bres], k_rrf=self.k_rrf, k=k)


# ----------------------------------------------------------------------------
# Cross-encoder reranker
# ----------------------------------------------------------------------------

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if not results:
            return []
        pairs = [[query, r.text] for r in results]
        scores = self._model.predict(pairs, show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))
        top_k = top_k or len(order)
        return [
            RetrievedChunk(
                chunk_id=results[int(i)].chunk_id,
                doc_id=results[int(i)].doc_id,
                text=results[int(i)].text,
                score=float(scores[int(i)]),
                retriever="reranked",
                rank=rank,
            )
            for rank, i in enumerate(order[:top_k])
        ]


# ----------------------------------------------------------------------------
# Query transforms: HyDE + multi-query
# ----------------------------------------------------------------------------

HYDE_PROMPT = (
    "Write a short, hypothetical passage (2-3 sentences) that would directly answer "
    "the following question. Write it in the voice of an authoritative policy document; "
    "do NOT hedge or say 'I don't know'.\n\n"
    "Question: {query}\n"
    "Hypothetical passage:"
)

MULTI_QUERY_PROMPT = (
    "Generate exactly {n} diverse reformulations of the following query. They should "
    "preserve the intent but use different wording or angles (synonyms, more specific, "
    "more general). Return them as a JSON array of strings with no other text.\n\n"
    "Original query: {query}\n"
    "Reformulations (JSON list):"
)


def hyde_transform(query: str, llm: LLMClient, embedder: Embedder) -> np.ndarray:
    """HyDE (Gao et al. 2022): generate a hypothetical answer and embed it."""
    resp = llm.complete(
        [LLMMessage(role="user", content=HYDE_PROMPT.format(query=query))],
        max_tokens=160,
        temperature=0.3,
    )
    return embedder.embed_query(resp.text)


def multi_query_rewrite(query: str, llm: LLMClient, n: int = 3) -> list[str]:
    """Return the original query plus LLM-generated reformulations (n+1 total)."""
    resp = llm.complete(
        [LLMMessage(role="user", content=MULTI_QUERY_PROMPT.format(query=query, n=n))],
        max_tokens=240,
        temperature=0.5,
    )
    raw = resp.text.strip()
    # Be robust to minor formatting drift
    start = raw.find("[")
    end = raw.rfind("]")
    variants: list[str] = []
    if 0 <= start < end:
        try:
            parsed = json.loads(raw[start : end + 1])
            variants = [str(v).strip() for v in parsed if str(v).strip()]
        except json.JSONDecodeError:
            pass
    # Always include the original query first, dedup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for q in [query] + variants:
        if q and q not in seen:
            out.append(q)
            seen.add(q)
        if len(out) >= n + 1:
            break
    return out
