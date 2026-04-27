"""Recursive character-level chunker, compatible in spirit with LangChain's
`RecursiveCharacterTextSplitter` but implemented here to keep the dep graph minimal.

Strategy: try to split at the most semantic separator that produces pieces smaller
than `chunk_size` (paragraph > sentence > word > character), then greedy-merge
adjacent pieces up to `chunk_size` with `chunk_overlap` tail carry-over.
"""
from __future__ import annotations

import re
from typing import Iterable

from .schemas import Chunk, Document

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: tuple[str, ...] = DEFAULT_SEPARATORS,
) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Pick the first separator that actually appears in the text (or fall back to char-level).
    chosen = ""
    remaining: tuple[str, ...] = separators
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            chosen = sep
            remaining = separators[i + 1:]
            break

    if chosen == "":
        # Character-level fallback
        pieces = [
            text[i : i + chunk_size]
            for i in range(0, len(text), max(1, chunk_size - chunk_overlap))
        ]
        return [p for p in pieces if p.strip()]

    raw_pieces = text.split(chosen)
    # Re-glue the separator back onto all but the last piece so the split is lossless
    pieces = [p + chosen for p in raw_pieces[:-1]] + raw_pieces[-1:]

    processed: list[str] = []
    for p in pieces:
        if len(p) > chunk_size:
            processed.extend(_recursive_split(p, chunk_size, chunk_overlap, remaining))
        elif p.strip():
            processed.append(p)

    return _merge(processed, chunk_size, chunk_overlap)


def _merge(pieces: list[str], max_size: int, overlap: int) -> list[str]:
    """Greedy packer with tail-carryover for overlap."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for piece in pieces:
        if current and current_len + len(piece) > max_size:
            chunks.append("".join(current).strip())
            # Carry overlap: keep tail pieces up to `overlap` chars
            tail: list[str] = []
            tail_len = 0
            for p in reversed(current):
                if tail_len + len(p) > overlap:
                    break
                tail.insert(0, p)
                tail_len += len(p)
            current = tail[:]
            current_len = tail_len
        current.append(piece)
        current_len += len(piece)

    if current:
        out = "".join(current).strip()
        if out:
            chunks.append(out)

    return chunks


def chunk_document(doc: Document, chunk_size: int = 512, chunk_overlap: int = 64) -> list[Chunk]:
    """Produce a list of Chunks from one Document."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    texts = _recursive_split(doc.text, chunk_size, chunk_overlap)
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}#c{i:03d}",
            doc_id=doc.doc_id,
            text=text.strip(),
            position=i,
            metadata=dict(doc.metadata),
        )
        for i, text in enumerate(texts)
        if text.strip()
    ]


def chunk_documents(
    docs: Iterable[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return out


def chunk_size_stats(chunks: list[Chunk]) -> dict[str, float]:
    """Summary stats on chunk lengths (characters)."""
    if not chunks:
        return {"n": 0, "mean_chars": 0, "p50_chars": 0, "p95_chars": 0, "max_chars": 0}
    lengths = sorted(len(c.text) for c in chunks)
    return {
        "n": len(chunks),
        "mean_chars": sum(lengths) / len(lengths),
        "p50_chars": lengths[len(lengths) // 2],
        "p95_chars": lengths[max(0, int(0.95 * len(lengths)) - 1)],
        "max_chars": lengths[-1],
    }
