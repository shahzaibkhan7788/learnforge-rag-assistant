"""Hybrid retrieval using reciprocal-rank fusion (RRF)."""

from __future__ import annotations

from typing import Iterable, List, Optional

from app.schemas.models import Chunk, SearchResult
from .bm25 import BM25Retriever
from .dense import DenseRetriever


class HybridRetriever:
    def __init__(
        self,
        chunks: Iterable[Chunk],
        *,
        bm25: Optional[BM25Retriever] = None,
        dense: Optional[DenseRetriever] = None,
        rrf_k: int = 60,
    ):
        self.chunks = list(chunks)
        self.bm25 = bm25 or BM25Retriever(self.chunks)
        self.dense = dense or DenseRetriever(self.chunks)
        self.rrf_k = rrf_k

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        lexical = self.bm25.search(query, limit=max(limit * 3, 10))
        dense = self.dense.search(query, limit=max(limit * 3, 10))
        fused = {}
        for rank, result in enumerate(lexical, 1):
            fused.setdefault(result.chunk.id, [result, 0.0])
            fused[result.chunk.id][1] += 1.0 / (self.rrf_k + rank)
        for rank, result in enumerate(dense, 1):
            fused.setdefault(result.chunk.id, [result, 0.0])
            fused[result.chunk.id][1] += 1.0 / (self.rrf_k + rank)
        values: List[SearchResult] = []
        for result, score in fused.values():
            values.append(SearchResult(result.chunk, score, "rrf", result.matched_terms))
        values.sort(key=lambda item: (-item.score, item.chunk.id))
        return values[: max(1, limit)]
