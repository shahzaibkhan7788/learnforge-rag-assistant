"""Index construction and an optional Qdrant adapter.

The lexical index is always available. Qdrant is intentionally imported lazily
so a fresh checkout can run offline with only the Python standard library.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

from app.schemas.models import Chunk, SearchResult
from app.retrieval.bm25 import BM25Retriever


class LexicalIndex:
    def __init__(self, chunks: Iterable[Chunk]):
        self.chunks = list(chunks)
        self.retriever = BM25Retriever(self.chunks)

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        return self.retriever.search(query, limit=limit)


def build_index(chunks: Iterable[Chunk]) -> LexicalIndex:
    """Build the guaranteed local index; deployments can swap in Qdrant."""
    return LexicalIndex(chunks)


class QdrantIndex:
    """Optional adapter; callers should use it only with a configured client."""

    def __init__(self, chunks: Iterable[Chunk], *, url: Optional[str] = None, collection: str = "support"):
        self.chunks = list(chunks)
        self.url = url or os.getenv("QDRANT_URL")
        self.collection = collection
        self.client = None
        if self.url:
            try:
                from qdrant_client import QdrantClient  # type: ignore

                self.client = QdrantClient(url=self.url, api_key=os.getenv("QDRANT_API_KEY"))
            except ImportError:
                self.client = None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def search(self, _query: str, limit: int = 10) -> List[SearchResult]:
        # Embedding creation is provider-specific; leave this adapter inert
        # until an embedding model is configured by the deployment.
        return []
