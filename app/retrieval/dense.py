"""Optional dense retrieval boundary.

No model is downloaded by default. A configured sentence-transformers/Qdrant
deployment can subclass this interface without changing the query pipeline.
"""

from __future__ import annotations

import os
from typing import Iterable, List

from app.schemas.models import Chunk, SearchResult


class DenseRetriever:
    def __init__(self, chunks: Iterable[Chunk]):
        self.chunks = list(chunks)
        self.enabled = bool(os.getenv("DENSE_RETRIEVAL_ENABLED")) and self._available()

    @staticmethod
    def _available() -> bool:
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def search(self, _query: str, limit: int = 10) -> List[SearchResult]:
        # Keeping the fallback empty is preferable to silently downloading a
        # model or making network calls in a local/offline deployment.
        return []
