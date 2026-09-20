"""Transparent reranking of fused candidates."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Iterable, List, Optional

from app.schemas.models import SearchResult
from .bm25 import tokenize

STALE_WORDS = ("older", "outdated", "obsolete", "archived", "previous", "retired")


def freshness(reviewed: Optional[str]) -> float:
    if not reviewed:
        return 0.58
    try:
        age = max(0, (date.today() - datetime.strptime(reviewed, "%Y-%m").date()).days)
        return max(0.2, 1.0 - age / 730.0)
    except ValueError:
        return 0.7


def rerank(query: str, results: Iterable[SearchResult], limit: int = 5) -> List[SearchResult]:
    query_terms = set(tokenize(query))
    ranked = []
    for result in results:
        metadata = result.chunk.metadata
        terms = set(tokenize(result.chunk.text))
        title_terms = set(tokenize(result.chunk.section))
        score = result.score
        score += 0.08 * len(query_terms.intersection(title_terms))
        score += 0.04 * len(query_terms.intersection(terms)) / max(1, len(query_terms))
        score += 0.05 * float(metadata.get("authority", 0.5))
        score += 0.04 * freshness(metadata.get("reviewed"))
        if metadata.get("kind") == "ticket":
            score -= 0.03
        if any(word in result.chunk.text.lower() for word in STALE_WORDS):
            score -= 0.04
        ranked.append(SearchResult(result.chunk, score, "reranked", result.matched_terms))
    ranked.sort(key=lambda item: (-item.score, item.chunk.id))
    return ranked[: max(1, limit)]
