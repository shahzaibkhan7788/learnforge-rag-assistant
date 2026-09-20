"""Confidence heuristic used for safe fallback and escalation."""

from __future__ import annotations

from typing import Iterable

from app.schemas.models import SearchResult


def confidence_score(results: Iterable[SearchResult]) -> float:
    values = list(results)
    if not values:
        return 0.0
    top = max(0.0, values[0].score)
    margin = (values[0].score - values[1].score) if len(values) > 1 else top
    # RRF scores are small, so combine rank evidence and normalized margin.
    score = min(1.0, 0.45 + min(0.35, top * 10) + min(0.2, max(0.0, margin) * 10))
    if values[0].chunk.metadata.get("kind") == "ticket":
        score -= 0.08
    return round(max(0.0, min(1.0, score)), 3)
