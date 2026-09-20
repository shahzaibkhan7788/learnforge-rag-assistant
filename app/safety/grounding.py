"""Simple evidence-overlap validation for generated answers."""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from app.retrieval.bm25 import tokenize
from app.schemas.models import SearchResult


def validate_grounding(answer: str, results: Sequence[SearchResult], minimum: float = 0.12) -> bool:
    if not results:
        return False
    evidence_terms = set(tokenize(" ".join(result.chunk.text for result in results[:3])))
    answer_terms = set(tokenize(answer))
    return bool(answer_terms) and len(answer_terms.intersection(evidence_terms)) / len(answer_terms) >= minimum
