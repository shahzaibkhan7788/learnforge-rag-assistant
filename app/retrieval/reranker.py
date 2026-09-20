"""Transparent reranking of fused candidates."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Iterable, List, Optional, Tuple

from app.schemas.models import Chunk, SearchResult
from .bm25 import tokenize

STALE_WORDS = ("older", "outdated", "obsolete", "archived", "previous", "retired")
FACT_QUERY_TERMS = {
    "amount", "date", "days", "duration", "how", "limit", "many", "period",
    "percentage", "price", "rate", "time", "when", "what", "weeks",
}
VALUE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|days?|weeks?|months?|years?|hours?|minutes?)\b"
    r"|\b(?:yes|no|never|always)\b",
    re.IGNORECASE,
)


def freshness(reviewed: Optional[str]) -> float:
    if not reviewed:
        return 0.58
    try:
        age = max(0, (date.today() - datetime.strptime(reviewed, "%Y-%m").date()).days)
        return max(0.2, 1.0 - age / 730.0)
    except ValueError:
        return 0.7


def _sentences(text: str) -> List[str]:
    """Keep evidence units small enough to answer the requested fact directly."""
    normalized = re.sub(r"\s+", " ", text).strip()
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
    ]


def _sentence_score(
    sentence: str,
    query_terms: set[str],
    *,
    fact_query: bool,
    authority: float,
) -> float:
    sentence_terms = set(tokenize(sentence))
    overlap = query_terms.intersection(sentence_terms)
    score = len(overlap) / max(1, len(query_terms))
    if fact_query and VALUE_PATTERN.search(sentence):
        score += 0.28
    if any(word in sentence.lower() for word in STALE_WORDS):
        score -= 0.35
    score += 0.08 * authority
    return score


def _focused_chunk(
    result: SearchResult,
    query_terms: set[str],
    *,
    fact_query: bool,
) -> Tuple[SearchResult, float]:
    sentences = _sentences(result.chunk.text)
    if len(sentences) <= 1:
        return result, 0.0
    authority = float(result.chunk.metadata.get("authority", 0.5))
    scored = sorted(
        (
            (_sentence_score(sentence, query_terms, fact_query=fact_query, authority=authority), index, sentence)
            for index, sentence in enumerate(sentences)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    best_score, best_index, best_sentence = scored[0]
    if best_score <= 0:
        return result, 0.0

    # Keep one nearby qualification, but never return an unrelated paragraph.
    selected = [best_sentence]
    for candidate_score, candidate_index, candidate_sentence in scored[1:]:
        if candidate_index == best_index + 1 and candidate_score >= best_score * 0.55:
            selected.append(candidate_sentence)
        break
    focused_text = " ".join(selected)
    focused_chunk = Chunk(
        id=result.chunk.id,
        document_id=result.chunk.document_id,
        text=focused_text,
        section=result.chunk.section,
        ordinal=result.chunk.ordinal,
        metadata={**result.chunk.metadata, "focused_evidence": True},
    )
    return SearchResult(
        focused_chunk,
        result.score,
        result.source,
        result.matched_terms,
    ), best_score


def rerank(query: str, results: Iterable[SearchResult], limit: int = 5) -> List[SearchResult]:
    query_terms = set(tokenize(query))
    fact_query = bool(query_terms.intersection(FACT_QUERY_TERMS))
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
            score -= 0.12
        if any(word in result.chunk.text.lower() for word in STALE_WORDS):
            score -= 0.04
        focused, evidence_score = _focused_chunk(
            result, query_terms, fact_query=fact_query
        )
        score += 0.18 * evidence_score
        ranked.append(SearchResult(focused.chunk, score, "reranked", result.matched_terms))
    ranked.sort(key=lambda item: (-item.score, item.chunk.id))
    return ranked[: max(1, limit)]
