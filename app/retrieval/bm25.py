"""Dependency-free BM25 implementation."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List, Sequence, Set

from app.schemas.models import Chunk, SearchResult

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.I)
STOP_WORDS = {
    "a", "an", "and", "are", "be", "can", "do", "for", "get", "i", "if",
    "in", "is", "it", "me", "my", "of", "on", "or", "our", "the", "this",
    "to", "was", "what", "when", "where", "will", "with", "you", "your",
}


def tokenize(value: str) -> List[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(value)
        if token.lower() not in STOP_WORDS and len(token) > 1
    ]


class BM25Retriever:
    def __init__(self, chunks: Iterable[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.tokens = [tokenize(c.text) for c in self.chunks]
        self.term_frequency = [Counter(tokens) for tokens in self.tokens]
        self.document_frequency = Counter(
            term for tokens in self.tokens for term in set(tokens)
        )
        self.average_length = (
            sum(len(tokens) for tokens in self.tokens) / len(self.tokens)
            if self.tokens else 1
        )

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        terms = tokenize(query)
        if not terms:
            return []
        unique_terms = set(terms)
        total = len(self.chunks)
        scored: List[SearchResult] = []
        for index, chunk in enumerate(self.chunks):
            length = len(self.tokens[index]) or 1
            score = 0.0
            matched: Set[str] = set()
            for term in unique_terms:
                frequency = self.term_frequency[index].get(term, 0)
                if not frequency:
                    continue
                matched.add(term)
                idf = math.log(1 + (total - self.document_frequency[term] + 0.5) /
                               (self.document_frequency[term] + 0.5))
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / self.average_length
                )
                score += idf * frequency * (self.k1 + 1) / denominator
            if matched:
                # Titles and authoritative current documents get transparent
                # tie-breaking boosts; the base score remains BM25.
                score += 0.08 * len(set(tokenize(chunk.section)).intersection(unique_terms))
                score += 0.05 * float(chunk.metadata.get("authority", 0.5))
                scored.append(
                    SearchResult(chunk, score, "bm25", tuple(sorted(matched)))
                )
        scored.sort(key=lambda item: (-item.score, item.chunk.id))
        return scored[: max(1, limit)]
