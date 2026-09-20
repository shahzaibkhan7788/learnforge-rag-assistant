"""Small, serialisable data contracts shared by the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    kind: str
    source_file: str
    text: str
    authority: float = 0.5
    reviewed: Optional[str] = None
    status: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    section: str = ""
    ordinal: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    source: str = "bm25"
    matched_terms: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        value = self.chunk.as_dict()
        value.update(
            score=round(self.score, 6),
            source=self.source,
            matched_terms=list(self.matched_terms),
        )
        return value


@dataclass
class AssistantResponse:
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    grounded: bool
    escalation: Optional[Dict[str, Any]] = None
    conversation_id: str = "default"
    rewritten_query: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
