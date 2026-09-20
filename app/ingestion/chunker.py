"""Chunk documents without losing source and section metadata."""

from __future__ import annotations

import re
from typing import Iterable, List

from app.schemas.models import Chunk, Document


def _paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n+", text)
    return [re.sub(r"\s+", " ", part).strip() for part in parts if part.strip()]


def chunk_document(document: Document, max_chars: int = 850, overlap: int = 120) -> List[Chunk]:
    chunks: List[Chunk] = []
    current = ""
    ordinal = 0
    for paragraph in _paragraphs(document.text):
        pieces = [paragraph]
        if len(paragraph) > max_chars:
            pieces = [
                paragraph[start : start + max_chars]
                for start in range(0, len(paragraph), max_chars - overlap)
            ]
        for piece in pieces:
            candidate = f"{current}\n{piece}".strip() if current else piece
            if current and len(candidate) > max_chars:
                chunks.append(
                    Chunk(
                        id=f"{document.id}:{ordinal}",
                        document_id=document.id,
                        text=current,
                        section=document.title,
                        ordinal=ordinal,
                        metadata={
                            "kind": document.kind,
                            "source_file": document.source_file,
                            "reviewed": document.reviewed,
                            "authority": document.authority,
                        },
                    )
                )
                ordinal += 1
                current = piece
            else:
                current = candidate
    if current:
        chunks.append(
            Chunk(
                id=f"{document.id}:{ordinal}",
                document_id=document.id,
                text=current,
                section=document.title,
                ordinal=ordinal,
                metadata={
                    "kind": document.kind,
                    "source_file": document.source_file,
                    "reviewed": document.reviewed,
                    "authority": document.authority,
                },
            )
        )
    return chunks


def chunk_documents(documents: Iterable[Document], **kwargs: int) -> List[Chunk]:
    result: List[Chunk] = []
    for document in documents:
        result.extend(chunk_document(document, **kwargs))
    return result
