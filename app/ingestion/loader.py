"""Corpus loading with deterministic file ordering."""

from pathlib import Path
from typing import Iterable, List, Union

from app.schemas.models import Document
from .parser import parse_markdown

CORPUS_FILES = ("faqs.md", "policies.md", "tickets.md")


def load_documents(data_dir: Union[str, Path]) -> List[Document]:
    root = Path(data_dir)
    documents: List[Document] = []
    for filename in CORPUS_FILES:
        path = root / filename
        if path.exists():
            documents.extend(parse_markdown(path))
    if not documents:
        raise FileNotFoundError(
            "No corpus files found; expected faqs.md, policies.md, or tickets.md "
            f"in {root}"
        )
    return documents
