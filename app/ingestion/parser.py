"""Structure-aware parsing for the supplied markdown corpus."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from app.schemas.models import Document

ENTRY_RE = re.compile(r"^#\s+((?:FAQ|POLICY|TICKET)-\d+)\s+[—-]\s*(.+?)\s*$", re.I)
DATE_RE = re.compile(
    r"(?:last reviewed|last updated|updated|reviewed|effective(?: date)?|"
    r"effective|reviewed)\s*:?\s*([A-Za-z]+\s+\d{4})",
    re.I,
)
KIND_AUTHORITY = {"policy": 1.0, "faq": 0.9, "ticket": 0.55}


def parse_reviewed(text: str) -> Optional[str]:
    match = DATE_RE.search(text)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%B %Y").strftime("%Y-%m")
    except ValueError:
        return None


def parse_markdown(path: Path) -> List[Document]:
    """Parse headings into records while retaining transcript labels/sections."""
    records: List[Document] = []
    current_id: Optional[str] = None
    title = ""
    lines: List[str] = []

    def finish() -> None:
        if not current_id:
            return
        body = "\n".join(lines).strip()
        kind = current_id.split("-", 1)[0].lower()
        status_match = re.search(r"^STATUS:\s*(.+)$", body, re.I | re.M)
        metadata = {
            "source": path.name,
            "authority": KIND_AUTHORITY.get(kind, 0.5),
            "has_status": bool(status_match),
        }
        records.append(
            Document(
                id=current_id,
                title=title.strip(),
                kind=kind,
                source_file=path.name,
                text=body,
                authority=KIND_AUTHORITY.get(kind, 0.5),
                reviewed=parse_reviewed(body),
                status=status_match.group(1).strip() if status_match else None,
                metadata=metadata,
            )
        )

    for line in path.read_text(encoding="utf-8").splitlines():
        match = ENTRY_RE.match(line)
        if match:
            finish()
            current_id, title, lines = match.group(1).upper(), match.group(2), []
        elif current_id and line.strip() != "---":
            lines.append(line)
    finish()
    return records
