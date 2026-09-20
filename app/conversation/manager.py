"""Bounded conversation memory and conservative query rewriting."""

from __future__ import annotations

import re
import threading
from typing import Dict, List


class ConversationManager:
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self._memory: Dict[str, List[Dict[str, str]]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def redact(value: str) -> str:
        pattern = (
            r"(?i)(?:\b(?:password|passcode|cvv|cvc|pin|security code|"
            r"authentication code|one[- ]time code)\b\s*[:=]?\s*\S+|"
            r"\b(?:\d[ -]?){13,19}\b)"
        )
        return re.sub(pattern, "[redacted]", value)

    def remember(self, conversation_id: str, role: str, content: str) -> None:
        with self._lock:
            history = self._memory.setdefault(conversation_id, [])
            history.append({"role": role, "content": self.redact(content)})
            del history[:-self.max_messages]

    def history(self, conversation_id: str) -> List[Dict[str, str]]:
        with self._lock:
            return list(self._memory.get(conversation_id, []))

    def rewrite(self, conversation_id: str, query: str) -> str:
        query = self.redact(query.strip())
        if not query:
            return ""
        history = self.history(conversation_id)
        if len(query.split()) >= 4 or not history:
            return query
        previous = " ".join(item["content"] for item in history[-4:] if item["role"] == "user")
        return f"{previous} {query}".strip() if previous else query
