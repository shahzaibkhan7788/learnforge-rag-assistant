"""Deterministic answerer plus an opt-in OpenAI-compatible adapter."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Dict, Optional, Sequence

from app.schemas.models import SearchResult
from .prompts import build_prompt


class Generator:
    def __init__(self, enabled: Optional[bool] = None):
        self.enabled = (
            bool(os.getenv("SUPPORT_LLM_API_KEY"))
            if enabled is None
            else enabled
        )

    def _remote(
        self, query: str, evidence: str, history: Sequence[Dict[str, str]]
    ) -> Optional[str]:
        key = os.getenv("SUPPORT_LLM_API_KEY")
        base_url = os.getenv("SUPPORT_LLM_BASE_URL")
        if not self.enabled or not key or not base_url:
            return None
        payload = {
            "model": os.getenv("SUPPORT_LLM_MODEL", "gpt-4o-mini"),
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "You are a careful support agent."},
                *history[-4:],
                {"role": "user", "content": build_prompt(query, evidence, list(history))},
            ],
        }
        request = urllib.request.Request(
            base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                value = json.loads(response.read().decode("utf-8"))
            return value["choices"][0]["message"]["content"].strip()
        except (OSError, ValueError, KeyError, IndexError, urllib.error.URLError):
            return None

    def generate(
        self,
        query: str,
        results: Sequence[SearchResult],
        history: Sequence[Dict[str, str]] = (),
    ) -> str:
        evidence = "\n".join(
            f"[{result.chunk.document_id}] {result.chunk.text}" for result in results[:3]
        )
        remote = self._remote(query, evidence, history)
        if remote:
            return remote
        if not results:
            return (
                "I couldn't find a reliable answer in the LearnForge guidance. "
                "Please provide the course or transaction details so Support can investigate."
            )
        bullets = "\n".join(f"- {result.chunk.text}" for result in results[:2])
        return (
            "Based on the current LearnForge guidance:\n\n"
            f"{bullets}\n\n"
            "Please do not send a password, full card number, CVV, PIN, or authentication code."
        )
