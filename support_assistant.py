"""Small, dependency-free retrieval and support-answering application.

The default mode deliberately uses only the Python standard library.  An
OpenAI-compatible provider can be enabled with environment variables, but the
same retrieval, safety, and escalation behaviour is retained in either mode.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


STOP_WORDS = {
    "a", "an", "and", "are", "be", "can", "do", "for", "get", "i", "if",
    "in", "is", "it", "me", "my", "of", "on", "or", "our", "the", "this",
    "to", "was", "what", "when", "where", "will", "with", "you", "your",
}
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.I)
ENTRY_RE = re.compile(r"^#\s+((?:FAQ|POLICY|TICKET)-\d+)\s+[—-]\s*(.+?)\s*$")
DATE_RE = re.compile(
    r"(?:last reviewed|last updated|updated|reviewed|effective(?: date)?)"
    r"\s*:\s*([A-Za-z]+\s+\d{4})",
    re.I,
)
SENSITIVE_RE = re.compile(
    r"(?i)(?:\b(?:password|passcode|cvv|cvc|pin|security code|"
    r"authentication code|one[- ]time code)\b\s*[:=]?\s*\S+|"
    r"\b(?:\d[ -]?){13,19}\b)"
)
STALE_WORDS = (
    "older", "outdated", "obsolete", "archived", "previous", "retired",
    "no longer", "old version", "superseded",
)


def _tokens(value: str) -> List[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(value)
        if token.lower() not in STOP_WORDS and len(token) > 1
    ]


def _redact(value: str) -> str:
    return SENSITIVE_RE.sub("[redacted]", value)


@dataclass(frozen=True)
class KnowledgeRecord:
    id: str
    title: str
    kind: str
    source_file: str
    text: str
    authority: float
    reviewed: Optional[str] = None
    status: Optional[str] = None

    @property
    def has_stale_language(self) -> bool:
        lowered = self.text.lower()
        return any(phrase in lowered for phrase in STALE_WORDS)

    def freshness(self, as_of: Optional[date] = None) -> float:
        """Return a bounded freshness score; undated tickets are deliberately lower."""
        if not self.reviewed:
            return 0.58 if self.kind == "ticket" else 0.72
        try:
            reviewed = datetime.strptime(self.reviewed, "%Y-%m").date()
        except ValueError:
            return 0.7
        today = as_of or date.today()
        age_days = max(0, (today - reviewed).days)
        return max(0.2, 1.0 - (age_days / 730.0))


@dataclass(frozen=True)
class SearchHit:
    record: KnowledgeRecord
    score: float
    matched_terms: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        result = asdict(self.record)
        result.update({
            "score": round(self.score, 4),
            "matched_terms": list(self.matched_terms),
            "freshness": round(self.record.freshness(), 3),
        })
        return result


def _parse_date(text: str) -> Optional[str]:
    match = DATE_RE.search(text)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%B %Y").strftime("%Y-%m")
    except ValueError:
        return None


def _records_from_markdown(path: Path) -> List[KnowledgeRecord]:
    lines = path.read_text(encoding="utf-8").splitlines()
    records: List[KnowledgeRecord] = []
    current_id: Optional[str] = None
    current_title = ""
    current_lines: List[str] = []

    def finish() -> None:
        if not current_id:
            return
        body = "\n".join(current_lines).strip()
        kind = current_id.split("-", 1)[0].lower()
        authority = {"policy": 1.0, "faq": 0.9, "ticket": 0.55}[kind]
        status_match = re.search(r"^STATUS:\s*(.+)$", body, re.I | re.M)
        records.append(KnowledgeRecord(
            id=current_id,
            title=current_title.strip(),
            kind=kind,
            source_file=path.name,
            text=body,
            authority=authority,
            reviewed=_parse_date(body),
            status=status_match.group(1).strip() if status_match else None,
        ))

    for line in lines:
        match = ENTRY_RE.match(line)
        if match:
            finish()
            current_id, current_title, current_lines = match.group(1), match.group(2), []
        elif current_id and line.strip() != "---":
            current_lines.append(line)
    finish()
    return records


class KnowledgeBase:
    """Loads the supplied markdown corpus and provides weighted keyword search."""

    def __init__(self, data_dir: Union[Path, str]):
        self.data_dir = Path(data_dir)
        self.records: List[KnowledgeRecord] = []
        for filename in ("faqs.md", "policies.md", "tickets.md"):
            path = self.data_dir / filename
            if path.exists():
                self.records.extend(_records_from_markdown(path))
        if not self.records:
            raise FileNotFoundError(
                f"No faqs.md, policies.md, or tickets.md found in {self.data_dir}"
            )

    def search(
        self,
        query: str,
        limit: int = 5,
        *,
        kinds: Optional[Iterable[str]] = None,
    ) -> List[SearchHit]:
        query_terms = set(_tokens(query))
        if not query_terms:
            return []
        allowed = {kind.lower() for kind in kinds} if kinds else None
        query_lower = query.lower()
        hits: List[SearchHit] = []
        for record in self.records:
            if allowed and record.kind not in allowed:
                continue
            title_terms = set(_tokens(record.title))
            body_terms = set(_tokens(record.text))
            matched = query_terms.intersection(body_terms | title_terms)
            if not matched:
                continue
            overlap = len(matched) / len(query_terms)
            title_bonus = len(query_terms.intersection(title_terms)) * 0.12
            phrase_bonus = 0.18 if len(query_terms) > 1 and query_lower in record.text.lower() else 0.0
            score = (
                overlap * 0.62
                + title_bonus
                + phrase_bonus
                + record.authority * 0.16
                + record.freshness() * 0.12
            )
            # Current policies and FAQs should win near-ties against historical
            # examples; tickets remain useful but are not authoritative guidance.
            score += {"policy": 0.18, "faq": 0.08, "ticket": 0.0}[record.kind]
            if record.kind == "ticket" and record.has_stale_language:
                score -= 0.08
            hits.append(SearchHit(record, score, tuple(sorted(matched))))
        hits.sort(key=lambda hit: (-hit.score, -hit.record.authority, hit.record.id))
        return hits[: max(1, limit)]


def _sentences(record: KnowledgeRecord, query: str, maximum: int = 3) -> List[str]:
    """Select concise, relevant sentences while avoiding transcript labels."""
    text = re.sub(r"^(?:QUESTION|ANSWER|USER|AGENT|STATUS):\s*", "", record.text,
                  flags=re.I | re.M)
    candidates = [
        re.sub(r"\s+", " ", item).strip(" -")
        for item in re.split(r"(?<=[.!?])\s+|\n+", text)
    ]
    query_terms = set(_tokens(query))
    scored: List[Tuple[int, int, str]] = []
    for index, sentence in enumerate(candidates):
        if len(sentence) < 25 or sentence.startswith("#"):
            continue
        overlap = len(query_terms.intersection(set(_tokens(sentence))))
        if overlap:
            scored.append((overlap, -index, sentence))
    if not scored:
        scored = [(0, -index, sentence) for index, sentence in enumerate(candidates)
                  if len(sentence) >= 25 and not sentence.startswith("#")]
    scored.sort(reverse=True)
    result: List[str] = []
    seen = set()
    for _, _, sentence in scored:
        if sentence not in seen:
            result.append(sentence)
            seen.add(sentence)
        if len(result) >= maximum:
            break
    return result


class SupportAssistant:
    """Grounded support assistant with memory, safety checks, and escalation."""

    def __init__(
        self, data_dir: Union[Path, str], *, llm_enabled: Optional[bool] = None
    ):
        self.knowledge = KnowledgeBase(data_dir)
        self._memory: Dict[str, List[Dict[str, str]]] = {}
        self._memory_lock = threading.Lock()
        self.llm_enabled = (
            bool(os.getenv("SUPPORT_LLM_API_KEY"))
            if llm_enabled is None else llm_enabled
        )

    def _remember(self, conversation_id: str, role: str, text: str) -> None:
        with self._memory_lock:
            messages = self._memory.setdefault(conversation_id, [])
            messages.append({"role": role, "content": _redact(text)})
            del messages[:-10]

    def _history(self, conversation_id: str) -> List[Dict[str, str]]:
        with self._memory_lock:
            return list(self._memory.get(conversation_id, []))

    @staticmethod
    def _needs_escalation(query: str, hits: Sequence[SearchHit]) -> Optional[Tuple[str, str]]:
        lowered = query.lower()
        if not hits:
            return "no_matching_guidance", "medium"
        unrecognized_payment = (
            ("unrecognized" in lowered or "don't recognize" in lowered
             or "do not recognize" in lowered)
            and "charge" in lowered
        )
        if unrecognized_payment or any(word in lowered for word in (
            "unauthorized", "account hacked", "compromised", "chargeback", "fraud",
        )):
            return "security_or_payment_risk", "high"
        if "refund" in lowered and not any(
            marker in lowered for marker in ("order", "transaction", "purchase", "course")
        ):
            return "refund_requires_transaction_review", "medium"
        if "delete my account" in lowered or "legal" in lowered:
            return "account_or_legal_request", "medium"
        return None

    def _escalation_payload(
        self,
        query: str,
        conversation_id: str,
        hits: Sequence[SearchHit],
        reason: str,
        priority: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "reason": reason,
            "priority": priority,
            "conversation_id": conversation_id,
            "customer_message": _redact(query),
            "metadata": {key: _redact(str(value)) for key, value in metadata.items()},
            "suggested_fields": [
                "account email", "order or transaction reference", "purchase date",
            ],
            "sources": [hit.record.id for hit in hits],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def _llm_answer(
        self,
        query: str,
        context: str,
        history: Sequence[Dict[str, str]],
        metadata: Dict[str, Any],
    ) -> Optional[str]:
        api_key = os.getenv("SUPPORT_LLM_API_KEY")
        if not self.llm_enabled or not api_key:
            return None
        provider = os.getenv("SUPPORT_LLM_PROVIDER", "openai-compatible").lower()
        base_url = os.getenv("SUPPORT_LLM_BASE_URL")
        if not base_url:
            base_url = "https://api.openai.com/v1/chat/completions" if provider == "openai" else ""
        if not base_url:
            return None
        model = os.getenv("SUPPORT_LLM_MODEL", "gpt-4o-mini")
        prompt = (
            "Answer the customer using only the supplied evidence. If evidence conflicts, "
            "prefer current policy over FAQs and tickets and say that verification is needed. "
            "Never invent account facts and never repeat secrets.\n\n"
            f"REQUEST CONTEXT:\n{json.dumps(metadata, ensure_ascii=False, default=str)}\n\n"
            f"EVIDENCE:\n{context}\n\nCUSTOMER:\n{_redact(query)}"
        )
        payload = {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "You are a careful LearnForge support agent."},
                *history[-4:],
                {"role": "user", "content": prompt},
            ],
        }
        request = urllib.request.Request(
            base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"].strip()
        except (OSError, ValueError, KeyError, IndexError, urllib.error.URLError):
            return None

    def answer(
        self,
        query: str,
        *,
        conversation_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        metadata = metadata or {}
        safe_query = _redact(query.strip())
        if not safe_query:
            return {
                "answer": "Please tell me what you need help with.",
                "sources": [],
                "escalation": None,
            }
        self._remember(conversation_id, "user", safe_query)
        hits = self.knowledge.search(safe_query, limit=limit)
        evidence_parts = []
        for hit in hits[:3]:
            excerpts = _sentences(hit.record, safe_query, maximum=2)
            if excerpts:
                evidence_parts.append(f"[{hit.record.id}] " + " ".join(excerpts))
        evidence = "\n".join(evidence_parts)

        if not hits:
            answer = (
                "I couldn't find a reliable answer in the LearnForge guidance. "
                "Please provide the course or transaction details so Support can investigate."
            )
        else:
            llm_answer = self._llm_answer(
                safe_query, evidence, self._history(conversation_id), metadata
            )
            if llm_answer:
                answer = llm_answer
            else:
                lead = "Based on the current LearnForge guidance"
                if hits[0].record.kind == "ticket":
                    lead += " (support examples are illustrative, not account-specific)"
                answer = lead + ":\n\n" + "\n".join(
                    f"- {sentence}" for hit in hits[:2]
                    for sentence in _sentences(hit.record, safe_query, maximum=2)
                )
                answer += "\n\nPlease do not send a password, full card number, CVV, PIN, or authentication code."
            relevant_stale = any(
                any(
                    phrase in " ".join(_sentences(hit.record, safe_query, maximum=3)).lower()
                    for phrase in STALE_WORDS
                )
                for hit in hits[:3]
            )
            if relevant_stale:
                answer += (
                    "\n\nSome retrieved material mentions older or archived guidance; "
                    "the current policy source takes precedence."
                )

        escalation_info = self._needs_escalation(safe_query, hits)
        escalation = None
        if escalation_info:
            reason, priority = escalation_info
            escalation = self._escalation_payload(
                safe_query, conversation_id, hits, reason, priority, metadata
            )
        sources = [
            {
                "id": hit.record.id,
                "kind": hit.record.kind,
                "authority": hit.record.authority,
                "reviewed": hit.record.reviewed,
                "freshness": round(hit.record.freshness(), 3),
                "score": round(hit.score, 4),
            }
            for hit in hits[:3]
        ]
        self._remember(conversation_id, "assistant", answer)
        return {
            "answer": answer,
            "sources": sources,
            "escalation": escalation,
            "conversation_id": conversation_id,
            "metadata": metadata,
        }


def create_handler(assistant: SupportAssistant):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: Dict[str, Any]) -> None:
            encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200, {"ok": True, "records": len(assistant.knowledge.records)})
            else:
                self._send(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/query":
                self._send(404, {"error": "not_found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(size).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("body must be an object")
                query = body.get("query", "")
                if not isinstance(query, str):
                    raise ValueError("query must be a string")
                result = assistant.answer(
                    query,
                    conversation_id=str(body.get("conversation_id", "default")),
                    metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else {},
                )
                self._send(200, result)
            except (ValueError, json.JSONDecodeError):
                self._send(400, {"error": "expected JSON with a string query"})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LearnForge deterministic support assistant")
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--query", help="answer one query and exit")
    parser.add_argument("--conversation-id", default="default")
    parser.add_argument("--json", action="store_true", help="emit a JSON response")
    parser.add_argument("--serve", action="store_true", help="start the HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    assistant = SupportAssistant(args.data_dir)

    if args.serve:
        server = ThreadingHTTPServer((args.host, args.port), create_handler(assistant))
        print(f"LearnForge support API listening on http://{args.host}:{args.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0

    if args.query:
        result = assistant.answer(args.query, conversation_id=args.conversation_id)
        print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else result["answer"])
        return 0

    print("LearnForge support assistant. Type a question, or Ctrl-C to exit.")
    try:
        while True:
            question = input("> ").strip()
            if question:
                print(assistant.answer(question)["answer"])
    except (KeyboardInterrupt, EOFError):
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
