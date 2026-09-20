"""Application composition root and command-line entry point."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from app.api.routes import create_handler
from app.conversation.manager import ConversationManager
from app.generation.llm import Generator
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_documents
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import rerank
from app.safety.confidence import confidence_score
from app.safety.grounding import validate_grounding
from app.schemas.models import AssistantResponse


class SupportAssistant:
    """Offline-first retrieval, generation, grounding, and escalation loop."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        *,
        llm_enabled: Optional[bool] = None,
        confidence_threshold: float = 0.38,
    ):
        self.data_dir = Path(data_dir)
        self.documents = load_documents(self.data_dir)
        self.chunks = chunk_documents(self.documents)
        self.retriever = HybridRetriever(self.chunks)
        self.conversations = ConversationManager()
        self.generator = Generator(enabled=llm_enabled)
        self.confidence_threshold = confidence_threshold

    def _escalation(
        self,
        query: str,
        conversation_id: str,
        reason: str,
        priority: str,
        hits: Sequence[Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "reason": reason,
            "priority": priority,
            "conversation_id": conversation_id,
            "customer_message": self.conversations.redact(query),
            "metadata": {
                key: self.conversations.redact(str(value))
                for key, value in metadata.items()
            },
            "suggested_fields": [
                "account email", "order or transaction reference", "purchase date",
            ],
            "sources": list(dict.fromkeys(hit.chunk.document_id for hit in hits)),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _risk_reason(query: str) -> Optional[tuple[str, str]]:
        lowered = query.lower()
        if lowered in {
            "cancel my learnforge",
            "cancel learnforge",
            "cancel my account",
            "cancel it",
        }:
            return "ambiguous_cancellation_intent", "medium"
        if "refund" in lowered and any(
            term in lowered for term in ("annual subscription", "subscription", "renewal")
        ):
            return "subscription_refund_requires_policy_review", "medium"
        if (
            ("charge" in lowered and ("unrecognized" in lowered or "don't recognize" in lowered or "do not recognize" in lowered))
            or any(word in lowered for word in ("unauthorized", "fraud", "hacked", "compromised", "chargeback"))
        ):
            return "security_or_payment_risk", "high"
        if "delete my account" in lowered or "legal" in lowered:
            return "account_or_legal_request", "medium"
        if "refund" in lowered and not any(
            marker in lowered for marker in ("order", "transaction", "purchase", "course", "subscription")
        ):
            return "refund_requires_transaction_review", "medium"
        return None

    @staticmethod
    def _clarification(query: str) -> Optional[str]:
        normalized = " ".join(query.lower().split()).rstrip(".!?")
        if normalized in {
            "cancel my learnforge",
            "cancel learnforge",
            "cancel my account",
            "cancel it",
        }:
            return (
                "Do you want to stop future subscription payments, request a refund "
                "for a payment already made, cancel a course enrollment, or delete "
                "your account?"
            )
        return None

    @staticmethod
    def _small_talk(query: str) -> Optional[str]:
        normalized = " ".join(query.lower().split()).strip(" .!?")
        greetings = {
            "hi",
            "hello",
            "hey",
            "hi there",
            "hello there",
            "hey there",
            "good morning",
            "good afternoon",
            "good evening",
        }
        capability_questions = (
            "how can you help",
            "what can you help",
            "what can you do",
            "how do you help",
        )
        if normalized in greetings:
            return (
                "Hi! I’m the LearnForge support assistant. "
                "I can help with courses, payments, refunds, subscriptions, "
                "progress, certificates, accessibility, and account questions. "
                "What would you like help with?"
            )
        if any(phrase in normalized for phrase in capability_questions):
            return (
                "I am the LearnForge support assistant. I can help with courses, "
                "payments, refunds, "
                "subscriptions, progress, certificates, accessibility, and "
                "account questions. Tell me what happened and I'll guide you "
                "using the support knowledge base."
            )
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
        rewritten = self.conversations.rewrite(conversation_id, query)
        if not rewritten:
            response = AssistantResponse(
                "Please tell me what you need help with.", [], 0.0, False,
                conversation_id=conversation_id,
            )
            return response.as_dict()
        self.conversations.remember(conversation_id, "user", rewritten)
        small_talk = self._small_talk(rewritten)
        if small_talk:
            self.conversations.remember(conversation_id, "assistant", small_talk)
            return AssistantResponse(
                answer=small_talk,
                sources=[],
                confidence=1.0,
                grounded=True,
                escalation=None,
                conversation_id=conversation_id,
                rewritten_query=rewritten,
            ).as_dict()
        clarification = self._clarification(rewritten)
        if clarification:
            escalation = self._escalation(
                rewritten,
                conversation_id,
                "ambiguous_cancellation_intent",
                "medium",
                [],
                metadata,
            )
            self.conversations.remember(conversation_id, "assistant", clarification)
            return AssistantResponse(
                answer=clarification,
                sources=[],
                confidence=1.0,
                grounded=True,
                escalation=escalation,
                conversation_id=conversation_id,
                rewritten_query=rewritten,
            ).as_dict()
        candidates = self.retriever.search(rewritten, limit=max(limit * 2, 8))
        hits = rerank(rewritten, candidates, limit=limit)
        confidence = confidence_score(hits)
        answer = self.generator.generate(rewritten, hits, self.conversations.history(conversation_id))
        grounded = validate_grounding(answer, hits)
        escalation = None
        risk = self._risk_reason(rewritten)
        if not hits or confidence < self.confidence_threshold or not grounded:
            reason = "low_confidence_or_ungrounded"
            priority = "medium"
            if not hits:
                reason = "no_matching_guidance"
            escalation = self._escalation(rewritten, conversation_id, reason, priority, hits, metadata)
        elif risk:
            escalation = self._escalation(
                rewritten, conversation_id, risk[0], risk[1], hits, metadata
            )
        if any(
            any(word in hit.chunk.text.lower() for word in ("outdated", "archived", "obsolete"))
            for hit in hits[:3]
        ):
            answer += "\n\nSome material is archived or outdated; current policy takes precedence."
        self.conversations.remember(conversation_id, "assistant", answer)
        response = AssistantResponse(
            answer=answer,
            sources=[
                {
                    "id": hit.chunk.document_id,
                    "chunk_id": hit.chunk.id,
                    "kind": hit.chunk.metadata.get("kind"),
                    "reviewed": hit.chunk.metadata.get("reviewed"),
                    "authority": hit.chunk.metadata.get("authority"),
                    "score": round(hit.score, 4),
                }
                for hit in hits
            ],
            confidence=confidence,
            grounded=grounded,
            escalation=escalation,
            conversation_id=conversation_id,
            rewritten_query=rewritten,
        )
        return response.as_dict()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LearnForge offline support assistant")
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--query")
    parser.add_argument("--conversation-id", default="default")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--serve", action="store_true")
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
