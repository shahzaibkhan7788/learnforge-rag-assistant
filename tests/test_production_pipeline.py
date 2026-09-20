from pathlib import Path

from app.main import SupportAssistant
from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.reranker import rerank


ROOT = Path(__file__).resolve().parents[1]


def test_structure_preserves_corpus_and_metadata():
    documents = load_documents(ROOT)
    assert len(documents) == 40
    chunks = chunk_documents(documents)
    assert chunks
    assert all(chunk.document_id and chunk.metadata["kind"] for chunk in chunks)


def test_follow_up_is_rewritten_and_sensitive_data_is_redacted():
    assistant = SupportAssistant(ROOT, llm_enabled=False)
    first = assistant.answer("My course progress is not saving", conversation_id="prod")
    second = assistant.answer(
        "My password: secret-password and card 4111 1111 1111 1111",
        conversation_id="prod",
    )
    assert first["sources"]
    assert "secret-password" not in second["answer"]
    assert "4111" not in second["answer"]
    assert second["rewritten_query"]


def test_unrecognized_charge_escalates_high_priority():
    result = SupportAssistant(ROOT, llm_enabled=False).answer(
        "I don't recognize a LearnForge charge"
    )
    assert result["escalation"]["priority"] == "high"
    assert result["grounded"]


def test_ambiguous_cancellation_asks_for_clarification():
    result = SupportAssistant(ROOT, llm_enabled=False).answer("Cancel my LearnForge")
    assert "subscription payments" in result["answer"]
    assert result["escalation"]["reason"] == "ambiguous_cancellation_intent"


def test_subscription_refund_is_escalated_for_policy_review():
    result = SupportAssistant(ROOT, llm_enabled=False).answer(
        "I want a refund for my annual subscription"
    )
    assert result["escalation"]["reason"] == "subscription_refund_requires_policy_review"


def test_greeting_does_not_retrieve_unrelated_knowledge():
    result = SupportAssistant(ROOT, llm_enabled=False).answer("Hi, how can you help me?")
    assert result["sources"] == []
    assert result["escalation"] is None
    assert "support assistant" in result["answer"].lower()


def test_exact_value_query_prioritizes_focused_authoritative_evidence():
    assistant = SupportAssistant(ROOT, llm_enabled=False)
    candidates = assistant.retriever.search(
        "What is the standard refund period?", limit=8
    )
    hits = rerank("What is the standard refund period?", candidates, limit=5)

    assert hits[0].chunk.document_id == "POLICY-02"
    assert hits[0].chunk.id == "POLICY-02:1"
    assert "14 days" in hits[0].chunk.text
    assert len(hits[0].chunk.text) < len(
        next(chunk for chunk in assistant.chunks if chunk.id == "POLICY-02:1").text
    )
