from pathlib import Path

from support_assistant import KnowledgeBase, SupportAssistant


ROOT = Path(__file__).resolve().parents[1]


def test_all_corpus_entries_are_loaded():
    knowledge = KnowledgeBase(ROOT)
    assert len(knowledge.records) == 40
    assert sum(record.kind == "faq" for record in knowledge.records) == 15
    assert sum(record.kind == "policy" for record in knowledge.records) == 10
    assert sum(record.kind == "ticket" for record in knowledge.records) == 15


def test_policy_ranks_above_historical_ticket_for_refund():
    assistant = SupportAssistant(ROOT, llm_enabled=False)
    hits = assistant.knowledge.search("refund annual subscription")
    assert hits
    assert hits[0].record.kind in {"policy", "faq"}
    assert any(hit.record.id == "POLICY-02" for hit in hits)


def test_answer_is_grounded_and_escalates_unrecognized_charge():
    assistant = SupportAssistant(ROOT, llm_enabled=False)
    result = assistant.answer("I don't recognize a LearnForge charge")
    assert "full card number" in result["answer"].lower()
    assert "FAQ-15" in {source["id"] for source in result["sources"]}
    assert result["escalation"]["priority"] == "high"


def test_sensitive_input_is_not_repeated():
    assistant = SupportAssistant(ROOT, llm_enabled=False)
    result = assistant.answer("My password: secret-password and card 4111 1111 1111 1111")
    assert "secret-password" not in result["answer"]
    assert "4111" not in result["answer"]


def test_conversation_memory_keeps_a_bounded_history():
    assistant = SupportAssistant(ROOT, llm_enabled=False)
    for index in range(20):
        assistant.answer(f"question about progress {index}", conversation_id="test")
    assert len(assistant._history("test")) <= 10
