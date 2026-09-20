"""Prompt templates used by optional providers and deterministic generation."""

SYSTEM_PROMPT = (
    "You are a careful LearnForge support agent. Answer only from EVIDENCE. "
    "Never invent account facts, and never request or repeat passwords, full "
    "card numbers, CVV/PINs, or authentication codes. Current policy wins "
    "over FAQs and historical tickets."
)


def build_prompt(query: str, evidence: str, history: list[dict[str, str]]) -> str:
    context = "\n".join(
        f"{message['role']}: {message['content']}" for message in history[-4:]
    )
    return f"{SYSTEM_PROMPT}\n\nHISTORY:\n{context}\n\nEVIDENCE:\n{evidence}\n\nCUSTOMER:\n{query}"
