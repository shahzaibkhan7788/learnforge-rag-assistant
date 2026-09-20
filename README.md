# LearnForge Customer Support Assistant

Offline-first customer-support RAG prototype for the supplied `faqs.md`,
`policies.md`, and `tickets.md` corpus. The default runtime uses only the
Python standard library and never needs an API key or external service.

## Run exactly

```powershell
cd E:\Mydata\Desktop\Customer_Support_Assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py --query "Can I get a refund for a course?"
python app.py --query "I don't recognize a LearnForge charge" --json
python -m pytest -q
python evaluation\evaluate.py
```

Start the local API and the included browser interface:

```powershell
python app.py --serve --host 127.0.0.1 --port 8000
Invoke-RestMethod http://127.0.0.1:8000/health
$body = @{query="How do I watch courses offline?"; conversation_id="demo"} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/query -Method Post -ContentType "application/json" -Body $body
```

`app.py` remains a compatibility entry point; the implementation is in
`app/`. `python -m app.main --query "..."` is equivalent.

Open `http://127.0.0.1:8000/` for the interactive LearnForge support workspace.
It includes guided example prompts, conversation memory, source chips,
confidence status, and human-review indicators. The UI is served from
`frontend/` and does not change the retrieval pipeline.

The UI uses `POST /query/stream`, which returns Server-Sent Events: incremental
`token` events followed by a final `done` event containing sources, confidence,
and escalation metadata. The deterministic response is grounded first and
then streamed in chunks, so safety behavior is unchanged.

## Architecture and flow

```text
Markdown corpus -> loader/parser -> metadata-preserving chunks
                                  -> BM25 + optional dense/Qdrant -> RRF -> reranker
user query -> redaction + conversation rewrite --------------------^
                         -> confidence/evidence checks
                         -> deterministic answer (optional LLM adapter)
                         -> grounding validation + escalation -> JSON/API/CLI
```

`Document` stores `id`, `title`, `kind`, `source_file`, text, authority,
review date, status, and metadata. `Chunk` stores a stable `document_id:ordinal`,
section, text, and inherited metadata. This schema is suitable for a vector
record or relational table.

## Failure handling and safety

* BM25 is always available. Dense retrieval and Qdrant are lazy, opt-in
  adapters and remain inert when packages or environment variables are absent.
* Low confidence, no evidence, or failed evidence-overlap grounding creates a
  redacted escalation payload instead of claiming certainty.
* Current policy authority/freshness beats FAQ and historical ticket evidence;
  stale-language matches are explicitly disclosed.
* Bounded in-process conversation memory rewrites short follow-up questions.
  Passwords, payment-card numbers, CVV/PINs, and authentication codes are
  redacted before memory, generation, and escalation.
* Security/payment risk, legal/account operations, ambiguous refunds, and
  unknown requests escalate. This prototype does not authenticate users or
  execute billing/account actions.

## Evaluation

`evaluation/dataset.json` contains a small labelled query set. The evaluator
reports retrieval Recall@5, MRR, and query count without network calls:

```powershell
python evaluation\evaluate.py
```

For production, add answer-level human labels, citation precision, grounded
claim rate, escalation precision/recall, stale-document detection, and
latency/cost monitoring. Retrieval metrics alone do not measure hallucination.

## Optional generation

Copy `.env.example` to `.env` (never commit `.env`) and set
`SUPPORT_LLM_API_KEY`, `SUPPORT_LLM_BASE_URL`, and optionally
`SUPPORT_LLM_MODEL`/`SUPPORT_LLM_PROVIDER` to use an OpenAI-compatible endpoint.
Only redacted query context and retrieved evidence are sent. Network errors,
invalid responses, and missing configuration automatically fall back to the
deterministic generator.

For OpenRouter:

```powershell
Copy-Item .env.example .env
$env:SUPPORT_LLM_PROVIDER = "openrouter"
$env:SUPPORT_LLM_API_KEY = "your-openrouter-key"
$env:SUPPORT_LLM_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
$env:SUPPORT_LLM_MODEL = "openrouter/free"
python app.py --serve
```

The repository shows the model/provider configuration but contains only an
empty key placeholder. If a real key was ever pasted into `.env.example`,
revoke it in OpenRouter immediately, create a replacement, and keep the
replacement only in a local `.env` or deployment secret manager.

## Trade-offs and limitations

The standard-library BM25/RRF path was chosen for reproducibility, offline
execution, and transparent scoring. It is less semantically capable than
embeddings and does not persist an index. A production deployment should add
authenticated users, durable conversation storage, a managed vector index,
embedding/version refresh jobs, policy effective-date filtering, rate limits,
structured observability, and human-reviewed answer tests. Qdrant and dense
retrieval hooks are provided but intentionally do not download models.
