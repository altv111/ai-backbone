# AI Backbone (Layer 1)

Layer 1 is a stateless, provider-agnostic AI backbone. It exposes primitives and avoids business workflows.

Layer 2 (future) composes Layer 1 into product tools like Jira generators, NL->JQL, Confluence assistants, and code assistants.

## Provider Types

- `LLMProvider`: chat and embeddings
- `RetrievalProvider`: index/retrieve collections
- `KnowledgeProvider`: composite knowledge systems

OMNI-like systems belong in `KnowledgeProvider` because they orchestrate knowledge behavior (often combining LLM and retrieval), not raw retrieval only.

## Endpoints

All routes are under `/v1`:
- `GET /v1/health`
- `GET /v1/health/providers`
- `GET /v1/providers`
- `GET /v1/collections`
- `POST /v1/llm/chat`
- `POST /v1/llm/chat/stream` (local-gemma only)
- `POST /v1/llm/embed`
- `POST /v1/rag/index`
- `POST /v1/rag/retrieve`
- `POST /v1/knowledge/answer`
- `POST /v1/knowledge/search`

## Run Locally

```bash
cd ai_backbone
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload
```

## Run Tests

```bash
cd ai_backbone
pytest -q
```

## Minimal Audit JSON

Layer 1 emits simple JSON audit events for chat requests (for example `email`, `ask`, `wait_time_ms`, `latency_ms`, `status`, `request_id`).

Configure via:
- `AI_BACKBONE_AUDIT_ENABLED=true|false`
- `AI_BACKBONE_AUDIT_LOG_PATH=/path/to/audit.log` (empty means JSON is emitted via logger)

## Central LLM (`central-llm`)

Enable with:
- `CENTRAL_LLM_ENABLED=true`
- `CENTRAL_LLM_API_URL=...`
- `CENTRAL_LLM_API_KEY=...`

`central-llm` supports chat only. Embeddings return `unsupported_operation`.

### Exact payload mapping

Backbone chat request is mapped to this organizational payload:

```json
{
  "email": "...",
  "apikey": "...",
  "data_classification": "...",
  "message": "...",
  "kannon_id": "...",
  "model_name": "..."
}
```

Mapping:
- `request.options["email"] -> email`
- `request.options["api_key"] or CENTRAL_LLM_API_KEY -> apikey`
- `request.options.get("data_classification") or CENTRAL_LLM_DEFAULT_DATA_CLASSIFICATION -> data_classification`
- `request.messages -> message`
- `request.options["kannon_id"] -> kannon_id`
- `request.model or CENTRAL_LLM_DEFAULT_MODEL -> model_name`

Message formatting:
- Exactly one user message: use that user content directly.
- Otherwise: include ordered role lines (`system: ...`, `user: ...`, etc).

### Example cURL (through backbone)

```bash
curl -s -X POST http://127.0.0.1:8000/v1/llm/chat \
  -H 'content-type: application/json' \
  -d '{
    "provider":"central-llm",
    "model":"gemini-flash",
    "messages":[{"role":"user","content":"Summarize this Jira ticket"}],
    "options":{"email":"dev@example.com","kannon_id":"kan-1"}
  }'
```

Streaming (local-gemma only):

```bash
curl -N -X POST http://127.0.0.1:8000/v1/llm/chat/stream \
  -H 'content-type: application/json' \
  -d '{
    "provider":"local-gemma",
    "messages":[{"role":"user","content":"Explain VaR in simple terms"}]
  }'
```

## Local Gemma (`local-gemma`) Worker Pool

`local-gemma` is one logical provider backed by multiple worker machines.

Configure in backbone:
- `GEMMA_ENABLED=true`
- `GEMMA_WORKERS_JSON=[{"id":"gemma-01","url":"http://host1:8888","max_concurrent":1}]`

Backbone selects workers in-memory (v1):
- healthy workers only
- `active_requests < max_concurrent`
- lowest `active_requests`
- tie-break by lowest `avg_latency_ms`

For CPU hosts, each worker is intentionally run at one active generation (`max_concurrent=1`).

Note: this in-memory state is suitable for a single API process. Future multi-process deployment should move worker-state coordination to Redis-backed state or an async job queue.

### Example cURL (through backbone)

```bash
curl -s -X POST http://127.0.0.1:8000/v1/llm/chat \
  -H 'content-type: application/json' \
  -d '{
    "provider":"local-gemma",
    "model":"gemma-12b-it",
    "messages":[{"role":"user","content":"Explain VaR in simple terms"}],
    "temperature":0.2,
    "max_tokens":512
  }'
```

## Gemma Worker Service (Separate Deployment)

Location:
- `worker_services/gemma_worker/`

This service is independent from backbone deployment and exposes:
- `GET /health`
- `POST /generate`

### Run mock mode

```bash
cd ai_backbone
export GEMMA_WORKER_MOCK_MODE=true
python -m worker_services.gemma_worker.app
```

### Run real mode

```bash
cd ai_backbone
export GEMMA_WORKER_MOCK_MODE=false
export GEMMA_WORKER_MODEL_NAME=gemma-12b-it
export GEMMA_WORKER_MODEL_PATH=/models/gemma-12b-it
python -m worker_services.gemma_worker.app
```

### Direct worker cURL

```bash
curl -s -X POST http://127.0.0.1:8888/generate \
  -H 'content-type: application/json' \
  -d '{
    "request_id":"req-1",
    "model_name":"gemma-12b-it",
    "prompt":"Explain VaR in simple terms",
    "temperature":0.2,
    "max_new_tokens":128
  }'
```

## Add a New Provider (High-Level)

1. Implement `LLMProvider`, `RetrievalProvider`, or `KnowledgeProvider`.
2. Return contract models (`app/contracts/*`) only.
3. Register in `app/bootstrap.py`.
4. Add tests for happy paths and structured errors.
5. Keep Layer 1 stateless and workflow-agnostic.
