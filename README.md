# AI Backbone Quickstart

This quickstart runs the full local stack from repo root:
- Backbone API service (`ai_backbone/app/main.py`)
- One or more Gemma worker services (`ai_backbone/worker_services/gemma_worker/app.py`)

## 1) Setup Python Environment

```bash
cd /home/alpha/ai-backbone
python3 -m venv .venv
source .venv/bin/activate
pip install -e ./ai_backbone
```

## 2) Configure Environment

```bash
cp ai_backbone/.env.example ai_backbone/.env
```

Edit `ai_backbone/.env` and set at least:

```env
GEMMA_ENABLED=true
GEMMA_WORKERS_JSON=[{"id":"gemma-01","url":"http://127.0.0.1:8888","max_concurrent":1}]
GEMMA_WORKER_MOCK_MODE=true
```

## 3) Start Gemma Worker(s)

Open terminal A:

```bash
cd /home/alpha/ai-backbone
source .venv/bin/activate
export GEMMA_WORKER_MOCK_MODE=true
export GEMMA_WORKER_PORT=8888
python -m worker_services.gemma_worker.app
```

Optional second worker (terminal B):

```bash
cd /home/alpha/ai-backbone
source .venv/bin/activate
export GEMMA_WORKER_MOCK_MODE=true
export GEMMA_WORKER_PORT=8889
python -m worker_services.gemma_worker.app
```

If you run 2 workers, update `GEMMA_WORKERS_JSON` in `ai_backbone/.env`:

```env
GEMMA_WORKERS_JSON=[{"id":"gemma-01","url":"http://127.0.0.1:8888","max_concurrent":1},{"id":"gemma-02","url":"http://127.0.0.1:8889","max_concurrent":1}]
```

## 4) Start Backbone API

Open terminal C:

```bash
cd /home/alpha/ai-backbone
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir ai_backbone
```

## 5) Quick Validation cURL

Backbone health:

```bash
curl -s http://127.0.0.1:8000/v1/health
```

Provider health:

```bash
curl -s http://127.0.0.1:8000/v1/health/providers
```

Registered providers:

```bash
curl -s http://127.0.0.1:8000/v1/providers
```

Worker health:

```bash
curl -s http://127.0.0.1:8888/health
```

Local Gemma chat (non-streaming):

```bash
curl -s -X POST http://127.0.0.1:8000/v1/llm/chat \
  -H 'content-type: application/json' \
  -d '{
    "provider":"local-gemma",
    "messages":[{"role":"user","content":"Explain VaR in simple terms"}],
    "temperature":0.2,
    "max_tokens":128
  }'
```

Local Gemma chat (streaming):

```bash
curl -N -X POST http://127.0.0.1:8000/v1/llm/chat/stream \
  -H 'content-type: application/json' \
  -d '{
    "provider":"local-gemma",
    "messages":[{"role":"user","content":"Give me 3 bullet points on VaR"}]
  }'
```

Direct worker streaming test:

```bash
curl -N -X POST http://127.0.0.1:8888/generate/stream \
  -H 'content-type: application/json' \
  -d '{
    "request_id":"quickstart-1",
    "model_name":"gemma-12b-it",
    "prompt":"Explain VaR in simple terms",
    "temperature":0.2,
    "max_new_tokens":128
  }'
```

## 6) Run Tests

```bash
cd /home/alpha/ai-backbone/ai_backbone
source ../.venv/bin/activate
pytest -q
```
