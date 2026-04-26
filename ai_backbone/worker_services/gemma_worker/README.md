# Gemma Worker Service

This service is deployed independently from the backbone API. Each Linux machine hosting local Gemma runs one worker instance.

## Endpoints

- `GET /health`
- `POST /generate`
- `POST /generate/stream`

## Run in Mock Mode (default)

```bash
export GEMMA_WORKER_MOCK_MODE=true
python -m worker_services.gemma_worker.app
```

## Run in Real Mode

```bash
export GEMMA_WORKER_MOCK_MODE=false
export GEMMA_WORKER_MODEL_NAME=gemma-12b-it
export GEMMA_WORKER_MODEL_PATH=/models/gemma-12b-it
export GEMMA_WORKER_MAX_CONCURRENT=1
export GEMMA_WORKER_NUM_THREADS=8
export GEMMA_WORKER_NUM_INTEROP_THREADS=1
python -m worker_services.gemma_worker.app
```

`TransformersGemmaRunner` contains TODO hooks for real model load/inference integration.
`GEMMA_WORKER_MAX_CONCURRENT` is intentionally forced to `1` for CPU stability.

## Example cURL

```bash
curl -s http://127.0.0.1:8888/health
```

```bash
curl -s -X POST http://127.0.0.1:8888/generate \
  -H 'content-type: application/json' \
  -d '{"request_id":"req-1","model_name":"gemma-12b-it","prompt":"Explain VaR","temperature":0.2,"max_new_tokens":64}'
```

```bash
curl -N -X POST http://127.0.0.1:8888/generate/stream \
  -H 'content-type: application/json' \
  -d '{"request_id":"req-1","model_name":"gemma-12b-it","prompt":"Explain VaR","temperature":0.2,"max_new_tokens":64}'
```

## Backbone Integration

Configure backbone `GEMMA_WORKERS_JSON` with one or more workers:

```json
[
  {"id":"gemma-01","url":"http://host1:8888","max_concurrent":1},
  {"id":"gemma-02","url":"http://host2:8888","max_concurrent":1}
]
```

The backbone `local-gemma` provider treats these workers as one logical provider and handles worker selection.

## CPU Tuning Checklist

- Keep `GEMMA_WORKER_MAX_CONCURRENT=1` (enforced in code).
- Set `GEMMA_WORKER_NUM_THREADS` to match your physical CPU core budget per worker process.
- Keep `GEMMA_WORKER_NUM_INTEROP_THREADS` low (often `1`).
- Start with lower `max_new_tokens` for interactive UX.
- Monitor p50/p95 latency and tune threads before increasing machine count.
- Prefer horizontal scale (more workers) over in-process concurrency on CPU-only nodes.
