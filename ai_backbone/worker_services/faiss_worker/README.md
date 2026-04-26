# FAISS Worker Service

This service is deployed independently from the backbone API. It owns embeddings, FAISS indexes, sidecar documents, and retrieval logic.

## Endpoints

- `GET /health`
- `POST /index`
- `POST /retrieve`
- `GET /collections`

## Run in Mock Mode (default)

```bash
export FAISS_WORKER_MOCK_MODE=true
python -m worker_services.faiss_worker.app
```

## Run in Real Mode

```bash
export FAISS_WORKER_MOCK_MODE=false
export FAISS_WORKER_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export FAISS_WORKER_INDEX_ROOT=./data/faiss_indexes
python -m worker_services.faiss_worker.app
```

Real mode uses:
- `SentenceTransformer.encode(..., convert_to_numpy=True)`
- `faiss.normalize_L2(...)`
- `faiss.IndexFlatIP`
- `faiss.write_index(...)` / `faiss.read_index(...)`

## Collection Files

Per collection directory:

- `index.faiss`
- `documents.json`
- `manifest.json`

## Example cURL

```bash
curl -s http://127.0.0.1:8890/health
```

```bash
curl -s -X POST http://127.0.0.1:8890/index \
  -H 'content-type: application/json' \
  -d '{
    "collection":"sample-docs",
    "documents":[{"id":"doc-1","text":"Value at Risk measures potential loss.","metadata":{"source":"manual","category":"risk"}}],
    "mode":"append"
  }'
```

```bash
curl -s -X POST http://127.0.0.1:8890/retrieve \
  -H 'content-type: application/json' \
  -d '{
    "collection":"sample-docs",
    "query":"What is VaR?",
    "top_k":5,
    "filters":{}
  }'
```
