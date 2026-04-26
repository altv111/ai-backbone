import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from worker_services.faiss_worker.embedding import EmbeddingService
from worker_services.faiss_worker.schemas import CollectionInfo, IndexDocument, RetrievedDocument


class MockIndexStore:
    def __init__(self) -> None:
        self._collections: dict[str, list[IndexDocument]] = {}
        self._lock = asyncio.Lock()

    async def index_documents(self, collection: str, documents: list[IndexDocument], mode: str) -> dict[str, Any]:
        async with self._lock:
            existing = [] if mode == "replace" else list(self._collections.get(collection, []))
            merged = _merge_documents(existing, documents)
            self._collections[collection] = merged
            return {
                "collection": collection,
                "indexed_count": len(documents),
                "total_count": len(merged),
                "metadata": {"mode": mode, "mock_mode": True},
            }

    async def retrieve(self, collection: str, query: str, top_k: int, filters: dict[str, Any]) -> list[RetrievedDocument]:
        async with self._lock:
            if collection not in self._collections:
                raise KeyError(collection)
            documents = self._collections[collection]

        query_terms = {term for term in query.lower().split() if term}
        scored: list[tuple[IndexDocument, float]] = []
        for idx, doc in enumerate(documents):
            terms = set(doc.text.lower().split())
            overlap = len(query_terms & terms)
            score = overlap / max(len(query_terms), 1)
            if score <= 0:
                score = max(0.0, 0.1 - idx * 0.001)
            scored.append((doc, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        filtered = [
            RetrievedDocument(id=doc.id, text=doc.text, score=float(score), metadata=doc.metadata)
            for doc, score in scored
            if _matches_filters(doc.metadata, filters)
        ]
        return filtered[:top_k]

    async def list_collections(self) -> list[CollectionInfo]:
        async with self._lock:
            return [
                CollectionInfo(
                    name=name,
                    document_count=len(documents),
                    index_exists=bool(documents),
                    metadata={"mock_mode": True},
                )
                for name, documents in sorted(self._collections.items())
            ]


class FaissIndexStore:
    def __init__(self, index_root: str, embedding_model_name: str, embedding_service: EmbeddingService) -> None:
        self.index_root = Path(index_root)
        self.embedding_model_name = embedding_model_name
        self.embedding_service = embedding_service
        self._lock = asyncio.Lock()
        self.index_root.mkdir(parents=True, exist_ok=True)

    async def index_documents(self, collection: str, documents: list[IndexDocument], mode: str) -> dict[str, Any]:
        async with self._lock:
            collection_dir = self._collection_dir(collection)
            collection_dir.mkdir(parents=True, exist_ok=True)

            existing_docs = [] if mode == "replace" else self._load_documents(collection)
            merged_docs = _merge_documents(existing_docs, documents)

            texts = [doc.text for doc in merged_docs]
            embeddings = self.embedding_service.encode(texts, show_progress_bar=True)
            import numpy as np

            embeddings = embeddings.astype(np.float32)

            # Real-mode FAISS pattern: normalize then IndexFlatIP.
            import faiss

            faiss.normalize_L2(embeddings)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
            faiss.write_index(index, str(self._index_path(collection)))

            self._write_documents(collection, merged_docs)
            self._write_manifest(collection, len(merged_docs))

            return {
                "collection": collection,
                "indexed_count": len(documents),
                "total_count": len(merged_docs),
                "metadata": {"mode": mode, "index_type": "IndexFlatIP", "normalized": True},
            }

    async def retrieve(self, collection: str, query: str, top_k: int, filters: dict[str, Any]) -> list[RetrievedDocument]:
        async with self._lock:
            if not self._index_path(collection).exists() or not self._documents_path(collection).exists():
                raise KeyError(collection)

            import faiss

            index = faiss.read_index(str(self._index_path(collection)))
            documents = self._load_documents(collection)

            query_embedding = self.embedding_service.encode([query], show_progress_bar=False).astype(np.float32)
            faiss.normalize_L2(query_embedding)

            candidate_k = min(max(top_k * 5, top_k), 100, len(documents))
            scores, indices = index.search(query_embedding, candidate_k)

            results: list[RetrievedDocument] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or idx >= len(documents):
                    continue
                doc = documents[int(idx)]
                if not _matches_filters(doc.metadata, filters):
                    continue
                results.append(
                    RetrievedDocument(
                        id=doc.id,
                        text=doc.text,
                        score=float(score),
                        metadata=doc.metadata,
                    )
                )
                if len(results) >= top_k:
                    break
            return results

    async def list_collections(self) -> list[CollectionInfo]:
        collections: list[CollectionInfo] = []
        for path in sorted(self.index_root.iterdir() if self.index_root.exists() else []):
            if not path.is_dir():
                continue
            name = path.name
            document_count = len(self._load_documents(name)) if self._documents_path(name).exists() else 0
            collections.append(
                CollectionInfo(
                    name=name,
                    document_count=document_count,
                    index_exists=self._index_path(name).exists(),
                    metadata={"path": str(path)},
                )
            )
        return collections

    def _collection_dir(self, collection: str) -> Path:
        return self.index_root / collection

    def _index_path(self, collection: str) -> Path:
        return self._collection_dir(collection) / "index.faiss"

    def _documents_path(self, collection: str) -> Path:
        return self._collection_dir(collection) / "documents.json"

    def _manifest_path(self, collection: str) -> Path:
        return self._collection_dir(collection) / "manifest.json"

    def _load_documents(self, collection: str) -> list[IndexDocument]:
        path = self._documents_path(collection)
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [IndexDocument(**item) for item in payload]

    def _write_documents(self, collection: str, documents: list[IndexDocument]) -> None:
        payload = [doc.model_dump() for doc in documents]
        self._documents_path(collection).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_manifest(self, collection: str, document_count: int) -> None:
        payload = {
            "collection": collection,
            "embedding_model": self.embedding_model_name,
            "document_count": document_count,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "index_type": "IndexFlatIP",
            "normalized": True,
            "similarity": "cosine_via_inner_product",
        }
        self._manifest_path(collection).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _merge_documents(existing: list[IndexDocument], incoming: list[IndexDocument]) -> list[IndexDocument]:
    merged: dict[str, IndexDocument] = {doc.id: doc for doc in existing}
    for doc in incoming:
        merged[doc.id] = doc
    return list(merged.values())


def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    if not filters:
        return True
    for key, value in filters.items():
        if metadata.get(key) != value:
            return False
    return True
