import hashlib
from typing import Protocol


class EmbeddingService(Protocol):
    def encode(self, texts: list[str], show_progress_bar: bool = False):
        ...


class MockEmbeddingService:
    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension

    def encode(self, texts: list[str], show_progress_bar: bool = False):
        import numpy as np

        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([digest[i] / 255.0 for i in range(self.dimension)])
        return np.array(vectors, dtype=np.float32)


class SentenceTransformerEmbeddingService:
    def __init__(self, model_name: str) -> None:
        # Heavy import kept local to real mode.
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: list[str], show_progress_bar: bool = False):
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        import numpy as np

        return vectors.astype(np.float32)
