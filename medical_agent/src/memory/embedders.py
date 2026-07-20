"""Embedding backends used by episodic and semantic memory."""
from __future__ import annotations

from typing import Iterable, List, Optional


class Embedder:
    """Small embedding interface with enough metadata to prevent vector mixing."""

    model_id = "unknown"
    dim: Optional[int] = None

    def embed(self, text: str) -> List[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]


class MockEmbedder(Embedder):
    """Deterministic dependency-free embedder used by tests and mock mode."""

    model_id = "semantic-mock-v1"

    def __init__(self):
        from .mock_embedder import SemanticMockEmbedder

        self._impl = SemanticMockEmbedder()
        self.dim = self._impl.dim

    def embed(self, text: str) -> List[float]:
        return self._impl.embed(text)

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        return [self._impl.embed(text) for text in texts]


class BGEEmbedder(Embedder):
    """Lazy SentenceTransformer wrapper for BAAI/bge-m3.

    The model is loaded on first use so importing the mock path remains free of
    torch and sentence-transformers dependencies.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        batch_size: int = 8,
        use_fp16: bool = True,
        local_files_only: bool = False,
    ):
        self.model_name = model_name
        self.model_id = model_name
        self.device = device
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.local_files_only = local_files_only
        self.dim = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "BGEEmbedder requires sentence-transformers; install project production dependencies"
            ) from exc

        kwargs = {}
        if self.device:
            kwargs["device"] = self.device
        if self.local_files_only:
            kwargs["local_files_only"] = True
        self._model = SentenceTransformer(self.model_name, **kwargs)
        if self.use_fp16 and str(self._model.device).startswith("cuda"):
            self._model.half()
        self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        values = [str(text or "") for text in texts]
        if not values:
            return []
        model = self._load()
        vectors = model.encode(
            values,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32", copy=False).tolist()


def create_embedder(
    name: str,
    *,
    model_name: str = "BAAI/bge-m3",
    device: Optional[str] = None,
    batch_size: int = 8,
    use_fp16: bool = True,
    local_files_only: bool = False,
) -> Embedder:
    normalized = name.strip().lower()
    if normalized == "mock":
        return MockEmbedder()
    if normalized in {"bge", "bge-m3"}:
        return BGEEmbedder(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            use_fp16=use_fp16,
            local_files_only=local_files_only,
        )
    raise ValueError(f"unsupported memory embedder: {name}")
