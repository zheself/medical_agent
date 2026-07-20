"""Optional cross-encoder rerankers for episodic memory."""
from __future__ import annotations

from typing import Iterable, List, Optional


class Reranker:
    model_id = "unknown"

    def score(self, query: str, documents: Iterable[str]) -> List[float]:
        raise NotImplementedError


class IdentityReranker(Reranker):
    """Deterministic reranker for tests; preserves candidate order."""

    model_id = "identity-v1"

    def score(self, query: str, documents: Iterable[str]) -> List[float]:
        docs = list(documents)
        return [float(len(docs) - index) for index in range(len(docs))]


class CrossEncoderReranker(Reranker):
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        batch_size: int = 8,
        local_files_only: bool = False,
        use_fp16: bool = True,
    ):
        self.model_name = model_name
        self.model_id = model_name
        self.device = device
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self.use_fp16 = use_fp16
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("CrossEncoderReranker requires sentence-transformers") from exc
            kwargs = {"trust_remote_code": True}
            if self.device:
                kwargs["device"] = self.device
            if self.local_files_only:
                kwargs["local_files_only"] = True
            self._model = CrossEncoder(self.model_name, **kwargs)
            if self.use_fp16 and str(self._model.model.device).startswith("cuda"):
                self._model.model.half()
        return self._model

    def score(self, query: str, documents: Iterable[str]) -> List[float]:
        docs = [str(document or "") for document in documents]
        if not docs:
            return []
        scores = self._load().predict(
            [(query, document) for document in docs],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]


def create_reranker(
    name: str,
    *,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: Optional[str] = None,
    batch_size: int = 8,
    local_files_only: bool = False,
) -> Optional[Reranker]:
    normalized = name.strip().lower()
    if normalized in {"", "none"}:
        return None
    if normalized == "identity":
        return IdentityReranker()
    if normalized in {"cross-encoder", "bge-reranker"}:
        return CrossEncoderReranker(model_name, device, batch_size, local_files_only)
    raise ValueError(f"unsupported memory reranker: {name}")
