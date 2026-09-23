"""Embedding and similarity logic for log classification."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np

from .config import EmbedSimilarityConfig


class EmbeddingBackend(Protocol):
    """Minimal embedding interface for classification and tests."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into embedding vectors."""


class SentenceTransformerBackend:
    """BGE-M3 embedding backend via sentence-transformers."""

    def __init__(self, model_name: str, batch_size: int) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - exercised in real runtime.
                raise RuntimeError(
                    "sentence-transformers is required to run embedding similarity classification."
                ) from exc

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure_model()
        embeddings = model.encode(  # type: ignore[no-untyped-call]
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


def load_embedder(config: EmbedSimilarityConfig) -> EmbeddingBackend:
    """Create the embedding backend for a run."""

    return SentenceTransformerBackend(model_name=config.model_name, batch_size=config.batch_size)
