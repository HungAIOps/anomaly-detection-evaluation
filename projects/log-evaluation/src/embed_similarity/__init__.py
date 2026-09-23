"""Embedding-similarity log classification package."""

from __future__ import annotations

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_INPUT_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_PATH,
    EmbedSimilarityConfig,
    EVENT_TYPE_DEFINITIONS,
    KnowledgeCard,
)
from .evaluation import classify_dataframe, run_evaluation
from .model import EmbeddingBackend, SentenceTransformerBackend, load_embedder
from .preprocessing import load_knowledge_cards, normalize_text, resolve_text_column
from .cli import build_parser, main


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_INPUT_PATH",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_OUTPUT_PATH",
    "EmbeddingBackend",
    "EmbedSimilarityConfig",
    "EVENT_TYPE_DEFINITIONS",
    "KnowledgeCard",
    "SentenceTransformerBackend",
    "build_parser",
    "classify_dataframe",
    "load_embedder",
    "load_knowledge_cards",
    "main",
    "normalize_text",
    "resolve_text_column",
    "run_evaluation",
]
