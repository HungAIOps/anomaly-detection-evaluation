"""Evaluation and CSV export for embedding-similarity classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import EmbedSimilarityConfig, MIN_MARGIN_THRESHOLD, MIN_SIMILARITY_THRESHOLD, UNSPECIFIED_EVENT_TYPE
from .model import EmbeddingBackend, load_embedder
from .preprocessing import load_knowledge_cards, normalize_text, resolve_text_column


def classify_dataframe(
    dataframe: pd.DataFrame,
    config: EmbedSimilarityConfig,
    embedder: EmbeddingBackend,
) -> pd.DataFrame:
    """Classify each log row by similarity to the knowledge cards."""

    cards = load_knowledge_cards(config.knowledge_card_dir)
    text_column = resolve_text_column(dataframe, config.text_column)
    output = dataframe.copy()
    log_texts = [normalize_text(value) for value in output[text_column].tolist()]

    card_texts = [card.text for card in cards]
    card_embeddings = embedder.encode(card_texts)
    log_embeddings = embedder.encode(log_texts)

    if card_embeddings.ndim != 2:
        raise ValueError("Knowledge-card embeddings must be a 2D matrix.")
    if log_embeddings.ndim != 2:
        raise ValueError("Log embeddings must be a 2D matrix.")
    if card_embeddings.shape[1] != log_embeddings.shape[1]:
        raise ValueError("Log and knowledge-card embeddings must have the same dimensionality.")

    similarity_matrix = log_embeddings @ card_embeddings.T
    
    # Sort similarities in descending order
    sorted_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1]
    best_indices = sorted_indices[:, 0]
    second_indices = sorted_indices[:, 1]

    best_scores = similarity_matrix[np.arange(len(output)), best_indices]
    second_scores = similarity_matrix[np.arange(len(output)), second_indices]

    margins = best_scores - second_scores
    predicted_event_types = []

    for best_index, best_score, margin in zip(
        best_indices,
        best_scores,
        margins,
        strict=True,
    ):
        if (
            best_score < MIN_SIMILARITY_THRESHOLD
            or margin < MIN_MARGIN_THRESHOLD
        ):
            predicted_event_types.append(UNSPECIFIED_EVENT_TYPE)
        else:
            predicted_event_types.append(cards[best_index].event_type)

    output["predicted_event_type"] = predicted_event_types
    output["predicted_similarity"] = best_scores
    output["predicted_text_column"] = text_column
    # output["second_best_similarity"] = second_scores
    # output["similarity_margin"] = margins

    return output


def run_evaluation(config: EmbedSimilarityConfig) -> dict[str, Any]:
    """Run the embedding-similarity classification pipeline and export a CSV artifact."""

    if not config.input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {config.input_path}")

    dataframe = pd.read_csv(config.input_path)
    embedder = load_embedder(config)
    classified = classify_dataframe(dataframe, config, embedder)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    classified.to_csv(config.output_path, index=False)

    event_type_counts = classified["predicted_event_type"].value_counts().to_dict()
    return {
        "input_path": str(config.input_path),
        "output_path": str(config.output_path),
        "rows": int(len(classified)),
        "text_column": resolve_text_column(dataframe, config.text_column),
        "event_type_counts": event_type_counts,
    }
