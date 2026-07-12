"""Input preparation for embedding-similarity classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from .config import DEFAULT_TEXT_COLUMN_CANDIDATES, EVENT_TYPE_DEFINITIONS, KnowledgeCard

LOGGER = logging.getLogger(__name__)


def load_knowledge_cards(card_dir: Path) -> list[KnowledgeCard]:
    """Load the configured knowledge cards from text files."""

    if not card_dir.exists():
        raise FileNotFoundError(f"Knowledge-card directory does not exist: {card_dir}")
    if not card_dir.is_dir():
        raise NotADirectoryError(f"Knowledge-card path is not a directory: {card_dir}")

    cards: list[KnowledgeCard] = []
    expected_event_types = set(EVENT_TYPE_DEFINITIONS)
    for event_type in EVENT_TYPE_DEFINITIONS:
        card_path = card_dir / f"{event_type}.txt"
        if not card_path.exists():
            raise FileNotFoundError(f"Missing knowledge card file: {card_path}")
        card_text = card_path.read_text(encoding="utf-8").strip()
        if not card_text:
            raise ValueError(f"Knowledge card is empty: {card_path}")
        cards.append(KnowledgeCard(event_type=event_type, path=card_path, text=card_text))

    for extra_path in sorted(card_dir.glob("*.txt")):
        if extra_path.stem not in expected_event_types:
            LOGGER.warning("Ignoring unexpected knowledge card file: %s", extra_path)

    return cards


def resolve_text_column(dataframe: pd.DataFrame, preferred: str | None = None) -> str:
    """Choose the column that contains the raw log text."""

    if preferred is not None:
        if preferred not in dataframe.columns:
            raise KeyError(f"Requested text column not found: {preferred}")
        return preferred

    for candidate in DEFAULT_TEXT_COLUMN_CANDIDATES:
        if candidate in dataframe.columns:
            return candidate

    textual_columns = [
        column
        for column in dataframe.columns
        if is_object_dtype(dataframe[column]) or is_string_dtype(dataframe[column])
    ]
    if len(textual_columns) == 1:
        return textual_columns[0]
    if textual_columns:
        LOGGER.warning("Multiple textual columns found; using the first one: %s", textual_columns[0])
        return textual_columns[0]
    if len(dataframe.columns) == 1:
        return str(dataframe.columns[0])

    raise ValueError(
        "Unable to infer the log-text column. Pass --text-column to specify the source column."
    )


def normalize_text(value: Any) -> str:
    """Convert a CSV field value into a clean log string."""

    if pd.isna(value):
        return ""
    return str(value).strip()
