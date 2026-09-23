"""Input preparation for Drain3-based log template mining."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from .config import DEFAULT_TEXT_COLUMN_CANDIDATES

LOGGER = logging.getLogger(__name__)


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
        LOGGER.warning(
            "Multiple textual columns found; using the first one: %s",
            textual_columns[0],
        )
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