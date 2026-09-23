"""Data preparation helpers for STL metric evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import MissingValueStrategy


@dataclass(frozen=True, slots=True)
class PreparedDataFrame:
    """Prepared dataframe ready for STL fitting or scoring."""

    dataframe: pd.DataFrame
    order_column: str


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file from disk."""

    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")
    return pd.read_csv(path)


def prepare_dataframe(
    dataframe: pd.DataFrame,
    *,
    timestamp_column: str,
    metric_columns: tuple[str, ...],
    missing_value_strategy: MissingValueStrategy,
) -> PreparedDataFrame:
    """Clean timestamps and metric columns without leaking between datasets."""

    missing_columns = [column for column in (timestamp_column, *metric_columns) if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    cleaned = dataframe.copy()
    order_column = "__stl_row_order__"
    cleaned[order_column] = range(len(cleaned))
    cleaned[timestamp_column] = pd.to_datetime(cleaned[timestamp_column], errors="coerce")

    metric_frame = cleaned.loc[:, list(metric_columns)].apply(pd.to_numeric, errors="coerce")
    cleaned.loc[:, list(metric_columns)] = metric_frame

    if missing_value_strategy == "fail":
        invalid_rows = cleaned[timestamp_column].isna()
        invalid_rows |= metric_frame.isna().any(axis=1)
        if invalid_rows.any():
            raise ValueError("Encountered missing or non-numeric values with fail strategy.")
    elif missing_value_strategy == "drop":
        cleaned = cleaned.dropna(subset=[timestamp_column, *metric_columns])
    elif missing_value_strategy == "interpolate":
        cleaned = cleaned.dropna(subset=[timestamp_column])
        cleaned = cleaned.sort_values(timestamp_column, kind="mergesort")
        cleaned.loc[:, list(metric_columns)] = (
            cleaned.loc[:, list(metric_columns)]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )
    else:
        raise ValueError(f"Unsupported missing value strategy: {missing_value_strategy}")

    if cleaned.empty:
        raise ValueError("No rows remain after preprocessing.")

    cleaned = cleaned.sort_values(timestamp_column, kind="mergesort").reset_index(drop=True)
    return PreparedDataFrame(dataframe=cleaned, order_column=order_column)


def restore_original_order(dataframe: pd.DataFrame, order_column: str) -> pd.DataFrame:
    """Restore the original row order after scoring."""

    restored = dataframe.sort_values(order_column, kind="mergesort").reset_index(drop=True)
    return restored.drop(columns=[order_column])