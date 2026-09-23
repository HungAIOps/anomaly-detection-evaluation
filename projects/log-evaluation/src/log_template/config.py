"""Configuration for Drain3-based log template mining."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_INPUT_PATH = Path("data/test.csv")
DEFAULT_OUTPUT_PATH = Path("artifacts/log_template_output.csv")
DEFAULT_TEXT_COLUMN_CANDIDATES = ("log", "message", "content", "text", "raw_log")
DEFAULT_TEMPLATE_COLUMN = "template"
DEFAULT_CLUSTER_ID_COLUMN = "template_cluster_id"
DEFAULT_SIMILARITY_THRESHOLD = 0.4
DEFAULT_DEPTH = 4
DEFAULT_MAX_CHILDREN = 100
DEFAULT_MAX_CLUSTERS = 1000
DEFAULT_EXTRA_DELIMITERS = ("_", ":", "/")


@dataclass(frozen=True, slots=True)
class LogTemplateConfig:
    """Configuration for a log-template mining run."""

    input_path: Path = DEFAULT_INPUT_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH
    text_column: str | None = None
    template_column: str = DEFAULT_TEMPLATE_COLUMN
    cluster_id_column: str = DEFAULT_CLUSTER_ID_COLUMN
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    depth: int = DEFAULT_DEPTH
    max_children: int = DEFAULT_MAX_CHILDREN
    max_clusters: int = DEFAULT_MAX_CLUSTERS
    extra_delimiters: tuple[str, ...] = DEFAULT_EXTRA_DELIMITERS