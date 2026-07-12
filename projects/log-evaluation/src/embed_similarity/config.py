"""Fixed configuration for embedding-similarity classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_INPUT_PATH = Path("data/logs.csv")
DEFAULT_OUTPUT_PATH = Path("artifacts/embed_similarity_predictions.csv")
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_BATCH_SIZE = 32
DEFAULT_TEXT_COLUMN_CANDIDATES = ("log", "message", "content", "text", "raw_log")
MIN_SIMILARITY_THRESHOLD = 0.4
MIN_MARGIN_THRESHOLD = 0.0001

EVENT_TYPE_DEFINITIONS = (
    "InboundAuthError",
    "InboundValidationError",
    "InternalApplicationError",
    "DatabaseError",
    "OutboundClientCallError",
)
UNSPECIFIED_EVENT_TYPE = "UnspecifiedError"


@dataclass(frozen=True, slots=True)
class KnowledgeCard:
    """Loaded knowledge card text for a single event type."""

    event_type: str
    path: Path
    text: str


@dataclass(frozen=True, slots=True)
class EmbedSimilarityConfig:
    """Configuration for the embedding-similarity log classifier."""

    input_path: Path = DEFAULT_INPUT_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH
    text_column: str | None = None
    model_name: str = DEFAULT_MODEL_NAME
    knowledge_card_dir: Path = Path(__file__).resolve().parent / "knowledge_cards"
    batch_size: int = DEFAULT_BATCH_SIZE
