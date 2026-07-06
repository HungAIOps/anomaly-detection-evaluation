from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Configuration for log evaluation runs."""

    input_path: Path
    output_path: Path | None = None
    random_seed: int = 42


def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    """Run the log evaluation pipeline."""

    raise NotImplementedError("Implement the log evaluation pipeline.")
