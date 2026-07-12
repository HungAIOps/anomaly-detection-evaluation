"""Command-line interface for embeding similarity model."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_INPUT_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_PATH,
    EmbedSimilarityConfig
)
from .evaluation import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for local runs."""

    parser = argparse.ArgumentParser(description="Classify error logs with BGE-M3 similarity.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input CSV path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output CSV path.")
    parser.add_argument("--text-column", type=str, default=None, help="Raw log text column.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Embedding model.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size.")
    return parser


def main() -> int:
    """Run the embedding-similarity pipeline from the package entry point."""

    args = build_parser().parse_args()
    run_evaluation(
        EmbedSimilarityConfig(
            input_path=args.input,
            output_path=args.output,
            text_column=args.text_column,
            model_name=args.model_name,
            batch_size=args.batch_size,
        )
    )
    return 0