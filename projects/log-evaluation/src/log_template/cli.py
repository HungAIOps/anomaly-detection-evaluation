"""Command-line interface for Drain3-based log template mining."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    DEFAULT_CLUSTER_ID_COLUMN,
    DEFAULT_DEPTH,
    DEFAULT_EXTRA_DELIMITERS,
    DEFAULT_INPUT_PATH,
    DEFAULT_MAX_CHILDREN,
    DEFAULT_MAX_CLUSTERS,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TEMPLATE_COLUMN,
    LogTemplateConfig,
)
from .evaluation import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for local runs."""

    parser = argparse.ArgumentParser(description="Mine log templates with Drain3.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input CSV path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output CSV path.")
    parser.add_argument("--text-column", type=str, default=None, help="Raw log text column.")
    parser.add_argument(
        "--template-column",
        type=str,
        default=DEFAULT_TEMPLATE_COLUMN,
        help="Output column for the mined template.",
    )
    parser.add_argument(
        "--cluster-id-column",
        type=str,
        default=DEFAULT_CLUSTER_ID_COLUMN,
        help="Output column for the template cluster identifier.",
    )
    parser.add_argument(
        "--sim-th",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Drain3 similarity threshold.",
    )
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="Drain3 tree depth.")
    parser.add_argument(
        "--max-children",
        type=int,
        default=DEFAULT_MAX_CHILDREN,
        help="Maximum children per Drain3 node.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=DEFAULT_MAX_CLUSTERS,
        help="Maximum tracked Drain3 clusters.",
    )
    parser.add_argument(
        "--extra-delimiters",
        nargs="*",
        default=list(DEFAULT_EXTRA_DELIMITERS),
        help="Additional Drain3 token delimiters.",
    )
    return parser


def main() -> int:
    """Run the log-template pipeline from the package entry point."""

    args = build_parser().parse_args()
    run_evaluation(
        LogTemplateConfig(
            input_path=args.input,
            output_path=args.output,
            text_column=args.text_column,
            template_column=args.template_column,
            cluster_id_column=args.cluster_id_column,
            similarity_threshold=args.sim_th,
            depth=args.depth,
            max_children=args.max_children,
            max_clusters=args.max_clusters,
            extra_delimiters=tuple(args.extra_delimiters),
        )
    )
    return 0