"""Command-line interface for the STL metric evaluation project."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import build_run_config
from .evaluation import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for local STL evaluation runs."""

    parser = argparse.ArgumentParser(description="Run STL-based metric anomaly evaluation.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file.")
    parser.add_argument("--historical", type=Path, default=None, help="Historical CSV path.")
    parser.add_argument("--evaluation", type=Path, default=None, help="Evaluation CSV path.")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path.")
    parser.add_argument("--timestamp-column", type=str, default=None, help="Timestamp column name.")
    parser.add_argument("--metrics", type=str, default=None, help="Comma-separated metric names.")
    parser.add_argument("--period", type=int, default=None, help="Shared seasonal period in rows.")
    parser.add_argument(
        "--direction",
        choices=["upper", "lower"],
        default=None,
        help="Shared point-anomaly direction.",
    )
    parser.add_argument(
        "--threshold-method",
        choices=["std", "mad"],
        default=None,
        help="Shared residual threshold method.",
    )
    parser.add_argument("--threshold-k", type=float, default=None, help="Shared residual threshold multiplier.")
    parser.add_argument("--robust", dest="robust", action="store_true", help="Enable robust STL fitting.")
    parser.add_argument("--no-robust", dest="robust", action="store_false", help="Disable robust STL fitting.")
    parser.set_defaults(robust=None)
    parser.add_argument(
        "--missing-value-strategy",
        choices=["drop", "interpolate", "fail"],
        default=None,
        help="How to handle missing or non-numeric values.",
    )
    parser.add_argument(
        "--minimum-history-mode",
        choices=["warn", "fail"],
        default=None,
        help="Warn or fail when history is shorter than two seasonal cycles.",
    )
    parser.add_argument(
        "--sequence-method",
        choices=["none", "consecutive", "cusum"],
        default=None,
        help="Optional sequence aggregation method.",
    )
    parser.add_argument(
        "--min-consecutive-points",
        type=int,
        default=None,
        help="Minimum consecutive point anomalies for sequence labeling.",
    )
    parser.add_argument(
        "--cusum-threshold",
        type=float,
        default=None,
        help="CUSUM threshold for sequence labeling.",
    )
    parser.add_argument(
        "--cusum-drift",
        type=float,
        default=None,
        help="CUSUM drift for sequence labeling.",
    )
    parser.add_argument(
        "--overall-combination",
        choices=["any", "all"],
        default=None,
        help="How to aggregate metric labels into the final row label.",
    )
    parser.add_argument(
        "--metric-configs",
        type=str,
        default=None,
        help="Inline JSON or a JSON file path with per-metric overrides.",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser


def main() -> int:
    """Run the STL evaluation pipeline from the package entry point."""

    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    config = build_run_config(args)
    summary = run_evaluation(config)
    logging.getLogger(__name__).info(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0