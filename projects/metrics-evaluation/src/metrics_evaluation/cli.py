from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import EvaluationConfig, run_evaluation


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for metrics evaluation."""

    parser = argparse.ArgumentParser(description="Run metrics anomaly evaluation.")
    parser.add_argument("--input", type=Path, required=True, help="Input dataset path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results.json"),
        help="Output JSON path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def main() -> int:
    """Entry point for the metrics evaluation CLI."""

    parser = build_parser()
    args = parser.parse_args()
    result = run_evaluation(
        EvaluationConfig(input_path=args.input, output_path=args.output, random_seed=args.seed)
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
