from pathlib import Path

from log_evaluation.cli import build_parser


def test_build_parser_parses_paths() -> None:
    parser = build_parser()
    args = parser.parse_args(["--input", "sample.json"])

    assert args.input == Path("sample.json")
    assert args.output == Path("results.json")
