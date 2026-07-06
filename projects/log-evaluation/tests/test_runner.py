from pathlib import Path

from log_evaluation.runner import EvaluationConfig


def test_evaluation_config_keeps_input_path() -> None:
    config = EvaluationConfig(input_path=Path("logs.jsonl"))

    assert config.input_path == Path("logs.jsonl")
    assert config.random_seed == 42
