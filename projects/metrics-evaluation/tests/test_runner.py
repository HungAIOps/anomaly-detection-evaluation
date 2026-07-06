from pathlib import Path

from metrics_evaluation.runner import EvaluationConfig


def test_evaluation_config_keeps_input_path() -> None:
    config = EvaluationConfig(input_path=Path("metrics.jsonl"))

    assert config.input_path == Path("metrics.jsonl")
    assert config.random_seed == 42
