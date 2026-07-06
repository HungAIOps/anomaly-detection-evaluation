from pathlib import Path

from trace_evaluation.runner import EvaluationConfig


def test_evaluation_config_keeps_input_path() -> None:
    config = EvaluationConfig(input_path=Path("traces.jsonl"))

    assert config.input_path == Path("traces.jsonl")
    assert config.random_seed == 42
