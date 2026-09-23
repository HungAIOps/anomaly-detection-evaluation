"""Configuration helpers for the STL metric evaluation project."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

Direction = Literal["upper", "lower"]
ThresholdMethod = Literal["std", "mad"]
MissingValueStrategy = Literal["drop", "interpolate", "fail"]
SequenceMethod = Literal["none", "consecutive", "cusum"]
CombinationRule = Literal["any", "all"]
HistoryValidationMode = Literal["warn", "fail"]

DEFAULT_HISTORICAL_PATH = Path("data/historical.csv")
DEFAULT_EVALUATION_PATH = Path("data/evaluation.csv")
DEFAULT_OUTPUT_PATH = Path("artifacts/stl_labeled.csv")
DEFAULT_TIMESTAMP_COLUMN = "timestamp"
DEFAULT_METRIC_PERIOD = 24
DEFAULT_DIRECTION: Direction = "upper"
DEFAULT_THRESHOLD_METHOD: ThresholdMethod = "std"
DEFAULT_THRESHOLD_K = 3.0


@dataclass(frozen=True, slots=True)
class MetricConfig:
    """Per-metric STL configuration."""

    period: int
    direction: Direction = DEFAULT_DIRECTION
    threshold_method: ThresholdMethod = DEFAULT_THRESHOLD_METHOD
    threshold_k: float = DEFAULT_THRESHOLD_K


@dataclass(frozen=True, slots=True)
class SequenceConfig:
    """Optional sequence aggregation configuration."""

    method: SequenceMethod = "none"
    min_consecutive_points: int = 3
    cusum_threshold: float = 5.0
    cusum_drift: float = 0.0


@dataclass(frozen=True, slots=True)
class STLRunConfig:
    """End-to-end configuration for STL anomaly evaluation."""

    historical_path: Path = DEFAULT_HISTORICAL_PATH
    evaluation_path: Path = DEFAULT_EVALUATION_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN
    metrics: tuple[str, ...] = field(default_factory=tuple)
    period: int = DEFAULT_METRIC_PERIOD
    direction: Direction = DEFAULT_DIRECTION
    threshold_method: ThresholdMethod = DEFAULT_THRESHOLD_METHOD
    threshold_k: float = DEFAULT_THRESHOLD_K
    robust: bool = True
    missing_value_strategy: MissingValueStrategy = "interpolate"
    minimum_history_mode: HistoryValidationMode = "warn"
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    overall_combination: CombinationRule = "any"
    metric_configs: dict[str, MetricConfig] = field(default_factory=dict)


def load_json_mapping(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a JSON object: {path}")
    return data


def parse_json_mapping(value: str) -> dict[str, Any]:
    """Parse an inline JSON object or a JSON file path."""

    candidate_path = Path(value)
    if candidate_path.exists():
        return load_json_mapping(candidate_path)
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("Metric configuration must be a JSON object.")
    return data


def parse_metrics(value: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Normalize metric names from CLI or configuration input."""

    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(metric.strip() for metric in value.split(",") if metric.strip())
    return tuple(str(metric).strip() for metric in value if str(metric).strip())


def _coalesce(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def _parse_bool(value: Any, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def _parse_metric_config(
    raw_config: Mapping[str, Any] | None,
    *,
    shared_defaults: MetricConfig,
) -> MetricConfig:
    if raw_config is None:
        return shared_defaults
    return MetricConfig(
        period=int(raw_config.get("period", shared_defaults.period)),
        direction=raw_config.get("direction", shared_defaults.direction),
        threshold_method=raw_config.get(
            "threshold_method", shared_defaults.threshold_method
        ),
        threshold_k=float(raw_config.get("threshold_k", shared_defaults.threshold_k)),
    )


def build_run_config(namespace: argparse.Namespace) -> STLRunConfig:
    """Build a strongly typed run config from CLI arguments and optional JSON config."""

    file_config: dict[str, Any] = {}
    if namespace.config is not None:
        file_config = load_json_mapping(namespace.config)

    metrics = parse_metrics(_coalesce(namespace.metrics, file_config.get("metrics")))
    if not metrics:
        raise ValueError("At least one metric must be provided.")

    default_metric_config = MetricConfig(
        period=int(_coalesce(namespace.period, file_config.get("period", DEFAULT_METRIC_PERIOD))),
        direction=_coalesce(namespace.direction, file_config.get("direction", DEFAULT_DIRECTION)),
        threshold_method=_coalesce(
            namespace.threshold_method,
            file_config.get("threshold_method", DEFAULT_THRESHOLD_METHOD),
        ),
        threshold_k=float(_coalesce(namespace.threshold_k, file_config.get("threshold_k", DEFAULT_THRESHOLD_K))),
    )

    sequence_config_data = file_config.get("sequence", {})
    if not isinstance(sequence_config_data, dict):
        raise ValueError("The 'sequence' configuration must be a JSON object.")
    sequence_method = _coalesce(namespace.sequence_method, sequence_config_data.get("method", "none"))
    sequence_config = SequenceConfig(
        method=sequence_method,
        min_consecutive_points=int(
            _coalesce(namespace.min_consecutive_points, sequence_config_data.get("min_consecutive_points", 3))
        ),
        cusum_threshold=float(
            _coalesce(namespace.cusum_threshold, sequence_config_data.get("cusum_threshold", 5.0))
        ),
        cusum_drift=float(
            _coalesce(namespace.cusum_drift, sequence_config_data.get("cusum_drift", 0.0))
        ),
    )

    metric_overrides: dict[str, Mapping[str, Any]] = {}
    file_metric_configs = file_config.get("metric_configs", {})
    if file_metric_configs and not isinstance(file_metric_configs, dict):
        raise ValueError("The 'metric_configs' configuration must be a JSON object.")
    for metric_name, metric_config in file_metric_configs.items():
        if not isinstance(metric_config, Mapping):
            raise ValueError(f"Metric configuration for {metric_name!r} must be a JSON object.")
        metric_overrides[str(metric_name)] = metric_config

    inline_metric_configs = namespace.metric_configs
    if inline_metric_configs is not None:
        parsed_metric_configs = parse_json_mapping(inline_metric_configs)
        for metric_name, metric_config in parsed_metric_configs.items():
            if not isinstance(metric_config, Mapping):
                raise ValueError(f"Metric configuration for {metric_name!r} must be a JSON object.")
            metric_overrides[str(metric_name)] = metric_config

    metric_configs = {
        metric_name: _parse_metric_config(metric_overrides.get(metric_name), shared_defaults=default_metric_config)
        for metric_name in metrics
    }

    historical_path = Path(_coalesce(namespace.historical, file_config.get("historical_path", DEFAULT_HISTORICAL_PATH)))
    evaluation_path = Path(_coalesce(namespace.evaluation, file_config.get("evaluation_path", DEFAULT_EVALUATION_PATH)))
    output_path = Path(_coalesce(namespace.output, file_config.get("output_path", DEFAULT_OUTPUT_PATH)))
    timestamp_column = _coalesce(
        namespace.timestamp_column,
        file_config.get("timestamp_column", DEFAULT_TIMESTAMP_COLUMN),
    )

    return STLRunConfig(
        historical_path=historical_path,
        evaluation_path=evaluation_path,
        output_path=output_path,
        timestamp_column=timestamp_column,
        metrics=metrics,
        period=default_metric_config.period,
        direction=default_metric_config.direction,
        threshold_method=default_metric_config.threshold_method,
        threshold_k=default_metric_config.threshold_k,
        robust=_parse_bool(_coalesce(namespace.robust, file_config.get("robust", True)), True),
        missing_value_strategy=_coalesce(
            namespace.missing_value_strategy,
            file_config.get("missing_value_strategy", "interpolate"),
        ),
        minimum_history_mode=_coalesce(
            namespace.minimum_history_mode,
            file_config.get("minimum_history_mode", "warn"),
        ),
        sequence=sequence_config,
        overall_combination=_coalesce(
            namespace.overall_combination,
            file_config.get("overall_combination", "any"),
        ),
        metric_configs=metric_configs,
    )


def config_to_dict(config: STLRunConfig) -> dict[str, Any]:
    """Convert a run config into a JSON-serializable dictionary."""

    payload = asdict(config)
    payload["historical_path"] = str(config.historical_path)
    payload["evaluation_path"] = str(config.evaluation_path)
    payload["output_path"] = str(config.output_path)
    payload["metrics"] = list(config.metrics)
    payload["metric_configs"] = {
        metric_name: asdict(metric_config)
        for metric_name, metric_config in config.metric_configs.items()
    }
    return payload