"""Drain3 integration for log template mining."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .config import LogTemplateConfig


@dataclass(frozen=True, slots=True)
class LogTemplateResult:
    """Normalized template-mining result for a single log line."""

    cluster_id: int
    cluster_size: int
    template: str
    change_type: str
    cluster_count: int


class LogTemplateBackend(Protocol):
    """Minimal interface for template mining backends."""

    def add_log_message(self, log_message: str) -> Mapping[str, Any]:
        """Mine a template for a single log line."""


class Drain3TemplateMinerBackend:
    """Lazy wrapper around Drain3's TemplateMiner."""

    def __init__(self, config: LogTemplateConfig) -> None:
        self._config = config
        self._miner: Any | None = None

    def _ensure_miner(self) -> Any:
        if self._miner is None:
            try:
                from drain3 import TemplateMiner
                from drain3.template_miner_config import TemplateMinerConfig
            except ImportError as exc:  # pragma: no cover - exercised in real runtime.
                raise RuntimeError("drain3 is required to run log template mining.") from exc

            miner_config = TemplateMinerConfig()
            miner_config.drain_sim_th = self._config.similarity_threshold
            miner_config.drain_depth = self._config.depth
            miner_config.drain_max_children = self._config.max_children
            miner_config.drain_max_clusters = self._config.max_clusters
            miner_config.extra_delimiters = list(self._config.extra_delimiters)
            self._miner = TemplateMiner(config=miner_config)

        return self._miner

    def add_log_message(self, log_message: str) -> Mapping[str, Any]:
        miner = self._ensure_miner()
        result = miner.add_log_message(log_message)
        if not isinstance(result, Mapping):
            raise TypeError("Drain3 returned an unexpected result type.")
        return result


def _extract_result(result: Mapping[str, Any]) -> LogTemplateResult:
    """Convert a Drain3 result dictionary into a typed record."""

    try:
        return LogTemplateResult(
            cluster_id=int(result["cluster_id"]),
            cluster_size=int(result["cluster_size"]),
            template=str(result["template_mined"]),
            change_type=str(result["change_type"]),
            cluster_count=int(result["cluster_count"]),
        )
    except KeyError as exc:
        raise KeyError(f"Drain3 result is missing an expected field: {exc.args[0]}") from exc


def load_template_miner(config: LogTemplateConfig) -> Drain3TemplateMinerBackend:
    """Create the default Drain3 backend for a run."""

    return Drain3TemplateMinerBackend(config=config)


def mine_templates(texts: Sequence[str], backend: LogTemplateBackend) -> list[LogTemplateResult]:
    """Mine templates for a sequence of log messages."""

    mined_results: list[LogTemplateResult] = []
    for text in texts:
        mined_results.append(_extract_result(backend.add_log_message(text)))
    return mined_results