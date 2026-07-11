"""CSV export for Drain3-based log template mining."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import LogTemplateConfig
from .algo import LogTemplateBackend, load_template_miner, mine_templates
from .preprocessing import normalize_text, resolve_text_column


def annotate_dataframe(
    dataframe: pd.DataFrame,
    config: LogTemplateConfig,
    backend: LogTemplateBackend,
) -> pd.DataFrame:
    """Annotate each log row with its Drain3 cluster and template."""

    text_column = resolve_text_column(dataframe, config.text_column)
    output = dataframe.copy()
    log_texts = [normalize_text(value) for value in output[text_column].tolist()]
    mined_results = mine_templates(log_texts, backend)

    output[config.cluster_id_column] = [result.cluster_id for result in mined_results]
    # Place the mined template into the configured output column name.
    output[config.template_column] = [result.template for result in mined_results]
    # output["template_cluster_size"] = [result.cluster_size for result in mined_results]
    # output["template_change_type"] = [result.change_type for result in mined_results]
    # output["template_cluster_count"] = [result.cluster_count for result in mined_results]
    # output["template_text_column"] = text_column
    return output


def run_evaluation(config: LogTemplateConfig) -> dict[str, Any]:
    """Run the log-template pipeline and export a CSV artifact."""

    if not config.input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {config.input_path}")

    dataframe = pd.read_csv(config.input_path)
    backend = load_template_miner(config)
    annotated = annotate_dataframe(dataframe, config, backend)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(config.output_path, index=False)

    return {
        "input_path": str(config.input_path),
        "output_path": str(config.output_path),
        "rows": int(len(annotated)),
        "text_column": resolve_text_column(dataframe, config.text_column),
        "cluster_count": int(annotated[config.cluster_id_column].nunique()),
    }