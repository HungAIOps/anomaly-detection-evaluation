from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_prophet_algo_module():
	module_path = Path(__file__).with_name("prophet.py")
	module_name = "prophet_algo"
	spec = importlib.util.spec_from_file_location(module_name, module_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load module from {module_path}")
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	spec.loader.exec_module(module)
	return module


prophet_algo = _load_prophet_algo_module()
ProphetTrainingConfig = prophet_algo.ProphetTrainingConfig
train_model = prophet_algo.train_model
export_training_artifacts = prophet_algo.export_training_artifacts
evaluate_from_artifacts = prophet_algo.evaluate_from_artifacts

LOGGER = logging.getLogger(__name__)


def generate_train_data() -> pd.DataFrame:
	"""Generate clean historical metric data for Prophet training."""
	rng = np.random.default_rng(42)
	periods = 7 * 24 * 12
	timestamp = pd.date_range("2026-07-01 00:00:00", periods=periods, freq="5min")

	hours = np.arange(periods) / 12.0
	trend = 95.0 + 0.06 * hours
	daily = 12.0 * np.sin(2.0 * np.pi * hours / 24.0)
	weekly = 7.0 * np.sin(2.0 * np.pi * hours / (24.0 * 7.0))
	noise = rng.normal(0.0, 1.8, size=periods)
	metric = trend + daily + weekly + noise
	return pd.DataFrame({"timestamp": timestamp, "metric": metric})


def generate_eval_data() -> pd.DataFrame:
	"""Generate evaluation data with injected spikes and drops."""
	rng = np.random.default_rng(7)
	periods = 24 * 12
	timestamp = pd.date_range("2026-07-08 00:00:00", periods=periods, freq="5min")

	hours = np.arange(periods) / 12.0 + 24.0 * 7.0
	trend = 95.0 + 0.06 * hours
	daily = 12.0 * np.sin(2.0 * np.pi * hours / 24.0)
	weekly = 7.0 * np.sin(2.0 * np.pi * hours / (24.0 * 7.0))
	noise = rng.normal(0.0, 2.1, size=periods)
	metric = trend + daily + weekly + noise

	for pos in [35, 88, 151, 210, 250]:
		metric[pos] += 24.0 + 3.0 * rng.normal()
	metric[120] -= 21.0

	return pd.DataFrame({"timestamp": timestamp, "metric": metric})


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

	train_df = generate_train_data()
	config = ProphetTrainingConfig(
		growth="linear",
		seasonality_mode="additive",
		interval_width=0.95,
		changepoint_prior_scale=0.05,
		seasonality_prior_scale=10.0,
		yearly_seasonality=False,
		weekly_seasonality=True,
		daily_seasonality=True,
		threshold_z=3.5,
	)

	model, summary, train_output_df = train_model(
		train_df=train_df,
		time_field="timestamp",
		value_field="metric",
		config=config,
	)

	model_path = Path(__file__).with_name("prophet_model.json")
	summary_path = Path(__file__).with_name("prophet_training_summary.json")
	train_output_path = Path(__file__).with_name("prophet_train_output.csv")

	export_training_artifacts(
		model=model,
		summary=summary,
		train_output_df=train_output_df,
		model_path=model_path,
		summary_path=summary_path,
		train_output_path=train_output_path,
	)
	LOGGER.info("Exported training model artifact to %s", model_path)
	LOGGER.info("Exported training summary artifact to %s", summary_path)
	LOGGER.info("Exported training output to %s", train_output_path)

	eval_df = generate_eval_data()
	scored_df, events_df = evaluate_from_artifacts(
		eval_df=eval_df,
		model_path=model_path,
		summary_path=summary_path,
	)

	scored_output_path = Path(__file__).with_name("prophet_eval_output.csv")
	events_output_path = Path(__file__).with_name("prophet_eval_events.csv")
	scored_df.to_csv(scored_output_path, index=False)
	events_df.to_csv(events_output_path, index=False)

	LOGGER.info("Detected %s anomalies in evaluation phase", len(events_df))
	LOGGER.info("Exported evaluation output to %s", scored_output_path)
	LOGGER.info("Exported anomaly events to %s", events_output_path)


if __name__ == "__main__":
	main()
