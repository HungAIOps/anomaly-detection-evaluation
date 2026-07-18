from __future__ import annotations

import json
import importlib
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _import_prophet_dependencies() -> tuple[Any, Any, Any]:
	"""Import Prophet safely even when this file name shadows the package name."""
	current_dir = Path(__file__).resolve().parent
	filtered_paths = [
		path
		for path in sys.path
		if Path(path or ".").resolve() != current_dir
	]

	try:
		import importlib.machinery
		import importlib.util

		package_spec = importlib.machinery.PathFinder.find_spec(
			"prophet", filtered_paths
		)
		if package_spec is None or package_spec.loader is None:
			raise ModuleNotFoundError("prophet")

		package_module = importlib.util.module_from_spec(package_spec)
		sys.modules["prophet"] = package_module
		package_spec.loader.exec_module(package_module)

		serialize_module = importlib.import_module("prophet.serialize")
		prophet_cls = getattr(package_module, "Prophet")
		model_to_json = getattr(serialize_module, "model_to_json")
		model_from_json = getattr(serialize_module, "model_from_json")
	except ModuleNotFoundError as exc:
		raise ModuleNotFoundError(
			"Missing dependency 'prophet'. Install it with: pip install prophet"
		) from exc

	return prophet_cls, model_to_json, model_from_json


@dataclass(frozen=True)
class ProphetTrainingConfig:
	"""Configuration for training and scoring with Prophet."""

	growth: str = "linear"
	seasonality_mode: str = "additive"
	interval_width: float = 0.95
	changepoint_prior_scale: float = 0.05
	seasonality_prior_scale: float = 10.0
	yearly_seasonality: bool = True
	weekly_seasonality: bool = True
	daily_seasonality: bool = True
	threshold_z: float = 3.5


@dataclass(frozen=True)
class ProphetTrainingSummary:
	"""Serializable summary exported after training."""

	time_field: str
	value_field: str
	threshold_z: float
	residual_mean: float
	residual_std: float
	train_rows: int


def _prepare_series(
	df: pd.DataFrame,
	time_field: str,
	value_field: str,
) -> pd.DataFrame:
	"""Validate and normalize source data into Prophet's ds/y format."""
	if time_field not in df.columns:
		raise ValueError(f"time_field '{time_field}' not found in dataframe columns")
	if value_field not in df.columns:
		raise ValueError(f"value_field '{value_field}' not found in dataframe columns")

	prepared = (
		df[[time_field, value_field]]
		.rename(columns={time_field: "ds", value_field: "y"})
		.copy()
	)
	prepared["ds"] = pd.to_datetime(prepared["ds"], errors="coerce")
	prepared["y"] = pd.to_numeric(prepared["y"], errors="coerce")

	if prepared["ds"].isna().any() or prepared["y"].isna().any():
		raise ValueError(
			"Input contains invalid timestamps or metric values after conversion."
		)

	prepared = prepared.sort_values("ds", kind="mergesort").reset_index(drop=True)
	return prepared


def train_model(
	train_df: pd.DataFrame,
	time_field: str,
	value_field: str,
	config: ProphetTrainingConfig | None = None,
) -> tuple[Any, ProphetTrainingSummary, pd.DataFrame]:
	"""Train Prophet and return model, residual calibration, and annotated train output."""
	config = config or ProphetTrainingConfig()
	prophet_cls, _, _ = _import_prophet_dependencies()
	prepared = _prepare_series(train_df, time_field=time_field, value_field=value_field)

	model = prophet_cls(
		growth=config.growth,
		seasonality_mode=config.seasonality_mode,
		interval_width=config.interval_width,
		changepoint_prior_scale=config.changepoint_prior_scale,
		seasonality_prior_scale=config.seasonality_prior_scale,
		yearly_seasonality=config.yearly_seasonality,
		weekly_seasonality=config.weekly_seasonality,
		daily_seasonality=config.daily_seasonality,
	)
	model.fit(prepared)

	forecast = model.predict(prepared[["ds"]])
	residuals = (
		prepared["y"].to_numpy(dtype=float)
		- forecast["yhat"].to_numpy(dtype=float)
	)
	residual_mean = float(np.mean(residuals))
	residual_std = float(np.std(residuals))
	residual_std = max(residual_std, 1e-6)

	z_score = np.abs((residuals - residual_mean) / residual_std)
	train_output = pd.DataFrame(
		{
			time_field: prepared["ds"],
			value_field: prepared["y"],
			f"{value_field}_forecast": forecast["yhat"].to_numpy(dtype=float),
			f"{value_field}_residual": residuals,
			f"{value_field}_z_score": z_score,
			f"{value_field}_is_anomaly": z_score > config.threshold_z,
		}
	)

	summary = ProphetTrainingSummary(
		time_field=time_field,
		value_field=value_field,
		threshold_z=config.threshold_z,
		residual_mean=residual_mean,
		residual_std=residual_std,
		train_rows=int(len(prepared)),
	)
	LOGGER.info(
		"Trained Prophet model on %s rows with residual_std=%.6f",
		summary.train_rows,
		summary.residual_std,
	)
	return model, summary, train_output


def export_training_artifacts(
	model: Any,
	summary: ProphetTrainingSummary,
	train_output_df: pd.DataFrame,
	model_path: str | Path,
	summary_path: str | Path,
	train_output_path: str | Path,
) -> None:
	"""Persist trained model and calibration outputs for later evaluation."""
	_, model_to_json, _ = _import_prophet_dependencies()
	model_path = Path(model_path)
	summary_path = Path(summary_path)
	train_output_path = Path(train_output_path)

	model_path.write_text(model_to_json(model), encoding="utf-8")
	summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
	train_output_df.to_csv(train_output_path, index=False)


def load_training_artifacts(
	model_path: str | Path,
	summary_path: str | Path,
) -> tuple[Any, ProphetTrainingSummary]:
	"""Load serialized model and summary exported from the training phase."""
	_, _, model_from_json = _import_prophet_dependencies()
	model_path = Path(model_path)
	summary_path = Path(summary_path)

	if not model_path.exists():
		raise FileNotFoundError(f"Model artifact not found: {model_path}")
	if not summary_path.exists():
		raise FileNotFoundError(f"Training summary artifact not found: {summary_path}")

	model = model_from_json(model_path.read_text(encoding="utf-8"))
	payload = json.loads(summary_path.read_text(encoding="utf-8"))
	summary = ProphetTrainingSummary(
		time_field=str(payload["time_field"]),
		value_field=str(payload["value_field"]),
		threshold_z=float(payload["threshold_z"]),
		residual_mean=float(payload["residual_mean"]),
		residual_std=max(float(payload["residual_std"]), 1e-6),
		train_rows=int(payload["train_rows"]),
	)
	return model, summary


def evaluate_from_artifacts(
	eval_df: pd.DataFrame,
	model_path: str | Path,
	summary_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Score an evaluation dataframe by loading trained Prophet artifacts from disk."""
	model, summary = load_training_artifacts(
		model_path=model_path,
		summary_path=summary_path,
	)
	prepared = _prepare_series(
		eval_df,
		time_field=summary.time_field,
		value_field=summary.value_field,
	)
	forecast = model.predict(prepared[["ds"]])

	values = prepared["y"].to_numpy(dtype=float)
	yhat = forecast["yhat"].to_numpy(dtype=float)
	residual = values - yhat
	z_score = np.abs((residual - summary.residual_mean) / summary.residual_std)
	is_anomaly = z_score > summary.threshold_z

	result_df = pd.DataFrame(
		{
			summary.time_field: prepared["ds"],
			summary.value_field: values,
			f"{summary.value_field}_forecast": yhat,
			f"{summary.value_field}_residual": residual,
			f"{summary.value_field}_z_score": z_score,
			f"{summary.value_field}_is_anomaly": is_anomaly,
		}
	)

	events_df = pd.DataFrame(
		{
			"timestamp": result_df.loc[is_anomaly, summary.time_field].to_numpy(),
			"index": np.flatnonzero(is_anomaly),
			"value": result_df.loc[is_anomaly, summary.value_field].to_numpy(
				dtype=float
			),
			"forecast": result_df.loc[
				is_anomaly, f"{summary.value_field}_forecast"
			].to_numpy(dtype=float),
			"residual": result_df.loc[
				is_anomaly, f"{summary.value_field}_residual"
			].to_numpy(dtype=float),
			"z_score": result_df.loc[
				is_anomaly, f"{summary.value_field}_z_score"
			].to_numpy(dtype=float),
		}
	)
	return result_df, events_df
