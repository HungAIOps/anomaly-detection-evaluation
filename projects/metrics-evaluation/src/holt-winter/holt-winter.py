from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class HoltWintersDetector:
    """Offline-trained Holt-Winters model for real-time anomaly scoring.

    The model is trained once on historical data to estimate a stable level,
    trend, and seasonality. After training, the detector freezes these
    parameters and only scores fresh points against the learned baseline.
    """

    season_period: int
    alpha: float = 0.2
    beta: float = 0.1
    gamma: float = 0.15
    threshold_z: float = 3.5
    residual_window: int = 48

    level: float = 0.0
    trend: float = 0.0
    seasonal: List[float] = field(default_factory=list)
    residual_mean: float = 0.0
    residual_std: float = 1.0
    last_forecast: Optional[float] = None
    last_residual: Optional[float] = None
    last_z_score: Optional[float] = None

    def __post_init__(self) -> None:
        if self.season_period <= 1:
            raise ValueError("season_period must be greater than 1")
        if not self.seasonal:
            self.seasonal = [0.0 for _ in range(self.season_period)]

    def fit(self, values: Sequence[float]) -> "HoltWintersDetector":
        """Estimate level, trend, and seasonal pattern from historical data."""
        arr = np.asarray(values, dtype=float)
        if arr.size < self.season_period:
            raise ValueError(
                "Training data must contain at least one full seasonal cycle."
            )

        level = float(np.mean(arr[: self.season_period]))
        trend = 0.0
        if arr.size > self.season_period:
            first_window = arr[: min(arr.size, self.season_period + 1)]
            if first_window.size > 1:
                trend = float(np.mean(np.diff(first_window)))

        seasonal = np.zeros(self.season_period, dtype=float)
        for phase in range(self.season_period):
            phase_values = arr[phase:: self.season_period]
            seasonal[phase] = (
                float(np.mean(phase_values)) - level if phase_values.size else 0.0
            )

        forecast_values = np.empty(arr.size, dtype=float)
        residuals = np.empty(arr.size, dtype=float)
        for idx, value in enumerate(arr):
            forecast_values[idx] = self._predict_from_params(idx, level, trend, seasonal)
            residuals[idx] = float(value - forecast_values[idx])

        residual_mean = float(np.mean(residuals))
        residual_std = float(np.std(residuals))
        if residual_std <= 1e-6:
            residual_std = 1e-6

        self.level = level
        self.trend = trend
        self.seasonal = seasonal.tolist()
        self.residual_mean = residual_mean
        self.residual_std = residual_std
        self.last_forecast = None
        self.last_residual = None
        self.last_z_score = None
        return self

    def _predict_from_params(
        self,
        offset: int,
        level: float,
        trend: float,
        seasonal: Sequence[float],
    ) -> float:
        phase = offset % self.season_period
        return level + (trend * offset) + seasonal[phase]

    def forecast(self, offset: int) -> float:
        """Forecast the value for a point offset from the start of the series."""
        phase = offset % self.season_period
        return self.level + (self.trend * offset) + self.seasonal[phase]

    def score_values(self, values: Sequence[float]) -> Dict[str, np.ndarray]:
        """Score a new series without updating the learned parameters."""
        arr = np.asarray(values, dtype=float)
        forecasts = np.empty(arr.size, dtype=float)
        residuals = np.empty(arr.size, dtype=float)
        z_scores = np.empty(arr.size, dtype=float)
        is_anomaly = np.zeros(arr.size, dtype=bool)

        for idx, value in enumerate(arr):
            forecast = self.forecast(idx)
            residual = float(value - forecast)
            z_score = abs(residual - self.residual_mean) / self.residual_std
            forecasts[idx] = forecast
            residuals[idx] = residual
            z_scores[idx] = z_score
            is_anomaly[idx] = bool(z_score > self.threshold_z)

        self.last_forecast = float(forecasts[-1]) if forecasts.size else None
        self.last_residual = float(residuals[-1]) if residuals.size else None
        self.last_z_score = float(z_scores[-1]) if z_scores.size else None

        return {
            "forecast": forecasts,
            "residual": residuals,
            "z_score": z_scores,
            "is_anomaly": is_anomaly,
        }

    def save_to_file(self, path: str | Path) -> None:
        """Persist the trained model so it can be loaded later for detection."""
        payload = {
            "season_period": self.season_period,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "threshold_z": self.threshold_z,
            "residual_window": self.residual_window,
            "level": self.level,
            "trend": self.trend,
            "seasonal": self.seasonal,
            "residual_mean": self.residual_mean,
            "residual_std": self.residual_std,
        }
        path = Path(path)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_from_file(cls, path: str | Path) -> "HoltWintersDetector":
        """Reload a trained model from a JSON file."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            season_period=int(payload["season_period"]),
            alpha=float(payload.get("alpha", 0.2)),
            beta=float(payload.get("beta", 0.1)),
            gamma=float(payload.get("gamma", 0.15)),
            threshold_z=float(payload.get("threshold_z", 3.5)),
            residual_window=int(payload.get("residual_window", 48)),
        )
        model.level = float(payload["level"])
        model.trend = float(payload["trend"])
        model.seasonal = [float(v) for v in payload["seasonal"]]
        model.residual_mean = float(payload.get("residual_mean", 0.0))
        model.residual_std = float(payload.get("residual_std", 1.0))
        return model


def detect_anomalies(
    df: pd.DataFrame,
    time_field: str,
    value_field: str,
    model: Optional[HoltWintersDetector] = None,
    season_period: int = 24,
    alpha: float = 0.2,
    beta: float = 0.1,
    gamma: float = 0.15,
    threshold_z: float = 3.5,
    residual_window: int = 48,
    model_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a new series with a frozen Holt-Winters model.

    Detection is strictly real-time scoring: the model parameters are not updated
    as each point is processed. A fitted model is loaded from disk or passed in.
    """
    if model is None:
        if model_path is None:
            model = HoltWintersDetector(
                season_period=season_period,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                threshold_z=threshold_z,
                residual_window=residual_window,
            )
        else:
            model = HoltWintersDetector.load_from_file(model_path)

    if time_field not in df.columns:
        raise ValueError(f"time_field '{time_field}' not found in dataframe columns")
    if value_field not in df.columns:
        raise ValueError(f"value_field '{value_field}' not found in dataframe columns")

    result_df = df.sort_values(by=time_field, kind="mergesort").reset_index(drop=True).copy()
    values = result_df[value_field].to_numpy(dtype=float)
    scored = model.score_values(values)

    result_df[f"{value_field}_forecast"] = scored["forecast"]
    result_df[f"{value_field}_residual"] = scored["residual"]
    result_df[f"{value_field}_z_score"] = scored["z_score"]
    result_df[f"{value_field}_is_anomaly"] = scored["is_anomaly"]

    events: List[dict] = []
    for idx, is_anomaly in enumerate(scored["is_anomaly"]):
        if not is_anomaly:
            continue
        events.append(
            {
                "timestamp": result_df.at[idx, time_field],
                "index": idx,
                "value": float(result_df.at[idx, value_field]),
                "forecast": float(scored["forecast"][idx]),
                "residual": float(scored["residual"][idx]),
                "z_score": float(scored["z_score"][idx]),
            }
        )

    events_df = pd.DataFrame(events)
    if not events_df.empty:
        events_df = events_df[
            ["timestamp", "index", "value", "forecast", "residual", "z_score"]
        ]

    return result_df, events_df


__all__ = ["HoltWintersDetector", "detect_anomalies"]
