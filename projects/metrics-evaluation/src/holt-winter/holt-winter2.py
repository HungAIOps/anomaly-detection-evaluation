from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .constants import ADAPTIVE_MODE, FREEZE_ON_ANOMALY_MODE, FROZEN_MODE


@dataclass
class HoltWintersDetector:
    """Additive Holt-Winters model for seasonal time series.

    Supports two usage patterns:

    1. Continuous online learning: call `update(value)` (default `adapt=True`).
       Every point, anomalous or not, updates level/trend/seasonal state.

    2. Train-once, freeze, detect-only: call `fit_initial_state(...)` on a
       historical batch, then call `score(value)` for real-time scoring.
       State never changes until you explicitly call `fit_initial_state`
       again (e.g. on a nightly retrain schedule). This avoids anomalies
       corrupting the learned seasonal pattern.

    Residuals are standardized using a robust median/MAD z-score rather
    than mean/std, since mean/std are themselves sensitive to the outliers
    you're trying to detect.
    """

    season_period: int
    alpha: float = 0.2
    beta: float = 0.1
    gamma: float = 0.15
    threshold_z: float = 3.5
    residual_window: int = 48
    min_history_cycles: int = 2  # cycles required before fit_initial_state accepts data

    level: float = 0.0
    trend: float = 0.0
    seasonal: List[float] = field(default_factory=list)
    history: List[float] = field(default_factory=list)
    residuals: List[float] = field(default_factory=list)
    n_seen: int = 0
    last_forecast: Optional[float] = None
    last_residual: Optional[float] = None
    last_z_score: Optional[float] = None
    is_initialized: bool = False
    previous_level: Optional[float] = None

    def __post_init__(self) -> None:
        if self.season_period <= 1:
            raise ValueError("season_period must be greater than 1")
        self.seasonal = [0.0 for _ in range(self.season_period)]

    def fit_initial_state(self, values: List[float] | np.ndarray) -> None:
        """(Re)initialize the detector from a warmup/training window.

        Call this once for train-then-freeze usage, or periodically
        (e.g. nightly) to retrain on a rolling window in that mode.
        """
        min_len = self.season_period * self.min_history_cycles
        if len(values) < min_len:
            raise ValueError(
                f"At least {self.min_history_cycles} full seasonal cycle(s) "
                f"({min_len} points) are required to initialize the model, "
                f"got {len(values)}"
            )

        arr = np.asarray(values, dtype=float)
        if np.isnan(arr).any():
            raise ValueError("fit_initial_state received NaN values; clean or impute first")

        level = float(np.mean(arr[: self.season_period]))
        trend = 0.0
        if len(arr) >= self.season_period + 1:
            trend = float(np.mean(np.diff(arr[: self.season_period + 1])))

        seasonal = np.zeros(self.season_period, dtype=float)
        for phase in range(self.season_period):
            seasonal[phase] = float(np.mean(arr[phase:: self.season_period] - level))

        self.level = level
        self.trend = trend
        self.seasonal = seasonal.tolist()
        self.history = arr.tolist()
        self.n_seen = len(self.history)
        self.is_initialized = True
        self.previous_level = self.level
        # Reset residual window on (re)fit so stale pre-retrain residuals
        # don't bleed into the new model's z-scores.
        self.residuals = []

    def _robust_z(self, residual: float) -> float:
        """Median/MAD-based z-score — robust to outliers already in the window."""
        if len(self.residuals) >= self.residual_window:
            self.residuals.pop(0)
        self.residuals.append(residual)

        if not self.residuals:
            return 0.0

        median = float(np.median(self.residuals))
        mad = float(np.median(np.abs(np.asarray(self.residuals) - median)))
        mad = mad if mad > 1e-6 else 1e-6
        # 0.6745 scales MAD to be comparable to standard deviation under normality
        return 0.6745 * abs(residual - median) / mad

    def score(self, value: float) -> Dict[str, Any]:
        """Score one point against the CURRENT (FROZEN_MODE) model. No state mutation.

        Use this for train-once/freeze-then-detect real-time inference.
        Advances the phase counter so seasonal alignment stays correct,
        but does not touch level/trend/seasonal values.
        """
        if not self.is_initialized:
            raise RuntimeError("Call fit_initial_state(...) before score()")

        value = float(value)
        phase = self.n_seen % self.season_period
        forecast = self.level + self.trend + self.seasonal[phase]
        residual = value - forecast
        z_score = self._robust_z(residual)
        is_anomaly = bool(z_score > self.threshold_z)

        self.last_forecast = forecast
        self.last_residual = residual
        self.last_z_score = z_score
        self.n_seen += 1  # advance phase only; level/trend/seasonal untouched

        return {
            "value": value,
            "forecast": forecast,
            "residual": residual,
            "z_score": z_score,
            "is_anomaly": is_anomaly,
            "level": self.level,
            "trend": self.trend,
            "seasonal": list(self.seasonal),
            "adapted": False,
        }

    def update(self, value: float, adapt: bool | str = True) -> Dict[str, Any]:
        """Process one new point and return a result dictionary.

        adapt=True   : classic online learning — state updates every point.
        adapt=False  : score only, never mutate state (equivalent to score()).
        adapt="skip_anomalies": score first; only feed the update equations
                     if the point is NOT flagged anomalous. This is the
                     "don't let outliers corrupt the model" mode.
        """
        value = float(value)
        if not self.is_initialized:
            self.history.append(value)
            if len(self.history) >= self.season_period * self.min_history_cycles:
                self.fit_initial_state(self.history)
            return {
                "value": value,
                "forecast": None,
                "residual": None,
                "z_score": None,
                "is_anomaly": False,
                "level": self.level,
                "trend": self.trend,
                "seasonal": list(self.seasonal),
                "adapted": False,
            }

        phase = self.n_seen % self.season_period
        forecast = self.level + self.trend + self.seasonal[phase]
        residual = value - forecast
        z_score = self._robust_z(residual)
        is_anomaly = bool(z_score > self.threshold_z)

        self.last_forecast = forecast
        self.last_residual = residual
        self.last_z_score = z_score

        do_adapt = (adapt is True) or (adapt == "skip_anomalies" and not is_anomaly)

        if do_adapt:
            previous_level = self.level
            self.level = self.alpha * (value - self.seasonal[phase]) + (1.0 - self.alpha) * (self.level + self.trend)
            self.trend = self.beta * (self.level - previous_level) + (1.0 - self.beta) * self.trend
            self.seasonal[phase] = self.gamma * (value - self.level) + (1.0 - self.gamma) * self.seasonal[phase]
            self.previous_level = previous_level
            self.history.append(value)

        self.n_seen += 1

        return {
            "value": value,
            "forecast": forecast,
            "residual": residual,
            "z_score": z_score,
            "is_anomaly": is_anomaly,
            "level": self.level,
            "trend": self.trend,
            "seasonal": list(self.seasonal),
            "adapted": do_adapt,
        }


def detect_anomalies(
    df: pd.DataFrame,
    time_field: str,
    value_field: str,
    season_period: int = 24,
    alpha: float = 0.2,
    beta: float = 0.1,
    gamma: float = 0.15,
    threshold_z: float = 3.5,
    residual_window: int = 48,
    mode: str = "ADAPTIVE_MODE",  # "ADAPTIVE_MODE" = learn from every point (incl. anomalies)
                              # "FREEZE_ON_ANOMALY_MODE" = skip state update when flagged
                              # "FROZEN_MODE" = fit once on warmup, never update again
    warmup_cycles: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a Holt-Winters detector over a single metric column.

    mode="ADAPTIVE_MODE": matches the original streaming behavior (learns from everything).
    mode="FREEZE_ON_ANOMALY_MODE": learns online, but skips the update step for points
        flagged anomalous, so outliers don't distort future level/trend/seasonal.
    mode="FROZEN_MODE": fits once on the first `warmup_cycles` seasonal cycles, then
        scores every remaining point against that fixed model (train-once, detect-only).
    """
    if time_field not in df.columns:
        raise ValueError(f"time_field '{time_field}' not found in dataframe columns")
    if value_field not in df.columns:
        raise ValueError(f"value_field '{value_field}' not found in dataframe columns")
    if mode not in {"ADAPTIVE_MODE", "FREEZE_ON_ANOMALY_MODE", "FROZEN_MODE"}:
        raise ValueError("mode must be 'ADAPTIVE_MODE', 'FREEZE_ON_ANOMALY_MODE', or 'FROZEN_MODE'")

    result_df = df.sort_values(by=time_field, kind="mergesort").reset_index(drop=True).copy()
    values = result_df[value_field].to_numpy(dtype=float)

    detector = HoltWintersDetector(
        season_period=season_period,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        threshold_z=threshold_z,
        residual_window=residual_window,
        min_history_cycles=warmup_cycles,
    )

    start_idx = 0
    if mode == "FROZEN_MODE":
        warmup_n = season_period * warmup_cycles
        if len(values) < warmup_n:
            raise ValueError("Not enough data to cover the requested warmup_cycles")
        detector.fit_initial_state(values[:warmup_n])
        start_idx = warmup_n

    for col, dtype in [
        (f"{value_field}_forecast", float),
        (f"{value_field}_residual", float),
        (f"{value_field}_z_score", float),
        (f"{value_field}_level", float),
        (f"{value_field}_trend", float),
    ]:
        result_df[col] = np.full(len(result_df), np.nan, dtype=dtype)
    result_df[f"{value_field}_is_anomaly"] = np.zeros(len(result_df), dtype=bool)
    result_df[f"{value_field}_adapted"] = np.zeros(len(result_df), dtype=bool)

    events: List[dict] = []
    for idx in range(start_idx, len(values)):
        value = values[idx]

        if mode == "FROZEN_MODE":
            out = detector.score(value)
        elif mode == "FREEZE_ON_ANOMALY_MODE":
            out = detector.update(value, adapt="skip_anomalies")
        else:
            out = detector.update(value, adapt=True)

        result_df.at[idx, f"{value_field}_forecast"] = out["forecast"]
        result_df.at[idx, f"{value_field}_residual"] = out["residual"]
        result_df.at[idx, f"{value_field}_z_score"] = out["z_score"]
        result_df.at[idx, f"{value_field}_is_anomaly"] = out["is_anomaly"]
        result_df.at[idx, f"{value_field}_level"] = out["level"]
        result_df.at[idx, f"{value_field}_trend"] = out["trend"]
        result_df.at[idx, f"{value_field}_adapted"] = out.get("adapted", False)

        if out["is_anomaly"]:
            events.append(
                {
                    "timestamp": result_df.at[idx, time_field],
                    "value": value,
                    "forecast": out["forecast"],
                    "residual": out["residual"],
                    "z_score": out["z_score"],
                    "index": idx,
                }
            )

    events_df = pd.DataFrame(events)
    if not events_df.empty:
        events_df = events_df[["timestamp", "index", "value", "forecast", "residual", "z_score"]]

    return result_df, events_df


__all__ = ["HoltWintersDetector", "detect_anomalies"]