"""Minimal STL compatibility layer for the metrics-evaluation project."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class STLResult:
    """Decomposition result matching the subset of the statsmodels API we use."""

    trend: np.ndarray
    seasonal: np.ndarray
    resid: np.ndarray


class STL:
    """Simplified STL implementation with the same public constructor shape."""

    def __init__(self, endog: np.ndarray | pd.Series, period: int, robust: bool = False) -> None:
        if period < 2:
            raise ValueError("period must be at least 2")
        self._endog = np.asarray(endog, dtype=float)
        self.period = period
        self.robust = robust

    def fit(self) -> STLResult:
        values = self._endog.astype(float, copy=True)
        series = pd.Series(values)
        window = min(len(series), max(3, self.period if self.period % 2 == 1 else self.period + 1))
        trend = series.rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)

        for _ in range(2):
            detrended = values - trend
            seasonal_profile = np.zeros(self.period, dtype=float)
            for position in range(self.period):
                samples = detrended[np.arange(len(detrended)) % self.period == position]
                if samples.size == 0:
                    seasonal_profile[position] = 0.0
                elif self.robust:
                    seasonal_profile[position] = float(np.median(samples))
                else:
                    seasonal_profile[position] = float(np.mean(samples))
            seasonal = seasonal_profile[np.arange(len(values)) % self.period]
            trend = pd.Series(values - seasonal).rolling(
                window=window,
                center=True,
                min_periods=1,
            ).mean().to_numpy(dtype=float)

        resid = values - trend - seasonal
        return STLResult(trend=trend, seasonal=seasonal, resid=resid)