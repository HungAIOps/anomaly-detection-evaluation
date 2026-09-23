"""
STL-based anomaly detection algorithm for multi-metric time-series data.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def _robust_std(residuals: np.ndarray) -> float:
    """MAD-based robust std, more resistant to outliers skewing the noise band itself."""
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    return 1.4826 * mad if mad > 0 else np.std(residuals) or 1e-9


def detect_anomalies(
    hist_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    metric_cols: list[str],
    time_col: str = "time",
    period: int = 24,          # e.g. 24 for hourly data w/ daily seasonality
    k: float = 3.0,            # sigma multiplier for anomaly threshold
    robust: bool = True,       # use STL's robust mode (down-weights outliers during fitting)
) -> pd.DataFrame:
    """
    Returns eval_df with added columns per metric:
      {metric}_trend, {metric}_seasonal, {metric}_residual, {metric}_is_abnormal
    """
    hist_df = hist_df.sort_values(time_col).reset_index(drop=True)
    eval_df = eval_df.sort_values(time_col).reset_index(drop=True)
    out = eval_df.copy()

    n_hist = len(hist_df)
    n_eval = len(eval_df)

    for metric in metric_cols:
        # Combine history + eval so STL has continuity/context for the eval window.
        combined = pd.concat(
            [hist_df[[time_col, metric]], eval_df[[time_col, metric]]],
            ignore_index=True,
        )
        series = combined[metric].astype(float).interpolate(limit_direction="both")

        stl_result = STL(series, period=period, robust=robust).fit()

        trend = stl_result.trend.to_numpy()
        seasonal = stl_result.seasonal.to_numpy()
        resid = stl_result.resid.to_numpy()

        # Noise band learned from historical residuals only (avoid eval anomalies inflating threshold).
        hist_resid = resid[:n_hist]
        threshold = k * _robust_std(hist_resid)

        eval_trend = trend[n_hist:]
        eval_seasonal = seasonal[n_hist:]
        eval_resid = resid[n_hist:]
        is_abnormal = np.abs(eval_resid) > threshold

        out[f"{metric}_trend"] = eval_trend
        out[f"{metric}_seasonal"] = eval_seasonal
        out[f"{metric}_residual"] = eval_resid
        out[f"{metric}_threshold"] = threshold
        out[f"{metric}_is_abnormal"] = is_abnormal

    return out


if __name__ == "__main__":
    # ---- Build a synthetic example: hourly latency & cpu_usage over 10 days history + 1 day eval ----
    rng = np.random.default_rng(42)

    hist_hours = 24 * 10
    eval_hours = 24

    hist_time = pd.date_range("2026-06-01", periods=hist_hours, freq="h")
    eval_time = pd.date_range(hist_time[-1] + pd.Timedelta(hours=1), periods=eval_hours, freq="h")

    def daily_pattern(n, base, amp, noise_scale):
        hours = np.arange(n) % 24
        seasonal = amp * np.sin(2 * np.pi * hours / 24)
        return base + seasonal + rng.normal(0, noise_scale, n)

    hist_df = pd.DataFrame({
        "time": hist_time,
        "latency_p99": daily_pattern(hist_hours, base=150, amp=20, noise_scale=5),
        "cpu_usage": daily_pattern(hist_hours, base=40, amp=15, noise_scale=3),
    })

    eval_df = pd.DataFrame({
        "time": eval_time,
        "latency_p99": daily_pattern(eval_hours, base=150, amp=20, noise_scale=5),
        "cpu_usage": daily_pattern(eval_hours, base=40, amp=15, noise_scale=3),
    })
    # Inject a real anomaly: latency spike at hour 5 of eval window
    eval_df.loc[5, "latency_p99"] += 120

    result = detect_anomalies(
        hist_df, eval_df,
        metric_cols=["latency_p99", "cpu_usage"],
        time_col="time",
        period=24,
        k=3.0,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(result[["time", "latency_p99", "latency_p99_residual", "latency_p99_is_abnormal",
                   "cpu_usage", "cpu_usage_residual", "cpu_usage_is_abnormal"]])