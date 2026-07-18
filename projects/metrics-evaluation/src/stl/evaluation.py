import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from train_baseline import train_baseline


def evaluate(
    eval_df: pd.DataFrame,
    baseline: dict,
) -> pd.DataFrame:
    """
    Score eval_df against a baseline produced by train_baseline().

    Re-fits STL on (hist_df + eval_df) for continuity/context, same as the original
    combined function, but reuses the threshold computed once at train time.

    Returns eval_df with added columns per metric:
      {metric}_trend, {metric}_seasonal, {metric}_residual, {metric}_is_abnormal
    """
    cfg = baseline["_config"]
    hist_df, time_col, period, robust = cfg["hist_df"], cfg["time_col"], cfg["period"], cfg["robust"]

    eval_df = eval_df.sort_values(time_col).reset_index(drop=True)
    out = eval_df.copy()

    n_hist = len(hist_df)
    metric_cols = [m for m in baseline if m != "_config"]

    for metric in metric_cols:
        combined = pd.concat(
            [hist_df[[time_col, metric]], eval_df[[time_col, metric]]],
            ignore_index=True,
        )
        series = combined[metric].astype(float).interpolate(limit_direction="both")

        stl_result = STL(series, period=period, robust=robust).fit()

        trend = stl_result.trend.to_numpy()
        seasonal = stl_result.seasonal.to_numpy()
        resid = stl_result.resid.to_numpy()

        threshold = baseline[metric]["threshold"]  # reuse threshold from train_baseline, not recomputed

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
    eval_df.loc[5, "latency_p99"] += 120  # inject anomaly

    baseline = train_baseline(hist_df, metric_cols=["latency_p99", "cpu_usage"], time_col="time", period=24)
    result = evaluate(eval_df, baseline)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(result[["time", "latency_p99", "latency_p99_residual", "latency_p99_is_abnormal",
                   "cpu_usage", "cpu_usage_residual", "cpu_usage_is_abnormal"]])