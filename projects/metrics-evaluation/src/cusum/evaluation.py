from typing import List, Tuple

import numpy as np
import pandas as pd
from cusumstate import CusumState

def detect_spikes(
    df: pd.DataFrame,
    time_field: str,
    metric_fields: List[str],
    alpha: float = 0.2,
    beta: float = 0.2,
    k: float = 0.5,
    h: float = 5.0,
    warmup_periods: int = 10,
    max_score_cap: float | None = 6.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run streaming CUSUM spike detection over one or more metric columns.

    The tuning parameters mirror the CusumState defaults so the detector can be
    configured consistently from the sample script or higher-level callers.
    """
    if time_field not in df.columns:
        raise ValueError(f"time_field '{time_field}' not found in dataframe columns")
    missing = [m for m in metric_fields if m not in df.columns]
    if missing:
        raise ValueError(f"metric_fields not found in dataframe: {missing}")

    result_df = df.sort_values(by=time_field, kind="mergesort").reset_index(drop=True).copy()
    timestamps = result_df[time_field]

    events: List[dict] = []

    for metric in metric_fields:
        state = CusumState(
            alpha=alpha,
            beta=beta,
            k=k,
            h=h,
            warmup_periods=warmup_periods,
            max_score_cap=max_score_cap,
        )

        baseline_col = np.full(len(result_df), np.nan)
        score_col = np.full(len(result_df), np.nan)
        is_spike_col = np.zeros(len(result_df), dtype=bool)
        direction_col = np.array([None] * len(result_df), dtype=object)

        values = result_df[metric].to_numpy(dtype=float)

        for idx in range(len(result_df)):
            ts = timestamps.iloc[idx]
            out = state.update(values[idx], idx, ts)

            baseline_col[idx] = out["baseline"] if out["baseline"] is not None else np.nan
            score_col[idx] = out["score"]
            is_spike_col[idx] = out["is_spike"]
            direction_col[idx] = out["direction"]

            if out["event"] == "end":
                if state.spike_start_time is None or state.spike_start_idx is None:
                    state.spike_direction = None
                    state.spike_start_idx = None
                    state.spike_start_time = None
                    state.spike_peak_value = None
                    state.spike_peak_score = 0.0
                    state.in_spike = False
                    state.s_pos = 0.0
                    state.s_neg = 0.0
                    continue
                events.append(
                    {
                        "metric": metric,
                        "direction": state.spike_direction,
                        "start_time": state.spike_start_time,
                        "end_time": ts,
                        "start_index": state.spike_start_idx,
                        "end_index": idx,
                        "peak_value": state.spike_peak_value,
                        "peak_score": state.spike_peak_score,
                    }
                )
                # reset episode tracking after logging
                state.spike_direction = None
                state.spike_start_idx = None
                state.spike_start_time = None
                state.spike_peak_value = None
                state.spike_peak_score = 0.0

        # If the series ends while still inside a spike, close it out
        # using the last available timestamp/index.
        if state.in_spike and state.spike_start_idx is not None:
            last_idx = len(result_df) - 1
            events.append(
                {
                    "metric": metric,
                    "direction": state.spike_direction,
                    "start_time": state.spike_start_time,
                    "end_time": timestamps.iloc[last_idx],
                    "start_index": state.spike_start_idx,
                    "end_index": last_idx,
                    "peak_value": state.spike_peak_value,
                    "peak_score": state.spike_peak_score,
                }
            )

        result_df[f"{metric}_baseline"] = baseline_col
        result_df[f"{metric}_score"] = score_col
        result_df[f"{metric}_is_spike"] = is_spike_col
        result_df[f"{metric}_direction"] = direction_col

    events_df = pd.DataFrame(events)
    if not events_df.empty:
        events_df["duration_seconds"] = (
            pd.to_datetime(events_df["end_time"]) - pd.to_datetime(events_df["start_time"])
        ).dt.total_seconds()
        events_df = events_df.sort_values(by="start_time").reset_index(drop=True)
        events_df = events_df[
            [
                "metric",
                "direction",
                "start_time",
                "end_time",
                "duration_seconds",
                "peak_value",
                "peak_score",
                "start_index",
                "end_index",
            ]
        ]

    return result_df, events_df

if __name__ == "__main__":
    # Build a synthetic example: normal noise with two injected spikes.
    rng = np.random.default_rng(42)
    n = 300
    timestamps = pd.date_range("2026-07-12 00:00:00", periods=n, freq="10s")

    latency = rng.normal(loc=50, scale=5, size=n)   # baseline ~50ms
    latency[120:135] += rng.normal(loc=120, scale=10, size=15)  # spike episode
    latency[220:225] += rng.normal(loc=200, scale=15, size=5)   # short sharp spike

    tps = rng.normal(loc=500, scale=20, size=n)      # baseline ~500 TPS
    tps[60:70] += rng.normal(loc=800, scale=30, size=10)        # spike episode

    example_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "latency_ms": latency,
            "tps": tps,
        }
    )

    result_df, events_df = detect_spikes(
        example_df,
        time_field="timestamp",
        metric_fields=["latency_ms", "tps"],
        alpha=0.2,
        beta=0.2,
        k=0.5,
        h=10.0,
        warmup_periods=10,
        max_score_cap=15.0
    )

    print("=== Detected spike episodes ===")
    if events_df.empty:
        print("No spikes detected.")
    else:
        print(events_df.to_string(index=False))

    print("\n=== Sample of annotated result_df ===")
    cols_to_show = [
        "timestamp",
        "latency_ms",
        "latency_ms_score",
        "latency_ms_is_spike",
    ]
    print(result_df.loc[115:140, cols_to_show].to_string(index=False))

    # Set single-point flag equal to per-point is_spike and do not export group column
    for metric in ["latency_ms", "tps"]:
        is_spike = result_df[f"{metric}_is_spike"].to_numpy(dtype=bool)
        result_df[f"{metric}_is_single_point_spike"] = is_spike

    # Export annotated results and events
    out_results = "cusum_example_output.csv"
    out_events = "cusum_example_events.csv"
    result_df.to_csv(out_results, index=False)
    events_df.to_csv(out_events, index=False)
    print(f"Exported annotated results to {out_results} and events to {out_events}")