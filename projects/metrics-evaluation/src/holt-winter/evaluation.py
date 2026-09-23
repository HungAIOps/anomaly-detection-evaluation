from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_holtwinters_module():
    module_path = Path(__file__).with_name("holt-winter.py")
    module_name = "holtwinter_algo"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


holtwinters = _load_holtwinters_module()
HoltWintersDetector = holtwinters.HoltWintersDetector
detect_anomalies = holtwinters.detect_anomalies


def generate_train_data() -> pd.DataFrame:
    """Create clean seasonal data used to fit the model offline."""
    rng = np.random.default_rng(42)
    periods = 240
    timestamp = pd.date_range("2026-07-12 00:00:00", periods=periods, freq="5min")

    base = 100.0 + 0.15 * np.arange(periods)
    seasonal = 18.0 * np.sin(2 * np.pi * np.arange(periods) / 24.0)
    noise = rng.normal(0.0, 3.0, size=periods)
    series = base + seasonal + noise

    return pd.DataFrame({"timestamp": timestamp, "metric": series})


def generate_test_data() -> pd.DataFrame:
    """Create seasonal data with several injected anomalies for real-time detection."""
    rng = np.random.default_rng(7)
    periods = 240
    timestamp = pd.date_range("2026-07-12 00:00:00", periods=periods, freq="5min")

    base = 100.0 + 0.15 * np.arange(periods)
    seasonal = 18.0 * np.sin(2 * np.pi * np.arange(periods) / 24.0)
    noise = rng.normal(0.0, 3.0, size=periods)
    series = base + seasonal + noise

    anomaly_positions = [30, 55, 118, 170, 205]
    for pos in anomaly_positions:
        series[pos] += 30.0 + 8.0 * rng.normal()

    series[42] -= 22.0
    series[150] += 18.0

    return pd.DataFrame({"timestamp": timestamp, "metric": series})


def main() -> None:
    train_df = generate_train_data()
    model = HoltWintersDetector(
        season_period=24,
        alpha=0.25,
        beta=0.1,
        gamma=0.2,
        threshold_z=3.5,
        residual_window=48,
    ).fit(train_df["metric"].to_numpy(dtype=float))

    model_path = Path(__file__).with_name("holt_winter_model.json")
    model.save_to_file(model_path)
    loaded_model = HoltWintersDetector.load_from_file(model_path)

    test_df = generate_test_data()
    result_df, events_df = detect_anomalies(
        test_df,
        time_field="timestamp",
        value_field="metric",
        model=loaded_model,
    )

    print("=== Trained model summary ===")
    print(f"level={loaded_model.level:.4f}")
    print(f"trend={loaded_model.trend:.4f}")
    print(f"seasonal={loaded_model.seasonal[:6]}")
    print(f"residual_std={loaded_model.residual_std:.4f}")

    print("\n=== Detected anomalies ===")
    if events_df.empty:
        print("No anomalies detected.")
    else:
        print(events_df.to_string(index=False))

    print("\n=== Sample annotated rows ===")
    cols = [
        "timestamp",
        "metric",
        "metric_forecast",
        "metric_residual",
        "metric_z_score",
        "metric_is_anomaly",
    ]
    print(result_df.loc[0:15, cols].to_string(index=False))

    result_df.to_csv("holt_winter_example_output.csv", index=False)
    events_df.to_csv("holt_winter_example_events.csv", index=False)
    print("\nExported results to holt_winter_example_output.csv and holt_winter_example_events.csv")
    print(f"Saved trained params to {model_path}")


if __name__ == "__main__":
    main()
