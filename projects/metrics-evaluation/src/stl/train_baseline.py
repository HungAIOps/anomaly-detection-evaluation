import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
import pickle
from .utils import robust_std

def train_baseline(
    hist_df: pd.DataFrame,
    metric_cols: list[str],
    time_col: str = "time",
    period: int = 24,          # e.g. 24 for hourly data w/ daily seasonality
    k: float = 3.0,            # sigma multiplier for anomaly threshold
    robust: bool = True,       # use STL's robust mode (down-weights outliers during fitting)
    output_baseline_path: str | None = None,
) -> dict:
    """
    Fit STL on historical data per metric and compute the anomaly threshold.
    """
    hist_df = hist_df.sort_values(time_col).reset_index(drop=True)
    baseline = {"_config": {"hist_df": hist_df, "time_col": time_col, "period": period, "robust": robust}}

    for metric in metric_cols:
        series = hist_df[metric].astype(float).interpolate(limit_direction="both")
        stl_result = STL(series, period=period, robust=robust).fit()
        resid = stl_result.resid.to_numpy()
        threshold = k * robust_std(resid)
        baseline[metric] = {"threshold": threshold}
    
    if output_baseline_path is not None:
        with open(output_baseline_path, "w") as f:
            json.dump(baseline, f, default=str)

    return baseline