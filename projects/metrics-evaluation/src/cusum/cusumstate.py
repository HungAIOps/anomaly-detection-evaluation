from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from constants import SPIKE_DOWN, SPIKE_UP


@dataclass
class CusumState:
    """Holds the running state for one metric's streaming CUSUM detector.

    Standard two-sided CUSUM on an EWMA-estimated baseline:
        z_t     = (x_t - mean) / std
        s_pos_t = max(0, s_pos_{t-1} + z_t - k)
        s_neg_t = max(0, s_neg_{t-1} - z_t - k)
        score_t = max(s_pos_t, s_neg_t)
        spike declared when score_t > h

    A spike episode STARTS the first time score crosses above h, and ENDS
    the first time score drops back to <= h
    """

    alpha: float = 0.2          # EWMA smoothing factor for the baseline mean
    beta: float = 0.2           # EWMA smoothing factor for the baseline std
    k: float = 0.5              # Slack (in std units) allowed before accumulating
    h: float = 5.0              # Decision threshold (in std units) to declare a spike
    warmup_periods: int = 10    # Number of initial points used to seed baseline
    # Formular: max_score_cap = minimum steps to ease/k 
    max_score_cap: Optional[float] = None  # Cap on s_pos/s_neg so long spikes don't accumulate an unbounded score

    # Internal running state (populated as data streams in)
    mean: Optional[float] = None
    std: Optional[float] = None
    s_pos: float = 0.0          # Upward cumulative sum
    s_neg: float = 0.0          # Downward cumulative sum
    n_seen: int = 0

    # Spike episode tracking
    in_spike: bool = False
    spike_direction: Optional[str] = None   # SPIKE_UP or SPIKE_DOWN
    spike_start_idx: Optional[int] = None
    spike_start_time = None
    spike_peak_value: Optional[float] = None
    spike_peak_score: float = 0.0

    # Values seen during the current/most recent spike episode, in order.
    # Used by reset_baseline_from_spike() to rebase the baseline if the
    # spike turns out to be a legitimate shift (e.g. an app release) rather
    # than a true anomaly.
    _spike_values: List[float] = field(default_factory=list)
    _last_spike_values: List[float] = field(default_factory=list)

    # Warmup buffer
    _warmup_values: List[float] = field(default_factory=list)

    def _seed_baseline(self):
        """Initialize mean/std from the warmup buffer using simple stats."""
        vals = np.array(self._warmup_values, dtype=float)
        self.mean = float(np.mean(vals))
        self.std = float(np.std(vals))
        if self.std == 0.0 or np.isnan(self.std):
            # Avoid divide-by-zero for flat/constant series
            self.std = 1e-6

    def update(self, value: float, idx: int, timestamp) -> dict:
        """Feed one new data point into the detector."""

        self.n_seen += 1

        if np.isnan(value):
            # Missing data: don't update baseline, don't accumulate.
            return {
                "index": idx,
                "value": value,
                "baseline": self.mean,
                "std": self.std,
                "score": 0.0,
                "is_spike": False,
                "event": "none",
                "direction": None,
            }

        # ---- Warmup phase: collect points, then seed baseline ----
        if self.mean is None:
            self._warmup_values.append(value)
            if len(self._warmup_values) >= self.warmup_periods:
                self._seed_baseline()
            return {
                "index": idx,
                "value": value,
                "baseline": self.mean,
                "std": self.std,
                "score": 0.0,
                "is_spike": False,
                "event": "none",
                "direction": None,
            }

        baseline = self.mean
        std = self.std if self.std > 1e-6 else 1e-6

        # ---- Standard CUSUM update (in std-normalized units) ----
        z = (value - baseline) / std
        self.s_pos = max(0.0, self.s_pos + z - self.k)
        self.s_neg = max(0.0, self.s_neg - z - self.k)

        # Clip the accumulators so a very long spike can't push the tally
        # up indefinitely -- otherwise recovery afterwards (each normal
        # point only drains ~k) could take an unrealistic number of points.
        if self.max_score_cap is not None:
            self.s_pos = min(self.s_pos, self.max_score_cap)
            self.s_neg = min(self.s_neg, self.max_score_cap)

        score = max(self.s_pos, self.s_neg)
        direction = SPIKE_UP if self.s_pos >= self.s_neg else SPIKE_DOWN

        was_in_spike = self.in_spike
        is_spike_point = score > self.h
        event = "none"

        if is_spike_point:
            if not was_in_spike:
                # New spike episode begins
                self.in_spike = True
                self.spike_direction = direction
                self.spike_start_idx = idx
                self.spike_start_time = timestamp
                self.spike_peak_value = value
                self.spike_peak_score = score
                self._spike_values = [value]
                event = "start"
            else:
                # Ongoing spike; track peak
                if self.value_is_more_extreme(value, self.spike_peak_value, direction):
                    self.spike_peak_value = value
                self.spike_peak_score = max(self.spike_peak_score, score)
                self._spike_values.append(value)
                event = "ongoing"
        else:
            if was_in_spike:
                # Spike just ended (score dropped back to <= h): reset
                # accumulators and stash episode values for optional rebase.
                event = "end"
                self.in_spike = False
                self.s_pos = 0.0
                self.s_neg = 0.0
                self._last_spike_values = list(self._spike_values)
                self._spike_values = []
                self.spike_direction = None
                self.spike_start_idx = None
                self.spike_start_time = None
                self.spike_peak_value = None
                self.spike_peak_score = 0.0
            else:
                event = "none"

        # ---- Update adaptive EWMA baseline ----
        # Only adapt the baseline using "normal" (non-spike) points, so the
        # baseline doesn't chase the spike itself while it's ongoing.
        if not is_spike_point:
            self.mean = self.alpha * value + (1 - self.alpha) * self.mean
            self.std = self.beta * abs(value - self.mean) + (1 - self.beta) * self.std
            if self.std < 1e-6:
                self.std = 1e-6

        return {
            "index": idx,
            "value": value,
            "baseline": baseline,
            "std": std,
            "score": score,
            "is_spike": is_spike_point,
            "event": event,
            "direction": direction if is_spike_point else None,
        }

    def value_is_more_extreme(self, new_val: float, current_peak: float, direction: str) -> bool:
        if direction == SPIKE_UP:
            return new_val > current_peak
        return new_val < current_peak

    def reset_baseline_from_spike(self, extra_values: Optional[List[float]] = None) -> bool:
        """
        Manually declare the current/most recent spike episode as the new
        normal (e.g. it coincided with a known app release) and rebase the
        detector's baseline onto it.
        """
        values = list(self._spike_values) if self.in_spike else list(self._last_spike_values)
        if extra_values:
            values.extend(extra_values)

        if not values:
            return False

        vals = np.array(values, dtype=float)
        self.mean = float(np.mean(vals))
        self.std = float(np.std(vals))
        if self.std == 0.0 or np.isnan(self.std):
            self.std = 1e-6

        # Clear all spike / accumulator state so detection resumes cleanly.
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.in_spike = False
        self.spike_direction = None
        self.spike_start_idx = None
        self.spike_start_time = None
        self.spike_peak_value = None
        self.spike_peak_score = 0.0
        self._spike_values = []
        self._last_spike_values = []

        return True