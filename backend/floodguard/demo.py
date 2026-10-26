"""Interactive risk assessment for HYPOTHETICAL user inputs (clearly separated from historical evaluation).

The reliability checks and the rise / rain features are defined on a time series, not on one reading. A short,
documented history consistent with the user's inputs is therefore constructed, the chosen sensor-quality
scenario is applied to its last readings, and the REAL fitted models are run on it. Only the last step is
reported.

History (N = 48 steps): level flat, then a linear ramp over the last 6 steps ending at the entered level with the
entered rise rate; rain = the entered accumulation spread evenly over the rain window ending now, zero before.
Every rain sensor reports the same rain (in its own units, i.e. undoing the train-split bias factor).

Sensor-quality scenarios (the TRUE values are what the user entered; the scenario changes what the sensors report):
  normal         sensors report the true values
  noisy          Gaussian noise (sd = 0.25 x train level sd; 0.25 x rain scale) on the last 12 readings, seed 7
  missing_level  the last level reading is missing
  stuck_level    the level sensor has been frozen for the last 8 steps at the value it had 8 steps ago
  spike          the last level reading is the true level + 6 x train level sd
  rain_conflict  the first rain sensor reports zero over the rain window while the others report rain
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .datasets.loaders import add_derived

SENSOR_SCENARIOS = {
    "normal": "sensors report the true values",
    "noisy": "Gaussian noise on the last 12 readings of every sensor (fixed seed)",
    "missing_level": "the latest water-level reading is missing",
    "stuck_level": "level sensor frozen for the last 8 steps at its old value",
    "spike": "latest level reading = true level + 6 x training level sd",
    "rain_conflict": "first rain sensor reports zero while the others report rain",
}
N = 48


def validate(pipe, level: float, rise: float, rain: float) -> list[str]:
    cal = pipe.cal
    errs = []
    lo, hi = cal.cfg.level_bounds
    if not np.isfinite(level) or not lo <= level <= hi:
        errs.append(f"water level must be between {lo:g} and {hi:g} m")
    rlo, rhi = cal.bounds["rise"]
    if not np.isfinite(rise) or not rlo <= rise <= rhi:
        errs.append(f"rate of rise must be between {rlo:.3f} and {rhi:.3f} m/h (twice the training range)")
    if not np.isfinite(rain) or not 0 <= rain <= cal.bounds["rain"][1]:
        errs.append(f"rainfall must be between 0 and {cal.bounds['rain'][1]:.1f} (twice the training maximum)")
    return errs


def build_history(pipe, level: float, rise: float, rain: float, scenario: str) -> pd.DataFrame:
    cal, cfg = pipe.cal, pipe.cal.cfg
    dt, W = cfg.step_hours, cfg.rain_window_steps
    k = np.arange(N)
    ramp = np.clip(N - 1 - k, 0, 6)
    # earlier levels are clipped to the plausible range (a fast rise from a low level starts at the lower bound)
    true_level = np.clip(level - rise * dt * ramp, *cfg.level_bounds)
    per_step = np.where(k >= N - W, rain / W, 0.0)
    df = pd.DataFrame({"t_hours": k * dt, "timestamp": [""] * N if dt >= 24 else
                       pd.date_range("2000-01-01", periods=N, freq=f"{int(dt * 60)}min").astype(str)})
    df["level"] = true_level.copy()
    for s in cfg.rain_sensors:
        df[f"rain__{s}"] = per_step / cal.rain_bias.get(s, 1.0)
    rng = np.random.default_rng(7)
    lsd = cal.level_sd
    if scenario == "noisy":
        tail = k >= N - 12
        df.loc[tail, "level"] += rng.normal(0, 0.25 * lsd, tail.sum())
        for s in cfg.rain_sensors:
            scale = cal.rel["rain_scale_per_step"] / cal.rain_bias.get(s, 1.0)
            df.loc[tail, f"rain__{s}"] = np.clip(df.loc[tail, f"rain__{s}"] + rng.normal(0, 0.25 * scale, tail.sum()), 0, None)
    elif scenario == "missing_level":
        df.loc[N - 1, "level"] = np.nan
    elif scenario == "stuck_level":
        df.loc[N - 8:, "level"] = true_level[N - 9]
    elif scenario == "spike":
        df.loc[N - 1, "level"] = level + 6 * lsd
    elif scenario == "rain_conflict":
        df.loc[k >= N - W, f"rain__{cfg.rain_sensors[0]}"] = 0.0
    df["level"] = df["level"].clip(*cfg.level_bounds)
    return add_derived(df, dt), true_level
