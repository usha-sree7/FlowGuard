"""Per-sensor, per-time-step reliability assessment. Causal: every check at time t uses only data <= t.

Reliability r in [0, 1] = prior_health * prod(1 - penalty_k); r = 0 when the reading is missing or outside the
configured plausibility range. Reliability describes the SENSOR READING, not the flood risk.

Level checks : missing, out-of-range, spike (jump vs. recent median, rain-aware), stuck (flat run while rain
               falls), noise (roughness of 2nd differences above the train 99th percentile), device health prior.
Rain checks  : missing, out-of-range, stuck (identical non-zero run), cross-sensor inconsistency, health prior.
Conflicts    : evidence that cannot be attributed to one sensor (two-sensor rain disagreement, level rising
               fast with no rain anywhere) is NOT turned into a reliability penalty; it is returned as an
               unresolved-conflict score and widens the fused input intervals instead.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .calibration import Calibration, rain_matrix, run_lengths


def _health(df: pd.DataFrame, name: str) -> np.ndarray:
    col = f"health__{name}"
    return np.clip(df[col].to_numpy(float), 0, 1) if col in df else np.ones(len(df))


def assess(df: pd.DataFrame, cal: Calibration) -> dict:
    cfg, p = cal.cfg, cal.rel
    n = len(df)
    step_ok = df["step_ok"].to_numpy(bool)

    # ---------------- rain sensors
    R = rain_matrix(df, cfg, cal.rain_bias)
    rain = {}
    for j, s in enumerate(cfg.rain_sensors):
        x = R[:, j]
        miss = np.isnan(x)
        oor = ~miss & ((x < 0) | (x > p["rain_max_per_step"][j]))
        xr = np.where(R[:, j] > 0, R[:, j], np.nan)
        stuck = np.isfinite(xr) & (run_lengths(xr) >= p["rain_stuck_steps"])
        rain[s] = {"x": x, "missing": miss, "out_of_range": oor, "stuck": stuck}

    Rv = R.copy()
    for j, s in enumerate(cfg.rain_sensors):
        Rv[rain[s]["missing"] | rain[s]["out_of_range"], j] = np.nan
    k = len(cfg.rain_sensors)
    conflict_rain = np.zeros(n)
    for j, s in enumerate(cfg.rain_sensors):
        if k >= 3:  # attributable: compare with the median of the other sensors
            cons = np.nanmedian(np.delete(Rv, j, axis=1), axis=1)
            z = np.abs(Rv[:, j] - cons) / (1.0 + cons) / p["disagreement_q99"]
            pen_inc = np.clip(0.8 * (z - 1.0), 0, 0.8)
        elif k == 2:  # not attributable: mild penalty on both, the rest is unresolved conflict
            other = Rv[:, 1 - j]
            cons = other
            z = np.abs(Rv[:, j] - other) / (1.0 + np.fmin(Rv[:, j], other)) / p["disagreement_q99"]
            pen_inc = np.clip(0.3 * (z - 1.0), 0, 0.3)
            conflict_rain = np.fmax(conflict_rain, np.nan_to_num(np.clip(z - 1.0, 0, 1)))
        else:
            z = np.zeros(n)
            pen_inc = np.zeros(n)
        pen_inc = np.nan_to_num(pen_inc)
        inconsistent = pen_inc > 0
        r = _health(df, s) * (1 - pen_inc) * np.where(rain[s]["stuck"], 0.3, 1.0)
        r = np.where(rain[s]["missing"] | rain[s]["out_of_range"], 0.0, r)
        rain[s].update({"inconsistent": inconsistent, "r": r})

    # Raw (unweighted) rain context for level checks.
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-missing rows -> NaN -> 0
        raw_rain = np.nan_to_num(np.nanmean(Rv, axis=1)) if k else np.zeros(n)
    rain_ctx = pd.Series(raw_rain).rolling(cfg.rain_window_steps * 3, min_periods=1).sum().to_numpy()
    light = cal.anchors["rain"][0]

    # ---------------- level sensor
    x = df["level"].to_numpy(float)
    lo_b, hi_b = cfg.level_bounds
    miss = np.isnan(x)
    oor = ~miss & ((x < lo_b) | (x > hi_b))
    valid = ~miss & ~oor
    xs = pd.Series(np.where(valid, x, np.nan))
    ref = xs.shift(1).rolling(4, min_periods=1).median().to_numpy()
    jump = np.abs(x - ref)
    spike = valid & step_ok & (jump > p["jump_m_per_step"])
    spike_pen = np.where(spike, np.where(rain_ctx < light, 0.9, 0.4), 0.0)

    runs = run_lengths(np.where(valid, x, np.nan))
    cum_rain = np.cumsum(raw_rain)
    start_idx = np.clip(np.arange(n) - runs + 1, 0, n - 1)
    rain_during_run = cum_rain - cum_rain[start_idx] + raw_rain[start_idx]
    stuck = valid & (runs >= p["stuck_min_steps"]) & (rain_during_run > p["stuck_rain_during_run"])
    stuck_pen = np.where(stuck, 0.85, 0.0)

    d2 = np.abs(pd.Series(np.where(valid, x, np.nan)).diff().diff())
    rough = d2.rolling(6, min_periods=3).mean().to_numpy()
    noise_pen = np.nan_to_num(np.clip(0.5 * (rough / max(p["rough_threshold"], 1e-9) - 1.0), 0, 0.7))
    noisy = valid & (noise_pen > 0)

    r_level = _health(df, "level") * (1 - spike_pen) * (1 - stuck_pen) * (1 - noise_pen)
    r_level = np.where(valid, r_level, 0.0)

    # Physical conflict: level rising fast with no rain anywhere in the extended window (unresolved).
    rise_raw = np.where(step_ok, np.diff(x, prepend=np.nan) / cfg.step_hours, np.nan)
    conflict_phys = np.where((rise_raw > cal.anchors["rise"][1]) & (rain_ctx < light) & ~spike, 0.5, 0.0)

    return {
        "level": {"x": x, "r": r_level, "missing": miss, "out_of_range": oor, "spike": spike, "stuck": stuck,
                  "noisy": noisy, "health": _health(df, "level")},
        "rain": rain,
        "rain_matrix": R,
        "conflict": np.fmax(conflict_rain, conflict_phys),
        "conflict_rain": conflict_rain,
        "conflict_phys": conflict_phys,
    }


def flagged(rel: dict, channel: str, threshold: float = 0.5) -> np.ndarray:
    """A reading is 'flagged' (detected as faulty) when its reliability falls below `threshold`."""
    r = rel["level"]["r"] if channel == "level" else rel["rain"][channel]["r"]
    return r < threshold
