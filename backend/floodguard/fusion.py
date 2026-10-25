"""Reliability-weighted evidence fusion -> interval-valued inputs for the IT2FIS.

Each fused input is an interval [lo, hi] that carries the UNRESOLVED uncertainty forward instead of
discarding it: a perfectly reliable single reading gives a singleton; lower reliability, staleness,
sensor disagreement and unattributable conflicts widen the interval.

level : centre c = r*x + (1-r)*p, where p = causal median of the last 4 trustworthy (r >= 0.5) readings.
        half-width = (1-r)*max(|x-p|, s0) + conflict*s0;  missing -> c = p, half-width 2*s0*sqrt(staleness).
rise  : from consecutive fused level centres; half-width = (h_t + h_{t-1}) / dt. While the level is missing
        the last known rise is carried for up to max_stale steps; unknown across timestamp gaps.
rain  : reliability-weighted mean of bias-adjusted sensors +/- reliability-weighted dispersion, widened by
        (1 - best reliability) * scale; all sensors missing -> [0, universe max]; accumulated over the window.

The fused CENTRES are the best estimate used for the flood-risk decision. The INTERVALS are propagated
through the IT2FIS separately to obtain a plausible risk range, which feeds only the uncertainty index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import Calibration


def fuse(df: pd.DataFrame, rel: dict, cal: Calibration) -> dict:
    cfg, p = cal.cfg, cal.rel
    n = len(df)
    s0 = p["level_step_scale_m"]
    lo_b, hi_b = cal.bounds["level"]
    conflict = rel["conflict"]

    # ---------------- level
    x, r = rel["level"]["x"], rel["level"]["r"]
    usable = r > 0
    reliable = r >= 0.5
    trusted = pd.Series(np.where(reliable, x, np.nan)).dropna()
    prior = (trusted.rolling(4, min_periods=1).median().reindex(range(n)).ffill().shift(1).to_numpy())
    idx = np.arange(n)
    last_rel = pd.Series(np.where(reliable, idx, np.nan)).ffill().shift(1).to_numpy()
    stale = np.where(np.isnan(last_rel), np.inf, idx - last_rel)

    has_prior = ~np.isnan(prior)
    xc = np.where(usable, x, np.nan)
    c = np.where(usable & has_prior, r * xc + (1 - r) * prior, np.where(usable, xc, prior))
    h = np.where(usable & has_prior, (1 - r) * np.maximum(np.abs(xc - prior), s0), 0.0)
    h = np.where(usable & ~has_prior, (1 - r) * s0 * 4, h)
    h = np.where(~usable & has_prior, 2 * s0 * np.sqrt(np.minimum(stale, 1e6)), h)
    h = h + conflict * s0
    level_lo = np.where(np.isnan(c), lo_b, np.clip(c - h, lo_b, hi_b))
    level_hi = np.where(np.isnan(c), hi_b, np.clip(c + h, lo_b, hi_b))
    level_c = np.where(np.isnan(c), np.nan, np.clip(c, lo_b, hi_b))

    # ---------------- rise rate
    rise_lo_b, rise_hi_b = cal.bounds["rise"]
    step_ok = df["step_ok"].to_numpy(bool)
    prev_c = np.roll(level_c, 1)
    prev_h = np.roll((level_hi - level_lo) / 2, 1)
    rc = (level_c - prev_c) / cfg.step_hours
    rh = ((level_hi - level_lo) / 2 + prev_h) / cfg.step_hours
    rise_known = step_ok & np.isfinite(rc) & usable & np.roll(usable, 1)
    rise_known[0] = False
    rise_lo = np.where(rise_known, np.clip(rc - rh, rise_lo_b, rise_hi_b), rise_lo_b)
    rise_hi = np.where(rise_known, np.clip(rc + rh, rise_lo_b, rise_hi_b), rise_hi_b)
    rise_c = np.where(rise_known, rc, np.nan)
    carried = pd.Series(rise_c).ffill(limit=p["max_stale_steps"]).to_numpy()
    rise_c_point = np.where(rise_known, rise_c, np.where(~usable, np.nan_to_num(carried), 0.0))

    # ---------------- rain
    R = rel["rain_matrix"]
    W = np.column_stack([rel["rain"][s]["r"] for s in cfg.rain_sensors])
    Rz = np.nan_to_num(R)
    wsum = W.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(wsum > 0, (W * Rz).sum(axis=1) / wsum, np.nan)
        disp = np.where(wsum > 0, np.sqrt((W * (Rz - mean[:, None]) ** 2).sum(axis=1) / wsum), np.nan)
    best = W.max(axis=1)
    scale = p["rain_scale_per_step"]
    per_lo = np.where(wsum > 0, np.maximum(mean - disp, 0) * best, 0.0)
    per_hi = np.where(wsum > 0, mean + disp + (1 - best) * scale + conflict * scale,
                      cal.bounds["rain"][1] / cfg.rain_window_steps)
    per_c = np.where(wsum > 0, mean, np.nan)
    roll = lambda a: pd.Series(a).rolling(cfg.rain_window_steps, min_periods=1).sum().to_numpy()  # noqa: E731
    rain_lo, rain_hi = roll(per_lo), roll(per_hi)
    rain_c = pd.Series(per_c).rolling(cfg.rain_window_steps, min_periods=1).sum().to_numpy()

    # ---------------- data sufficiency inputs (missing != unreliable)
    rain_missing_frac = np.mean(np.column_stack([rel["rain"][s]["missing"] | rel["rain"][s]["out_of_range"]
                                                 for s in cfg.rain_sensors]), axis=1)
    return {
        "level": (level_lo, level_hi), "level_c": level_c,
        "rise": (rise_lo, rise_hi), "rise_c": rise_c, "rise_point": rise_c_point,
        "rain": (rain_lo, rain_hi), "rain_c": rain_c, "rain_point": np.nan_to_num(rain_c),
        "r_level": r, "r_rain": W.mean(axis=1), "r_rain_sensors": W,
        "level_missing": ~usable, "rise_missing": ~rise_known, "rain_missing_frac": rain_missing_frac,
        "level_stale_steps": np.where(usable, 0, stale),
        "conflict": conflict,
    }
