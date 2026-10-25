"""Fault injection on COPIES of the processed test split.

Returns (corrupted_df, ground_truth) where
    corrupted_df  : model input only (original columns, corrupted values, optional health__* metadata)
    ground_truth  : dict channel -> bool mask of injected faults (evaluation only, never shown to models)
The original dataframe is never modified.

Magnitudes are expressed relative to train-split statistics so the same scenario definitions work for
both the 30-min and the daily dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import Calibration

SCENARIOS = (
    "clean", "noise_low", "noise_med", "noise_high", "missing_10", "missing_30", "spikes", "stuck",
    "conflict", "reduced_reliability", "combined",
)

SCENARIO_DOC = {
    "clean": "no corruption",
    "noise_low": "additive Gaussian noise on all sensors: level sd = 0.10*level_sd, rain sd = 0.10*rain scale",
    "noise_med": "as above with factor 0.25",
    "noise_high": "as above with factor 0.50",
    "missing_10": "10% of readings of every sensor set missing independently (MCAR)",
    "missing_30": "30% of readings of every sensor set missing independently (MCAR)",
    "spikes": "2% of level readings +/- U(4,8)*level_sd; 1% of each rain sensor + U(2,5)*rain scale",
    "stuck": "level and first rain sensor frozen at last value in random segments covering ~10% of steps",
    "conflict": "first rain sensor reads 0 during ~15% segments (clogged) and spurious rain at 1% of dry steps",
    "reduced_reliability": "device-health metadata of the level sensor drops to 0.4 in ~20% segments, with "
                           "noise sd 0.25*level_sd in those segments",
    "combined": "noise_med + missing_10 + spikes + stuck",
}


def _segments(n: int, seg_len: int, coverage: float, rng) -> np.ndarray:
    mask = np.zeros(n, bool)
    n_seg = max(1, int(round(coverage * n / seg_len)))
    for s in rng.integers(0, max(n - seg_len, 1), n_seg):
        mask[s:s + seg_len] = True
    return mask


def inject(df: pd.DataFrame, scenario: str, cal: Calibration, seed: int) -> tuple[pd.DataFrame, dict]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario}")
    rng = np.random.default_rng(seed)
    out = df.copy(deep=True)
    n = len(out)
    sensors = list(cal.cfg.rain_sensors)
    rain_cols = [f"rain__{s}" for s in sensors]
    gt = {"level": np.zeros(n, bool), **{s: np.zeros(n, bool) for s in sensors}}
    lsd = cal.level_sd
    # rain scale in ORIGINAL sensor units (undo bias adjustment)
    rscale = {s: cal.rel["rain_scale_per_step"] / cal.rain_bias.get(s, 1.0) for s in sensors}
    seg_len = int(max(3, round(12.0 / cal.cfg.step_hours))) if cal.cfg.step_hours < 24 else 5
    lo_b, hi_b = cal.cfg.level_bounds

    def noise(factor):
        out["level"] = np.clip(out["level"] + rng.normal(0, factor * lsd, n), lo_b, hi_b)
        gt["level"][:] = True
        for s, c in zip(sensors, rain_cols):
            out[c] = np.clip(out[c] + rng.normal(0, factor * rscale[s], n), 0, None)
            gt[s][:] = True

    def missing(frac):
        for ch, c in [("level", "level"), *zip(sensors, rain_cols)]:
            m = rng.random(n) < frac
            out.loc[m, c] = np.nan
            gt[ch] |= m

    def spikes():
        m = rng.random(n) < 0.02
        mag = rng.uniform(4, 8, n) * lsd * rng.choice([-1, 1], n)
        out.loc[m, "level"] = np.clip(out.loc[m, "level"] + mag[m], lo_b, hi_b)
        gt["level"] |= m
        for s, c in zip(sensors, rain_cols):
            m = rng.random(n) < 0.01
            out.loc[m, c] = out.loc[m, c] + rng.uniform(2, 5, m.sum()) * rscale[s]
            gt[s] |= m

    def stuck():
        for ch, c in [("level", "level"), (sensors[0], rain_cols[0])]:
            m = _segments(n, seg_len, 0.10, rng)
            vals = out[c].to_numpy(float).copy()
            frozen = pd.Series(np.where(m, np.nan, vals)).ffill().to_numpy()
            out[c] = np.where(m, frozen, vals)
            gt[ch] |= m  # fault present for the whole segment, even where frozen value == true value

    if scenario.startswith("noise_"):
        noise({"noise_low": 0.10, "noise_med": 0.25, "noise_high": 0.50}[scenario])
    elif scenario.startswith("missing_"):
        missing(int(scenario.split("_")[1]) / 100.0)
    elif scenario == "spikes":
        spikes()
    elif scenario == "stuck":
        stuck()
    elif scenario == "conflict":
        c, s = rain_cols[0], sensors[0]
        clog = _segments(n, seg_len, 0.15, rng)
        before = out[c].to_numpy(float).copy()
        out.loc[clog, c] = 0.0
        gt[s] |= clog & (before > 0)
        dry = (out[rain_cols].fillna(0).sum(axis=1) == 0).to_numpy()
        fake = dry & (rng.random(n) < 0.01)
        out.loc[fake, c] = rng.uniform(1, 3, fake.sum()) * rscale[s]
        gt[s] |= fake
    elif scenario == "reduced_reliability":
        seg = _segments(n, seg_len, 0.20, rng)
        out["health__level"] = np.where(seg, 0.4, 1.0)
        out.loc[seg, "level"] = np.clip(out.loc[seg, "level"] + rng.normal(0, 0.25 * lsd, seg.sum()), lo_b, hi_b)
        gt["level"] |= seg
    elif scenario == "combined":
        noise(0.25)
        missing(0.10)
        spikes()
        stuck()
    return out, gt
