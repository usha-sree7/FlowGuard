"""Uncertainty index and decision-support status. Kept strictly separate from flood risk.

    Q = sensor-quality deficit      = 1 - (0.6 * r_level + 0.4 * mean r_rain)
    D = data-sufficiency deficit    = level term (0.5 if unusable, 0.3 if unusable for <= max_stale steps)
                                      + 0.2 * [rise unknown while level present] + 0.3 * (fraction of rain sensors missing)
    F = inference (model) uncertainty = width of the KM type-reduced interval, normalised:
        clip((yr - yl - w0) / (w_ref - w0), 0, 1), w0 = median width on clean training data, w_ref = 50
    U = 100 * (wQ*Q + wD*D + wF*F)   weights 0.40 / 0.30 / 0.30 (provisional), renormalised over available terms.

U is an index for ranking how much to trust the current output. It is NOT a calibrated probability of error.

Status:
    Insufficient data : level unusable for more than max_stale steps, OR D >= 0.6
    Confident         : D == 0 (no essential input missing) AND U < 25
    Uncertain         : otherwise
"""
from __future__ import annotations

import numpy as np

from . import config


def components(fused: dict, yl: np.ndarray, yr: np.ndarray, width_ref: dict, max_stale: int) -> dict:
    Q = 1.0 - (0.6 * fused["r_level"] + 0.4 * fused["r_rain"])
    lvl_miss = fused["level_missing"]
    # A briefly missing level (<= max_stale steps) still has an informative persistence prior: 0.3 instead
    # of 0.5. Rise is counted separately only when the level itself is present (no double counting).
    lvl_term = np.where(lvl_miss, np.where(fused["level_stale_steps"] <= max_stale, 0.3, 0.5), 0.0)
    D = lvl_term + 0.2 * (fused["rise_missing"] & ~lvl_miss) + 0.3 * fused["rain_missing_frac"]
    w0, wref = width_ref.get("w0", 0.0), width_ref.get("w_ref", 50.0)
    width = yr - yl
    F = np.clip((width - w0) / max(wref - w0, 1e-9), 0, 1)
    return {"Q": np.clip(Q, 0, 1), "D": np.clip(D, 0, 1), "F": F, "width": width}


def index(Q, D, F, weights=None) -> np.ndarray:
    w = weights or config.U_WEIGHTS
    terms = [(w["Q"], Q), (w["D"], D), (w["F"], F)]
    num = np.zeros_like(np.asarray(Q, float))
    den = np.zeros_like(num)
    for wk, comp in terms:
        comp = np.asarray(comp, float)
        ok = np.isfinite(comp)
        num += np.where(ok, wk * comp, 0)
        den += np.where(ok, wk, 0)
    return np.where(den > 0, 100.0 * num / np.where(den > 0, den, 1), np.nan)


def band(U) -> np.ndarray:
    U = np.asarray(U, float)
    out = np.full(U.shape, "Very High", dtype=object)
    for edge, name in reversed(config.U_BANDS[:-1]):
        out = np.where(U < edge, name, out)
    return out


def status(U, D, fused: dict, max_stale: int) -> np.ndarray:
    U = np.asarray(U, float)
    insufficient = (fused["level_stale_steps"] > max_stale) | (D >= 0.6) | ~np.isfinite(U)
    confident = (D == 0) & (U < config.CONFIDENT_MAX_U)
    return np.where(insufficient, "Insufficient data", np.where(confident, "Confident", "Uncertain"))
