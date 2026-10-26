"""Genuine future water-level forecasting for the forecast-horizon comparison (continuous target).

  persistence : level(t + h) = level(t)  (the standard no-skill reference)
  ARX         : linear autoregressive model with exogenous rain, fitted by ridge least squares on the FIT block of
                the training split. It predicts the CHANGE level(t + h) - level(t) from causal features only:
                level(t), level changes over 1 / 2 / 4 steps, and the fused rain accumulated over 1, 4 and 12 steps.

Targets are the observed level exactly h steps later (NaN across timestamp gaps, see systems.future_level), so a
forecast is never scored against a reading from a different time. Forecast class = class of the forecast level
against the same train-split thresholds as the proxy label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ("level", "dlevel_1", "dlevel_2", "dlevel_4", "rain_1", "rain_4", "rain_12")


def features(level: np.ndarray, rain_step: np.ndarray) -> np.ndarray:
    L, P = pd.Series(level), pd.Series(rain_step)
    cols = [L, L - L.shift(1), L - L.shift(2), L - L.shift(4),
            P, P.rolling(4, min_periods=1).sum(), P.rolling(12, min_periods=1).sum()]
    return np.column_stack([c.fillna(0.0).to_numpy(float) for c in cols])


class ARX:
    def __init__(self, ridge: float = 1e-3):
        self.ridge = ridge

    def fit(self, F, level, target):
        ok = np.isfinite(target) & np.isfinite(F).all(1)
        A = np.column_stack([np.ones(ok.sum()), F[ok]])
        d = target[ok] - level[ok]
        self.coef = np.linalg.solve(A.T @ A + self.ridge * np.eye(A.shape[1]), A.T @ d)
        return self

    def predict(self, F, level):
        return level + np.column_stack([np.ones(len(F)), F]) @ self.coef

    def describe(self) -> dict:
        return {"intercept": round(float(self.coef[0]), 6),
                "coefficients": {n: round(float(c), 6) for n, c in zip(FEATURES, self.coef[1:])}, "ridge": self.ridge}


def regression_metrics(y_true, y_pred) -> dict:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    t, p = y_true[ok], y_pred[ok]
    if not len(t):
        return {"n": 0, "mae": None, "rmse": None, "r2": None}
    err = p - t
    ss_tot = ((t - t.mean()) ** 2).sum()
    return {"n": int(len(t)), "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "r2": float(1 - (err ** 2).sum() / ss_tot) if ss_tot > 0 else None}
