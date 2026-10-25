"""Karnik-Mendel (KM) type reduction for center-of-sets interval type-2 fuzzy systems.

Given rule firing intervals [f_i_lo, f_i_hi] and consequent centroid intervals [cl_i, cr_i]:
    y_l = min over f_i in [f_lo, f_hi] of sum(f_i cl_i) / sum(f_i)
    y_r = max over f_i in [f_lo, f_hi] of sum(f_i cr_i) / sum(f_i)
The optimum is a single switch point in the sorted consequent order (Karnik & Mendel, 2001).

`km_iterative` is the classical iterative algorithm (one sample at a time).
`km_exact` evaluates every switch point with cumulative sums (vectorised over samples); it returns the
same optimum as KM because KM is guaranteed to stop at the best switch point. Equality is unit-tested.
"""
from __future__ import annotations

import numpy as np


def km_iterative(c, f_lo, f_hi, right: bool, max_iter: int = 100) -> float:
    c = np.asarray(c, float)
    f_lo = np.asarray(f_lo, float)
    f_hi = np.asarray(f_hi, float)
    if f_hi.sum() <= 0:
        return float("nan")
    order = np.argsort(c)
    c, f_lo, f_hi = c[order], f_lo[order], f_hi[order]
    f = (f_lo + f_hi) / 2.0
    if f.sum() <= 0:
        f = f_hi.copy()
    y = float(np.dot(f, c) / f.sum())
    for _ in range(max_iter):
        k = int(np.searchsorted(c, y, side="right")) - 1  # c[k] <= y < c[k+1]
        k = min(max(k, 0), len(c) - 2) if len(c) > 1 else 0
        idx = np.arange(len(c))
        if right:   # lower firing left of switch, upper firing right of switch
            f = np.where(idx <= k, f_lo, f_hi)
        else:       # upper firing left of switch, lower firing right of switch
            f = np.where(idx <= k, f_hi, f_lo)
        s = f.sum()
        y_new = float(np.dot(f, c) / s) if s > 0 else y
        if abs(y_new - y) < 1e-12:
            return y_new
        y = y_new
    return y


def km_exact(c, f_lo, f_hi, right: bool) -> np.ndarray:
    """Vectorised exact solution. c: (R,), f_lo/f_hi: (N, R). Returns (N,)."""
    c = np.asarray(c, float)
    order = np.argsort(c)
    c = c[order]
    f_lo = np.asarray(f_lo, float)[:, order]
    f_hi = np.asarray(f_hi, float)[:, order]
    R = len(c)
    # Switch after position k (k = -1..R-1): left part uses A, right part uses B.
    A, B = (f_lo, f_hi) if right else (f_hi, f_lo)
    zeros = np.zeros((A.shape[0], 1))
    cumA_w = np.concatenate([zeros, np.cumsum(A, axis=1)], axis=1)          # (N, R+1)
    cumA_wc = np.concatenate([zeros, np.cumsum(A * c, axis=1)], axis=1)
    totB_w = B.sum(axis=1, keepdims=True)
    totB_wc = (B * c).sum(axis=1, keepdims=True)
    cumB_w = np.concatenate([zeros, np.cumsum(B, axis=1)], axis=1)
    cumB_wc = np.concatenate([zeros, np.cumsum(B * c, axis=1)], axis=1)
    num = cumA_wc + (totB_wc - cumB_wc)
    den = cumA_w + (totB_w - cumB_w)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.where(den > 0, num / den, np.nan)
    y = np.where(np.isnan(y), -np.inf if right else np.inf, y)
    out = y.max(axis=1) if right else y.min(axis=1)
    return np.where(np.isfinite(out), out, np.nan)  # NaN only when every firing strength is zero


def type_reduce(cl, cr, f_lo, f_hi):
    """Center-of-sets type reduction -> (y_l, y_r)."""
    return km_exact(cl, f_lo, f_hi, right=False), km_exact(cr, f_lo, f_hi, right=True)
