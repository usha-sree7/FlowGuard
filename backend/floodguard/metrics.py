"""Evaluation metrics for ordered risk classes (0 Normal, 1 Low, 2 Moderate, 3 High, 4 Critical).

Binary 'alert' = class >= High (config.ALERT_CLASS). Severity-weighted error uses an asymmetric ordinal cost:
under-warning costs 2 per class step, over-warning costs 1 per class step (provisional, documented choice).
Confidence intervals: moving-block bootstrap (Kunsch 1989) over the test period, so autocorrelated neighbouring
time steps are resampled together instead of being treated as independent.
"""
from __future__ import annotations

import numpy as np

from .config import ALERT_CLASS, N_CLASSES


def binary(y_true, y_pred) -> dict:
    t, p = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp, fp = int((t & p).sum()), int((~t & p).sum())
    fn, tn = int((t & ~p).sum()), int((~t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    # F1 is undefined (NaN) when the period contains no true positives at all: there is nothing to detect.
    # It is 0 when positives exist but none were caught.
    f1 = float("nan") if tp + fn == 0 else (2 * prec * rec / (prec + rec) if tp else 0.0)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1,
            "fpr": fp / (fp + tn) if fp + tn else float("nan"),
            "fnr": fn / (fn + tp) if fn + tp else float("nan"), "n": int(len(t))}


def ordinal(y_true, y_pred, k: int = N_CLASSES) -> dict:
    t, p = np.asarray(y_true, int), np.asarray(y_pred, int)
    cm = np.zeros((k, k), int)
    np.add.at(cm, (t, p), 1)
    diff = p - t
    cost = np.where(diff < 0, -2 * diff, diff)
    per_class_f1 = []
    for c in range(k):
        b = binary(t == c, p == c)
        per_class_f1.append(b["f1"] if (t == c).any() else float("nan"))
    n = cm.sum()
    pe = float((cm.sum(0) * cm.sum(1)).sum()) / n ** 2 if n else float("nan")
    acc = float(np.trace(cm)) / n if n else float("nan")
    return {"confusion_matrix": cm.tolist(), "accuracy": acc,
            "kappa": (acc - pe) / (1 - pe) if n and pe < 1 else float("nan"),
            "macro_f1": float(np.nanmean(per_class_f1)), "per_class_f1": per_class_f1,
            "mean_abs_class_error": float(np.abs(diff).mean()) if len(t) else float("nan"),
            "severity_weighted_error": float(cost.mean()) if len(t) else float("nan")}


def evaluate(y_true, y_pred, mask=None) -> dict:
    y_true = np.asarray(y_true, float)
    ok = ~np.isnan(y_true)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    t, p = y_true[ok].astype(int), np.asarray(y_pred)[ok].astype(int)
    return {"binary_alert": binary(t >= ALERT_CLASS, p >= ALERT_CLASS), "ordinal": ordinal(t, p)}


def block_bootstrap_f1(y_true, y_pred, block: int, n_boot: int = 1000, seed: int = 0) -> list[float]:
    """95 % percentile interval of the alert F1 under a moving-block bootstrap of the test period."""
    y_true = np.asarray(y_true, float)
    ok = ~np.isnan(y_true)
    t = np.where(ok, y_true, 0) >= ALERT_CLASS
    p = np.asarray(y_pred) >= ALERT_CLASS
    n = len(t)
    if not (t & ok).any():  # no true alerts in the test period: F1 and its interval are undefined
        return [float("nan"), float("nan")]
    block = max(1, min(block, n))
    starts = np.arange(n - block + 1)
    rng = np.random.default_rng(seed)
    offs = np.arange(block)
    f1 = np.empty(n_boot)
    for b in range(n_boot):
        idx = (rng.choice(starts, int(np.ceil(n / block)))[:, None] + offs).ravel()[:n]
        m = ok[idx]
        tt, pp = t[idx][m], p[idx][m]
        tp = (tt & pp).sum()
        f1[b] = 2 * tp / (tt.sum() + pp.sum()) if tt.sum() + pp.sum() else 0.0
    return [float(np.quantile(f1, 0.025)), float(np.quantile(f1, 0.975))]


def detection(gt_mask, flagged) -> dict:
    g, f = np.asarray(gt_mask, bool), np.asarray(flagged, bool)
    tp, fp, fn = int((g & f).sum()), int((~g & f).sum()), int((g & ~f).sum())
    return {"injected": int(g.sum()), "flagged": int(f.sum()), "tp": tp,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan")}
