"""Evaluation metrics for ordered risk classes (0 Normal, 1 Watch, 2 Warning, 3 Critical).

Binary 'alert' = class >= Warning. Severity-weighted error uses an asymmetric ordinal cost:
under-warning costs 2 per class step, over-warning costs 1 per class step (provisional, documented choice).
"""
from __future__ import annotations

import numpy as np


def binary(y_true, y_pred) -> dict:
    t, p = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp, fp = int((t & p).sum()), int((~t & p).sum())
    fn, tn = int((t & ~p).sum()), int((~t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (tp and prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1,
            "fpr": fp / (fp + tn) if fp + tn else float("nan"),
            "fnr": fn / (fn + tp) if fn + tp else float("nan"), "n": int(len(t))}


def ordinal(y_true, y_pred, k: int = 4) -> dict:
    t, p = np.asarray(y_true, int), np.asarray(y_pred, int)
    cm = np.zeros((k, k), int)
    np.add.at(cm, (t, p), 1)
    diff = p - t
    cost = np.where(diff < 0, -2 * diff, diff)
    per_class_f1 = []
    for c in range(k):
        b = binary(t == c, p == c)
        per_class_f1.append(b["f1"] if (t == c).any() else float("nan"))
    return {"confusion_matrix": cm.tolist(), "accuracy": float((t == p).mean()) if len(t) else float("nan"),
            "macro_f1": float(np.nanmean(per_class_f1)), "per_class_f1": per_class_f1,
            "mean_abs_class_error": float(np.abs(diff).mean()) if len(t) else float("nan"),
            "severity_weighted_error": float(cost.mean()) if len(t) else float("nan")}


def evaluate(y_true, y_pred, mask=None) -> dict:
    y_true = np.asarray(y_true, float)
    ok = ~np.isnan(y_true)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    t, p = y_true[ok].astype(int), np.asarray(y_pred)[ok].astype(int)
    return {"binary_alert": binary(t >= 2, p >= 2), "ordinal": ordinal(t, p)}


def detection(gt_mask, flagged) -> dict:
    g, f = np.asarray(gt_mask, bool), np.asarray(flagged, bool)
    tp, fp, fn = int((g & f).sum()), int((~g & f).sum()), int((g & ~f).sum())
    return {"injected": int(g.sum()), "flagged": int(f.sum()), "tp": tp,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan")}
