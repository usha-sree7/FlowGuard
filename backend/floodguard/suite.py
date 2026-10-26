"""HydroSentinel AI model suite: the four compared techniques, fitted once and run on identical observations.

  it2_fis    Interval Type-2 FIS, 80 expert rules, KM type reduction; anchors from train-split statistics (untuned)
  pso_it2    the same IT2FIS with membership-function anchors and FOU width tuned by PSO on the validation block
  som        Self-Organising Map: input state -> best-matching prototype -> risk class learned from the fit block
  anfis      first-order TSK neuro-fuzzy system (80 rules) trained by hybrid least-squares / gradient learning
  threshold  REFERENCE ONLY: class of the current level vs. the train-split thresholds (the rule the proxy labels
             are built from, so it is near-perfect at short horizons on clean data by construction)

Every model runs on the same three inputs x = [level, rise, rain] in one of two input modes:
  raw    conventional pre-processing (missing level -> last value, rise from consecutive levels, rain = plain mean)
  fused  the reliability layer (reliability.py + fusion.py): each reading is checked (missing, spike, stuck, noisy,
         out-of-range, cross-sensor conflict), down-weighted by its reliability and fused; the models then see the
         reliability-weighted best estimates. The layer also gives an uncertainty index and a confidence status.

Data use: calibration on the whole TRAIN split (first 70 %); SOM codebook / risk mapping, ANFIS and the ARX level
forecaster on its first 80 % (FIT block); SOM size, ANFIS ridge and the PSO genes chosen on its last 20 %
(VALIDATION block). Fit rows whose horizon-h target lies inside the validation block are purged. The TEST split
(last 30 %) is only used for the final evaluation.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from . import config
from .anfis import ANFIS, select_anfis
from .calibration import rain_matrix
from .forecast import ARX, features
from .patterns import SOM, select_som
from .systems import Pipeline, future_level, proxy_labels, risk_class
from .tuning import TunedIT2

MODELS = ("it2_fis", "pso_it2", "som", "anfis")
REFERENCE = "threshold"
RISK_MODELS = (*MODELS, REFERENCE)
MODES = ("raw", "fused")
MODEL_NAMES = {"it2_fis": "IT2FIS", "pso_it2": "PSO-tuned IT2FIS", "som": "SOM", "anfis": "ANFIS",
               "threshold": "Threshold rule (reference)"}
CANDIDATE_HORIZONS_H = (0.5, 1.0, 3.0)  # 30 min, 1 h, 3 h


def horizons(step_hours: float) -> dict:
    """Supported horizons (steps -> label) and the requested ones the sampling interval cannot represent."""
    sup, unsup = {}, []
    for h in CANDIDATE_HORIZONS_H:
        k = h / step_hours
        if abs(k - round(k)) < 1e-9 and round(k) >= 1:
            sup[int(round(k))] = f"{int(h * 60)} min" if h < 1 else f"{h:g} h"
        else:
            unsup.append(f"{int(h * 60)} min" if h < 1 else f"{h:g} h")
    if step_hours >= 24:
        sup[1] = "1 day"
    return {"supported": sup, "unsupported": unsup}


def matrix(inputs: dict) -> np.ndarray:
    return np.column_stack([np.asarray(inputs[k], float) for k in ("level", "rise", "rain")])


def rain_per_step(df: pd.DataFrame, pipe: Pipeline) -> np.ndarray:
    R = rain_matrix(df, pipe.cal.cfg, pipe.cal.rain_bias)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nan_to_num(np.nanmean(np.where(R < 0, np.nan, R), axis=1)) if R.size else np.zeros(len(df))


def _timed(fn, store: dict, key: str, n: int):
    t0 = time.perf_counter()
    out = fn()
    store[key] = store.get(key, 0.0) + (time.perf_counter() - t0) * 1000.0 / max(n, 1)
    return out


class ModelSuite:
    def __init__(self, pipe: Pipeline, seed: int = config.DEFAULT_SEED):
        self.pipe, self.seed = pipe, seed
        self.horizons = horizons(pipe.cal.cfg.step_hours)["supported"]
        self.som, self.pso, self.anfis, self.arx = {}, {}, {}, {}
        self.info = {"som": {}, "pso_it2": {}, "anfis": {}}
        self.fit_seconds = {m: {} for m in MODELS}

    @classmethod
    def fit(cls, train: pd.DataFrame, cfg, seed: int = config.DEFAULT_SEED, saved: dict | None = None,
            progress=None) -> "ModelSuite":
        """saved: the `fitted` block of a stored comparison run. When given, the validation searches are not
        repeated: PSO genes are reused and only the selected SOM grid / ANFIS ridge is refitted (deterministic,
        so the models are identical to the evaluated ones)."""
        say = progress or (lambda m: None)
        saved = saved or {}
        t0 = time.perf_counter()
        suite = cls(Pipeline.fit(train, cfg), seed)
        pipe = suite.pipe
        calib_s = time.perf_counter() - t0
        X = matrix(pipe.naive_inputs(train))
        n_fit = int(0.8 * len(train))
        idx = np.arange(len(train))
        suite.split_info = {"fit_rows": n_fit, "validation_rows": len(train) - n_fit,
                            "fit_last": train["timestamp"].iloc[n_fit - 1] or f"index {n_fit - 1}",
                            "validation_first": train["timestamp"].iloc[n_fit] or f"index {n_fit}"}
        lvl, rstep = train["level"].to_numpy(float), rain_per_step(train, pipe)
        F = features(lvl, rstep)
        for h in suite.horizons:
            hl = suite.horizons[h]
            y = proxy_labels(train, pipe.cal, h)
            ok = ~np.isnan(y)
            # purge: the last h fit rows have targets inside the validation block, so they are not used for fitting
            fm, vm = ok & (idx < n_fit - h), ok & (idx >= n_fit)
            yi = np.nan_to_num(y).astype(int)
            suite.fit_seconds["it2_fis"][h] = calib_s  # expert rules: only the train-split calibration is "fitted"

            say(f"SOM ({hl})")
            t = time.perf_counter()
            prev = saved.get("som", {}).get(str(h), {})
            if "state" in prev:  # stored fitted model: load instead of refitting
                suite.som[h] = SOM.from_state(prev["state"])
                suite.info["som"][h] = {k: prev[k] for k in ("selected", "validation_fitness")}
                prev = {}
            grid = (tuple(int(v) for v in prev["selected"].split("x")),) if "selected" in prev else None
            if h not in suite.som:
                suite.som[h], suite.info["som"][h] = select_som(X[fm], yi[fm], X[vm], y[vm], seed=seed,
                                                               **({"grid": grid} if grid else {}))
            if grid:
                suite.info["som"][h] = {k: prev[k] for k in ("selected", "validation_fitness")}
            suite.fit_seconds["som"][h] = saved.get("fit_seconds", {}).get("som", {}).get(str(h))                 if "state" in saved.get("som", {}).get(str(h), {}) else time.perf_counter() - t

            say(f"PSO-tuned IT2FIS ({hl})")
            t = time.perf_counter()
            g = TunedIT2(pipe, "pso", seed=seed)
            prev = saved.get("pso_it2", {}).get(str(h))
            if prev and len(prev.get("genes", [])) == len(g.BASE):
                g.set_genes(prev["genes"], prev.get("validation_fitness_untuned"), prev.get("validation_fitness_tuned"))
                g.history = prev.get("best_fitness_per_iteration", [])
                suite.fit_seconds["pso_it2"][h] = prev.get("fit_seconds")
            else:
                g.fit(X[vm], y[vm])
                suite.fit_seconds["pso_it2"][h] = time.perf_counter() - t + calib_s
            suite.pso[h] = g
            suite.info["pso_it2"][h] = {**g.describe(), "fit_seconds": suite.fit_seconds["pso_it2"][h]}

            say(f"ANFIS ({hl})")
            t = time.perf_counter()
            prev = saved.get("anfis", {}).get(str(h), {})
            ridge = {"grid": (prev["selected_ridge"],)} if "selected_ridge" in prev else {}
            if "state" in prev:
                suite.anfis[h] = ANFIS.from_state(pipe.cal, prev["state"])
            else:
                suite.anfis[h], suite.info["anfis"][h] = select_anfis(pipe.cal, X[fm], yi[fm], X[vm], y[vm], **ridge)
            if ridge:
                suite.info["anfis"][h] = {k: prev[k] for k in ("selected_ridge", "validation_fitness")}
            suite.fit_seconds["anfis"][h] = saved.get("fit_seconds", {}).get("anfis", {}).get(str(h))                 if "state" in prev else time.perf_counter() - t

            target = future_level(train, h, cfg.step_hours)
            suite.arx[h] = ARX().fit(F[:n_fit - h], lvl[:n_fit - h], target[:n_fit - h])
        return suite

    # ------------------------------------------------------------------ inference
    def run_shared(self, df: pd.DataFrame) -> dict:
        """Horizon-independent outputs: inputs in both modes, the reliability layer, the untuned IT2FIS and the
        threshold reference."""
        n, ms = len(df), {}
        pipe = self.pipe
        Xn = _timed(lambda: matrix(pipe.naive_inputs(df)), ms, "raw_inputs", n)
        p = _timed(lambda: pipe.run_proposed(df), ms, "reliability_layer", n)
        Xf = matrix(p["point"])
        X = {"raw": Xn, "fused": Xf}
        it2 = {m: _timed(lambda m=m: pipe.fis.infer_it2({k: (X[m][:, j], X[m][:, j]) for j, k in
                                                          enumerate(("level", "rise", "rain"))}), ms, f"it2_fis|{m}", n)
               for m in MODES}
        thr = {m: np.digitize(X[m][:, 0], pipe.cal.thresholds) for m in MODES}
        return {"X": X, "proposed": p, "it2": it2, "threshold": thr, "ms": ms, "n": n}

    def run_horizon(self, sh: dict, h: int) -> dict:
        ms, n = sh["ms"], sh["n"]
        cls, score, node = {}, {}, {}
        for m in MODES:
            X = sh["X"][m]
            y_pso = _timed(lambda: self.pso[h].predict(X), ms, f"pso_it2|{m}", n)
            y_anf = _timed(lambda: self.anfis[h].predict(X), ms, f"anfis|{m}", n)
            node[m] = _timed(lambda: self.som[h].bmu(X), ms, f"som|{m}", n)
            score[m] = {"it2_fis": sh["it2"][m]["y"], "pso_it2": y_pso, "anfis": y_anf}
            cls[m] = {"it2_fis": risk_class(sh["it2"][m]["y"]), "pso_it2": risk_class(y_pso),
                      "som": self.som[h].node_class[node[m]], "anfis": risk_class(y_anf),
                      "threshold": sh["threshold"][m]}
        return {"cls": cls, "score": score, "som_node": node}

    def forecast(self, df: pd.DataFrame, h: int) -> np.ndarray:
        lvl = pd.Series(df["level"].to_numpy(float)).ffill().bfill().to_numpy()
        return self.arx[h].predict(features(lvl, rain_per_step(df, self.pipe)), lvl)

    def describe(self) -> dict:
        return {"horizons": {str(k): v for k, v in self.horizons.items()},
                "split": getattr(self, "split_info", {}),
                "it2_fis": {"rules": 80, "anchors": self.pipe.cal.anchors, "fou_half_widths": self.pipe.cal.fou_eps,
                            "thresholds_m": self.pipe.cal.thresholds, "risk_edges": config.RISK_EDGES},
                "som": {str(h): {**self.info["som"][h], **self.som[h].describe(), "state": self.som[h].state()}
                        for h in self.horizons},
                "pso_it2": {str(h): v for h, v in self.info["pso_it2"].items()},
                "anfis": {str(h): {**self.info["anfis"][h], **self.anfis[h].describe(), "state": self.anfis[h].state()}
                          for h in self.horizons},
                "fit_seconds": {m: {str(h): v for h, v in d.items()} for m, d in self.fit_seconds.items()},
                "arx": {str(h): m.describe() for h, m in self.arx.items()}}
