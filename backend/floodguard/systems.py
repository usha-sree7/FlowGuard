"""The four compared systems, run on identical observations.

  threshold    : class of the current level vs. the station thresholds (conventional gauge alarm)
  t1_fis       : Type-1 FIS, same rules/variables, embedded T1 MFs, crisp inputs
  it2_fis      : IT2FIS (KM type reduction) with crisp (singleton) inputs, no reliability assessment
  proposed     : reliability assessment -> reliability-weighted fusion (interval inputs) -> IT2FIS
                 + separate uncertainty index / status

Systems 1-3 are reliability-unaware and use the conventional naive pre-processing: missing level is
replaced by the last observation (forward fill), rise is the difference of consecutive (filled) levels,
rain is the plain mean of available bias-adjusted sensors (missing -> 0).
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from . import MODEL_VERSION, config, uncertainty
from .calibration import Calibration, build_fis, calibrate, rain_matrix
from .fusion import fuse
from .reliability import assess

SYSTEMS = ("threshold", "t1_fis", "it2_fis", "proposed")


def risk_class(score) -> np.ndarray:
    return np.digitize(np.asarray(score, float), config.RISK_EDGES)


class Pipeline:
    def __init__(self, cal: Calibration):
        self.cal = cal
        self.fis = build_fis(cal)

    @classmethod
    def fit(cls, train: pd.DataFrame, cfg) -> "Pipeline":
        pipe = cls(calibrate(train, cfg))
        # F normalisation: median KM interval width of the proposed system on CLEAN training data.
        out = pipe.run_proposed(train, _calibrating=True)
        pipe.cal.width_ref = {"w0": float(np.nanmedian(out["yr"] - out["yl"])), "w_ref": 50.0}
        return pipe

    # ---------------- naive inputs for reliability-unaware systems
    def naive_inputs(self, df: pd.DataFrame) -> dict:
        cfg = self.cal.cfg
        lvl = pd.Series(df["level"].to_numpy(float)).ffill().bfill().to_numpy()
        step_ok = df["step_ok"].to_numpy(bool)
        rise = np.where(step_ok, np.diff(lvl, prepend=lvl[0]) / cfg.step_hours, 0.0)
        R = rain_matrix(df, cfg, self.cal.rain_bias)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all sensors missing -> NaN -> 0
            per = np.nan_to_num(np.nanmean(np.where(R < 0, np.nan, R), axis=1)) if R.size else np.zeros(len(df))
        rain = pd.Series(per).rolling(cfg.rain_window_steps, min_periods=1).sum().to_numpy()
        clip = lambda a, k: np.clip(a, *self.cal.bounds[k])  # noqa: E731
        return {"level": clip(lvl, "level"), "rise": clip(rise, "rise"), "rain": clip(rain, "rain")}

    def run_threshold(self, df):
        x = self.naive_inputs(df)["level"]
        cls = np.digitize(x, self.cal.thresholds)
        return {"cls": cls, "score": np.array([10.0, 37.5, 62.5, 90.0])[cls]}

    def run_t1(self, df):
        y = self.fis.infer_t1(self.naive_inputs(df))["y"]
        return {"cls": risk_class(y), "score": y}

    def run_it2(self, df):
        x = self.naive_inputs(df)
        out = self.fis.infer_it2({k: (v, v) for k, v in x.items()})
        return {"cls": risk_class(out["y"]), "score": out["y"], "yl": out["yl"], "yr": out["yr"]}

    def run_proposed(self, df, _calibrating: bool = False):
        rel = assess(df, self.cal)
        fused = fuse(df, rel, self.cal)
        lo_b = self.cal.bounds["level"][0]
        point = {"level": np.nan_to_num(fused["level_c"], nan=lo_b), "rise": fused["rise_point"],
                 "rain": fused["rain_point"]}
        # Flood risk: IT2FIS on the reliability-weighted best estimates (singleton inputs).
        best = self.fis.infer_it2({k: (v, v) for k, v in point.items()})
        # Plausible risk range: IT2FIS on the fused INTERVAL inputs. Feeds the uncertainty index only,
        # so input uncertainty is never silently converted into a higher (or lower) risk class.
        rng = self.fis.infer_it2({k: fused[k] for k in ("level", "rise", "rain")})
        yl, yr = np.fmin(rng["yl"], best["yl"]), np.fmax(rng["yr"], best["yr"])
        res = {"cls": risk_class(best["y"]), "score": best["y"], "yl": yl, "yr": yr,
               "rel": rel, "fused": fused, "point": point}
        if _calibrating:
            return res
        ms = self.cal.rel["max_stale_steps"]
        comp = uncertainty.components(fused, yl, yr, self.cal.width_ref, ms)
        U = uncertainty.index(comp["Q"], comp["D"], comp["F"])
        res.update(comp)
        res["U"] = U
        res["U_band"] = uncertainty.band(U)
        res["status"] = uncertainty.status(U, comp["D"], fused, ms)
        return res

    def run_all(self, df) -> dict:
        out, timing = {}, {}
        for name, fn in (("threshold", self.run_threshold), ("t1_fis", self.run_t1),
                         ("it2_fis", self.run_it2), ("proposed", self.run_proposed)):
            t0 = time.perf_counter()
            out[name] = fn(df)
            timing[name] = (time.perf_counter() - t0) * 1000.0 / max(len(df), 1)  # ms per observation
        return {"outputs": out, "ms_per_obs": timing}

    def describe(self) -> dict:
        return {"model_version": MODEL_VERSION, "calibration": self.cal.to_dict(), "fis": self.fis.describe()}


def future_level(df: pd.DataFrame, horizon_steps: int, step_hours: float) -> np.ndarray:
    """Observed level exactly horizon_steps * step_hours later (exact time match; NaN across gaps)."""
    t = df["t_hours"].to_numpy(float)
    lookup = pd.Series(df["level"].to_numpy(float), index=np.round(t, 6))
    lookup = lookup[~lookup.index.duplicated()]
    target = np.round(t + horizon_steps * step_hours, 6)
    return lookup.reindex(target).to_numpy()


def proxy_labels(df_original: pd.DataFrame, cal: Calibration) -> np.ndarray:
    """PROXY label (not an observed flood record): class of the FUTURE observed level, from the ORIGINAL
    (uncorrupted) data. Returns float array with NaN where the future observation is unavailable."""
    fut = future_level(df_original, cal.cfg.horizon_steps, cal.cfg.step_hours)
    cls = np.digitize(fut, cal.thresholds).astype(float)
    cls[np.isnan(fut)] = np.nan
    return cls
