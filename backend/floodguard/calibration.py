"""Station-specific parameters estimated ONLY from the chronological training split (clean, original data).

Nothing here is a universal flood threshold. Level thresholds follow the statistical rule found in the
Soares et al. UAE-ML code (mean + 2 sd = 'flood threshold'), extended by user-configured k = 1 and k = 3.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .fuzzy import FIS, partition

LEVEL_K = (0.5, 1.0, 2.0, 3.0)  # Low, Moderate, High (paper rule), Critical


@dataclass
class RunConfig:
    dataset: str
    step_hours: float
    horizon_steps: int          # label = class of the observed level horizon_steps ahead
    rain_window_steps: int      # fused rain is accumulated over this causal window
    rain_sensors: tuple
    level_bounds: tuple
    train_fraction: float = 0.70


DEFAULT_RUN = {
    # 1 h ahead, 2 h rain accumulation for the 30-min flash-flood catchment
    "bengalas": dict(step_hours=0.5, horizon_steps=2, rain_window_steps=4),
    # 1 day ahead, 2 day rain accumulation for the daily river series
    "malwathu": dict(step_hours=24.0, horizon_steps=1, rain_window_steps=2),
}


@dataclass
class Calibration:
    cfg: RunConfig
    level_mean: float
    level_sd: float
    thresholds: list            # [Low, Moderate, High, Critical] lower edges in metres
    threshold_source: dict
    rain_bias: dict             # multiplicative bias per sensor (radar -> gauge scale); 1.0 otherwise
    anchors: dict               # input -> 3 crossover anchors
    fou_eps: dict               # input -> 3 FOU half-shifts (std error of anchor across blocks)
    bounds: dict                # input -> (lower, upper) universe bounds
    rel: dict                   # reliability-check parameters
    width_ref: dict = field(default_factory=dict)  # F normalisation (filled by the pipeline)

    def to_dict(self):
        d = asdict(self)
        d["cfg"]["rain_sensors"] = list(self.cfg.rain_sensors)
        return d


def split(df: pd.DataFrame, frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = int(len(df) * frac)
    return df.iloc[:n].reset_index(drop=True), df.iloc[n:].reset_index(drop=True)


def blocks(df: pd.DataFrame) -> np.ndarray:
    """Calendar year when timestamps exist, otherwise consecutive 365-step blocks (daily data)."""
    if df["timestamp"].iloc[0]:
        return pd.to_datetime(df["timestamp"]).dt.year.to_numpy()
    return (np.arange(len(df)) // 365)


def run_lengths(x: np.ndarray) -> np.ndarray:
    """Length of the run of identical consecutive values ending at each position (NaN breaks runs)."""
    s = pd.Series(x)
    new = (s != s.shift(1)) | s.isna()
    grp = new.cumsum()
    return s.groupby(grp).cumcount().to_numpy() + 1


def rain_matrix(df: pd.DataFrame, cfg: RunConfig, bias: dict) -> np.ndarray:
    return np.column_stack([df[f"rain__{s}"].to_numpy(float) * bias.get(s, 1.0) for s in cfg.rain_sensors])


def _anchor_se(values_by_block: list[np.ndarray], fn, n: int = 3) -> np.ndarray:
    est = np.array([fn(v) for v in values_by_block if len(v) > 10])
    if len(est) < 2:
        return np.zeros(n)
    return est.std(axis=0, ddof=1) / np.sqrt(len(est))


def calibrate(train: pd.DataFrame, cfg: RunConfig) -> Calibration:
    lvl = train["level"].to_numpy(float)
    mu, sd = float(np.nanmean(lvl)), float(np.nanstd(lvl))
    thresholds = [mu + k * sd for k in LEVEL_K]
    blk = blocks(train)

    # Radar -> gauge mean-field bias (sum ratio over the training period); 1.0 for all other sensors.
    bias = {s: 1.0 for s in cfg.rain_sensors}
    if "radar" in cfg.rain_sensors and "gauge" in cfg.rain_sensors:
        bias["radar"] = float(train["rain__gauge"].sum() / max(train["rain__radar"].sum(), 1e-9))
    R = rain_matrix(train, cfg, bias)
    fused = np.nanmean(R, axis=1)
    acc = pd.Series(fused).rolling(cfg.rain_window_steps, min_periods=1).sum().to_numpy()
    wet_q = lambda a: np.quantile(a[a > 0], [0.75, 0.95, 0.99]) if (a > 0).sum() > 10 else np.full(3, np.nan)  # noqa: E731
    rise = train["rise_rate"].to_numpy(float)
    rise_q = lambda a: np.nanquantile(a, [0.01, 0.99, 0.999])  # noqa: E731
    level_q = lambda a: np.array([np.nanmean(a) + k * np.nanstd(a) for k in LEVEL_K])  # noqa: E731

    anchors = {"level": np.array(thresholds), "rise": rise_q(rise), "rain": wet_q(acc)}
    if not anchors["rise"][1] > 0:  # guarantee a positive 'rising' crossover
        anchors["rise"][1] = max(anchors["rise"][2] / 3, 1e-3)
    by = lambda a: [a[blk == b] for b in np.unique(blk)]  # noqa: E731
    eps = {
        "level": _anchor_se(by(lvl), level_q, len(LEVEL_K)),
        "rise": _anchor_se(by(rise), rise_q),
        "rain": _anchor_se(by(acc), wet_q),
    }
    eps = {k: np.nan_to_num(v) for k, v in eps.items()}
    lo_b, hi_b = cfg.level_bounds
    bounds = {
        "level": (float(lo_b), float(hi_b)),
        "rise": (float(np.nanmin(rise) * 2), float(np.nanmax(rise) * 2)),
        "rain": (0.0, float(np.nanmax(acc) * 2)),
    }

    # Reliability-check parameters (all from clean training data).
    d1 = np.abs(np.diff(lvl))
    d1 = d1[np.isfinite(d1)]
    rough = pd.Series(np.abs(np.diff(lvl, n=2, prepend=np.nan, append=np.nan)[:len(lvl)])
                      ).shift(1).rolling(6, min_periods=3).mean()
    # Stuck level: a flat run of >= 3 h (>= 3 steps) is only suspicious if more rain fell during it than in
    # 99.5% of such flat runs in the clean training data.
    stuck_min = int(max(3, round(3.0 / cfg.step_hours)))
    rl = run_lengths(lvl)
    cum = np.cumsum(np.nan_to_num(fused))
    st = np.clip(np.arange(len(lvl)) - rl + 1, 0, len(lvl) - 1)
    rain_run = (cum - cum[st] + np.nan_to_num(fused)[st])[rl >= stuck_min]
    stuck_rain = float(max(np.quantile(rain_run, 0.995), anchors["rain"][1])) if len(rain_run) else anchors["rain"][1]
    rain_runs = []
    for j in range(R.shape[1]):
        x = np.where(R[:, j] > 0, R[:, j], np.nan)
        rain_runs.append(run_lengths(x)[np.isfinite(x)])
    rain_runs = np.concatenate(rain_runs) if rain_runs else np.array([1])
    # Inter-sensor disagreement statistic on clean data -> normalises the consistency check.
    disagree = _disagreement(R)
    rel = {
        "jump_m_per_step": float(1.5 * np.quantile(d1, 0.999)),
        "level_step_scale_m": float(max(np.quantile(d1, 0.95), 0.01)),
        "rough_threshold": float(np.nanquantile(rough, 0.99)),
        "stuck_min_steps": stuck_min,
        "stuck_rain_during_run": stuck_rain,
        "rain_stuck_steps": int(max(3, np.quantile(rain_runs, 0.999) + 1)),
        "rain_max_per_step": [float(np.nanmax(R[:, j]) * 1.5) for j in range(R.shape[1])],
        "disagreement_q99": float(np.nanquantile(disagree, 0.99)) if np.isfinite(disagree).any() else 1.0,
        "rain_scale_per_step": float(max(anchors["rain"][1] / cfg.rain_window_steps, 1e-6)),
        "max_stale_steps": 2,
    }
    return Calibration(
        cfg=cfg, level_mean=mu, level_sd=sd, thresholds=thresholds,
        threshold_source={
            "Low": "mean + 0.5*sd - project extension (not from the paper)",
            "Moderate": "mean + 1*sd - project extension (not from the paper)",
            "High": "mean + 2*sd of train-split level: statistical 'flood threshold' rule in the UAE-ML "
                    "uncertainty-analysis code (authors computed it on the full dataset)",
            "Critical": "mean + 3*sd - project extension (not from the paper)",
        },
        rain_bias=bias, anchors={k: v.tolist() for k, v in anchors.items()},
        fou_eps={k: v.tolist() for k, v in eps.items()}, bounds=bounds, rel=rel,
    )


def _disagreement(R: np.ndarray) -> np.ndarray:
    """Per-row max over sensors of |x_i - consensus_i| / (1 + consensus_i) where consensus_i is the
    median of the other sensors. Only rows where at least one sensor reports rain are returned."""
    n = R.shape[1]
    if n < 2:
        return np.array([np.nan])
    out = np.zeros(len(R))
    for i in range(n):
        others = np.delete(R, i, axis=1)
        cons = np.nanmedian(others, axis=1)
        out = np.fmax(out, np.abs(R[:, i] - cons) / (1.0 + cons))
    wet = np.nanmax(R, axis=1) > 0
    return out[wet]


def build_fis(cal: Calibration) -> FIS:
    mfs = {}
    for name in ("level", "rise", "rain"):
        lo, hi = cal.bounds[name]
        terms, _, _ = partition(cal.anchors[name], cal.fou_eps[name], lower_bound=lo, upper_bound=hi)
        mfs[name] = terms
    return FIS(mfs=mfs)
