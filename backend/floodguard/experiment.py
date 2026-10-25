"""Reproducible, leak-free robustness experiment.

1. Load the processed (original) data; chronological split (train first, test last).
2. Fit every data-derived parameter on the clean TRAIN split only.
3. For each scenario: corrupt a COPY of the test split (model input), keep the injection ground truth
   separately, and compute PROXY labels from the ORIGINAL test data (future observed level).
4. Run the four systems on the identical corrupted input; evaluate clean and corrupted separately.
5. Export results/runs/<run_id>/{results.json, summary.csv, corrupted/*.csv.gz, ground_truth/*.csv.gz}.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import MODEL_VERSION, config
from .calibration import DEFAULT_RUN, RunConfig, split
from .datasets import DATASETS, load_processed, load_quality
from .datasets.loaders import processed_path
from .faults import SCENARIO_DOC, SCENARIOS, inject
from .metrics import detection, evaluate
from .reliability import flagged
from .systems import SYSTEMS, Pipeline, proxy_labels


def make_config(ds_id: str, **overrides) -> RunConfig:
    spec = DATASETS[ds_id]
    kw = {**DEFAULT_RUN[ds_id], **{k: v for k, v in overrides.items() if v is not None}}
    return RunConfig(dataset=ds_id, rain_sensors=spec.rain_sensors, level_bounds=spec.level_bounds_m, **kw)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_json(o):
    if isinstance(o, dict):
        return {str(k): _clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean_json(o.tolist())
    return o


def run_experiment(ds_id: str, scenarios=SCENARIOS, mode: str = "full", seed: int = config.DEFAULT_SEED,
                   sample_rows: int = 10_000, train_fraction: float = config.TRAIN_FRACTION,
                   save_inputs: bool = True, progress=None, **cfg_overrides) -> dict:
    say = progress or (lambda m: None)
    t_start = time.time()
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{ds_id}-{mode}-s{seed}"
    out_dir = config.RESULTS_DIR / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_processed(ds_id, mode=mode, sample_rows=sample_rows, seed=seed)
    cfg = make_config(ds_id, train_fraction=train_fraction, **cfg_overrides)
    train, test = split(df, cfg.train_fraction)
    say("calibrating on train split")
    pipe = Pipeline.fit(train, cfg)
    labels = proxy_labels(test, pipe.cal)  # from ORIGINAL test data only
    q = load_quality(ds_id) or {}

    meta = {
        "run_id": run_id, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "model_version": MODEL_VERSION,
        "dataset": ds_id, "dataset_raw_sha256": q.get("raw_sha256"),
        "processed_sha256": _file_sha(processed_path(ds_id)), "mode": mode, "seed": seed,
        "sample_rows": sample_rows if mode == "sample" else None, "rows_used": len(df),
        "train": {"rows": len(train), "first": train["timestamp"].iloc[0] or "index 0",
                  "last": train["timestamp"].iloc[-1] or f"index {len(train) - 1}"},
        "test": {"rows": len(test), "first": test["timestamp"].iloc[0] or f"index {len(train)}",
                 "last": test["timestamp"].iloc[-1] or f"index {len(df) - 1}"},
        "labels": {"kind": "PROXY (future observed level vs. train-split mean + k*sd thresholds)",
                   "horizon_hours": cfg.horizon_steps * cfg.step_hours,
                   "test_class_counts": {c: int((labels == i).sum()) for i, c in enumerate(config.RISK_CLASSES)},
                   "test_unlabelled_rows": int(np.isnan(labels).sum())},
        "config": {"run": pipe.cal.cfg.__dict__ | {"rain_sensors": list(cfg.rain_sensors)},
                   "U_weights": config.U_WEIGHTS, "risk_edges": config.RISK_EDGES},
        "model": pipe.describe(),
        "scenario_definitions": {s: SCENARIO_DOC[s] for s in scenarios},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }

    results, rows = {}, []
    clean_f1 = {}
    for i, scen in enumerate(scenarios):
        say(f"scenario {i + 1}/{len(scenarios)}: {scen}")
        corrupted, gt = inject(test, scen, pipe.cal, seed=seed + i)
        if save_inputs:
            (out_dir / "corrupted").mkdir(exist_ok=True)
            (out_dir / "ground_truth").mkdir(exist_ok=True)
            corrupted.to_csv(out_dir / "corrupted" / f"{scen}.csv.gz", index=False)
            pd.DataFrame(gt).to_csv(out_dir / "ground_truth" / f"{scen}.csv.gz", index=False)
        run = pipe.run_all(corrupted)
        any_fault = np.zeros(len(test), bool)
        for m in gt.values():
            any_fault |= m
        scen_res = {"systems": {}, "ms_per_obs": run["ms_per_obs"]}
        for name in SYSTEMS:
            o = run["outputs"][name]
            ev = {"all": evaluate(labels, o["cls"])}
            if any_fault.any() and not any_fault.all():
                ev["faulted_steps"] = evaluate(labels, o["cls"], any_fault)
                ev["unfaulted_steps"] = evaluate(labels, o["cls"], ~any_fault)
            if name == "proposed":
                not_insuff = o["status"] != "Insufficient data"
                ev["excluding_insufficient"] = evaluate(labels, o["cls"], not_insuff)
                ev["coverage"] = float(not_insuff.mean())
            scen_res["systems"][name] = ev
            f1 = ev["all"]["binary_alert"]["f1"]
            if scen == "clean":
                clean_f1[name] = f1
            b, od = ev["all"]["binary_alert"], ev["all"]["ordinal"]
            rows.append({"dataset": ds_id, "scenario": scen, "system": name, "precision": b["precision"],
                         "recall": b["recall"], "f1": f1, "fpr": b["fpr"], "fnr": b["fnr"],
                         "f1_drop_vs_clean": (clean_f1.get(name, np.nan) - f1) if clean_f1 else np.nan,
                         "accuracy_4class": od["accuracy"], "macro_f1_4class": od["macro_f1"],
                         "severity_weighted_error": od["severity_weighted_error"],
                         "ms_per_obs": run["ms_per_obs"][name]})
        p = run["outputs"]["proposed"]
        status_counts = pd.Series(p["status"]).value_counts().to_dict()
        scen_res["uncertainty"] = {
            "U_mean": float(np.nanmean(p["U"])), "U_p90": float(np.nanquantile(p["U"], 0.9)),
            "U_mean_faulted_steps": float(np.nanmean(p["U"][any_fault])) if any_fault.any() else None,
            "U_mean_unfaulted_steps": float(np.nanmean(p["U"][~any_fault])) if (~any_fault).any() else None,
            "Q_mean": float(np.mean(p["Q"])), "D_mean": float(np.mean(p["D"])), "F_mean": float(np.nanmean(p["F"])),
            "status_counts": {k: int(v) for k, v in status_counts.items()},
            "U_band_counts": {k: int(v) for k, v in pd.Series(p["U_band"]).value_counts().items()},
            "confident_while_level_missing": int(((p["status"] == "Confident") & p["fused"]["level_missing"]).sum()),
        }
        scen_res["fault_detection"] = {ch: detection(gt[ch], flagged(p["rel"], ch)) for ch in gt} \
            if scen != "clean" else {ch: {"false_flag_rate": float(flagged(p["rel"], ch).mean())} for ch in gt}
        results[scen] = scen_res

    meta["elapsed_s"] = round(time.time() - t_start, 2)
    payload = _clean_json({"meta": meta, "results": results})
    (out_dir / "results.json").write_text(json.dumps(payload, indent=1))
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    say("done")
    return payload


def list_runs() -> list[dict]:
    base = config.RESULTS_DIR / "runs"
    if not base.exists():
        return []
    out = []
    for d in sorted(base.iterdir(), reverse=True):
        f = d / "results.json"
        if f.exists():
            m = json.loads(f.read_text())["meta"]
            out.append({k: m.get(k) for k in ("run_id", "created", "dataset", "mode", "seed", "rows_used",
                                               "model_version", "elapsed_s")})
    return out


def load_run(run_id: str) -> dict:
    f = config.RESULTS_DIR / "runs" / run_id / "results.json"
    if not f.exists() or ".." in run_id:
        raise FileNotFoundError(run_id)
    return json.loads(f.read_text())
