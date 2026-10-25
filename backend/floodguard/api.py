"""FastAPI backend: dataset explorer/import, model inspection, replay dashboard, experiments.

Run:  uvicorn floodguard.api:app --port 8000   (from backend/)
The built frontend (frontend/dist) is served at / when present.
"""
from __future__ import annotations

import threading
import traceback
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import MODEL_VERSION, config
from .calibration import split
from .datasets import DATASETS, NOT_INTEGRATED, DatasetError, import_dataset, is_loaded, load_processed, load_quality
from .datasets.loaders import manual_instructions
from .experiment import list_runs, load_run, make_config, run_experiment
from .faults import SCENARIO_DOC, SCENARIOS, inject
from .fuzzy import INPUTS, OUTPUT_TERMS, TERMS
from .systems import Pipeline, proxy_labels

app = FastAPI(title="FloodGuard AI", version=MODEL_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _set_job(key, **kw):
    with _lock:
        _jobs.setdefault(key, {}).update(kw)


def _spec(ds_id):
    if ds_id not in DATASETS:
        raise HTTPException(404, f"unknown dataset {ds_id}")
    return DATASETS[ds_id]


def _num(a):
    """JSON-safe list (NaN/inf -> None), rounded."""
    a = np.asarray(a)
    if a.dtype.kind in "fc":
        return [None if not np.isfinite(v) else round(float(v), 4) for v in a]
    if a.dtype.kind == "b":
        return [bool(v) for v in a]
    return a.tolist()


# --------------------------------------------------------------------------- datasets

@app.get("/api/health")
def health():
    return {"ok": True, "model_version": MODEL_VERSION}


@app.get("/api/datasets")
def datasets():
    out = []
    for ds_id, spec in DATASETS.items():
        q = load_quality(ds_id) if is_loaded(ds_id) else None
        out.append({**spec.to_dict(), "loaded": bool(q and q.get("loaded")),
                    "quality": q, "import_job": _jobs.get(f"import:{ds_id}")})
    return {"datasets": out, "not_integrated": NOT_INTEGRATED}


@app.post("/api/datasets/{ds_id}/import")
def start_import(ds_id: str):
    spec = _spec(ds_id)
    key = f"import:{ds_id}"
    if _jobs.get(key, {}).get("state") == "running":
        return _jobs[key]

    def work():
        try:
            rep = import_dataset(ds_id, progress=lambda m: _set_job(key, message=m))
            _pipeline.cache_clear()
            _replay.cache_clear()
            _set_job(key, state="done", message=f"loaded and validated {rep['rows']} rows")
        except DatasetError as e:
            _set_job(key, state="failed", message=str(e).splitlines()[0], manual=manual_instructions(spec))
        except Exception as e:  # unexpected: surface it instead of pretending success
            _set_job(key, state="failed", message=repr(e), manual=manual_instructions(spec),
                     trace=traceback.format_exc()[-2000:])

    _set_job(key, state="running", message="starting", manual=None)
    threading.Thread(target=work, daemon=True).start()
    return _jobs[key]


@app.get("/api/datasets/{ds_id}/import")
def import_status(ds_id: str):
    _spec(ds_id)
    return _jobs.get(f"import:{ds_id}", {"state": "idle"})


@app.get("/api/datasets/{ds_id}/series")
def series(ds_id: str, max_points: int = 1500):
    _spec(ds_id)
    if not is_loaded(ds_id):
        raise HTTPException(409, "dataset not imported")
    df = load_processed(ds_id)
    step = max(1, len(df) // max_points)
    d = df.iloc[::step]
    rain_cols = [c for c in df.columns if c.startswith("rain__")]
    n_train = int(len(df) * config.TRAIN_FRACTION)
    return {"step": step, "n": len(df), "train_end_index": n_train,
            "index": d.index.tolist(), "timestamp": d["timestamp"].tolist(), "level": _num(d["level"]),
            "rain": {c[6:]: _num(d[c]) for c in rain_cols}}


# --------------------------------------------------------------------------- model

@lru_cache(maxsize=4)
def _pipeline(ds_id: str):
    if not is_loaded(ds_id):
        raise HTTPException(409, f"dataset '{ds_id}' not imported yet - use the Dataset Explorer")
    df = load_processed(ds_id)
    cfg = make_config(ds_id)
    train, test = split(df, cfg.train_fraction)
    return Pipeline.fit(train, cfg), test


@app.get("/api/model/{ds_id}")
def model(ds_id: str):
    _spec(ds_id)
    pipe, _ = _pipeline(ds_id)
    desc = pipe.describe()
    curves = {}
    for name in INPUTS:
        lo, hi = pipe.cal.bounds[name]
        a = pipe.cal.anchors[name]
        span = a[2] - a[0]
        x0 = max(lo, a[0] - 0.6 * span)
        x1 = min(hi, a[2] + 0.6 * span)
        xs = np.linspace(x0, x1, 240)
        curves[name] = {"x": _num(xs), "terms": [
            {"term": TERMS[name][i], "lower": _num(m.lower(xs)), "upper": _num(m.upper(xs)),
             "type1": _num(m.embedded(xs))} for i, m in enumerate(pipe.fis.mfs[name])]}
    return {**desc, "curves": curves, "output_terms": OUTPUT_TERMS}


class InferRequest(BaseModel):
    level: list[float]  # [lo, hi]; lo == hi for a crisp reading
    rise: list[float]
    rain: list[float]


@app.post("/api/infer/{ds_id}")
def infer(ds_id: str, req: InferRequest):
    _spec(ds_id)
    pipe, _ = _pipeline(ds_id)
    iv = {k: (np.array([min(v)]), np.array([max(v)])) for k, v in req.model_dump().items()}
    it2 = pipe.fis.infer_it2(iv)
    mid = {k: np.array([(v[0][0] + v[1][0]) / 2]) for k, v in iv.items()}
    t1 = pipe.fis.infer_t1(mid)
    from .fuzzy import rule_table
    rules = rule_table()
    top = np.argsort(-it2["f_hi"][0])[:6]
    return {"it2": {"yl": float(it2["yl"][0]), "yr": float(it2["yr"][0]), "y": float(it2["y"][0])},
            "t1": {"y": float(t1["y"][0])},
            "class_it2": config.RISK_CLASSES[int(np.digitize(it2["y"][0], config.RISK_EDGES))],
            "top_rules": [{**rules[i], "firing": [round(float(it2["f_lo"][0][i]), 4),
                                                   round(float(it2["f_hi"][0][i]), 4)]} for i in top]}


# --------------------------------------------------------------------------- replay dashboard

@lru_cache(maxsize=24)
def _replay(ds_id: str, scenario: str, seed: int):
    pipe, test = _pipeline(ds_id)
    corrupted, gt = inject(test, scenario, pipe.cal, seed)
    run = pipe.run_all(corrupted)
    return pipe, test, corrupted, gt, run, proxy_labels(test, pipe.cal)


@app.get("/api/scenarios")
def scenarios():
    return {"scenarios": [{"id": s, "description": SCENARIO_DOC[s]} for s in SCENARIOS]}


@app.get("/api/replay/{ds_id}")
def replay(ds_id: str, scenario: str = "clean", offset: int = 0, limit: int = 96, seed: int = config.DEFAULT_SEED):
    _spec(ds_id)
    if scenario not in SCENARIOS:
        raise HTTPException(400, "unknown scenario")
    pipe, test, corrupted, gt, run, labels = _replay(ds_id, scenario, seed)
    n = len(test)
    offset = int(np.clip(offset, 0, max(n - 1, 0)))
    sl = slice(offset, min(n, offset + max(1, min(limit, 2000))))
    p = run["outputs"]["proposed"]
    f, rel = p["fused"], p["rel"]
    sensors = list(pipe.cal.cfg.rain_sensors)
    any_fault = np.zeros(n, bool)
    for m in gt.values():
        any_fault |= m
    return {
        "n": n, "offset": offset, "scenario": scenario, "seed": seed,
        "note": "Replay of the held-out historical test split (not a live feed). Labels are PROXY labels.",
        "thresholds": pipe.cal.thresholds, "risk_classes": config.RISK_CLASSES,
        "timestamp": [t or f"day {offset + i}" for i, t in enumerate(test["timestamp"].iloc[sl])],
        "level_input": _num(corrupted["level"].to_numpy()[sl]), "level_original": _num(test["level"].to_numpy()[sl]),
        "level_fused": _num(f["level_c"][sl]), "level_lo": _num(f["level"][0][sl]), "level_hi": _num(f["level"][1][sl]),
        "rise_point": _num(f["rise_point"][sl]), "rain_point": _num(f["rain_point"][sl]),
        "rain_lo": _num(f["rain"][0][sl]), "rain_hi": _num(f["rain"][1][sl]),
        "rain_input": {s: _num(corrupted[f"rain__{s}"].to_numpy()[sl]) for s in sensors},
        "r_level": _num(f["r_level"][sl]),
        "r_rain": {s: _num(rel["rain"][s]["r"][sl]) for s in sensors},
        "flags": {k: _num(rel["level"][k][sl]) for k in ("missing", "out_of_range", "spike", "stuck", "noisy")},
        "conflict": _num(f["conflict"][sl]),
        "risk_score": _num(p["score"][sl]), "risk_yl": _num(p["yl"][sl]), "risk_yr": _num(p["yr"][sl]),
        "risk_class": _num(p["cls"][sl]),
        "U": _num(p["U"][sl]), "U_band": p["U_band"][sl].tolist(), "status": p["status"][sl].tolist(),
        "Q": _num(p["Q"][sl]), "D": _num(p["D"][sl]), "F": _num(p["F"][sl]),
        "baselines": {k: _num(run["outputs"][k]["cls"][sl]) for k in ("threshold", "t1_fis", "it2_fis")},
        "proxy_label": _num(labels[sl]), "injected_fault": _num(any_fault[sl]),
        "event_index": _num(np.flatnonzero(labels >= 2)[::max(1, int((labels >= 2).sum()) // 40)]),
    }


# --------------------------------------------------------------------------- experiments

class ExperimentRequest(BaseModel):
    dataset: str = "bengalas"
    mode: str = "full"
    seed: int = config.DEFAULT_SEED
    scenarios: list[str] | None = None


@app.get("/api/experiments")
def experiments():
    return {"runs": list_runs(), "jobs": {k: v for k, v in _jobs.items() if k.startswith("exp:")}}


@app.post("/api/experiments")
def start_experiment(req: ExperimentRequest):
    _spec(req.dataset)
    if not is_loaded(req.dataset):
        raise HTTPException(409, "dataset not imported")
    scen = req.scenarios or list(SCENARIOS)
    if any(s not in SCENARIOS for s in scen) or req.mode not in ("full", "sample"):
        raise HTTPException(400, "bad scenarios or mode")
    key = f"exp:{req.dataset}:{req.mode}:{req.seed}"
    if _jobs.get(key, {}).get("state") == "running":
        return {"job": key, **_jobs[key]}

    def work():
        try:
            res = run_experiment(req.dataset, scenarios=scen, mode=req.mode, seed=req.seed,
                                 progress=lambda m: _set_job(key, message=m))
            _set_job(key, state="done", run_id=res["meta"]["run_id"], message="done")
        except Exception as e:
            _set_job(key, state="failed", message=repr(e))

    _set_job(key, state="running", message="starting", run_id=None)
    threading.Thread(target=work, daemon=True).start()
    return {"job": key, **_jobs[key]}


@app.get("/api/experiments/{run_id}")
def experiment(run_id: str):
    try:
        return load_run(run_id)
    except FileNotFoundError:
        raise HTTPException(404, "run not found")


_dist = config.ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
