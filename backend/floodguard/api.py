"""HydroSentinel AI FastAPI backend: datasets, interactive assessment, historical replay, model comparison.

Run:  uvicorn floodguard.api:app --port 8000   (from backend/)
The built frontend (frontend/dist) is served at / when present.
"""
from __future__ import annotations

import json
import threading
import traceback
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import MODEL_VERSION, config
from .calibration import split
from .datasets import DATASETS, NOT_INTEGRATED, DatasetError, import_dataset, is_loaded, load_processed, load_quality
from .datasets import custom as lab
from .datasets.loaders import manual_instructions
from .experiment import _clean_json, list_runs, load_run, make_config, run_experiment
from .faults import SCENARIO_DOC, SCENARIOS, inject
from .fuzzy import INPUTS, OUTPUT_TERMS, TERMS
from .systems import Pipeline, future_level, proxy_labels
from . import alerts, demo
from .references import DATA_SOURCES_CHECKED, REFERENCES
from .suite import MODEL_NAMES, MODES, RISK_MODELS, ModelSuite, horizons

app = FastAPI(title="HydroSentinel AI", version=MODEL_VERSION)
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


def _writable():
    if config.READ_ONLY:
        raise HTTPException(403, "This hosted demo is read-only. Import datasets and run new "
                                 "experiments on a local install (see README).")


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
    return {"ok": True, "model_version": MODEL_VERSION, "read_only": config.READ_ONLY,
            "lab_max_upload_mb": round(lab.MAX_UPLOAD_BYTES / 1e6, 1)}


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
    _writable()
    key = f"import:{ds_id}"
    if _jobs.get(key, {}).get("state") == "running":
        return _jobs[key]

    def work():
        try:
            rep = import_dataset(ds_id, progress=lambda m: _set_job(key, message=m))
            _pipeline.cache_clear()
            _suite.cache_clear()
            _suite_replay.cache_clear()
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
            # rise and rain: maximum within each plotted bucket, so short peaks survive the down-sampling
            "rise": _num(df["rise_rate"].rolling(step, min_periods=1).max().iloc[::step]),
            "rain": {c[6:]: _num(df[c].rolling(step, min_periods=1).max().iloc[::step]) for c in rain_cols}}


@app.get("/api/datasets/{ds_id}/preview")
def preview(ds_id: str, rows: int = 12):
    """First rows of the processed file + per-column summary (computed from the file on disk)."""
    spec = _spec(ds_id)
    if not is_loaded(ds_id):
        raise HTTPException(409, "dataset not imported")
    df = load_processed(ds_id)
    cfg = make_config(ds_id)
    train, test = split(df, cfg.train_fraction)
    n_fit = int(0.8 * len(train))
    cols = [c for c in df.columns if c != "step_ok"]
    head = df[cols].head(max(1, min(rows, 50)))
    summary = {}
    for c in cols:
        if c == "timestamp":
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        summary[c] = {"missing": int(x.isna().sum()), "min": float(x.min()), "max": float(x.max()),
                      "mean": float(x.mean()), "zeros_pct": round(float((x == 0).mean() * 100), 2)}

    def ts(d, i):
        return d["timestamp"].iloc[i] or f"index {d.index[i]}"

    return {"columns": cols, "rows": json.loads(head.to_json(orient="records")), "summary": summary,
            "n": len(df), "step_hours": spec.step_hours, "rain_window_hours": cfg.rain_window_steps * cfg.step_hours,
            "split": {"fit": {"rows": n_fit, "first": ts(train, 0), "last": ts(train, n_fit - 1)},
                      "validation": {"rows": len(train) - n_fit, "first": ts(train, n_fit),
                                     "last": ts(train, len(train) - 1)},
                      "test": {"rows": len(test), "first": test["timestamp"].iloc[0] or f"index {len(train)}",
                               "last": test["timestamp"].iloc[-1] or f"index {len(df) - 1}"}},
            "horizons": horizons(spec.step_hours)}


# --------------------------------------------------------------------------- model

@lru_cache(maxsize=4)
def _pipeline(ds_id: str):
    if not is_loaded(ds_id):
        raise HTTPException(409, f"dataset '{ds_id}' not imported yet")
    df = load_processed(ds_id)
    cfg = make_config(ds_id)
    train, test = split(df, cfg.train_fraction)
    return Pipeline.fit(train, cfg), test


def _saved(ds_id: str) -> dict:
    f = config.RESULTS_DIR / f"comparison_{ds_id}.json"
    return json.loads(f.read_text()).get("fitted", {}) if f.exists() else {}


@lru_cache(maxsize=2)
def _suite(ds_id: str) -> ModelSuite:
    """Fitted model suite. The validation searches stored with the comparison run (PSO genes, selected SOM grid and
    ANFIS ridge) are reused, so the served models are the evaluated ones; all fitting is repeated on train only."""
    if not is_loaded(ds_id):
        raise HTTPException(409, f"dataset '{ds_id}' not imported yet")
    df = load_processed(ds_id)
    cfg = make_config(ds_id)
    train, _ = split(df, cfg.train_fraction)
    return ModelSuite.fit(train, cfg, saved=_saved(ds_id))


def _curves(fis, cal) -> dict:
    curves = {}
    for name in INPUTS:
        lo, hi = cal.bounds[name]
        a = cal.anchors[name]
        span = a[-1] - a[0]
        xs = np.linspace(max(lo, a[0] - 0.6 * span), min(hi, a[-1] + 0.6 * span), 240)
        curves[name] = {"x": _num(xs), "terms": [
            {"term": TERMS[name][i], "lower": _num(m.lower(xs)), "upper": _num(m.upper(xs)),
             "type1": _num(m.embedded(xs))} for i, m in enumerate(fis.mfs[name])]}
    return curves


@app.get("/api/model/{ds_id}")
def model(ds_id: str, horizon: int | None = None):
    """IT2FIS description and membership curves; with `horizon`, also the PSO-tuned curves for that horizon."""
    _spec(ds_id)
    pipe, _ = _pipeline(ds_id)
    out = {**pipe.describe(), "curves": _curves(pipe.fis, pipe.cal), "output_terms": OUTPUT_TERMS}
    if horizon is not None:
        suite = _suite(ds_id)
        if horizon not in suite.horizons:
            raise HTTPException(400, f"unsupported horizon {horizon}; supported: {list(suite.horizons)}")
        out["tuned_curves"] = _curves(suite.pso[horizon].fis, suite.pso[horizon].cal)
    return out


# --------------------------------------------------------------------------- historical replay

@lru_cache(maxsize=24)
def _suite_replay(ds_id: str, scenario: str, seed: int):
    suite = _suite(ds_id)
    _, test = _pipeline(ds_id)
    corrupted, gt = inject(test, scenario, suite.pipe.cal, seed)
    return test, corrupted, gt, suite.run_shared(corrupted)


def _check(suite, horizon, mode, model):
    h = min(suite.horizons) if horizon is None else horizon
    errs = []
    if h not in suite.horizons:
        errs.append(f"unsupported horizon {h}; supported: {list(suite.horizons)}")
    if mode not in MODES:
        errs.append(f"unknown input mode {mode}")
    if model not in RISK_MODELS:
        errs.append(f"unknown model {model}")
    return h, errs


@app.get("/api/scenarios")
def scenarios():
    return {"scenarios": [{"id": s, "description": SCENARIO_DOC[s]} for s in SCENARIOS]}


@app.get("/api/episodes/{ds_id}")
def episodes(ds_id: str, limit: int = 40):
    """High-water episodes in the held-out test period (runs of observed level >= the High threshold).
    When the test period never reaches High, the highest level that does occur is used instead and reported."""
    _spec(ds_id)
    pipe, test = _pipeline(ds_id)
    cls = np.digitize(test["level"].to_numpy(float), pipe.cal.thresholds)
    min_class = min(config.ALERT_CLASS, int(cls.max()))
    if min_class == 0:
        return {"episodes": [], "total": 0, "episode_class": None}
    hi = cls >= min_class
    starts = np.flatnonzero(hi & ~np.r_[False, hi[:-1]])
    out = []
    for s0 in starts:
        e = s0
        while e + 1 < len(hi) and hi[e + 1]:
            e += 1
        seg = test["level"].iloc[s0:e + 1]
        out.append({"start": int(s0), "end": int(e), "timestamp": test["timestamp"].iloc[s0] or f"day {s0}",
                    "peak_level": float(seg.max()), "peak_class": config.RISK_CLASSES[int(cls[s0:e + 1].max())],
                    "steps": int(e - s0 + 1)})
    out.sort(key=lambda r: -r["peak_level"])
    return {"episodes": out[:limit], "total": len(out), "episode_class": config.RISK_CLASSES[min_class]}


@app.get("/api/replay/{ds_id}")
def replay(ds_id: str, scenario: str = "clean", offset: int = 0, limit: int = 96, seed: int = config.DEFAULT_SEED,
           horizon: int | None = None, mode: str = "raw", model: str = "it2_fis", at: int | None = None):
    """A window of the held-out test period with every model's alert level per step, the proxy label and the
    explanation of every model's alert at step `at` (default: the last step of the window)."""
    _spec(ds_id)
    if scenario not in SCENARIOS:
        raise HTTPException(400, "unknown scenario")
    suite = _suite(ds_id)
    h, errs = _check(suite, horizon, mode, model)
    if errs:
        raise HTTPException(400, "; ".join(errs))
    test, corrupted, gt, sh = _suite_replay(ds_id, scenario, seed)
    pipe = suite.pipe
    hz = suite.run_horizon(sh, h)
    labels = proxy_labels(test, pipe.cal, h)
    fut = future_level(test, h, pipe.cal.cfg.step_hours)
    n = len(test)
    offset = int(np.clip(offset, 0, max(n - 1, 0)))
    sl = slice(offset, min(n, offset + max(1, min(limit, 2000))))
    i = sl.stop - 1 if at is None else int(np.clip(at, sl.start, sl.stop - 1))
    p = sh["proposed"]
    f, rel = p["fused"], p["rel"]
    sensors = list(pipe.cal.cfg.rain_sensors)
    any_fault = np.zeros(n, bool)
    for m in gt.values():
        any_fault |= m
    X = sh["X"][mode]
    return {
        "n": n, "offset": offset, "at": i, "scenario": scenario, "seed": seed, "horizon": h,
        "horizon_label": suite.horizons[h], "horizons": {str(k): v for k, v in suite.horizons.items()},
        "mode": mode, "model": model,
        "note": "Replay of the held-out historical test split (not a live feed). Labels are PROXY labels.",
        "thresholds": pipe.cal.thresholds, "risk_classes": config.RISK_CLASSES,
        "timestamp": [t or f"day {offset + k}" for k, t in enumerate(test["timestamp"].iloc[sl])],
        "level_original": _num(test["level"].to_numpy()[sl]), "level_sensor": _num(corrupted["level"].to_numpy()[sl]),
        "level_fused": _num(f["level_c"][sl]), "level_lo": _num(f["level"][0][sl]), "level_hi": _num(f["level"][1][sl]),
        "level_used": _num(X[sl, 0]), "rise_used": _num(X[sl, 1]), "rain_used": _num(X[sl, 2]),
        "rain_sensor": {s: _num(corrupted[f"rain__{s}"].to_numpy()[sl]) for s in sensors},
        "rain_original": {s: _num(test[f"rain__{s}"].to_numpy()[sl]) for s in sensors},
        "r_level": _num(f["r_level"][sl]), "r_rain": {s: _num(rel["rain"][s]["r"][sl]) for s in sensors},
        "flags": {k: _num(rel["level"][k][sl]) for k in ("missing", "out_of_range", "spike", "stuck", "noisy")},
        "status": p["status"][sl].tolist(), "U": _num(p["U"][sl]),
        "models": {k: _num(hz["cls"][mode][k][sl]) for k in RISK_MODELS}, "model_names": MODEL_NAMES,
        "proxy_label": _num(labels[sl]), "future_level": _num(fut[sl]), "injected_fault": _num(any_fault[sl]),
        "explanations": {m: alerts.explain(suite, sh, hz, h, i, m, mode) for m in RISK_MODELS},
        "primary": alerts.explain(suite, sh, hz, h, i, alerts.PRIMARY_MODEL, alerts.PRIMARY_MODE),
        "outcome": {"timestamp": test["timestamp"].iloc[i] or f"day {i}", "future_level": _num(fut[i:i + 1])[0],
                    "proxy_label": _num(labels[i:i + 1])[0],
                    "proxy_class": None if np.isnan(labels[i]) else config.RISK_CLASSES[int(labels[i])]},
    }


# --------------------------------------------------------------------------- interactive assessment

class AssessRequest(BaseModel):
    level: float
    rise: float
    rain: float
    horizon: int | None = None
    scenario: str = "normal"
    mode: str = "raw"


@app.get("/api/assess/{ds_id}/options")
def assess_options(ds_id: str):
    spec = _spec(ds_id)
    suite = _suite(ds_id)
    cal = suite.pipe.cal
    win = cal.cfg.rain_window_steps * cal.cfg.step_hours
    return {"scenarios": demo.SENSOR_SCENARIOS, "models": {m: MODEL_NAMES[m] for m in RISK_MODELS},
            "horizons": {str(k): v for k, v in suite.horizons.items()},
            "unsupported_horizons": horizons(cal.cfg.step_hours)["unsupported"],
            "default_horizon": min(suite.horizons), "anchors": cal.anchors, "thresholds": cal.thresholds,
            "bounds": {"level": list(cal.cfg.level_bounds), "rise": list(cal.bounds["rise"]),
                       "rain": [0.0, cal.bounds["rain"][1]]},
            "rain_window_hours": win, "step_hours": cal.cfg.step_hours, "risk_classes": config.RISK_CLASSES,
            "units": {"level": "m", "rise": "m/h",
                      "rain": f"accumulated over the last {win:g} h ({spec.rain_unit})"}}


@app.post("/api/assess/{ds_id}")
def assess(ds_id: str, req: AssessRequest):
    _spec(ds_id)
    suite = _suite(ds_id)
    h, errs = _check(suite, req.horizon, req.mode, "it2_fis")
    errs = demo.validate(suite.pipe, req.level, req.rise, req.rain) + errs
    if req.scenario not in demo.SENSOR_SCENARIOS:
        errs.append(f"unknown sensor scenario {req.scenario}")
    if errs:
        raise HTTPException(422, "; ".join(errs))
    hist, true_level = demo.build_history(suite.pipe, req.level, req.rise, req.rain, req.scenario)
    sh = suite.run_shared(hist)
    hz = suite.run_horizon(sh, h)
    i = len(hist) - 1
    p = sh["proposed"]
    return {
        "kind": "HYPOTHETICAL DEMONSTRATION: constructed input history, not an evaluated prediction",
        "horizon": h, "horizon_label": suite.horizons[h], "scenario": req.scenario, "mode": req.mode,
        "scenario_description": demo.SENSOR_SCENARIOS[req.scenario],
        "explanations": {m: alerts.explain(suite, sh, hz, h, i, m, req.mode, demo=True) for m in RISK_MODELS},
        "primary": alerts.explain(suite, sh, hz, h, i, alerts.PRIMARY_MODEL, alerts.PRIMARY_MODE, demo=True),
        "history": {"level_true": _num(true_level), "level_sensor": _num(hist["level"].to_numpy()),
                    "level_fused": _num(p["fused"]["level_c"]),
                    "rain_sensors": {s: _num(hist[f"rain__{s}"].to_numpy()) for s in suite.pipe.cal.cfg.rain_sensors}},
    }


# --------------------------------------------------------------------------- dataset lab (upload your own CSV)
# Stateless by design (see custom.py's module docstring): every request carries the CSV content it needs,
# nothing is cached server-side between calls. This is required for correctness on Vercel's serverless
# architecture, not an optimisation. Lab never writes to data/raw, data/processed or results/, so it cannot
# overwrite or be confused with the three curated datasets above or their results -- that is why these
# routes do not call _writable() and are available even when the hosted demo is otherwise read-only.

class LabMapRequest(BaseModel):
    csv_text: str
    timestamp: str | None = None
    level: str
    rain: list[str]
    step_hours: float | None = None


class LabAnalyzeRequest(LabMapRequest):
    filename: str = "upload.csv"
    horizon_steps: int | None = None


@app.post("/api/lab/upload")
async def lab_upload(file: UploadFile = File(...)):
    data = await file.read()
    try:
        return lab.preview_from_bytes(file.filename or "upload.csv", data)
    except lab.LabError as e:
        raise HTTPException(422, str(e))


@app.post("/api/lab/map")
def lab_map(req: LabMapRequest):
    try:
        return lab.validate_mapping(req.csv_text, req.timestamp, req.level, req.rain, req.step_hours)
    except lab.LabError as e:
        raise HTTPException(422, str(e))


@app.post("/api/lab/analyze")
def lab_analyze(req: LabAnalyzeRequest):
    try:
        return lab.analyze(req.csv_text, req.filename, req.timestamp, req.level, req.rain,
                           req.step_hours, req.horizon_steps)
    except lab.LabError as e:
        raise HTTPException(422, str(e))


# --------------------------------------------------------------------------- references

@app.get("/api/references")
def references():
    return {"references": REFERENCES, "data_sources_checked": DATA_SOURCES_CHECKED}


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
    _writable()
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


# --------------------------------------------------------------------------- model comparison

@app.get("/api/comparison/{ds_id}")
def comparison(ds_id: str):
    _spec(ds_id)
    f = config.RESULTS_DIR / f"comparison_{ds_id}.json"
    if not f.exists():
        raise HTTPException(404, f"no comparison yet: run scripts/compare_models.py {ds_id}")
    return json.loads(f.read_text())


_dist = config.ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
