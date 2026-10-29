"""Dataset Lab: upload an arbitrary CSV, map its columns, validate it, and run the PRIMARY alert model
(IT2FIS on reliability-fused inputs) on it using the exact same calibration / reliability / fuzzy-inference
code as the three curated datasets (`calibration.py`, `systems.Pipeline`, `metrics.py`). Nothing here
re-implements model maths; it only adapts an unknown CSV into the standard schema those modules expect
(`t_hours`, `timestamp`, `level`, `rain__<sensor>`, `step_ok`, `rise_rate`) and calls them.

STATELESS BY DESIGN. Every function here takes the CSV content as an explicit argument and returns its
result; nothing is cached or stored server-side between calls. This is required, not a style choice: the
hosted demo runs on Vercel Functions, which (like any Lambda-based serverless platform) do not guarantee
that two separate HTTP requests land on the same process, so a module-level "remember the last upload"
dict (the previous design) could intermittently 404 a user's own upload between the preview, mapping and
analysis steps. Nothing here ever touches disk either way (data/raw, data/processed, results/ are never
written to), so none of this can overwrite or be confused with the curated datasets, independent of the
read-only flag -- that is why Dataset Lab's endpoints do not call `_writable()`.

Vercel also hard-caps a function's request AND response body at 4.5 MB (undocumented in the project
README; confirmed against Vercel's own Functions limits page). `MAX_UPLOAD_BYTES` is sized well under
that for the hosted demo so the CSV content can safely round-trip through the mapping/analysis requests
as plain text; a larger limit is used for local development, which has no such constraint.

Scope, by design (student project, limited time before review):
  - CSV only (no Excel): kept out to avoid shipping an unreliable parser under time pressure.
  - Only the PRIMARY model (IT2FIS, reliability-fused) is fitted and run. PSO-tuned IT2FIS, SOM and ANFIS all
    need iterative training/validation search (the README times PSO+SOM+ANFIS fitting in minutes on Bengalas);
    running all four for an ad-hoc upload inside one HTTP request is not attempted here. The offline
    `scripts/import_dataset.py` + `scripts/compare_models.py` workflow remains the way to get the full
    4-model comparison, after a dataset is reviewed and added to `datasets/registry.py`.
  - Classification metrics (precision/recall/F1) are only ever computed from the PROXY label rule already
    used for the curated datasets (future observed level vs. train-split thresholds) -- never invented, and
    reported as undefined (None) exactly like the existing datasets when the test split has no alert.
"""
from __future__ import annotations

import io
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .. import config
from ..calibration import RunConfig, split
from ..metrics import evaluate
from ..suite import horizons as candidate_horizons
from ..systems import Pipeline, proxy_labels
from . import loaders

# 3 MB keeps a round-tripped CSV (content sent back to the client, then resent on /map and /analyze as
# plain JSON text) safely under Vercel's 4.5 MB request/response body limit even after JSON-string
# escaping overhead. Local development (long-lived uvicorn process, no such platform limit) gets a much
# larger allowance.
MAX_UPLOAD_BYTES = 3 * 1024 * 1024 if config.READ_ONLY else 20 * 1024 * 1024
MAX_ROWS = 300_000
MAX_COLS = 300
MIN_ROWS_FOR_MAPPING = 60             # need at least this many rows before mapping can be validated
MIN_TRAIN_ROWS, MIN_TEST_ROWS = 40, 20
MIN_WET_ROWS, MIN_RISE_ROWS = 20, 20  # minimum usable points to calibrate rain / rise membership functions


class LabError(RuntimeError):
    pass


def _clean(o):
    """JSON-safe: NaN/inf -> None, numpy scalars/arrays -> plain Python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def _safe_name(name: str) -> str:
    name = (name or "upload.csv").replace("\\", "/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._ -]", "_", name)[:120] or "upload.csv"


def _text_to_df(csv_text: str) -> pd.DataFrame:
    """Shared, defensive CSV parse + size/shape guard. Every entry point (upload, map, analyze) calls this
    independently -- none of them trusts a size/row check that supposedly already happened in an earlier,
    separate request, because on a stateless/serverless backend there is no such thing as "earlier"."""
    size = len(csv_text.encode("utf-8"))
    if size == 0:
        raise LabError("The uploaded file is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise LabError(f"File is {size / 1e6:.1f} MB; the limit on this server is {MAX_UPLOAD_BYTES / 1e6:.1f} MB.")
    try:
        df = pd.read_csv(io.StringIO(csv_text), low_memory=False)
    except Exception as exc:  # malformed file: never execute it, never leak a path, just report it failed
        raise LabError(f"Could not read this file as CSV ({type(exc).__name__}). "
                        "Check that it is a plain comma-separated file with a header row.") from exc
    if df.shape[1] == 0 or df.shape[0] == 0:
        raise LabError("No data rows or columns were found in this file.")
    if df.shape[1] > MAX_COLS:
        raise LabError(f"{df.shape[1]} columns found; the limit is {MAX_COLS}.")
    if df.shape[0] > MAX_ROWS:
        raise LabError(f"{df.shape[0]:,} rows found; the limit is {MAX_ROWS:,}. "
                        "Trim the file and re-upload rather than uploading a sample (so nothing is silently dropped).")
    df.columns = [str(c) for c in df.columns]
    return df


# --------------------------------------------------------------------------- upload + raw preview

def preview_from_bytes(filename: str, data: bytes) -> dict:
    """Validate + preview an uploaded file. Returns the decoded `csv_text` so the caller (the frontend) can
    resend it on /map and /analyze -- this function stores nothing server-side."""
    safe = _safe_name(filename)
    if not safe.lower().endswith(".csv"):
        raise LabError(f"Only .csv files are supported in this version ('{safe}' was not recognised as CSV). "
                        "Export your spreadsheet as CSV and try again.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise LabError(f"File is {len(data) / 1e6:.1f} MB; the limit on this server is {MAX_UPLOAD_BYTES / 1e6:.1f} MB.")
    csv_text = data.decode("utf-8", errors="replace")
    df = _text_to_df(csv_text)
    return {"filename": safe, "csv_text": csv_text, **_preview_fields(df)}


def _column_profile(s: pd.Series) -> dict:
    n = len(s)
    missing = int(s.isna().sum())
    num = pd.to_numeric(s, errors="coerce")
    numeric_rate = float((num.notna() & s.notna()).sum() / max(n - missing, 1)) if n > missing else 0.0
    ts = pd.to_datetime(s, errors="coerce") if s.dtype == object or numeric_rate < 0.5 else None
    datetime_rate = float(ts.notna().sum() / max(n - missing, 1)) if ts is not None and n > missing else 0.0
    return {"missing": missing, "numeric_rate": round(numeric_rate, 3), "datetime_rate": round(datetime_rate, 3),
            "sample": [None if pd.isna(v) else str(v)[:60] for v in s.head(3).tolist()]}


def _preview_fields(df: pd.DataFrame) -> dict:
    profiles = {c: _column_profile(df[c]) for c in df.columns}
    n = len(df)
    duplicate_rows = int(df.duplicated().sum())
    head = df.head(10)
    rows = [{c: (None if pd.isna(v) else str(v)[:80]) for c, v in rec.items()} for rec in head.to_dict("records")]

    def best(pred, key):
        cands = [c for c in df.columns if pred(c, profiles[c])]
        return sorted(cands, key=lambda c: -profiles[c][key])

    ts_name_hit = lambda c, p: p["datetime_rate"] > 0.9 and re.search(r"time|date|stamp", c, re.I)  # noqa: E731
    ts_any = lambda c, p: p["datetime_rate"] > 0.9  # noqa: E731
    level_name_hit = lambda c, p: p["numeric_rate"] > 0.9 and re.search(r"level|stage|wl|gauge\b|height", c, re.I)  # noqa: E731
    rain_name_hit = lambda c, p: p["numeric_rate"] > 0.9 and re.search(r"rain|precip|rr\b", c, re.I)  # noqa: E731
    ts_cands = best(ts_name_hit, "datetime_rate") or best(ts_any, "datetime_rate")
    level_cands = best(level_name_hit, "numeric_rate")
    rain_cands = best(rain_name_hit, "numeric_rate")
    suggested = {"timestamp": ts_cands[0] if ts_cands else None,
                 "level": level_cands[0] if level_cands else None,
                 "rain": rain_cands[:3]}

    return {"rows": n, "columns": list(df.columns), "column_profiles": _clean(profiles),
            "duplicate_rows": duplicate_rows, "sample_rows": rows, "suggested_mapping": suggested,
            "note": "This is a RAW preview of the file as uploaded. Nothing has been validated, cleaned or "
                    "analysed yet -- confirm or correct the suggested column mapping below, then validate.",
            "min_rows_required": MIN_ROWS_FOR_MAPPING, "enough_rows": n >= MIN_ROWS_FOR_MAPPING}


# --------------------------------------------------------------------------- column mapping + validation

def _build_standard(raw: pd.DataFrame, timestamp: str | None, level: str, rain: list[str],
                    step_hours: float | None) -> dict:
    """Pure function: raw DataFrame + mapping in, {errors/warnings/quality/std_df/cfg_info} out. Called by
    both `validate_mapping` (client-facing preview of the result) and `analyze` (which needs the same
    standardised frame to run the model on) -- each call re-derives everything from scratch, by design."""
    errors: list[str] = []
    warnings: list[str] = []

    for col, label in [(level, "water level"), *[(c, "rainfall") for c in rain]]:
        if col not in raw.columns:
            errors.append(f"Column '{col}' mapped as {label} was not found in the file.")
    if timestamp is not None and timestamp not in raw.columns:
        errors.append(f"Column '{timestamp}' mapped as timestamp was not found in the file.")
    if not rain:
        errors.append("Map at least one rainfall column -- the reliability layer and the fuzzy model both "
                       "require a rain input (this matches every curated dataset in this project).")
    if timestamp is None and not step_hours:
        errors.append("Either map a timestamp column, or give a fixed sampling interval in hours -- the "
                       "system needs to know the time step between readings.")
    if step_hours is not None and step_hours <= 0:
        errors.append("The sampling interval must be a positive number of hours.")
    if len(raw) < MIN_ROWS_FOR_MAPPING:
        errors.append(f"Only {len(raw)} rows in the file; at least {MIN_ROWS_FOR_MAPPING} are needed to "
                       "calibrate anything meaningfully.")
    if errors:
        return {"ready": False, "errors": errors, "warnings": warnings, "std_df": None, "cfg_info": None}

    n0 = len(raw)
    level_num = pd.to_numeric(raw[level], errors="coerce")
    level_missing_original = int(raw[level].isna().sum())
    level_non_numeric = int(level_num.isna().sum() - level_missing_original)

    rain_cols, rain_report = {}, {}
    for c in rain:
        v = pd.to_numeric(raw[c], errors="coerce")
        missing0 = int(raw[c].isna().sum())
        non_numeric = int(v.isna().sum() - missing0)
        negative = int((v < 0).sum())
        rain_cols[c] = v
        rain_report[c] = {"missing_in_file": missing0, "non_numeric_values": non_numeric, "negative_values": negative}

    dropped_unparseable_timestamps = 0
    dropped_duplicate_timestamps = 0
    reordered = False
    if timestamp is not None:
        ts = pd.to_datetime(raw[timestamp], errors="coerce")
        bad = ts.isna()
        dropped_unparseable_timestamps = int(bad.sum())
        if dropped_unparseable_timestamps > 0.2 * n0:
            return {"ready": False, "warnings": warnings, "std_df": None, "cfg_info": None, "errors": [
                f"{dropped_unparseable_timestamps} of {n0} values in '{timestamp}' could not be parsed as a "
                "date/time (more than 20%). Check the column mapping or the date format in the file."]}
        if dropped_unparseable_timestamps:
            warnings.append(f"{dropped_unparseable_timestamps} rows had an unparseable timestamp and were "
                             "excluded (not imputed).")
        keep = ~bad
        order = np.argsort(ts[keep].to_numpy(), kind="stable")
        if not np.all(np.diff(order) >= 0):
            reordered = True
            warnings.append("Rows were not in chronological order and have been sorted by timestamp.")
        ts_k = ts[keep].to_numpy()[order]
        lvl_k = level_num[keep].to_numpy()[order]
        rain_k = {c: v[keep].to_numpy()[order] for c, v in rain_cols.items()}
        dup = pd.Series(ts_k).duplicated()
        dropped_duplicate_timestamps = int(dup.sum())
        if dropped_duplicate_timestamps:
            warnings.append(f"{dropped_duplicate_timestamps} duplicate timestamps found; the first occurrence "
                             "of each was kept.")
        keep2 = ~dup.to_numpy()
        ts_k, lvl_k = ts_k[keep2], lvl_k[keep2]
        rain_k = {c: v[keep2] for c, v in rain_k.items()}
        ts_series = pd.Series(ts_k)
        t_hours = (ts_series - ts_series.iloc[0]).dt.total_seconds().to_numpy() / 3600.0
        dt = np.diff(t_hours)
        dt_pos = dt[dt > 0]
        nominal = float(np.median(dt_pos)) if len(dt_pos) else (step_hours or 1.0)
        if step_hours and abs(step_hours - nominal) > 0.05 * max(step_hours, nominal):
            warnings.append(f"The given sampling interval ({step_hours:g} h) does not match the interval "
                             f"detected from timestamps ({nominal:g} h); the detected value was used.")
        step_hours_final = round(nominal, 6)
        timestamp_strs = ts_series.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
    else:
        step_hours_final = float(step_hours)
        n = len(level_num)
        t_hours = np.arange(n) * step_hours_final
        lvl_k = level_num.to_numpy()
        rain_k = {c: v.to_numpy() for c, v in rain_cols.items()}
        timestamp_strs = np.array([""] * n)

    std = pd.DataFrame({"t_hours": t_hours, "timestamp": timestamp_strs, "level": lvl_k,
                        **{f"rain__{c}": v for c, v in rain_k.items()}})
    std = loaders.add_derived(std, step_hours_final)

    lo = float(np.nanmin(std["level"])) if std["level"].notna().any() else 0.0
    hi = float(np.nanmax(std["level"])) if std["level"].notna().any() else 1.0
    margin = max(0.5, 0.2 * (hi - lo))
    level_bounds = (round(lo - margin, 3), round(hi + margin, 3))

    fake_spec = SimpleNamespace(id="upload", step_hours=step_hours_final, level_bounds_m=level_bounds)
    info = {"column_mapping": {"timestamp": timestamp, "level": level, "rain": rain},
            "level_missing_in_source_file": level_missing_original, "level_non_numeric_values": level_non_numeric,
            "rain_columns": rain_report, "dropped_unparseable_timestamps": dropped_unparseable_timestamps,
            "dropped_duplicate_timestamps": dropped_duplicate_timestamps, "rows_reordered_to_be_chronological": reordered,
            "level_bounds_m_is_data_derived": True,
            "level_bounds_m_note": "No physical sensor range is known for this upload; bounds are the "
                                    "observed min/max plus a 20% margin, NOT a verified physical limit."}
    quality = loaders.quality_report(std, fake_spec, info)
    quality["level_bounds_source"] = "data-derived (unverified)"

    if level_non_numeric:
        warnings.append(f"{level_non_numeric} water-level values were not numeric and are treated as missing.")
    for c, r in rain_report.items():
        if r["non_numeric_values"]:
            warnings.append(f"{r['non_numeric_values']} values in rainfall column '{c}' were not numeric and are treated as missing.")
        if r["negative_values"]:
            warnings.append(f"{r['negative_values']} negative values in rainfall column '{c}' (kept as-is; not corrected).")
    pct_nominal = round(100.0 * quality["steps_at_nominal"] / max(quality["rows"] - 1, 1), 1)
    if pct_nominal < 50:
        warnings.append(f"Only {pct_nominal}% of steps are exactly one nominal interval apart -- this series has many gaps.")

    sup = candidate_horizons(step_hours_final)
    if not sup["supported"]:
        errors.append(f"No usable forecast horizon at a {step_hours_final:g} h sampling interval "
                       "(need a whole number of steps at 30 min, 1 h or 3 h).")
        return {"ready": False, "errors": errors, "warnings": warnings, "quality": quality, "std_df": None, "cfg_info": None}

    cfg_info = {"step_hours": step_hours_final, "rain_sensors": tuple(rain), "level_bounds": level_bounds,
                "rain_window_steps": max(1, round(2.0 / step_hours_final)),
                "horizons": sup["supported"], "unsupported_horizons": sup["unsupported"]}
    return {"ready": True, "errors": [], "warnings": warnings, "quality": quality, "std_df": std, "cfg_info": cfg_info}


def validate_mapping(csv_text: str, timestamp: str | None, level: str, rain: list[str],
                     step_hours: float | None) -> dict:
    """Client-facing preview of `_build_standard`'s result (no model run yet)."""
    raw = _text_to_df(csv_text)
    r = _build_standard(raw, timestamp, level, rain, step_hours)
    out = {"ready": r["ready"], "errors": r["errors"], "warnings": r["warnings"]}
    if r.get("quality") is not None:
        out["quality"] = _clean(r["quality"])
    if r["ready"]:
        sup = r["cfg_info"]["horizons"]
        out.update(horizons={str(k): v for k, v in sup.items()},
                   unsupported_horizons=r["cfg_info"]["unsupported_horizons"], rows=len(r["std_df"]))
    return out


# --------------------------------------------------------------------------- analysis (primary model only)

def analyze(csv_text: str, filename: str, timestamp: str | None, level: str, rain: list[str],
           step_hours: float | None, horizon_steps: int | None) -> dict:
    raw = _text_to_df(csv_text)
    r = _build_standard(raw, timestamp, level, rain, step_hours)
    if not r["ready"]:
        raise LabError("; ".join(r["errors"]) or "This CSV and column mapping did not validate.")
    df, cfg_info = r["std_df"], r["cfg_info"]
    sup = cfg_info["horizons"]
    h = horizon_steps if horizon_steps in sup else min(sup)
    cfg = RunConfig(dataset="upload", step_hours=cfg_info["step_hours"], horizon_steps=h,
                    rain_window_steps=cfg_info["rain_window_steps"], rain_sensors=cfg_info["rain_sensors"],
                    level_bounds=cfg_info["level_bounds"], train_fraction=config.TRAIN_FRACTION)
    train, test = split(df, cfg.train_fraction)
    if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
        raise LabError(f"Not enough rows for a reliable chronological train/test split (have {len(df)} rows; "
                        f"need at least {MIN_TRAIN_ROWS + MIN_TEST_ROWS}).")
    wet_rows = int((train["rain__" + cfg.rain_sensors[0]].to_numpy() > 0).sum()) if cfg.rain_sensors else 0
    for s in cfg.rain_sensors[1:]:
        wet_rows = max(wet_rows, int((train[f"rain__{s}"].to_numpy() > 0).sum()))
    if wet_rows < MIN_WET_ROWS:
        raise LabError(f"Only {wet_rows} training rows have rain > 0; need at least {MIN_WET_ROWS} to calibrate "
                        "the rainfall membership functions. Analysis was not run (rather than produce unreliable fuzzy sets).")
    if int(train["rise_rate"].notna().sum()) < MIN_RISE_ROWS:
        raise LabError("Too few consecutive-step readings in the training split to calibrate the rate-of-rise "
                        "membership functions. Analysis was not run.")

    pipe = Pipeline.fit(train, cfg)
    out = pipe.run_proposed(test)
    raw_out = pipe.run_it2(test)
    labels = proxy_labels(test, pipe.cal, h)
    m = evaluate(labels, out["cls"])
    m_raw = evaluate(labels, raw_out["cls"])
    has_alert = bool(np.any(~np.isnan(labels) & (labels >= config.ALERT_CLASS)))

    i = len(test) - 1
    ts_i = test["timestamp"].iloc[i] or f"test row {i + 1} of {len(test)} (no timestamp mapped)"
    cls_i = int(out["cls"][i])
    sensors = []
    lv = out["rel"]["level"]
    sensors.append({"sensor": "water level", "reliability": round(float(lv["r"][i]), 3),
                    "flags": [k for k in ("missing", "out_of_range", "spike", "stuck", "noisy") if bool(lv[k][i])]})
    for s in cfg.rain_sensors:
        rs = out["rel"]["rain"][s]
        sensors.append({"sensor": f"rain ({s})", "reliability": round(float(rs["r"][i]), 3),
                        "flags": [k for k in ("missing", "out_of_range", "stuck", "inconsistent") if bool(rs[k][i])]})
    status, U = str(out["status"][i]), float(out["U"][i])
    score_i = float(out["score"][i])
    risk = config.RISK_CLASSES[cls_i]
    ADVICE = ["No action needed.", "Keep an eye on the river.", "Be prepared: the river is clearly above normal.",
              "Flood threshold exceeded or expected: follow local warnings.", "Severe: level far above the flood threshold."]
    reasons = [f"This is the alert for ONE specific reading -- the most recent one in your test split "
               f"({ts_i}, row {i + 1} of {len(test)}) -- not a summary of the whole file.",
               f"Water level {out['fused']['level_c'][i]:.3f}, "
               f"rate of rise {out['fused']['rise_point'][i]:+.3f}/h, rainfall (fused, windowed) "
               f"{out['fused']['rain_point'][i]:.3f}.",
               f"IT2FIS risk interval [{out['yl'][i]:.1f}, {out['yr'][i]:.1f}]; midpoint {score_i:.1f}/100 "
               f"falls in the {risk} band.",
               f"Confidence: {status} (uncertainty index U = {U:.0f}/100, an index, not a probability)."]

    n_all = len(df)
    step_show = max(1, n_all // 1500)
    # _clean(): pandas .where() on an all-"" string column (no timestamp mapped) silently upcasts to
    # float64 NaN instead of keeping None, which is not JSON-serialisable -- sanitise like every other array here.
    ts_all = _clean(df["timestamp"].where(df["timestamp"] != "", None).tolist())

    return {
        "filename": filename,
        "primary_model": "it2_fis", "primary_mode": "fused",
        "models_evaluated": ["IT2FIS (primary, reliability-fused)"],
        "models_not_run": ["PSO-tuned IT2FIS", "SOM", "ANFIS"],
        "models_not_run_reason": "These three need iterative training/validation search; not run for ad-hoc "
                                 "uploads in this version. Use scripts/import_dataset.py + "
                                 "scripts/compare_models.py locally for the full 4-model comparison once this "
                                 "dataset is reviewed and registered.",
        "horizon": h, "horizon_label": sup[h], "step_hours": cfg.step_hours,
        "rain_window_hours": cfg.rain_window_steps * cfg.step_hours,
        "thresholds": pipe.cal.thresholds, "risk_classes": config.RISK_CLASSES,
        "thresholds_note": "Computed from THIS upload's own training-split statistics (mean + k x standard "
                            "deviation of its water level), not a universal flood level -- see Dataset & Sensor "
                            "Reliability for the full rule. If your file has very little natural variability, "
                            "these bands sit close together and ordinary noise can cross them.",
        "train_level_mean": round(float(pipe.cal.level_mean), 4), "train_level_sd": round(float(pipe.cal.level_sd), 4),
        "level_bounds": list(cfg_info["level_bounds"]), "level_bounds_is_data_derived": True,
        "split": {"total_rows": n_all, "train_rows": len(train), "test_rows": len(test)},
        "has_alert_in_test": has_alert,
        "metrics": _clean({"fused": m, "raw": m_raw}) if has_alert else None,
        "metrics_note": None if has_alert else "No High/Critical level occurs in this uploaded test split, so "
                         "alert precision/recall/F1 are undefined (not computed) -- exactly as already done for "
                         "the curated CWC Chindnar dataset when its test period has no alert.",
        "primary": {"risk_class": cls_i, "risk": risk, "alert": cls_i >= config.ALERT_CLASS,
                    "score": round(score_i, 1), "status": status, "U": round(U, 1), "advice": ADVICE[cls_i],
                    "reasons": reasons, "sensors": sensors, "headline": f"{risk.upper()} -- IT2FIS, {sup[h]} ahead",
                    "timestamp": ts_i, "row_index": i + 1, "test_rows": len(test)},
        "series": {
            "timestamp": ts_all[::step_show],
            "level": _clean(df["level"].to_numpy()[::step_show]),
            "rain": {s: _clean(df[f"rain__{s}"].to_numpy()[::step_show]) for s in cfg.rain_sensors},
            "rise": _clean(df["rise_rate"].to_numpy()[::step_show]),
            "train_end_index": len(train),
            "downsample_step": step_show,
        },
        "disclaimer": "Experimental output from an uploaded dataset. This is a research prototype, not an "
                       "official flood warning. Labels are proxy labels (future level vs. statistical thresholds "
                       "from this upload's own training split), not observed flood events.",
    }
