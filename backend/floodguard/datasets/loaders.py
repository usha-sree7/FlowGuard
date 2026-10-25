"""Download, validate and standardise the verified datasets.

Original files are stored untouched under data/raw/<id>/ and checked against the SHA-256 recorded
in the registry. A standardised copy is written to data/processed/<id>.csv together with a quality
report. Nothing is reported as "loaded" until the processed copy has been read back and validated.

Standard processed schema (one row per observation, chronological):
    t_hours     float  hours since the first record (relative day index * 24 for Malwathu)
    timestamp   str    original timestamp, or "" when the source has none (Malwathu)
    level       float  water level [m]
    rain__<s>   float  one column per rainfall sensor, original units (see registry)
    step_ok     bool   previous row is exactly one nominal step earlier
    rise_rate   float  (level_t - level_{t-1}) / dt [m/h]; NaN when step_ok is False
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from .registry import DATASETS, DatasetSpec

BENGALAS_MEMBER = "bacia_rio_bengalas_2019_2023_out00h30m.csv"


class DatasetError(RuntimeError):
    pass


def raw_dir(ds_id: str) -> Path:
    return config.RAW_DIR / ds_id


def processed_path(ds_id: str) -> Path:
    return config.PROCESSED_DIR / f"{ds_id}.csv"


def quality_path(ds_id: str) -> Path:
    return config.PROCESSED_DIR / f"{ds_id}.quality.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manual_instructions(spec: DatasetSpec) -> str:
    lines = [f"Automatic download failed. Download manually and place the files in {raw_dir(spec.id)}:"]
    for f in spec.files:
        lines.append(f"  - {f.url}  ->  {raw_dir(spec.id) / f.name}  (sha256 {f.sha256})")
    lines.append(f"Then run: python backend/scripts/import_dataset.py {spec.id} --no-download")
    return "\n".join(lines)


def download(spec: DatasetSpec, progress=None) -> None:
    """Fetch every remote file that is missing or fails its checksum."""
    d = raw_dir(spec.id)
    d.mkdir(parents=True, exist_ok=True)
    for rf in spec.files:
        target = d / rf.name
        if target.exists() and sha256_of(target) == rf.sha256:
            continue
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            with urllib.request.urlopen(rf.url, timeout=60) as resp, open(tmp, "wb") as out:
                done = 0
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(f"downloading {rf.name}: {done / 1e6:.1f} / {rf.size / 1e6:.1f} MB")
        except Exception as exc:  # network, HTTP, disk
            tmp.unlink(missing_ok=True)
            raise DatasetError(f"{exc}\n{manual_instructions(spec)}") from exc
        tmp.replace(target)


def verify_raw(spec: DatasetSpec) -> dict:
    out = {}
    for rf in spec.files:
        p = raw_dir(spec.id) / rf.name
        if not p.exists():
            raise DatasetError(f"Missing raw file {p}.\n{manual_instructions(spec)}")
        digest = sha256_of(p)
        if digest != rf.sha256:
            raise DatasetError(f"Checksum mismatch for {p}: {digest} != {rf.sha256}")
        out[rf.name] = digest
    return out


# --------------------------------------------------------------------------- readers (actual schemas)

def read_bengalas_raw() -> tuple[pd.DataFrame, dict]:
    with zipfile.ZipFile(raw_dir("bengalas") / "dataset.zip") as z:
        if BENGALAS_MEMBER not in z.namelist():
            raise DatasetError(f"{BENGALAS_MEMBER} not found in dataset.zip")
        with z.open(BENGALAS_MEMBER) as f:
            raw = pd.read_csv(f)
    expected = {"datahora", "chuva_conselheiro", "nivel_conselheiro", "out00h30m"}
    missing = expected - set(raw.columns)
    if missing:
        raise DatasetError(f"Bengalas schema changed; missing columns {sorted(missing)}")
    radar_cols = [c for c in raw.columns if c.startswith("precip")]
    ts = pd.to_datetime(raw["datahora"])
    out = pd.DataFrame({
        "t_hours": (ts - ts.iloc[0]).dt.total_seconds() / 3600.0,
        "timestamp": raw["datahora"].astype(str),
        "level": raw["nivel_conselheiro"].astype(float),
        "rain__gauge": raw["chuva_conselheiro"].astype(float),
        "rain__radar": raw[radar_cols].to_numpy(dtype=float).mean(axis=1),
    })
    # Validate the authors' target definition instead of assuming it.
    nxt_ok = (ts.shift(-1) - ts) == pd.Timedelta(minutes=30)
    target_match = float((raw["out00h30m"][nxt_ok] == raw["nivel_conselheiro"].shift(-1)[nxt_ok]).mean())
    info = {
        "source_member": BENGALAS_MEMBER,
        "raw_shape": list(raw.shape),
        "radar_cells": len(radar_cols),
        "radar_cell_min": float(raw[radar_cols].min().min()),
        "radar_cell_max": float(raw[radar_cols].max().max()),
        "out00h30m_equals_next_level_fraction": target_match,
    }
    return out, info


def read_malwathu_raw() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(raw_dir("malwathu") / "malwathu_1_day.csv")
    stations = list(DATASETS["malwathu"].rain_sensors)
    expected = set(stations + [s + "-1" for s in stations] + ["Kappachichiya", "Kappachichiya-1"])
    missing = expected - set(raw.columns)
    if missing:
        raise DatasetError(f"Malwathu schema changed; missing columns {sorted(missing)}")
    # Verify lag structure before reconstructing the series.
    lag_ok = {s: bool(np.allclose(raw[s].to_numpy()[:-1], raw[s + "-1"].to_numpy()[1:])) for s in stations}
    lvl_scale = raw["Kappachichiya-1"].to_numpy()[1:] / np.where(raw["Kappachichiya"].to_numpy()[:-1] == 0, np.nan,
                                                                  raw["Kappachichiya"].to_numpy()[:-1])
    if not all(lag_ok.values()):
        raise DatasetError(f"Unexpected lag structure in malwathu_1_day.csv: {lag_ok}")
    first = {s: raw[s + "-1"].iloc[0] for s in stations}
    data = {f"rain__{s}": np.concatenate([[first[s]], raw[s].to_numpy(dtype=float)]) for s in stations}
    level = np.concatenate([[raw["Kappachichiya-1"].iloc[0] / 100.0], raw["Kappachichiya"].to_numpy(dtype=float)])
    n = len(level)
    out = pd.DataFrame({"t_hours": np.arange(n) * 24.0, "timestamp": [""] * n, "level": level, **data})
    info = {
        "source_file": "malwathu_1_day.csv",
        "raw_shape": list(raw.shape),
        "lag_columns_match_previous_row": lag_ok,
        "level_lag_to_target_ratio_median": float(np.nanmedian(lvl_scale)),
        "duplicate_raw_rows_not_removed": int(raw.duplicated().sum()),
        "reconstructed_days": n,
    }
    return out, info


READERS = {"bengalas": read_bengalas_raw, "malwathu": read_malwathu_raw}


# --------------------------------------------------------------------------- standardise + quality

def add_derived(df: pd.DataFrame, step_hours: float) -> pd.DataFrame:
    df = df.copy()
    dt = df["t_hours"].diff()
    df["step_ok"] = np.isclose(dt, step_hours)
    # Rise rate only across exactly one nominal step: never interpolated over gaps.
    df["rise_rate"] = np.where(df["step_ok"], df["level"].diff() / step_hours, np.nan)
    return df


def quality_report(df: pd.DataFrame, spec: DatasetSpec, info: dict) -> dict:
    rain_cols = [c for c in df.columns if c.startswith("rain__")]
    dt = df["t_hours"].diff().dropna()
    lo, hi = spec.level_bounds_m
    invalid_level = int(((df["level"] < lo) | (df["level"] > hi)).sum())
    invalid_rain = {c: int((df[c] < 0).sum()) for c in rain_cols}
    gaps = dt[~np.isclose(dt, spec.step_hours)]
    rep = {
        "dataset": spec.id,
        "rows": int(len(df)),
        "first": df["timestamp"].iloc[0] or "day 0",
        "last": df["timestamp"].iloc[-1] or f"day {len(df) - 1}",
        "nominal_step_hours": spec.step_hours,
        "steps_at_nominal": int(np.isclose(dt, spec.step_hours).sum()),
        "gaps": int(len(gaps)),
        "largest_gap_hours": float(gaps.max()) if len(gaps) else 0.0,
        "non_increasing_steps": int((dt <= 0).sum()),
        "duplicate_timestamps": int(df["timestamp"].replace("", np.nan).dropna().duplicated().sum()),
        "missing_values": {c: int(df[c].isna().sum()) for c in ["level", *rain_cols]},
        "invalid_level_outside_bounds": invalid_level,
        "level_bounds_m (configured plausibility range, not a flood threshold)": [lo, hi],
        "invalid_negative_rain": invalid_rain,
        "rise_rate_available": int(df["rise_rate"].notna().sum()),
        "stats": {c: {k: float(v) for k, v in df[c].describe().items()} for c in ["level", "rise_rate", *rain_cols]},
        "reader_checks": info,
    }
    rep["valid_rows"] = int(len(df) - df[["level", *rain_cols]].isna().any(axis=1).sum()
                            - invalid_level - sum(invalid_rain.values()))
    return rep


def import_dataset(ds_id: str, download_files: bool = True, progress=None) -> dict:
    """Download (optional) -> verify checksum -> read actual schema -> standardise -> write processed copy
    -> read back and validate. Returns the quality report."""
    spec = DATASETS[ds_id]
    config.ensure_dirs()
    say = progress or (lambda m: None)
    if download_files:
        say("downloading")
        download(spec, progress)
    say("verifying checksums")
    digests = verify_raw(spec)
    say("reading original files")
    df, info = READERS[ds_id]()
    df = add_derived(df, spec.step_hours)
    rep = quality_report(df, spec, info)
    rep["raw_sha256"] = digests
    rep["processing_steps"] = [
        "verified SHA-256 of original download",
        "read original file(s) with documented column mapping (see registry.column_mapping)",
        *(["aggregated 504 radar cells to their arithmetic mean (rain__radar)",
           "validated out00h30m == next-step level (not used as input)"] if ds_id == "bengalas" else
          ["reconstructed continuous daily series from lag-structured rows (4,765 days)",
           "used metre-scaled Kappachichiya target column as level"]),
        "computed t_hours, step_ok and rise_rate (only across one nominal step)",
        "no rows dropped, no values imputed, no outliers removed in the processed copy",
    ]
    df.to_csv(processed_path(ds_id), index=False)
    # Read back and validate before declaring it loaded.
    back = pd.read_csv(processed_path(ds_id), keep_default_na=True)
    if len(back) != len(df) or not np.allclose(back["level"], df["level"], equal_nan=True):
        raise DatasetError("Processed copy failed read-back validation")
    rep["loaded"] = True
    rep["imported_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    quality_path(ds_id).write_text(json.dumps(rep, indent=2))
    say("done")
    return rep


def is_loaded(ds_id: str) -> bool:
    return processed_path(ds_id).exists() and quality_path(ds_id).exists()


def load_quality(ds_id: str) -> dict | None:
    p = quality_path(ds_id)
    return json.loads(p.read_text()) if p.exists() else None


def load_processed(ds_id: str, mode: str = "full", sample_rows: int = 10_000,
                   seed: int = config.DEFAULT_SEED) -> pd.DataFrame:
    """mode='full' returns everything; mode='sample' returns a seeded, contiguous block (keeps time order)."""
    if not is_loaded(ds_id):
        raise DatasetError(f"Dataset '{ds_id}' has not been imported. Run: python backend/scripts/import_dataset.py {ds_id}")
    df = pd.read_csv(processed_path(ds_id), dtype={"timestamp": str})
    df["timestamp"] = df["timestamp"].fillna("")
    df["step_ok"] = df["step_ok"].astype(bool)
    if mode == "sample" and len(df) > sample_rows:
        start = int(np.random.default_rng(seed).integers(0, len(df) - sample_rows))
        df = df.iloc[start:start + sample_rows].reset_index(drop=True)
    return df
