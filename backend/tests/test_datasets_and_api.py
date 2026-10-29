"""Checks against the REAL imported files (skipped when a dataset has not been imported)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from floodguard.api import app
from floodguard.datasets import is_loaded, load_processed, load_quality

needs_bengalas = pytest.mark.skipif(not is_loaded("bengalas"), reason="bengalas not imported")
needs_malwathu = pytest.mark.skipif(not is_loaded("malwathu"), reason="malwathu not imported")


@needs_bengalas
def test_bengalas_verified_schema_and_counts():
    q = load_quality("bengalas")
    assert q["rows"] == 72011 and q["duplicate_timestamps"] == 0
    assert q["reader_checks"]["radar_cells"] == 504
    assert q["reader_checks"]["out00h30m_equals_next_level_fraction"] == 1.0
    df = load_processed("bengalas")
    assert df["timestamp"].iloc[0] == "2019-01-04 14:30:00" and df["timestamp"].iloc[-1] == "2023-12-31 16:00:00"
    assert np.isnan(df.loc[~df["step_ok"], "rise_rate"]).all()


@needs_bengalas
def test_sample_mode_is_seeded_and_contiguous():
    a = load_processed("bengalas", mode="sample", sample_rows=500, seed=3)
    b = load_processed("bengalas", mode="sample", sample_rows=500, seed=3)
    assert a.equals(b) and len(a) == 500 and a["t_hours"].is_monotonic_increasing


@needs_malwathu
def test_malwathu_reconstruction():
    q = load_quality("malwathu")
    assert q["rows"] == 4765
    assert all(q["reader_checks"]["lag_columns_match_previous_row"].values())
    assert q["reader_checks"]["level_lag_to_target_ratio_median"] == pytest.approx(100.0)


@needs_bengalas
def test_api_endpoints():
    c = TestClient(app)
    assert c.get("/api/health").json()["ok"]
    ds = c.get("/api/datasets").json()
    assert {d["id"] for d in ds["datasets"]} == {"bengalas", "malwathu", "cwc_chindnar"}
    m = c.get("/api/model/bengalas", params={"horizon": 2}).json()
    assert len(m["fis"]["rules"]) == 80 and len(m["curves"]["level"]["terms"]) == 5 and "tuned_curves" in m
    r = c.get("/api/replay/bengalas", params={"scenario": "spikes", "limit": 50, "mode": "fused"}).json()
    assert len(r["level_used"]) == 50 and set(r["status"]) <= {"Confident", "Uncertain", "Insufficient data"}
    assert set(r["models"]) == {"it2_fis", "pso_it2", "som", "anfis", "threshold"}
    assert c.get("/api/replay/nope").status_code == 404
    r = c.get("/api/replay/bengalas", params={"limit": 30, "horizon": 6, "model": "anfis", "at": 10}).json()
    assert r["horizon_label"] == "3 h" and r["at"] == 10 and r["explanations"]["anfis"]["reasons"]
    assert c.get("/api/replay/bengalas", params={"horizon": 5}).status_code == 400
    ep = c.get("/api/episodes/bengalas").json()
    assert ep["total"] > 0 and ep["episode_class"] == "High"
    for path in ("/api/references", "/api/datasets/bengalas/preview", "/api/assess/bengalas/options"):
        assert c.get(path).status_code == 200
    refs = c.get("/api/references").json()["references"]
    assert all(x["doi"] for x in refs)


def test_failed_download_gives_manual_instructions_and_is_not_loaded(tmp_path, monkeypatch):
    import dataclasses

    from floodguard import config
    from floodguard.datasets import loaders
    from floodguard.datasets.registry import MALWATHU

    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    bad = dataclasses.replace(MALWATHU, files=(dataclasses.replace(MALWATHU.files[0], url="http://127.0.0.1:9/x.csv"),))
    monkeypatch.setitem(loaders.DATASETS, "malwathu", bad)
    with pytest.raises(loaders.DatasetError) as e:
        loaders.import_dataset("malwathu")
    assert "Download manually" in str(e.value)
    assert not loaders.is_loaded("malwathu")


@pytest.mark.skipif(not is_loaded("cwc_chindnar"), reason="cwc_chindnar not imported")
def test_cwc_episodes_fall_back_to_highest_level_reached():
    ep = TestClient(app).get("/api/episodes/cwc_chindnar").json()
    assert ep["episode_class"] == "Moderate" and ep["total"] > 0  # no High level in the Chindnar test period


def _lab_csv_text() -> str:
    """A small synthetic CSV with the same column-naming pattern a real user would upload
    (timestamp / rainfall / water_level), independent of any curated dataset."""
    rng = np.random.default_rng(1)
    n = 400
    lines = ["timestamp,water_level,rainfall"]
    for i in range(n):
        ts = f"2024-01-{1 + i // 24:02d} {i % 24:02d}:00:00"
        rain = max(0.0, float(rng.gamma(0.4, 2.0) - 0.3)) if rng.random() < 0.4 else 0.0
        level = 5.0 + 0.02 * np.sin(i / 20) + 0.01 * rng.normal()
        lines.append(f"{ts},{level:.4f},{rain:.2f}")
    return "\n".join(lines) + "\n"


def test_lab_upload_map_analyze_flow():
    """Exercises the exact stateless Dataset Lab flow the frontend's Upload/Validate/Run-analysis
    buttons drive: POST /api/lab/upload (multipart), then POST /api/lab/map and POST /api/lab/analyze
    with the csv_text carried in the JSON body (no upload_id in the URL -- see custom.py's docstring
    on why this must be stateless). Also guards against the exact failure mode a stale '/api/lab/
    {upload_id}' GET route would cause: a POST to a literal 2-segment path like /api/lab/map getting
    matched by path but rejected for method (405), if such a parameterised route were ever reintroduced."""
    c = TestClient(app)
    csv_text = _lab_csv_text()

    up = c.post("/api/lab/upload", files={"file": ("sensors.csv", csv_text, "text/csv")})
    assert up.status_code == 200, up.text
    body = up.json()
    assert body["csv_text"] == csv_text
    assert body["suggested_mapping"]["level"] == "water_level"
    assert body["suggested_mapping"]["rain"] == ["rainfall"]
    assert body["suggested_mapping"]["timestamp"] == "timestamp"

    # the exact request shape MappingForm.validate() sends
    map_req = {"csv_text": csv_text, "timestamp": "timestamp", "level": "water_level",
              "rain": ["rainfall"], "step_hours": None}
    mapped = c.post("/api/lab/map", json=map_req)
    assert mapped.status_code == 200, mapped.text
    mres = mapped.json()
    assert mres["ready"] is True and mres["errors"] == []

    # the exact request shape Results.run() sends
    an_req = {**map_req, "filename": "sensors.csv", "horizon_steps": None}
    analyzed = c.post("/api/lab/analyze", json=an_req)
    assert analyzed.status_code == 200, analyzed.text
    ares = analyzed.json()
    assert ares["primary_model"] == "it2_fis" and ares["primary_mode"] == "fused"
    assert ares["models_not_run"] == ["PSO-tuned IT2FIS", "SOM", "ANFIS"]
    assert ares["primary"]["risk"] in ("Normal", "Low", "Moderate", "High", "Critical")

    # no stale upload_id-based route should exist: this exact path shape is what previously
    # produced a 405 when a leftover GET /api/lab/{upload_id} matched it by path but not by method
    assert c.get("/api/lab/map").status_code == 404
    assert c.get("/api/lab/anything").status_code == 404


def test_lab_no_timestamp_with_fixed_interval():
    """The timestamp-optional path: leaving timestamp unmapped is valid as long as a fixed sampling
    interval is given (see custom.py `_build_standard`); this must not require a timestamp column."""
    c = TestClient(app)
    csv_text = _lab_csv_text()
    req = {"csv_text": csv_text, "timestamp": None, "level": "water_level", "rain": ["rainfall"], "step_hours": 1.0}
    mapped = c.post("/api/lab/map", json=req)
    assert mapped.status_code == 200 and mapped.json()["ready"] is True

    analyzed = c.post("/api/lab/analyze", json={**req, "filename": "sensors.csv", "horizon_steps": None})
    assert analyzed.status_code == 200, analyzed.text
    assert "no timestamp mapped" in analyzed.json()["primary"]["timestamp"]

    # leaving BOTH timestamp and step_hours unset must be a clear validation error, not a crash.
    # Ordinary mapping-validation failures return 200 with ready=False (as MappingForm.validate()
    # expects); 422 is reserved for genuine errors like an unparseable file -- see custom.py/api.py.
    bad = c.post("/api/lab/map", json={**req, "step_hours": None})
    assert bad.status_code == 200
    bres = bad.json()
    assert bres["ready"] is False
    assert any("timestamp" in e.lower() for e in bres["errors"])
