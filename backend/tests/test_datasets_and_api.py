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
    assert {d["id"] for d in ds["datasets"]} == {"bengalas", "malwathu"}
    m = c.get("/api/model/bengalas", params={"horizon": 2}).json()
    assert len(m["fis"]["rules"]) == 80 and len(m["curves"]["level"]["terms"]) == 5 and "tuned_curves" in m
    r = c.get("/api/replay/bengalas", params={"scenario": "spikes", "limit": 50, "mode": "fused"}).json()
    assert len(r["level_used"]) == 50 and set(r["status"]) <= {"Confident", "Uncertain", "Insufficient data"}
    assert set(r["models"]) == {"it2_fis", "pso_it2", "som", "anfis", "threshold"}
    assert c.get("/api/replay/nope").status_code == 404
    r = c.get("/api/replay/bengalas", params={"limit": 30, "horizon": 6, "model": "anfis", "at": 10}).json()
    assert r["horizon_label"] == "3 h" and r["at"] == 10 and r["explanations"]["anfis"]["reasons"]
    assert c.get("/api/replay/bengalas", params={"horizon": 5}).status_code == 400
    assert c.get("/api/episodes/bengalas").json()["total"] > 0
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
