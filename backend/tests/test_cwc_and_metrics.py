"""CWC (India) reader on small hand-made files, the real imported CWC data (skipped if absent), and the
'undefined F1' rule for test periods without any alert."""
import math

import numpy as np
import pandas as pd
import pytest

from floodguard.datasets import is_loaded, load_processed, load_quality
from floodguard.datasets import loaders
from floodguard.metrics import binary, block_bootstrap_f1

LEVEL_HEADER = ["SlNo", "Station", "Latitude", "Longitude", "Data Acquisition Time", loaders.CWC_LEVEL_COL]
RAIN_HEADER = ["SlNo", "Station", "Latitude", "Longitude", "Data Acquisition Time", loaders.CWC_RAIN_COL]


def write_cwc(tmp_path, level_rows, rain_rows, rain_lat=19.08305556):
    lv = pd.DataFrame([[i, s, 19.08305556, 81.3, t, v] for i, (s, t, v) in enumerate(level_rows)], columns=LEVEL_HEADER)
    rn = pd.DataFrame([[i, s, rain_lat, 81.3, t, v] for i, (s, t, v) in enumerate(rain_rows)], columns=RAIN_HEADER)
    lv.to_csv(tmp_path / "rwl_tel_hr_cwc_006_2021_2025.csv", index=False)
    rn.to_csv(tmp_path / "rainfall_tel_hr_cwc_cg_2021_2025.csv", index=False)


@pytest.fixture
def cwc_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "raw_dir", lambda ds_id: tmp_path)
    return tmp_path


def test_cwc_reader_cleaning_rules(cwc_dir):
    write_cwc(cwc_dir,
              level_rows=[("Chindnar", "01-07-2023 00:00", 331.0),
                          ("Chindnar", "01-07-2023 01:30", 331.2),   # moved to 02:00 (next full hour)
                          ("Chindnar", "01-07-2023 03:00", 905.0),   # impossible -> removed
                          ("Chindnar", "01-07-2023 04:00", 331.5),
                          ("Jagdalpur", "01-07-2023 00:00", 535.3)],  # other station -> ignored
              rain_rows=[("Chindnar", "01-07-2023 00:00", 5.0),
                         ("Chindnar", "01-07-2023 04:00", 342.0),     # > 100 mm/h -> missing
                         ("Chindnar", "01-07-2023 09:00", 2.0)])      # no level that hour -> dropped
    df, info = loaders.read_cwc_chindnar_raw()
    assert list(df["timestamp"]) == ["2023-07-01 00:00:00", "2023-07-01 02:00:00", "2023-07-01 04:00:00"]
    assert list(df["level"]) == [331.0, 331.2, 331.5]
    assert df["rain__gauge"].iloc[0] == 5.0 and df["rain__gauge"].iloc[1] == 0.0  # no record -> assumed dry
    assert np.isnan(df["rain__gauge"].iloc[2])
    assert info["level_removed_outside_bounds"] == 1
    assert info["rain_records_above_limit_set_missing"] == 1
    assert info["rain_records_without_level_reading_dropped"] == 1
    assert info["level_file"]["readings_not_on_the_hour"] == 1


def test_cwc_reader_rejects_gauges_that_are_not_co_located(cwc_dir):
    write_cwc(cwc_dir, [("Chindnar", "01-07-2023 00:00", 331.0)], [("Chindnar", "01-07-2023 00:00", 1.0)],
              rain_lat=19.5)
    with pytest.raises(loaders.DatasetError, match="co-located"):
        loaders.read_cwc_chindnar_raw()


@pytest.mark.skipif(not is_loaded("cwc_chindnar"), reason="cwc_chindnar not imported")
def test_cwc_real_import_counts():
    q = load_quality("cwc_chindnar")
    assert q["rows"] == 22578 and q["duplicate_timestamps"] == 0
    rc = q["reader_checks"]
    assert rc["level_file"]["station_rows"] == 22847
    assert rc["level_removed_outside_bounds"] == 103
    assert rc["rain_records_above_limit_set_missing"] == 13
    df = load_processed("cwc_chindnar")
    assert df["timestamp"].iloc[0] == "2021-01-01 00:00:00" and df["timestamp"].iloc[-1] == "2025-05-17 11:00:00"
    assert df["level"].between(327.15, 350.0).all()
    assert np.isnan(df.loc[~df["step_ok"], "rise_rate"]).all()


def test_f1_is_undefined_without_true_alerts():
    no_alerts = np.zeros(50, bool)
    assert math.isnan(binary(no_alerts, no_alerts)["f1"])
    assert math.isnan(binary(no_alerts, np.r_[True, no_alerts[1:]])["f1"])  # false alarm still counted
    assert binary(no_alerts, np.r_[True, no_alerts[1:]])["fp"] == 1
    missed = np.r_[True, no_alerts[1:]]
    assert binary(missed, no_alerts)["f1"] == 0.0  # an alert existed and was missed -> 0, not undefined


def test_bootstrap_interval_undefined_without_true_alerts():
    labels = np.zeros(200)
    assert all(math.isnan(v) for v in block_bootstrap_f1(labels, np.zeros(200, int), block=24, n_boot=20))
