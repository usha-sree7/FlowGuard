import numpy as np

from floodguard import uncertainty
from floodguard.calibration import calibrate, split
from floodguard.faults import SCENARIOS, inject
from floodguard.metrics import binary, ordinal
from floodguard.reliability import assess
from floodguard.systems import proxy_labels


def test_calibration_uses_train_only(synth, cfg):
    train, test = split(synth, 0.7)
    a = calibrate(train, cfg)
    tampered = synth.copy()
    tampered.loc[len(train):, "level"] = 9.0  # change only the test part
    b = calibrate(split(tampered, 0.7)[0], cfg)
    assert a.thresholds == b.thresholds and a.anchors == b.anchors


def test_proxy_labels_come_from_future_original_level(synth, pipe):
    lab = proxy_labels(synth, pipe.cal)
    fut = synth["level"].shift(-pipe.cal.cfg.horizon_steps)
    expected = np.digitize(fut, pipe.cal.thresholds).astype(float)
    ok = ~np.isnan(fut.to_numpy())
    np.testing.assert_array_equal(lab[ok], expected[ok])
    assert np.isnan(lab[-1])


def test_fault_injection_does_not_modify_original_and_is_reproducible(synth, pipe):
    before = synth.copy()
    for s in SCENARIOS:
        a, gta = inject(synth, s, pipe.cal, seed=5)
        b, gtb = inject(synth, s, pipe.cal, seed=5)
        assert a.equals(b) and all(np.array_equal(gta[k], gtb[k]) for k in gta)
    assert synth.equals(before)


def test_missing_level_reliability_zero_and_never_confident(synth, pipe):
    corrupted, gt = inject(synth, "missing_30", pipe.cal, seed=1)
    out = pipe.run_proposed(corrupted)
    miss = np.isnan(corrupted["level"].to_numpy())
    assert miss.any()
    assert np.all(out["rel"]["level"]["r"][miss] == 0)
    assert not np.any((out["status"] == "Confident") & miss)
    assert np.all(out["D"][miss] > 0)


def test_long_level_outage_is_insufficient_data(synth, pipe):
    df = synth.copy()
    df.loc[3000:3010, "level"] = np.nan
    out = pipe.run_proposed(df)
    assert out["status"][3005] == "Insufficient data"
    assert out["status"][3000] != "Confident"


def test_spike_is_flagged_and_downweighted(synth, pipe):
    df = synth.copy()
    i = 2500
    df.loc[i, "level"] = df.loc[i, "level"] + 3.0
    rel = assess(df, pipe.cal)
    assert rel["level"]["spike"][i] and rel["level"]["r"][i] < 0.5
    out = pipe.run_proposed(df)
    assert out["fused"]["level_c"][i] < df.loc[i, "level"] - 1.0  # fused estimate not dragged by the spike


def test_health_prior_reduces_reliability(synth, pipe):
    df = synth.copy()
    df["health__level"] = 0.4
    rel = assess(df, pipe.cal)
    assert np.nanmax(rel["level"]["r"]) <= 0.4 + 1e-12


def test_uncertainty_index_renormalises_missing_components():
    U = uncertainty.index(np.array([0.5]), np.array([0.0]), np.array([np.nan]))
    assert U[0] == np.float64(100 * 0.4 * 0.5 / 0.7)
    assert uncertainty.band(np.array([10, 30, 60, 90])).tolist() == ["Low", "Moderate", "High", "Very High"]


def test_clean_run_is_mostly_confident_and_risk_separate_from_U(synth, pipe):
    _, test = split(synth, 0.7)
    out = pipe.run_proposed(test)
    assert (out["status"] == "Confident").mean() > 0.8
    assert np.all((out["yl"] <= out["score"] + 1e-9) & (out["score"] <= out["yr"] + 1e-9))


def test_metrics():
    b = binary([1, 1, 0, 0], [1, 0, 1, 0])
    assert (b["tp"], b["fp"], b["fn"], b["tn"]) == (1, 1, 1, 1) and b["f1"] == 0.5
    o = ordinal([0, 3, 2], [1, 1, 2])
    assert o["confusion_matrix"][3][1] == 1
    assert o["severity_weighted_error"] == (1 + 4 + 0) / 3
