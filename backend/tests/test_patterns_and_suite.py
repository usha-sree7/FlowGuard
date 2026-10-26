import numpy as np
import pytest
from fastapi.testclient import TestClient

from floodguard.calibration import split
from floodguard.datasets import is_loaded
from floodguard.patterns import COST, SOM, FuzzyART, select_rho, som_art_cls
from floodguard.suite import ModelSuite, horizons
from floodguard.tuning import TunedIT2


def _states(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    level = 0.8 + np.abs(rng.normal(0, 0.2, n))
    return np.column_stack([level, rng.normal(0, 0.05, n), rng.gamma(0.5, 2.0, n)])


def test_som_cost_mapping_prefers_higher_class_when_missing_it_is_expensive():
    X = _states()
    som = SOM(3, 3, seed=1).fit(X)
    node = som.bmu(X)
    y = np.zeros(len(X), int)
    hot = node == node[0]
    y[np.flatnonzero(hot)[: max(1, int(hot.sum() * 0.4))]] = 2  # 40 % Moderate in one node
    som.map_risk(X, y)
    # expected cost: predict Normal = 0.4*4 = 1.6 > predict Moderate = 0.6*2 = 1.2 -> Moderate (class 2)
    assert som.node_class[node[0]] == 2
    assert COST[2, 0] == 4 and COST[0, 2] == 2


def test_fuzzy_art_familiar_inside_novel_outside():
    X = _states()
    art = FuzzyART(rho=0.9).fit(X)
    assert art.novelty(X)["novel"].mean() < 0.01
    weird = np.array([[0.8, 3.0, 0.0]])  # huge rise with no rain, outside the training envelope
    out = art.novelty(weird)
    assert out["novel"][0] and out["outside_envelope"][0]


def test_art_vigilance_selected_without_labels_within_budget():
    X = _states()
    art, info = select_rho(X[:3000], X[3000:], budget=0.01)
    assert info["validation_novelty_rate"][str(art.rho)] <= 0.01 or art.rho == 0.80


def test_som_art_falls_back_to_threshold_only_when_novel():
    out = som_art_cls(np.array([0, 3, 1]), np.array([False, True, False]), np.array([2, 0, 2]))
    assert out.tolist() == [0, 0, 1]


def test_horizon_support_depends_on_sampling_interval():
    assert horizons(0.5)["supported"] == {1: "30 min", 2: "1 h", 6: "3 h"}
    daily = horizons(24.0)
    assert daily["supported"] == {1: "1 day"} and daily["unsupported"] == ["30 min", "1 h", "3 h"]


@pytest.fixture(scope="module")
def suite(synth, cfg):
    return ModelSuite.fit(split(synth, 0.7)[0], cfg)


def test_suite_ignores_test_data(synth, cfg, suite):
    """SOM, PSO genes and ANFIS parameters must not change when only the TEST part of the series changes."""
    tampered = synth.copy()
    tampered.loc[len(split(synth, 0.7)[0]):, "level"] = 9.0
    b = ModelSuite.fit(split(tampered, 0.7)[0], cfg)
    for h in suite.horizons:
        assert np.array_equal(suite.som[h].W, b.som[h].W) and np.array_equal(suite.pso[h].genes, b.pso[h].genes)
        assert np.array_equal(suite.anfis[h].theta, b.anfis[h].theta)


def test_pso_never_worse_than_untuned_on_validation_and_base_gene_is_untuned(suite, synth):
    for h, g in suite.pso.items():
        assert g.best_fitness >= g.base_fitness - 1e-12
        assert g.history == sorted(g.history)  # global best is monotone
    g = suite.pso[2]
    X = np.column_stack([np.linspace(0.7, 1.4, 50), np.linspace(-0.1, 0.4, 50), np.linspace(0, 20, 50)])
    base = TunedIT2(suite.pipe, "pso").set_genes(g.BASE)
    untuned = suite.pipe.fis.infer_it2({k: (X[:, i], X[:, i]) for i, k in enumerate(("level", "rise", "rain"))})["y"]
    np.testing.assert_allclose(base.predict(X), untuned)


def test_anfis_learns_and_explains(suite):
    m = suite.anfis[2]
    assert m.loss[-1] <= m.loss[0]
    for i, c in enumerate(m.c):  # term order and widths stay interpretable after training
        assert np.all(np.diff(c) >= 0) and np.all(m.s[i] >= m.s_lo[i] - 1e-12)
    rules = m.explain(np.array([1.0, 0.05, 3.0]))
    assert len(rules) == 4 and abs(sum(r["contribution"] for r in m.explain(np.array([1.0, 0.05, 3.0]), k=80))
                                   - m.predict(np.array([[1.0, 0.05, 3.0]]))[0]) < 0.05


def test_reliability_layer_shields_models_from_a_spike(synth, suite):
    _, test = split(synth, 0.7)
    test = test.copy()
    k = int(np.flatnonzero(np.digitize(test["level"], suite.pipe.cal.thresholds) == 0)[300])  # a calm step
    test.loc[test.index[k], "level"] += 3.0
    sh = suite.run_shared(test)
    assert sh["X"]["raw"][k, 0] > sh["X"]["fused"][k, 0] + 1.0
    assert sh["proposed"]["rel"]["level"]["r"][k] < 0.5  # flagged and down-weighted, not ignored: c = r*x + (1-r)*p
    hz = suite.run_horizon(sh, 2)
    for m in ("it2_fis", "anfis", "threshold"):
        assert hz["cls"]["fused"][m][k] <= hz["cls"]["raw"][m][k]


@pytest.mark.skipif(not is_loaded("bengalas"), reason="bengalas not imported")
def test_assess_endpoint_runs_real_models_and_validates():
    c = TestClient(__import__("floodguard.api", fromlist=["app"]).app)
    ok = c.post("/api/assess/bengalas", json={"level": 0.8, "rise": 0.0, "rain": 0.0, "horizon": 2,
                                               "scenario": "spike", "mode": "fused"})
    assert ok.status_code == 200
    j = ok.json()
    assert "HYPOTHETICAL" in j["kind"] and set(j["explanations"]) == {"it2_fis", "pso_it2", "som", "anfis", "threshold"}
    e = j["explanations"]["it2_fis"]
    level_sensor = next(s for s in e["sensors"] if s["sensor"] == "water level")
    assert "spike" in level_sensor["flags"] and level_sensor["reliability"] < 0.5
    assert e["risk"] == "Normal" and e["reasons"]  # the spike is not turned into an alarm
    raw = c.post("/api/assess/bengalas", json={"level": 0.8, "rise": 0.0, "rain": 0.0, "horizon": 2,
                                                "scenario": "spike", "mode": "raw"}).json()
    assert raw["explanations"]["threshold"]["risk"] == "Critical"  # without the reliability layer it is
    bad = c.post("/api/assess/bengalas", json={"level": 99, "rise": 0, "rain": -1})
    assert bad.status_code == 422 and "water level" in bad.json()["detail"]
    assert c.post("/api/assess/bengalas", json={"level": 1, "rise": 0, "rain": 0, "horizon": 5}).status_code == 422


def test_arx_fit_block_is_purged_of_targets_inside_validation(synth, cfg, suite):
    """ARX targets of the last h fit rows lie in the validation block; those rows must not influence the fit."""
    train, _ = split(synth, 0.7)
    a = suite
    n_fit = int(0.8 * len(train))
    tampered = train.copy()
    tampered.loc[tampered.index[n_fit:n_fit + 6], "level"] += 2.0  # first 3 h of validation
    b = ModelSuite.fit(tampered, cfg)
    for h in a.arx:
        assert np.allclose(a.arx[h].coef, b.arx[h].coef)
