import numpy as np

from floodguard.neural import FEATURES, SOM, HybridNet, candidate_features, ga_select
from floodguard.softcomputing import _fitness


def _toy(n=3000, seed=0):
    """Level random walk; class = level quartile two steps ahead (a learnable, causal task)."""
    rng = np.random.default_rng(seed)
    level = np.cumsum(rng.normal(0, 0.1, n)) + 5
    X = np.column_stack([level, np.diff(level, prepend=level[0]), rng.gamma(1, 1, n)])
    fut = np.roll(level, -2)
    y = np.digitize(fut, np.quantile(fut, [0.5, 0.8, 0.95]))
    return X, y


def test_candidate_features_are_causal_and_finite():
    X, _ = _toy()
    F = candidate_features(X)
    assert F.shape == (len(X), len(FEATURES)) and np.isfinite(F).all()
    # changing the future must not change past features
    X2 = X.copy()
    X2[2000:] += 100
    np.testing.assert_array_equal(candidate_features(X2)[:2000], F[:2000])


def test_som_assigns_every_row_to_a_node():
    X, _ = _toy()
    Z = (X - X.mean(0)) / X.std(0)
    som = SOM(seed=1).fit(Z)
    oh = som.onehot(Z)
    assert oh.shape == (len(Z), 16) and (oh.sum(1) == 1).all()


def test_hybrid_net_learns_and_ga_returns_valid_mask():
    X, y = _toy()
    F = candidate_features(X)
    tr, va = slice(0, 2400), slice(2400, 3000)
    m = HybridNet(True, seed=1).fit(F[tr], y[tr], F[va], y[va])
    assert (m.net.predict_cls(m._inputs(F[va])) == y[va]).mean() > 0.6
    mask, hist = ga_select(F[tr], y[tr], F[va], y[va], fitness=_fitness, pop=4, gens=2, seed=1)
    assert mask.dtype == bool and mask.any() and len(hist) == 3
