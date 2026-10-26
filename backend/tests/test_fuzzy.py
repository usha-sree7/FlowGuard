import itertools

import numpy as np
import pytest

from floodguard.fuzzy import FIS, RULES, Trap, km_exact, km_iterative, partition, rule_table, type_reduce


def test_trapezoid_values():
    t = Trap(0, 1, 2, 3)
    np.testing.assert_allclose(t([-1, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4]), [0, 0, 0.5, 1, 1, 1, 0.5, 0, 0])
    shoulder = Trap(-np.inf, -np.inf, 1, 2)
    np.testing.assert_allclose(shoulder([-100, 1, 1.5, 3]), [1, 1, 0.5, 0])


def test_invalid_trapezoid_rejected():
    with pytest.raises(ValueError):
        Trap(0, 2, 1, 3)


def test_partition_fou_ordering_and_embedded_inside():
    terms, delta, eps = partition([1.0, 2.0, 3.0], [0.1, 0.1, 0.1], lower_bound=0, upper_bound=5)
    xs = np.linspace(0, 5, 2001)
    for t in terms:
        lo, up, mid = t.lower(xs), t.upper(xs), t.embedded(xs)
        assert np.all(lo <= up + 1e-12), "LMF must lie below UMF (valid FOU)"
        assert np.all((mid >= lo - 1e-12) & (mid <= up + 1e-12)), "Type-1 MF must lie inside the FOU"
        assert np.any(up - lo > 1e-6), "FOU must be non-degenerate"
    # embedded Type-1 MFs form a partition of unity at the crossovers
    tot = sum(t.embedded(xs) for t in terms)
    np.testing.assert_allclose(tot, 1.0, atol=1e-9)


def test_interval_firing_is_exact_inf_and_sup():
    terms, *_ = partition([1.0, 2.0, 3.0], [0.15, 0.1, 0.2], lower_bound=0, upper_bound=5)
    rng = np.random.default_rng(1)
    for _ in range(300):
        a, b = np.sort(rng.uniform(0, 5, 2))
        dense = np.linspace(a, b, 4001)
        for t in terms:
            lo, hi = t.interval_firing(np.array([a]), np.array([b]))
            assert lo[0] == pytest.approx(t.lower(dense).min(), abs=2e-3)
            assert hi[0] == pytest.approx(t.upper(dense).max(), abs=2e-3)


def test_km_exact_matches_iterative_km():
    rng = np.random.default_rng(0)
    for _ in range(1500):
        R = int(rng.integers(1, 15))
        c = rng.uniform(0, 100, R)
        lo = rng.uniform(0, 1, R)
        hi = lo + rng.uniform(0, 1, R) * (1 - lo)
        lo[rng.random(R) < 0.3] = 0
        for right in (False, True):
            assert km_exact(c, lo[None], hi[None], right)[0] == pytest.approx(km_iterative(c, lo, hi, right), abs=1e-9)


def test_km_reduces_to_type1_when_no_uncertainty():
    c = np.array([10.0, 40.0, 60.0, 90.0])
    f = np.array([[0.2, 0.5, 0.1, 0.0]])
    yl, yr = type_reduce(c, c, f, f)
    expected = (f[0] * c).sum() / f[0].sum()
    assert yl[0] == pytest.approx(expected) and yr[0] == pytest.approx(expected)


def test_rule_base_complete_and_unique():
    keys = [(L, R, P) for L, R, P, _ in RULES]
    assert len(keys) == 80 and len(set(keys)) == 80
    assert set(keys) == set(itertools.product(range(5), range(4), range(4)))
    assert len(rule_table()) == 80
    # the level term alone sets the class when rise is steady and rain is light
    assert [s for L, R, P, s in RULES if R == 1 and P == 1] == [0, 1, 2, 3, 4]


def test_partition_with_four_anchors_gives_five_terms_and_survives_full_fou():
    terms, delta, _ = partition([1.0, 1.5, 2.0, 3.0], [1.0] * 4, 0, 10)  # eps clipped to delta / 2
    xs = np.linspace(0, 10, 4001)
    assert len(terms) == 5
    np.testing.assert_allclose(sum(t.embedded(xs) for t in terms), 1.0, atol=1e-9)


def _fis():
    mfs = {
        "level": partition([0.8, 1.0, 1.5, 2.0], [0.05] * 4, 0, 10)[0],
        "rise": partition([-0.1, 0.2, 0.8], [0.02] * 3, -3, 3)[0],
        "rain": partition([1.0, 6.0, 15.0], [0.3] * 3, 0, 100)[0],
    }
    return FIS(mfs=mfs)


def test_risk_monotone_in_level():
    """Type-1 output is monotone. IT2 (KM midpoint) monotonicity is NOT guaranteed by a monotone rule base
    alone; with the FOU capped at half the ramp (mf.partition) measured violations stay below 0.1 point on the
    0-100 scale, so a 0.1-point tolerance is asserted."""
    fis = _fis()
    lv = np.linspace(0.5, 3.0, 2000)
    for rise, rain in [(0.0, 0.0), (0.3, 7.0), (1.0, 20.0), (0.1, 3.0)]:
        crisp = {"level": lv, "rise": np.full_like(lv, rise), "rain": np.full_like(lv, rain)}
        y1 = fis.infer_t1(crisp)["y"]
        y2 = fis.infer_it2({k: (v, v) for k, v in crisp.items()})["y"]
        assert np.all(np.diff(y1) >= -1e-6)
        assert np.all(np.diff(y2) >= -0.1)


def test_type1_output_inside_it2_interval():
    fis = _fis()
    rng = np.random.default_rng(3)
    crisp = {"level": rng.uniform(0.5, 3, 500), "rise": rng.uniform(-0.5, 1.5, 500), "rain": rng.uniform(0, 30, 500)}
    y1 = fis.infer_t1(crisp)["y"]
    it2 = fis.infer_it2({k: (v, v) for k, v in crisp.items()})
    assert np.all(it2["yl"] <= y1 + 1e-9) and np.all(y1 <= it2["yr"] + 1e-9)


def test_wider_input_interval_never_narrows_output_interval():
    fis = _fis()
    base = {"level": (np.array([1.4]), np.array([1.4])), "rise": (np.array([0.1]), np.array([0.1])),
            "rain": (np.array([3.0]), np.array([3.0]))}
    wide = {**base, "level": (np.array([1.2]), np.array([1.7]))}
    a, b = fis.infer_it2(base), fis.infer_it2(wide)
    assert b["yl"][0] <= a["yl"][0] + 1e-9 and b["yr"][0] >= a["yr"][0] - 1e-9
