"""Trapezoidal membership functions (Type-1 and Interval Type-2) with exact interval-input bounds.

All MFs here are trapezoids (shoulders use +/-inf corners), hence quasi-concave. For a quasi-concave
function the infimum over an interval [lo, hi] is attained at an endpoint, and the supremum is the
height if the interval touches the plateau, else the larger endpoint value. These closed forms are
exact, not sampled approximations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INF = np.inf


@dataclass(frozen=True)
class Trap:
    a: float
    b: float
    c: float
    d: float
    h: float = 1.0

    def __post_init__(self):
        if not (self.a <= self.b <= self.c <= self.d):
            raise ValueError(f"invalid trapezoid {self}")

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        y = np.zeros_like(x)
        with np.errstate(invalid="ignore", divide="ignore"):
            rise = (x - self.a) / (self.b - self.a) if np.isfinite(self.a) else np.ones_like(x)
            fall = (self.d - x) / (self.d - self.c) if np.isfinite(self.d) else np.ones_like(x)
        y = np.where((x >= self.b) & (x <= self.c), 1.0, y)
        y = np.where((x > self.a) & (x < self.b), rise, y)
        y = np.where((x > self.c) & (x < self.d), fall, y)
        y = np.where(np.isnan(x), np.nan, y)
        return self.h * np.clip(y, 0.0, 1.0)

    def inf_on(self, lo, hi):
        return np.minimum(self(lo), self(hi))

    def sup_on(self, lo, hi):
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        touches = (hi >= self.b) & (lo <= self.c)
        return np.where(touches, self.h, np.maximum(self(lo), self(hi)))

    def to_dict(self):
        f = lambda v: None if not np.isfinite(v) else float(v)  # noqa: E731
        return {"a": f(self.a), "b": f(self.b), "c": f(self.c), "d": f(self.d), "h": self.h}


@dataclass(frozen=True)
class IT2Trap:
    """Interval Type-2 trapezoid: footprint of uncertainty between lower (LMF) and upper (UMF) MFs."""
    lower: Trap
    upper: Trap
    embedded: Trap  # principal Type-1 MF lying inside the FOU, used by the Type-1 baseline

    def __call__(self, x):
        return self.lower(x), self.upper(x)

    def interval_firing(self, lo, hi):
        """Bounds on the membership of an unknown true value known only to lie in [lo, hi]:
        [inf LMF, sup UMF] over the interval. Singleton input (lo == hi) reduces to [LMF(x), UMF(x)]."""
        return self.lower.inf_on(lo, hi), self.upper.sup_on(lo, hi)


def partition(anchors, eps, lower_bound=-INF, upper_bound=INF):
    """(n+1)-term partition (shoulder, n-1 trapezoids, shoulder) with crossovers at n increasing anchors.

    delta_k (ramp half-width) = 0.25 * smaller distance to the neighbouring anchor/bound, so ramps never overlap.
    eps_k (FOU half-shift)    = data-derived uncertainty of anchor k, clipped to [0, delta_k / 2]. The cap keeps
                                the lower MFs of both neighbouring terms positive at every crossover; with eps = delta
                                they both vanish there, the KM interval jumps open and the IT2 output stops being
                                monotone in the level (measured on Bengalas: dips of up to 1.06 points that lowered
                                the class in 9 of 182 tested rise/rain curves; with the cap: <= 0.09 points, none).
    UMF ramps are shifted outward by eps, LMF ramps inward by eps; the Type-1 MF is the unshifted midline.
    """
    a = [float(v) for v in anchors]
    n = len(a)
    if n < 2 or not all(a[k] < a[k + 1] for k in range(n - 1)):
        raise ValueError(f"anchors must be strictly increasing: {a}")
    left = a[0] - (a[1] - a[0]) if not np.isfinite(lower_bound) else lower_bound
    right = a[-1] + (a[-1] - a[-2]) if not np.isfinite(upper_bound) else upper_bound
    pts = [left, *a, right]
    delta = [0.25 * min(pts[k + 1] - pts[k], pts[k + 2] - pts[k + 1]) for k in range(n)]
    e = [float(np.clip(eps[k], 0.0, 0.5 * delta[k])) for k in range(n)]

    def mid_trap(p, q, u, v):
        # with eps = delta the LMF plateau shrinks to a point; rounding may then put q a few ulps above u
        if q > u and q - u <= 1e-9 * (1.0 + abs(u)):
            q = u = 0.5 * (q + u)
        return Trap(p, q, u, v)

    def build(s):  # s=+1 -> UMF (wider), s=-1 -> LMF (narrower), s=0 -> Type-1
        r = [(a[k] - delta[k], a[k] + delta[k]) for k in range(n)]
        terms = [Trap(-INF, -INF, r[0][0] + s * e[0], r[0][1] + s * e[0])]
        for k in range(1, n):
            terms.append(mid_trap(r[k - 1][0] - s * e[k - 1], r[k - 1][1] - s * e[k - 1], r[k][0] + s * e[k], r[k][1] + s * e[k]))
        terms.append(Trap(r[-1][0] - s * e[-1], r[-1][1] - s * e[-1], INF, INF))
        return terms

    up, lo, mid = build(+1), build(-1), build(0)
    return [IT2Trap(lower=lo[i], upper=up[i], embedded=mid[i]) for i in range(n + 1)], delta, e
