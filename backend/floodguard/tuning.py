"""PSO-tuned IT2FIS: particle swarm optimisation of the IT2FIS membership functions (GA kept as an alternative).

What PSO optimises
------------------
The untuned IT2FIS places its membership-function crossovers ("anchors") with fixed statistical rules computed on
the training split (level: mean + 0.5/1/2/3 sd; rise: 1 / 99 / 99.9 % quantiles; rain: 75 / 95 / 99 % wet-step
quantiles) and sizes the footprint of uncertainty (FOU) from the spread of those anchors across years. Nothing in
that construction looks at alert quality. PSO searches small corrections to it.

Decision vector (11 genes for the 5-term level / 4-term rise / 4-term rain partition):
  * one shift per anchor in [-1, 1], applied as shift x 0.4 x the smaller gap to the neighbouring anchor, so the
    anchors stay strictly ordered (4 level + 3 rise + 3 rain = 10 genes);
  * one FOU scale in [0, 3] multiplying every data-derived FOU half-width (0 = Type-1-like, 1 = untuned).
The rule base, consequents and inference (KM type reduction) are NOT changed: only where the fuzzy sets sit.

Objective (maximised): validation alert F1 + 0.25 x 5-class macro F1, computed on the VALIDATION block (last 20 %
of the training split) for the horizon being tuned. The test split is never seen. The all-zero-shift / scale-1
vector IS the untuned IT2FIS and is particle 0 of the initial swarm, so the tuned system can only be selected if it
scores at least as well on validation; whether that carries over to the test period is what the comparison reports.

PSO (Kennedy & Eberhart 1995, inertia weight Shi & Eberhart 1998): 20 particles, 20 iterations, inertia w = 0.7,
c1 = c2 = 1.5, velocity clamped to 20 % of each gene's range, positions clipped to the bounds, fixed seed.
GA alternative: tournament (3), BLX-0.25 arithmetic crossover, Gaussian mutation (15 %), elitism (2).
"""
from __future__ import annotations

import copy

import numpy as np

from .calibration import build_fis
from .fuzzy.fis import INPUTS
from .patterns import _fitness


def _risk_class(y):
    from .systems import risk_class
    return risk_class(y)


class TunedIT2:
    def __init__(self, pipe, method: str = "pso", pop: int = 20, gens: int = 20, seed: int = 0):
        if method not in ("pso", "ga"):
            raise ValueError(f"unknown tuning method {method}")
        self.pipe, self.method, self.pop, self.gens, self.seed = pipe, method, pop, gens, seed
        self.n_anchor = [len(pipe.cal.anchors[n]) for n in INPUTS]
        d = sum(self.n_anchor)
        self.LO = np.array([-1.0] * d + [0.0])
        self.HI = np.array([1.0] * d + [3.0])
        self.BASE = np.array([0.0] * d + [1.0])

    def gene_names(self) -> list[str]:
        return [f"{n} anchor {k + 1} shift" for n, m in zip(INPUTS, self.n_anchor) for k in range(m)] + ["FOU scale"]

    def _fis(self, g):
        cal = copy.deepcopy(self.pipe.cal)
        i0 = 0
        for name, m in zip(INPUTS, self.n_anchor):
            a = np.array(cal.anchors[name], float)
            gap = np.diff(a)
            room = 0.4 * np.minimum(np.r_[gap[0], gap], np.r_[gap, gap[-1]])
            cal.anchors[name] = (a + g[i0:i0 + m] * room).tolist()
            cal.fou_eps[name] = (np.array(cal.fou_eps[name]) * g[-1]).tolist()
            i0 += m
        return build_fis(cal), cal

    def _score(self, g, X, labels):
        try:
            fis, _ = self._fis(g)
        except ValueError:  # degenerate partition (anchors collided) -> infeasible candidate
            return -1.0
        y = fis.infer_it2({k: (X[:, i], X[:, i]) for i, k in enumerate(INPUTS)})["y"]
        return _fitness(labels, _risk_class(y))

    def fit(self, X, labels):
        rng = np.random.default_rng(self.seed)
        f = lambda g: self._score(g, X, labels)  # noqa: E731
        P = rng.uniform(self.LO, self.HI, (self.pop, len(self.LO)))
        P[0] = self.BASE  # the untuned IT2FIS is always in the initial population
        fit = np.array([f(p) for p in P])
        self.history = [float(fit.max())]
        if self.method == "pso":
            vmax = 0.2 * (self.HI - self.LO)
            V = np.zeros_like(P)
            pbest, pfit = P.copy(), fit.copy()
            for _ in range(self.gens):
                gbest = pbest[np.argmax(pfit)]
                r1, r2 = rng.random(P.shape), rng.random(P.shape)
                V = np.clip(0.7 * V + 1.5 * r1 * (pbest - P) + 1.5 * r2 * (gbest - P), -vmax, vmax)
                P = np.clip(P + V, self.LO, self.HI)
                fit = np.array([f(p) for p in P])
                better = fit > pfit
                pbest[better], pfit[better] = P[better], fit[better]
                self.history.append(float(pfit.max()))
            best, best_fit = pbest[np.argmax(pfit)], float(pfit.max())
        else:
            for _ in range(self.gens):
                elite = P[np.argsort(fit)[-2:]]
                children = []
                while len(children) < self.pop - 2:
                    i, j = (max(rng.choice(self.pop, 3), key=lambda k: fit[k]) for _ in range(2))
                    a = rng.uniform(-0.25, 1.25, len(self.LO))
                    c = a * P[i] + (1 - a) * P[j]
                    mut = rng.random(len(c)) < 0.15
                    c[mut] += rng.normal(0, 0.1 * (self.HI - self.LO))[mut]
                    children.append(np.clip(c, self.LO, self.HI))
                P = np.vstack([elite, children])
                fit = np.array([f(p) for p in P])
                self.history.append(float(fit.max()))
            best, best_fit = P[np.argmax(fit)], float(fit.max())
        return self.set_genes(best, base_fitness=f(self.BASE), best_fitness=best_fit)

    def set_genes(self, genes, base_fitness=None, best_fitness=None):
        self.genes = np.asarray(genes, float)
        self.fis, self.cal = self._fis(self.genes)
        self.base_fitness, self.best_fitness = base_fitness, best_fitness
        return self

    def predict(self, X):
        return self.fis.infer_it2({k: (X[:, i], X[:, i]) for i, k in enumerate(INPUTS)})["y"]

    def describe(self) -> dict:
        return {"method": self.method,
                "genes": self.genes.tolist(),  # full precision: rounded genes can collapse a trapezoid
                "gene_names": self.gene_names(),
                "original_anchors": self.pipe.cal.anchors, "tuned_anchors": self.cal.anchors,
                "fou_scale": round(float(self.genes[-1]), 4),
                "validation_fitness_untuned": self.base_fitness, "validation_fitness_tuned": self.best_fitness,
                "best_fitness_per_iteration": [round(h, 4) for h in getattr(self, "history", [])],
                "population": self.pop, "iterations": self.gens, "seed": self.seed,
                "objective": "validation alert F1 + 0.25 x 5-class macro F1"}
