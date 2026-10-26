"""Additional soft-computing models, compared against the original four systems on identical observations.

  anfis     : first-order TSK neuro-fuzzy system (Jang's ANFIS). 64 rules on the same 4x4x4 term grid,
              Gaussian MFs initialised from the calibrated anchors; hybrid learning = least-squares
              consequents + gradient-descent (Adam) premise MFs.
  fcm_tsk   : fuzzy C-means clustering of the input space; one zero-order TSK rule per cluster
              (Gaussian MFs from the fuzzy cluster spread, consequent = membership-weighted target).
  ga_it2    : the IT2FIS with anchors and FOU width tuned by a real-coded genetic algorithm.
  pso_it2   : the same IT2FIS search space tuned by particle swarm optimisation.

Every model uses the same three inputs (level, rise, rain) and the same proxy labels. Each is run on the
naive inputs (`<name>`) and on the reliability-weighted fused point inputs (`<name>+fusion`).
Training / tuning uses the clean TRAIN split only: models fit on its first 80 %, GA/PSO fitness is
evaluated on its last 20 % (chronological validation block). Nothing is tuned on the test split.
"""
from __future__ import annotations

import copy

import numpy as np

from .calibration import build_fis
from .fuzzy.fis import INPUTS, RULES
from .metrics import evaluate

CLASS_SCORE = np.array([10.0, 37.5, 62.5, 90.0])  # target score per risk class (same as the threshold system)
MODELS = ("anfis", "fcm_tsk", "ga_it2", "pso_it2")


def _risk_class(y):
    from .systems import risk_class
    return risk_class(y)


def _fitness(labels, y) -> float:
    ev = evaluate(labels, _risk_class(y))
    return ev["binary_alert"]["f1"] + 0.25 * np.nan_to_num(ev["ordinal"]["macro_f1"])


class Scaler:
    """Robust min-max to roughly [0, 1] per input (no clipping, so extremes stay ordered)."""

    def __init__(self, X):
        self.lo = np.nanquantile(X, 0.001, axis=0)
        self.span = np.maximum(np.nanquantile(X, 0.999, axis=0) - self.lo, 1e-9)

    def __call__(self, X):
        return (X - self.lo) / self.span


def _weights(y_cls):
    """Inverse square-root class frequency, so the rare Warning/Critical classes are not ignored."""
    freq = np.bincount(y_cls, minlength=4) / len(y_cls)
    return 1.0 / np.sqrt(np.maximum(freq, 1e-6))[y_cls]


# ---------------------------------------------------------------- ANFIS
class ANFIS:
    def __init__(self, cal, epochs: int = 40, lr: float = 0.01, ridge: float = 1e-3):
        self.cal, self.epochs, self.lr, self.ridge = cal, epochs, lr, ridge
        self.ant = np.array([[L, R, P] for L, R, P, _ in RULES])

    def _init_mfs(self):
        c, s = np.zeros((3, 4)), np.zeros((3, 4))
        for i, name in enumerate(INPUTS):
            a = (np.array(self.cal.anchors[name], float) - self.scaler.lo[i]) / self.scaler.span[i]
            g = np.diff(a).mean()
            c[i] = [a[0] - g / 2, (a[0] + a[1]) / 2, (a[1] + a[2]) / 2, a[2] + g / 2]
            s[i] = np.full(4, max(g / 2, 1e-3))
        return c, s

    def _firing(self, Z):
        mu = np.exp(-0.5 * ((Z[:, :, None] - self.c[None]) / self.s[None]) ** 2)  # (N, 3, 4)
        W = np.ones((len(Z), len(RULES)))
        for i in range(3):
            W *= mu[:, i, self.ant[:, i]]
        return W, W.sum(axis=1) + 1e-12

    def _design(self, Z, W, S):
        Wn = W / S[:, None]
        Z1 = np.column_stack([np.ones(len(Z)), Z])
        return (Wn[:, :, None] * Z1[:, None, :]).reshape(len(Z), -1), Z1

    def fit(self, X, y_cls):
        self.scaler = Scaler(X)
        Z, t, sw = self.scaler(X), CLASS_SCORE[y_cls], _weights(y_cls)
        self.c, self.s = self._init_mfs()
        m = {k: np.zeros_like(v) for k, v in (("c", self.c), ("s", self.s))}
        v = copy.deepcopy(m)
        for ep in range(self.epochs + 1):
            W, S = self._firing(Z)
            A, Z1 = self._design(Z, W, S)
            # forward pass: consequents by weighted ridge least squares
            Aw = A * sw[:, None]
            self.theta = np.linalg.solve(Aw.T @ A + self.ridge * np.eye(A.shape[1]), Aw.T @ t)
            if ep == self.epochs:
                break
            # backward pass: gradient of weighted squared error w.r.t. MF centres and widths
            F = Z1 @ self.theta.reshape(len(RULES), 4).T  # rule outputs (N, R)
            y = (W * F).sum(axis=1) / S
            G = (sw * (y - t) / len(Z))[:, None] * (F - y[:, None]) / S[:, None] * W  # dE/dlog w_r
            grads = {"c": np.zeros_like(self.c), "s": np.zeros_like(self.s)}
            for i in range(3):
                for j in range(4):
                    g = G[:, self.ant[:, i] == j].sum(axis=1)
                    d = Z[:, i] - self.c[i, j]
                    grads["c"][i, j] = (g * d / self.s[i, j] ** 2).sum()
                    grads["s"][i, j] = (g * d ** 2 / self.s[i, j] ** 3).sum()
            for k in ("c", "s"):  # Adam
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                step = self.lr * (m[k] / (1 - 0.9 ** (ep + 1))) / (np.sqrt(v[k] / (1 - 0.999 ** (ep + 1))) + 1e-8)
                setattr(self, k, getattr(self, k) - step)
            self.s = np.maximum(self.s, 1e-3)
        return self

    def predict(self, X):
        Z = self.scaler(X)
        W, S = self._firing(Z)
        A, _ = self._design(Z, W, S)
        return np.clip(A @ self.theta, 0.0, 100.0)


# ---------------------------------------------------------------- FCM-derived TSK
class FCMTSK:
    def __init__(self, n_clusters: int = 16, m: float = 2.0, iters: int = 150, seed: int = 0):
        self.k, self.m, self.iters, self.rng = n_clusters, m, iters, np.random.default_rng(seed)

    def _membership(self, Z):
        d2 = ((Z[:, None, :] - self.v[None]) ** 2).sum(axis=2) + 1e-12
        inv = d2 ** (-1.0 / (self.m - 1))
        return inv / inv.sum(axis=1, keepdims=True)

    def fit(self, X, y_cls):
        self.scaler = Scaler(X)
        Z, t, sw = self.scaler(X), CLASS_SCORE[y_cls], _weights(y_cls)
        # k-means++ style seeding so rare high-level states get their own clusters
        self.v = Z[[self.rng.integers(len(Z))]]
        while len(self.v) < self.k:
            d2 = ((Z[:, None, :] - self.v[None]) ** 2).sum(axis=2).min(axis=1)
            self.v = np.vstack([self.v, Z[self.rng.choice(len(Z), p=d2 / d2.sum())]])
        for _ in range(self.iters):
            um = self._membership(Z) ** self.m
            v_new = (um.T @ Z) / um.sum(axis=0)[:, None]
            shift = np.abs(v_new - self.v).max()
            self.v = v_new
            if shift < 1e-5:
                break
        um = self._membership(Z) ** self.m
        self.sigma = np.sqrt((um.T @ Z ** 2) / um.sum(axis=0)[:, None] - self.v ** 2).clip(1e-3)
        uw = um * sw[:, None]
        self.cons = (uw.T @ t) / uw.sum(axis=0)
        return self

    def predict(self, X):
        Z = self.scaler(X)
        W = np.exp(-0.5 * (((Z[:, None, :] - self.v[None]) / self.sigma[None]) ** 2).sum(axis=2))
        return (W @ self.cons) / (W.sum(axis=1) + 1e-12)


# ---------------------------------------------------------------- GA / PSO tuned IT2FIS
class TunedIT2:
    """Genes (10): per input, 3 anchor shifts in [-1, 1] x 0.4 x smaller neighbour gap; 1 FOU scale in
    [0, 3] applied to the data-derived FOU half-shifts. Gene vector of zeros + scale 1 = the original IT2FIS."""

    LO = np.array([-1.0] * 9 + [0.0])
    HI = np.array([1.0] * 9 + [3.0])
    BASE = np.array([0.0] * 9 + [1.0])

    def __init__(self, pipe, method: str, pop: int = 20, gens: int = 20, seed: int = 0):
        self.pipe, self.method, self.pop, self.gens = pipe, method, pop, gens
        self.rng = np.random.default_rng(seed)

    def _fis(self, g):
        cal = copy.deepcopy(self.pipe.cal)
        for i, name in enumerate(INPUTS):
            a = np.array(cal.anchors[name], float)
            gap = np.diff(a)
            room = 0.4 * np.minimum(np.r_[gap[0], gap], np.r_[gap, gap[-1]])
            cal.anchors[name] = (a + g[3 * i:3 * i + 3] * room).tolist()
            cal.fou_eps[name] = (np.array(cal.fou_eps[name]) * g[9]).tolist()
        return build_fis(cal)

    def _score(self, g, X, labels):
        try:
            fis = self._fis(g)
        except ValueError:  # degenerate partition (anchors collided) -> infeasible candidate
            return -1.0
        y =fis.infer_it2({k: (X[:, i], X[:, i]) for i, k in enumerate(INPUTS)})["y"]
        return _fitness(labels, y)

    def fit(self, X, labels):
        f = lambda g: self._score(g, X, labels)  # noqa: E731
        P = self.rng.uniform(self.LO, self.HI, (self.pop, len(self.LO)))
        P[0] = self.BASE  # the hand-calibrated system is always in the initial population
        fit = np.array([f(p) for p in P])
        if self.method == "ga":
            for _ in range(self.gens):
                elite = P[np.argsort(fit)[-2:]]
                children = []
                while len(children) < self.pop - 2:
                    i, j = (max(self.rng.choice(self.pop, 3), key=lambda k: fit[k]) for _ in range(2))
                    a = self.rng.uniform(-0.25, 1.25, len(self.LO))  # BLX-style arithmetic crossover
                    c = a * P[i] + (1 - a) * P[j]
                    mut = self.rng.random(len(c)) < 0.15  # Gaussian mutation
                    c[mut] += self.rng.normal(0, 0.1 * (self.HI - self.LO))[mut]
                    children.append(np.clip(c, self.LO, self.HI))
                P = np.vstack([elite, children])
                fit = np.array([f(p) for p in P])
            self.best = P[np.argmax(fit)]
        else:  # PSO, inertia 0.7, c1 = c2 = 1.5
            V = np.zeros_like(P)
            pbest, pfit = P.copy(), fit.copy()
            for _ in range(self.gens):
                gbest = pbest[np.argmax(pfit)]
                r1, r2 = self.rng.random(P.shape), self.rng.random(P.shape)
                V = 0.7 * V + 1.5 * r1 * (pbest - P) + 1.5 * r2 * (gbest - P)
                P = np.clip(P + V, self.LO, self.HI)
                fit = np.array([f(p) for p in P])
                better = fit > pfit
                pbest[better], pfit[better] = P[better], fit[better]
            self.best = pbest[np.argmax(pfit)]
        self.base_fitness, self.best_fitness = f(self.BASE), f(self.best)
        self.fis = self._fis(self.best)
        return self

    def predict(self, X):
        return self.fis.infer_it2({k: (X[:, i], X[:, i]) for i, k in enumerate(INPUTS)})["y"]


# ---------------------------------------------------------------- wrapper used by the comparison script
def _matrix(inputs: dict) -> np.ndarray:
    return np.column_stack([np.asarray(inputs[k], float) for k in INPUTS])


class SoftComputingModels:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.models: dict = {}
        self.info: dict = {}

    def fit(self, pipe, train, labels, progress=None):
        say = progress or (lambda m: None)
        X = _matrix(pipe.naive_inputs(train))
        ok = ~np.isnan(labels)
        n_fit = int(0.8 * len(train))
        fit_m, val_m = ok & (np.arange(len(train)) < n_fit), ok & (np.arange(len(train)) >= n_fit)
        y = np.nan_to_num(labels).astype(int)  # NaN rows are masked out by fit_m
        # one hyper-parameter per model, chosen on the validation block
        for name, make, grid in (("anfis", lambda r: ANFIS(pipe.cal, ridge=r), (0.01, 1.0, 10.0, 100.0)),
                                 ("fcm_tsk", lambda k: FCMTSK(n_clusters=k, seed=self.seed), (8, 16, 24, 32))):
            say(name)
            scored = []
            for h in grid:
                m = make(h).fit(X[fit_m], y[fit_m])
                scored.append((_fitness(labels[val_m], m.predict(X[val_m])), h, m))
            best = max(scored, key=lambda s: s[0])
            self.models[name] = best[2]
            self.info[name] = {"selected": best[1], "validation_fitness": {str(h): round(float(f), 3) for f, h, _ in scored}}
        for method in ("ga", "pso"):
            say(f"{method.upper()}-tuned IT2FIS")
            m = TunedIT2(pipe, method, seed=self.seed).fit(X[val_m], labels[val_m])
            self.models[f"{method}_it2"] = m
            self.info[f"{method}_it2"] = {"validation_fitness_original": float(m.base_fitness),
                                          "validation_fitness_tuned": float(m.best_fitness),
                                          "genes": m.best.round(3).tolist()}
        return self

    def predict_all(self, pipe, df) -> dict:
        naive = _matrix(pipe.naive_inputs(df))
        fused = _matrix(pipe.run_proposed(df, _calibrating=True)["point"])
        out = {}
        for name, m in self.models.items():
            out[name] = _risk_class(m.predict(naive))
            out[f"{name}+fusion"] = _risk_class(m.predict(fused))
        return out
