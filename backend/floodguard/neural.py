"""Option A hybrid: GA feature selection + SOM regime detection + ANN classifier, built as an ablation ladder.

  ann         : multilayer perceptron (1 hidden tanh layer, softmax over the 4 risk classes, Adam)
  som_ann     : + a self-organising map that clusters the input space into flow regimes; the winning SOM node
                (one-hot) is appended to the ANN input, so the network can behave differently per regime
  ga_som_ann  : + a binary genetic algorithm that selects which candidate features feed the SOM and the ANN

All three see candidate features derived ONLY from the three FIS inputs (level, rise, rain) and their recent past
(lags, changes, rolling statistics), computed causally, so they can run on naive or fused inputs alike.
predict() returns the class-centre score so the shared risk_class() mapping applies unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLASS_SCORE = np.array([10.0, 37.5, 62.5, 90.0])

FEATURES = ("level", "rise", "rain", "level_lag1", "level_lag2", "level_lag4", "dlevel_2", "dlevel_4",
            "dlevel_8", "rise_mean3", "level_max8", "rain_lag1", "rain_mean4", "rain_max12")


def candidate_features(X: np.ndarray) -> np.ndarray:
    """X: (N, 3) time-ordered [level, rise, rain] -> (N, len(FEATURES)). Causal (past values only)."""
    L, R, P = (pd.Series(X[:, i]) for i in range(3))
    cols = [L, R, P, L.shift(1), L.shift(2), L.shift(4), L - L.shift(2), L - L.shift(4), L - L.shift(8),
            R.rolling(3, min_periods=1).mean(), L.rolling(8, min_periods=1).max(), P.shift(1),
            P.rolling(4, min_periods=1).mean(), P.rolling(12, min_periods=1).max()]
    return np.column_stack([c.bfill().fillna(0.0).to_numpy(float) for c in cols])  # bfill only the warm-up rows


def _class_weights(y):
    freq = np.bincount(y, minlength=4) / len(y)
    return 1.0 / np.sqrt(np.maximum(freq, 1e-6))


class Standardizer:
    def __init__(self, F):
        self.mu, self.sd = F.mean(axis=0), F.std(axis=0) + 1e-9

    def __call__(self, F):
        return np.clip((F - self.mu) / self.sd, -8, 8)


# ---------------------------------------------------------------- ANN
class MLP:
    def __init__(self, hidden: int = 16, epochs: int = 60, lr: float = 3e-3, batch: int = 256, seed: int = 0):
        self.h, self.epochs, self.lr, self.batch = hidden, epochs, lr, batch
        self.rng = np.random.default_rng(seed)

    def _forward(self, Z):
        H = np.tanh(Z @ self.W1 + self.b1)
        logits = H @ self.W2 + self.b2
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return H, e / e.sum(axis=1, keepdims=True)

    def fit(self, Z, y, Zv=None, yv=None, patience: int = 8):
        d = Z.shape[1]
        self.W1 = self.rng.normal(0, 1 / np.sqrt(d), (d, self.h))
        self.b1 = np.zeros(self.h)
        self.W2 = self.rng.normal(0, 1 / np.sqrt(self.h), (self.h, 4))
        self.b2 = np.zeros(4)
        params = ["W1", "b1", "W2", "b2"]
        m = {p: np.zeros_like(getattr(self, p)) for p in params}
        v = {p: np.zeros_like(getattr(self, p)) for p in params}
        cw = _class_weights(y)
        Y = np.eye(4)[y]
        best, best_loss, wait, t = None, np.inf, 0, 0
        for _ in range(self.epochs):
            for idx in np.array_split(self.rng.permutation(len(Z)), max(1, len(Z) // self.batch)):
                H, Pr = self._forward(Z[idx])
                sw = cw[y[idx]][:, None] / len(idx)
                dlog = (Pr - Y[idx]) * sw  # weighted softmax cross-entropy gradient
                dH = (dlog @ self.W2.T) * (1 - H ** 2)
                grads = {"W2": H.T @ dlog, "b2": dlog.sum(0), "W1": Z[idx].T @ dH, "b1": dH.sum(0)}
                t += 1
                for p in params:  # Adam + small L2
                    g = grads[p] + (1e-4 * getattr(self, p) if p.startswith("W") else 0)
                    m[p] = 0.9 * m[p] + 0.1 * g
                    v[p] = 0.999 * v[p] + 0.001 * g ** 2
                    setattr(self, p, getattr(self, p) - self.lr * (m[p] / (1 - 0.9 ** t))
                            / (np.sqrt(v[p] / (1 - 0.999 ** t)) + 1e-8))
            if Zv is not None:  # early stopping on weighted validation loss
                _, Pv = self._forward(Zv)
                loss = -(cw[yv] * np.log(Pv[np.arange(len(yv)), yv] + 1e-12)).mean()
                if loss < best_loss - 1e-4:
                    best, best_loss, wait = {p: getattr(self, p).copy() for p in params}, loss, 0
                else:
                    wait += 1
                    if wait >= patience:
                        break
        if best:
            for p in params:
                setattr(self, p, best[p])
        return self

    def predict_cls(self, Z):
        return self._forward(Z)[1].argmax(axis=1)


# ---------------------------------------------------------------- SOM
class SOM:
    def __init__(self, rows: int = 4, cols: int = 4, iters: int = 20, seed: int = 0):
        self.rows, self.cols, self.iters = rows, cols, iters
        self.rng = np.random.default_rng(seed)
        self.grid = np.array([(r, c) for r in range(rows) for c in range(cols)], float)

    def bmu(self, Z):
        d = (Z ** 2).sum(1)[:, None] - 2 * Z @ self.W.T + (self.W ** 2).sum(1)[None]
        return d.argmin(axis=1)

    def fit(self, Z):
        """Batch SOM with a Gaussian neighbourhood shrinking from half the map to < 1 node."""
        k = len(self.grid)
        self.W = Z[self.rng.choice(len(Z), k, replace=False)].copy()
        g2 = ((self.grid[:, None] - self.grid[None]) ** 2).sum(-1)
        s0 = max(self.rows, self.cols) / 2
        for it in range(self.iters):
            sigma = s0 * (0.5 / s0) ** (it / max(self.iters - 1, 1))
            Hn = np.exp(-g2 / (2 * sigma ** 2))[self.bmu(Z)]  # (N, k) neighbourhood weights
            self.W = (Hn.T @ Z) / (Hn.sum(0)[:, None] + 1e-12)
        return self

    def onehot(self, Z):
        return np.eye(len(self.grid))[self.bmu(Z)]


# ---------------------------------------------------------------- the three ablation stages
class HybridNet:
    def __init__(self, use_som: bool, mask=None, hidden: int = 16, epochs: int = 60, seed: int = 0):
        self.use_som, self.hidden, self.epochs, self.seed = use_som, hidden, epochs, seed
        self.mask = np.ones(len(FEATURES), bool) if mask is None else np.asarray(mask, bool)

    def _inputs(self, F):
        Z = self.std(F[:, self.mask])
        return np.hstack([Z, self.som.onehot(Z)]) if self.use_som else Z

    def fit(self, F, y, Fv=None, yv=None):
        self.std = Standardizer(F[:, self.mask])
        if self.use_som:
            self.som = SOM(seed=self.seed).fit(self.std(F[:, self.mask]))
        self.net = MLP(hidden=self.hidden, epochs=self.epochs, seed=self.seed).fit(
            self._inputs(F), y, None if Fv is None else self._inputs(Fv), yv)
        return self

    def predict_features(self, F):
        return CLASS_SCORE[self.net.predict_cls(self._inputs(F))]

    def predict(self, X):  # X: (N, 3) time-ordered [level, rise, rain]
        return self.predict_features(candidate_features(X))


def ga_select(F, y, Fv, yv, fitness, pop: int = 12, gens: int = 8, seed: int = 0, sub: int = 15000, say=None):
    """Binary GA over feature masks. Fitness of a mask = validation fitness of a quickly trained SOM+ANN
    minus 0.002 per selected feature (prefers compact subsets). Returns (best mask, history)."""
    rng = np.random.default_rng(seed)
    n = len(FEATURES)
    idx = np.sort(rng.choice(len(F), min(sub, len(F)), replace=False))
    cache: dict = {}

    def score(mask):
        key = mask.tobytes()
        if key not in cache:
            if not mask.any():
                cache[key] = -1.0
            else:
                m = HybridNet(True, mask, epochs=25, seed=seed).fit(F[idx], y[idx], Fv, yv)
                cache[key] = fitness(yv, m.predict_features(Fv)) - 0.002 * mask.sum()
        return cache[key]

    P = rng.random((pop, n)) < 0.5
    P[0] = True  # all features
    P[1] = False
    P[1, :3] = True  # only the three FIS inputs
    fit = np.array([score(p) for p in P])
    history = [float(fit.max())]
    for g in range(gens):
        elite = P[np.argsort(fit)[-2:]]
        kids = []
        while len(kids) < pop - 2:
            a, b = (P[max(rng.choice(pop, 3), key=lambda k: fit[k])] for _ in range(2))
            child = np.where(rng.random(n) < 0.5, a, b)  # uniform crossover
            child ^= rng.random(n) < 1.0 / n  # bit-flip mutation
            kids.append(child)
        P = np.vstack([elite, kids])
        fit = np.array([score(p) for p in P])
        history.append(float(fit.max()))
        if say:
            say(f"GA generation {g + 1}/{gens}: best fitness {fit.max():.3f}")
    return P[np.argmax(fit)], history
