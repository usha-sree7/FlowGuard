"""Pattern recognition layer: Self-Organising Map (known hydrological regimes) and Fuzzy ART (unfamiliar patterns).

Both work on the same three FIS inputs x = [level, rise, rain] and are fitted on the FIT block of the clean
training split only; their few hyper-parameters are chosen on the VALIDATION block (last 20 % of training).

  SOM        Kohonen (1990) batch SOM on z-scored inputs. Every node is a prototype hydrological state.
             Node -> risk class is learned from the fit-block proxy labels by MINIMUM EXPECTED COST under the
             same asymmetric ordinal cost the evaluation uses (missing a class step costs 2, a false step 1),
             so rare High / Critical states are not out-voted by the majority Normal class.
  Fuzzy ART  Carpenter, Grossberg & Rosen (1991), complement coding, fast learning. Categories are hyperboxes
             around the training states. At run time the ART does NOT learn: a state is UNFAMILIAR when no
             category resonates at the vigilance rho, or when any input lies outside the training envelope.
             Familiarity is not flood risk and novelty is not a fault - it only means "this combination of
             readings was not seen in training".
  SOM + ART  SOM risk class where ART says the pattern is familiar; where it is unfamiliar the SOM mapping has no
             training evidence, so the output falls back to the physically grounded threshold rule.
"""
from __future__ import annotations

import numpy as np

from .config import N_CLASSES, RISK_EDGES
from .metrics import evaluate

# class-centre score on the 0-100 scale (centre of each RISK_EDGES band)
CLASS_SCORE = np.convolve(np.r_[0.0, RISK_EDGES, 100.0], [0.5, 0.5], "valid")
# cost[t, p] of predicting p when the proxy label is t: under-warning 2 per class step, over-warning 1 per step
COST = np.array([[(p - t) if p >= t else 2 * (t - p) for p in range(N_CLASSES)] for t in range(N_CLASSES)], float)


def _fitness(labels, cls) -> float:
    """Model-selection fitness on the validation block: alert F1 + 0.25 x 5-class macro F1."""
    ev = evaluate(labels, cls)
    return np.nan_to_num(ev["binary_alert"]["f1"]) + 0.25 * np.nan_to_num(ev["ordinal"]["macro_f1"])


# ---------------------------------------------------------------- SOM
class SOM:
    def __init__(self, rows: int = 6, cols: int = 6, iters: int = 25, seed: int = 0):
        self.rows, self.cols, self.iters, self.seed = rows, cols, iters, seed
        self.grid = np.array([(r, c) for r in range(rows) for c in range(cols)], float)

    def _z(self, X):
        return np.clip((X - self.mu) / self.sd, -12, 12)

    def bmu(self, X) -> np.ndarray:
        Z = self._z(X)
        d = (Z ** 2).sum(1)[:, None] - 2 * Z @ self.W.T + (self.W ** 2).sum(1)[None]
        return d.argmin(axis=1)

    def fit(self, X):
        """Batch SOM, Gaussian neighbourhood shrinking from half the map to 0.5 node; codebook initialised
        from random training samples (seeded)."""
        rng = np.random.default_rng(self.seed)
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        Z = self._z(X)
        k = len(self.grid)
        self.W = Z[rng.choice(len(Z), k, replace=False)].copy()
        g2 = ((self.grid[:, None] - self.grid[None]) ** 2).sum(-1)
        s0 = max(self.rows, self.cols) / 2
        for it in range(self.iters):
            sigma = s0 * (0.5 / s0) ** (it / max(self.iters - 1, 1))
            Hn = np.exp(-g2 / (2 * sigma ** 2))[self.bmu(X)]  # (N, k) neighbourhood weights
            self.W = (Hn.T @ Z) / (Hn.sum(0)[:, None] + 1e-12)
        b = self.bmu(X)
        self.hits = np.bincount(b, minlength=k)
        self.quant_error = float(np.sqrt(((Z - self.W[b]) ** 2).sum(1)).mean())
        return self

    def map_risk(self, X, y_cls):
        """Learn node -> risk class (minimum expected cost); nodes without labelled hits copy the class of the
        nearest node that has some (Euclidean distance between prototypes)."""
        b = self.bmu(X)
        k = len(self.grid)
        self.label_counts = np.zeros((k, N_CLASSES), int)
        np.add.at(self.label_counts, (b, y_cls), 1)
        has = self.label_counts.sum(1) > 0
        exp_cost = self.label_counts @ COST  # (k, K): sum over labels of cost of each prediction
        cls = exp_cost.argmin(axis=1)
        if not has.all():
            d = ((self.W[:, None] - self.W[None]) ** 2).sum(-1)
            d[:, ~has] = np.inf
            cls[~has] = cls[d[~has].argmin(axis=1)]
        self.node_class = cls
        return self

    def predict_cls(self, X) -> np.ndarray:
        return self.node_class[self.bmu(X)]

    def prototypes(self) -> np.ndarray:
        """Codebook in original units (level m, rise m/h, rain accumulated)."""
        return self.W * self.sd + self.mu

    STATE = ("W", "mu", "sd", "hits", "label_counts", "node_class")

    def state(self) -> dict:
        """Full fitted state (full precision), so a served model is identical to the evaluated one."""
        return {"rows": self.rows, "cols": self.cols, "iters": self.iters, "seed": self.seed,
                "quant_error": self.quant_error, **{k: getattr(self, k).tolist() for k in self.STATE}}

    @classmethod
    def from_state(cls, st: dict) -> "SOM":
        som = cls(st["rows"], st["cols"], st["iters"], st["seed"])
        som.quant_error = st["quant_error"]
        for k in cls.STATE:
            setattr(som, k, np.asarray(st[k], int if k in ("hits", "label_counts", "node_class") else float))
        return som

    def describe(self, names=("level", "rise", "rain")) -> dict:
        P = self.prototypes()
        return {"rows": self.rows, "cols": self.cols, "iters": self.iters, "seed": self.seed,
                "quantisation_error_z": round(self.quant_error, 4),
                "nodes": [{"id": i, "row": int(self.grid[i, 0]), "col": int(self.grid[i, 1]),
                           "prototype": {n: round(float(P[i, j]), 4) for j, n in enumerate(names)},
                           "hits": int(self.hits[i]),
                           "label_counts": self.label_counts[i].tolist() if hasattr(self, "label_counts") else None,
                           "risk_class": int(self.node_class[i]) if hasattr(self, "node_class") else None}
                          for i in range(len(self.grid))]}


# ---------------------------------------------------------------- Fuzzy ART
class FuzzyART:
    def __init__(self, rho: float = 0.9, alpha: float = 1e-3, beta: float = 1.0, epochs: int = 2):
        self.rho, self.alpha, self.beta, self.epochs = rho, alpha, beta, epochs

    def _scale(self, X):
        return (X - self.lo) / self.span

    def _code(self, A):
        A = np.clip(A, 0.0, 1.0)
        return np.hstack([A, 1.0 - A])  # complement coding: |I| = M for every input

    def fit(self, X):
        """X: training states (original units). Inputs are min-max scaled to the training envelope."""
        self.lo, hi = X.min(axis=0), X.max(axis=0)
        self.span = np.maximum(hi - self.lo, 1e-9)
        # identical states give identical updates: learn each distinct (rounded) state once per epoch
        I = np.unique(np.round(self._code(self._scale(X)), 4), axis=0)
        order = np.random.default_rng(0).permutation(len(I))
        M = X.shape[1]
        W = np.empty((0, 2 * M))
        for _ in range(self.epochs):
            changed = False
            for x in I[order]:
                if len(W):
                    inter = np.minimum(x, W).sum(1)
                    T = inter / (self.alpha + W.sum(1))
                    for j in np.argsort(-T):  # search categories in order of choice value
                        if inter[j] / M >= self.rho:  # resonance
                            new = self.beta * np.minimum(x, W[j]) + (1 - self.beta) * W[j]
                            changed |= not np.allclose(new, W[j])
                            W[j] = new
                            break
                    else:
                        W = np.vstack([W, x])
                        changed = True
                else:
                    W = x[None].copy()
                    changed = True
            if not changed:
                break
        self.W = W
        return self

    def familiarity(self, X) -> np.ndarray:
        """max_j |I ^ w_j| / |I|: 1 inside a learned box, decreasing with the distance outside it."""
        I = self._code(self._scale(X))
        best = np.zeros(len(I))
        for s in range(0, len(I), 4096):  # chunked to bound memory
            blk = I[s:s + 4096]
            best[s:s + 4096] = np.minimum(blk[:, None, :], self.W[None]).sum(2).max(1) / (I.shape[1] / 2)
        return best

    def outside_envelope(self, X) -> np.ndarray:
        A = self._scale(X)
        return ((A < 0) | (A > 1)).any(axis=1)

    def novelty(self, X) -> dict:
        fam = self.familiarity(X)
        out = self.outside_envelope(X)
        return {"familiarity": fam, "outside_envelope": out, "novel": (fam < self.rho) | out}

    def describe(self, names=("level", "rise", "rain")) -> dict:
        M = len(names)
        lo, hi = self.W[:, :M], 1.0 - self.W[:, M:]
        size = (hi - lo).sum(1)
        top = np.argsort(-size)[:12]
        return {"rho": self.rho, "alpha": self.alpha, "beta": self.beta, "categories": int(len(self.W)),
                "envelope": {n: [round(float(self.lo[j]), 4), round(float(self.lo[j] + self.span[j]), 4)]
                             for j, n in enumerate(names)},
                "largest_boxes": [{n: [round(float(self.lo[j] + lo[i, j] * self.span[j]), 4),
                                       round(float(self.lo[j] + hi[i, j] * self.span[j]), 4)]
                                   for j, n in enumerate(names)} for i in top]}


def select_rho(X_fit, X_val, budget: float = 0.01, grid=(0.80, 0.85, 0.90, 0.93, 0.95)) -> tuple[FuzzyART, dict]:
    """Largest vigilance whose UNFAMILIAR rate on the clean validation block stays within the false-novelty
    budget (default 1 %). Unsupervised: no flood labels and no simulated faults are used for this choice."""
    rates, best = {}, None
    for rho in grid:
        art = FuzzyART(rho=rho).fit(X_fit)
        rates[str(rho)] = round(float(art.novelty(X_val)["novel"].mean()), 4)
        if rates[str(rho)] <= budget:
            best = art
    if best is None:  # even the coarsest vigilance exceeds the budget: keep the coarsest and report it
        best = FuzzyART(rho=grid[0]).fit(X_fit)
    return best, {"budget": budget, "validation_novelty_rate": rates, "selected_rho": best.rho,
                  "categories": int(len(best.W))}


def select_som(X_fit, y_fit, X_val, y_val, seed: int, grid=((4, 4), (6, 6), (8, 8), (10, 10))) -> tuple[SOM, dict]:
    scored = []
    for r, c in grid:
        som = SOM(r, c, seed=seed).fit(X_fit).map_risk(X_fit, y_fit)
        scored.append((_fitness(y_val, som.predict_cls(X_val)), f"{r}x{c}", som))
    best = max(scored, key=lambda s: s[0])
    return best[2], {"selected": best[1], "validation_fitness": {g: round(float(f), 4) for f, g, _ in scored}}


def som_art_cls(som_cls, novel, threshold_cls) -> np.ndarray:
    """SOM + ART: trust the SOM regime mapping only for familiar patterns; fall back to the threshold rule."""
    return np.where(novel, threshold_cls, som_cls)
