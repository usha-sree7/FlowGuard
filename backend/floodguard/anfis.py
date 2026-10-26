"""ANFIS: first-order Takagi-Sugeno-Kang neuro-fuzzy system (Jang 1993), learned from data.

Structure (same inputs and term grid as the IT2FIS, so the two are directly comparable):
  layer 1  Gaussian membership functions, 5 level x 4 rise x 4 rain terms, initialised from the calibrated anchors
  layer 2  rule firing w_r = product of the three memberships (80 rules, the same antecedent grid as the IT2FIS)
  layer 3  normalisation  wn_r = w_r / sum(w)
  layer 4  linear consequent f_r = p_r0 + p_r1 * level + p_r2 * rise + p_r3 * rain   (inputs scaled to ~[0, 1])
  layer 5  output y = sum_r wn_r f_r  -> risk score 0-100 -> alert level by the fixed band edges

Hybrid learning (Jang's): each epoch, (forward) consequent parameters by weighted ridge least squares with the
premise MFs fixed, then (backward) one Adam gradient step on the MF centres and widths. The premise step is
constrained so the fuzzy sets keep their meaning: each centre stays within half the distance to its neighbouring
centres (so the term order Normal < ... < VeryHigh is preserved) and each width within 0.5-2x its initial value.
Without these bounds the gradient step was observed to collapse widths to spikes and move "Elevated" below
"Normal" (Malwathu), which made the rule names in the explanations meaningless. Target = the centre score of
the proxy class (10 / 30 / 50 / 70 / 90); samples are weighted by inverse square-root class frequency so the rare
High / Critical classes are not ignored.

Data use: fitted on the FIT block (first 80 % of the training split); the ridge strength is chosen on the VALIDATION
block (last 20 %) with the same fitness as every other model. The test split is never seen.
"""
from __future__ import annotations

import numpy as np

from .config import N_CLASSES
from .fuzzy.fis import INPUTS, RULES, TERMS
from .patterns import CLASS_SCORE, _fitness


def _risk_class(y):
    from .systems import risk_class
    return risk_class(y)


class Scaler:
    """Robust min-max to roughly [0, 1] per input (no clipping, so extremes stay ordered)."""

    def __init__(self, X):
        self.lo = np.nanquantile(X, 0.001, axis=0)
        self.span = np.maximum(np.nanquantile(X, 0.999, axis=0) - self.lo, 1e-9)

    def __call__(self, X):
        return (X - self.lo) / self.span


def class_weights(y_cls):
    """Inverse square-root class frequency."""
    freq = np.bincount(y_cls, minlength=N_CLASSES) / len(y_cls)
    return 1.0 / np.sqrt(np.maximum(freq, 1e-6))[y_cls]


class ANFIS:
    def __init__(self, cal, epochs: int = 40, lr: float = 0.01, ridge: float = 1e-3):
        self.cal, self.epochs, self.lr, self.ridge = cal, epochs, lr, ridge
        self.ant = np.array([[L, R, P] for L, R, P, _ in RULES])
        self.n_terms = [len(TERMS[n]) for n in INPUTS]

    def _init_mfs(self):
        c, s = [], []
        for i, name in enumerate(INPUTS):
            a = (np.array(self.cal.anchors[name], float) - self.scaler.lo[i]) / self.scaler.span[i]
            g = np.diff(a).mean()
            c.append(np.r_[a[0] - g / 2, (a[:-1] + a[1:]) / 2, a[-1] + g / 2])
            s.append(np.full(len(a) + 1, max(g / 2, 1e-3)))
        # interpretability bounds (see module docstring)
        self.c_lo, self.c_hi, self.s_lo, self.s_hi = [], [], [], []
        for ci, si in zip(c, s):
            d = np.diff(ci)
            left, right = np.r_[d[0], d], np.r_[d, d[-1]]
            self.c_lo.append(ci - left / 2)
            self.c_hi.append(ci + right / 2)
            self.s_lo.append(0.5 * si)
            self.s_hi.append(2.0 * si)
        return c, s

    def _mu(self, Z, i):
        return np.exp(-0.5 * ((Z[:, i, None] - self.c[i][None]) / self.s[i][None]) ** 2)  # (N, terms_i)

    def _firing(self, Z):
        W = np.ones((len(Z), len(RULES)))
        for i in range(len(INPUTS)):
            W *= self._mu(Z, i)[:, self.ant[:, i]]
        return W, W.sum(axis=1) + 1e-12

    def _design(self, Z, W, S):
        Wn = W / S[:, None]
        Z1 = np.column_stack([np.ones(len(Z)), Z])
        return (Wn[:, :, None] * Z1[:, None, :]).reshape(len(Z), -1), Z1

    def fit(self, X, y_cls):
        self.scaler = Scaler(X)
        Z, t, sw = self.scaler(X), CLASS_SCORE[y_cls], class_weights(y_cls)
        self.c, self.s = self._init_mfs()
        m = {k: [np.zeros_like(v) for v in getattr(self, k)] for k in ("c", "s")}
        v = {k: [np.zeros_like(x) for x in getattr(self, k)] for k in ("c", "s")}
        self.loss = []
        for ep in range(self.epochs + 1):
            W, S = self._firing(Z)
            A, Z1 = self._design(Z, W, S)
            # forward pass: consequents by weighted ridge least squares
            Aw = A * sw[:, None]
            self.theta = np.linalg.solve(Aw.T @ A + self.ridge * np.eye(A.shape[1]), Aw.T @ t)
            F = Z1 @ self.theta.reshape(len(RULES), -1).T  # rule outputs (N, R)
            y = (W * F).sum(axis=1) / S
            self.loss.append(float(np.sqrt(np.average((y - t) ** 2, weights=sw))))
            if ep == self.epochs:
                break
            # backward pass: gradient of the weighted squared error w.r.t. MF centres and widths
            G = (sw * (y - t) / len(Z))[:, None] * (F - y[:, None]) / S[:, None] * W  # dE/dlog w_r
            for i in range(len(INPUTS)):
                gc, gs = np.zeros_like(self.c[i]), np.zeros_like(self.s[i])
                for j in range(self.n_terms[i]):
                    g = G[:, self.ant[:, i] == j].sum(axis=1)
                    d = Z[:, i] - self.c[i][j]
                    gc[j] = (g * d / self.s[i][j] ** 2).sum()
                    gs[j] = (g * d ** 2 / self.s[i][j] ** 3).sum()
                for k, grad in (("c", gc), ("s", gs)):  # Adam
                    m[k][i] = 0.9 * m[k][i] + 0.1 * grad
                    v[k][i] = 0.999 * v[k][i] + 0.001 * grad ** 2
                    step = self.lr * (m[k][i] / (1 - 0.9 ** (ep + 1))) / (np.sqrt(v[k][i] / (1 - 0.999 ** (ep + 1))) + 1e-8)
                    getattr(self, k)[i] = getattr(self, k)[i] - step
                self.c[i] = np.clip(self.c[i], self.c_lo[i], self.c_hi[i])
                self.s[i] = np.clip(self.s[i], self.s_lo[i], self.s_hi[i])
        return self

    def predict(self, X):
        Z = self.scaler(X)
        W, S = self._firing(Z)
        A, _ = self._design(Z, W, S)
        return np.clip(A @ self.theta, 0.0, 100.0)

    def explain(self, x: np.ndarray, k: int = 4) -> list[dict]:
        """Rules with the largest normalised firing for one input vector, with their linear consequent output."""
        Z = self.scaler(np.asarray(x, float)[None])
        W, S = self._firing(Z)
        wn = W[0] / S[0]
        F = np.r_[1.0, Z[0]] @ self.theta.reshape(len(RULES), -1).T
        out = []
        for r in np.argsort(-wn)[:k]:
            L, R, P, _ = RULES[r]
            out.append({"if": {"level": TERMS["level"][L], "rise": TERMS["rise"][R], "rain": TERMS["rain"][P]},
                        "weight": round(float(wn[r]), 4), "rule_output": round(float(F[r]), 2),
                        "contribution": round(float(wn[r] * F[r]), 2)})
        return out

    def state(self) -> dict:
        """Full fitted state (full precision), so a served model is identical to the evaluated one."""
        L = lambda v: [np.asarray(x).tolist() for x in v]  # noqa: E731
        return {"epochs": self.epochs, "lr": self.lr, "ridge": self.ridge, "theta": self.theta.tolist(),
                "scaler_lo": self.scaler.lo.tolist(), "scaler_span": self.scaler.span.tolist(),
                "c": L(self.c), "s": L(self.s), "loss": self.loss}

    @classmethod
    def from_state(cls, cal, st: dict) -> "ANFIS":
        m = cls(cal, st["epochs"], st["lr"], st["ridge"])
        m.scaler = Scaler.__new__(Scaler)
        m.scaler.lo, m.scaler.span = np.asarray(st["scaler_lo"]), np.asarray(st["scaler_span"])
        m.c, m.s = [np.asarray(x) for x in st["c"]], [np.asarray(x) for x in st["s"]]
        m.theta, m.loss = np.asarray(st["theta"]), st["loss"]
        return m

    def describe(self) -> dict:
        mfs = {}
        for i, n in enumerate(INPUTS):
            lo, sp = self.scaler.lo[i], self.scaler.span[i]
            mfs[n] = [{"term": TERMS[n][j], "centre": round(float(lo + self.c[i][j] * sp), 4),
                       "sigma": round(float(self.s[i][j] * sp), 4)} for j in range(self.n_terms[i])]
        return {"rules": len(RULES), "consequent_parameters": int(self.theta.size),
                "premise_parameters": int(sum(2 * n for n in self.n_terms)), "epochs": self.epochs,
                "ridge": self.ridge, "learning_rate": self.lr, "membership_functions": mfs,
                "training_rmse_per_epoch": [round(x, 3) for x in self.loss]}


def select_anfis(cal, X_fit, y_fit, X_val, y_val, grid=(0.01, 1.0, 100.0)) -> tuple[ANFIS, dict]:
    """Ridge strength chosen on the validation block."""
    scored = []
    for r in grid:
        m = ANFIS(cal, ridge=r).fit(X_fit, y_fit)
        scored.append((_fitness(y_val, _risk_class(m.predict(X_val))), r, m))
    best = max(scored, key=lambda s: s[0])
    return best[2], {"selected_ridge": best[1], "validation_fitness": {str(r): round(float(f), 4) for f, r, _ in scored}}
