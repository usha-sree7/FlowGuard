"""Alert messages with explanations, generated from the ACTUAL model outputs at one time step.

Nothing here is a template with fixed example values: every number and sentence is derived from the inputs, the
model internals (fired fuzzy rules, SOM prototype, ANFIS rule outputs), the reliability flags and the confidence
status that the models produced. Alerts are decision-support demonstrations, not official warnings.
"""
from __future__ import annotations

import numpy as np

from . import config
from .fuzzy import INPUTS, TERMS
from .fuzzy.fis import rule_table
from .suite import MODEL_NAMES

ADVICE = ["No action needed.", "Keep an eye on the river.", "Be prepared: the river is clearly above normal.",
          "Flood threshold exceeded or expected: follow local warnings.", "Severe: level far above the flood threshold."]
FLAG_TEXT = {"missing": "is missing", "out_of_range": "is outside the plausible range", "spike": "looks like a spike",
             "stuck": "appears stuck (flat while rain falls)", "noisy": "is unusually noisy",
             "inconsistent": "disagrees with the other rain sensor(s)"}


def dominant_terms(fis, x: dict) -> dict:
    """Linguistic term with the highest embedded (Type-1) membership for each crisp input."""
    out = {}
    for name in INPUTS:
        mu = np.array([float(m.embedded(np.array([x[name]]))[0]) for m in fis.mfs[name]])
        out[name] = {"term": TERMS[name][int(mu.argmax())], "membership": round(float(mu.max()), 3),
                     "all": {TERMS[name][j]: round(float(v), 3) for j, v in enumerate(mu)}}
    return out


def top_rules(fis, x: dict, k: int = 4) -> list[dict]:
    it2 = fis.infer_it2({n: (np.array([x[n]]), np.array([x[n]])) for n in INPUTS})
    rules = rule_table()
    order = np.argsort(-it2["f_hi"][0])[:k]
    return [{**rules[i], "firing": [round(float(it2["f_lo"][0][i]), 4), round(float(it2["f_hi"][0][i]), 4)]}
            for i in order if it2["f_hi"][0][i] > 0]


def _level_position(level: float, th: list) -> str:
    names = config.RISK_CLASSES
    k = int(np.digitize(level, th))
    if k == 0:
        return f"below the Low threshold ({th[0]:.2f} m)"
    if k == len(th):
        return f"above the Critical threshold ({th[-1]:.2f} m)"
    return f"between the {names[k]} ({th[k - 1]:.2f} m) and {names[k + 1]} ({th[k]:.2f} m) thresholds"


def sensor_state(sh: dict, cal, i: int) -> tuple[list[dict], float]:
    rel = sh["proposed"]["rel"]
    lv = rel["level"]
    sensors = [{"sensor": "water level", "reliability": round(float(lv["r"][i]), 3),
                "flags": [k for k in ("missing", "out_of_range", "spike", "stuck", "noisy") if bool(lv[k][i])]}]
    for s in cal.cfg.rain_sensors:
        rs = rel["rain"][s]
        sensors.append({"sensor": f"rain ({s})", "reliability": round(float(rs["r"][i]), 3),
                        "flags": [k for k in ("missing", "out_of_range", "stuck", "inconsistent") if bool(rs[k][i])]})
    return sensors, float(sh["proposed"]["fused"]["conflict"][i])


def explain(suite, sh: dict, hz: dict, h: int, i: int, model: str, mode: str = "raw", demo: bool = False) -> dict:
    pipe = suite.pipe
    cal = pipe.cal
    X = sh["X"][mode]
    x = {n: float(X[i, j]) for j, n in enumerate(INPUTS)}
    cls = int(hz["cls"][mode][model][i])
    risk = config.RISK_CLASSES[cls]
    fis = suite.pso[h].fis if model == "pso_it2" else pipe.fis
    terms = dominant_terms(fis, x)
    rain_h = cal.cfg.rain_window_steps * cal.cfg.step_hours
    hl = suite.horizons[h]
    score = hz["score"][mode].get(model)
    score_i = None if score is None else float(score[i])

    reasons = [f"Inputs ({'reliability-fused estimates' if mode == 'fused' else 'raw sensor readings'}): water level "
               f"{x['level']:.2f} m ({terms['level']['term']}), {_level_position(x['level'], cal.thresholds)}; rate of "
               f"rise {x['rise']:+.3f} m/h ({terms['rise']['term']}); rainfall over the last {rain_h:g} h "
               f"{x['rain']:.1f} ({terms['rain']['term']})."]
    detail: dict = {}
    if model in ("it2_fis", "pso_it2"):
        rules = top_rules(fis, x)
        detail["rules"] = rules
        it2 = fis.infer_it2({n: (np.array([x[n]]), np.array([x[n]])) for n in INPUTS})
        detail["interval"] = [float(it2["yl"][0]), float(it2["yr"][0])]
        if rules:
            r = rules[0]
            reasons.append(f"Strongest rule: IF level is {r['if']['level']} AND rise is {r['if']['rise']} AND rain is "
                           f"{r['if']['rain']} THEN risk is {r['then']} (firing strength between {r['firing'][0]:.2f} "
                           f"and {r['firing'][1]:.2f}).")
        reasons.append(f"Type reduction gives the risk interval [{detail['interval'][0]:.1f}, {detail['interval'][1]:.1f}]; "
                       f"its midpoint {score_i:.1f}/100 falls in the {risk} band "
                       f"({_band(cls)}).")
        if model == "pso_it2":
            moved = [(n, o, t) for n in INPUTS for o, t in zip(cal.anchors[n], suite.pso[h].cal.anchors[n])
                     if abs(t - o) > 1e-9]
            if moved:
                n, o, t = max(moved, key=lambda m: abs(m[2] - m[1]) / max(abs(m[1]), 1e-9))
                reasons.append(f"PSO moved the membership functions for this horizon (largest relative change: a "
                               f"{n} crossover from {o:.3f} to {t:.3f}) and scaled the FOU by "
                               f"{suite.pso[h].genes[-1]:.2f}.")
    elif model == "som":
        som = suite.som[h]
        node = int(hz["som_node"][mode][i])
        proto = som.prototypes()[node]
        counts = som.label_counts[node].tolist()
        detail["som"] = {"node": node, "prototype": {n: float(proto[j]) for j, n in enumerate(INPUTS)},
                         "label_counts": counts, "grid": [som.rows, som.cols]}
        seen = ", ".join(f"{c} {n}" for c, n in zip(config.RISK_CLASSES, counts) if n) or "none"
        reasons.append(f"The closest learned prototype is SOM node {node} (level {proto[0]:.2f} m, rise "
                       f"{proto[1]:+.3f} m/h, rain {proto[2]:.1f}).")
        reasons.append(f"Training states matched to this node were later labelled: {seen}. The node is mapped to "
                       f"{risk} because that choice has the lowest expected cost (missing a level costs twice as much "
                       f"as a false step up).")
    elif model == "anfis":
        rules = suite.anfis[h].explain(X[i])
        detail["anfis_rules"] = rules
        r = rules[0]
        reasons.append(f"Most active rule: IF level is {r['if']['level']} AND rise is {r['if']['rise']} AND rain is "
                       f"{r['if']['rain']} (normalised weight {r['weight']:.2f}); its learned linear output is "
                       f"{r['rule_output']:.1f}.")
        reasons.append(f"The weighted sum of all 80 rule outputs is {score_i:.1f}/100, in the {risk} band ({_band(cls)}).")
    else:
        reasons.append(f"The reference rule reports the class of the current level: {risk}.")

    sensors, conflict = sensor_state(sh, cal, i)
    for s in sensors:
        if s["flags"] or s["reliability"] < 0.5:
            what = "; ".join(FLAG_TEXT.get(f, f) for f in s["flags"]) or "has low reliability"
            reasons.append(f"The {s['sensor']} sensor {what} (reliability {s['reliability']:.2f})"
                           + (", so the reliability layer down-weighted it before the model ran." if mode == "fused"
                              else "; in raw mode the reading is used as-is."))
    if conflict > 0:
        reasons.append(f"Rain sensors disagree in a way that cannot be pinned on one sensor (conflict {conflict:.2f}).")
    status, U = None, None
    if mode == "fused":
        p = sh["proposed"]
        status, U = str(p["status"][i]), float(p["U"][i])
        reasons.append(f"Confidence: {status} (uncertainty index U = {U:.0f}/100, an index, not a probability).")
    head = f"{risk.upper()} — {MODEL_NAMES[model]}, risk {hl} ahead"
    if demo:
        reasons.append("Hypothetical input for demonstration; not an official warning.")
    return {"model": model, "model_name": MODEL_NAMES[model], "mode": mode, "risk_class": cls, "risk": risk,
            "alert": cls >= config.ALERT_CLASS, "headline": head, "advice": ADVICE[cls], "reasons": reasons,
            "message": head + ". " + " ".join(reasons), "inputs_used": x, "terms": terms, "score": score_i,
            "detail": detail, "sensors": sensors, "conflict": conflict, "status": status, "U": U, "horizon": hl}


def _band(cls: int) -> str:
    e = [0.0, *config.RISK_EDGES, 100.0]
    return f"{e[cls]:g}–{e[cls + 1]:g}"


def summary(suite, sh: dict, hz: dict, i: int, mode: str) -> dict:
    out = {}
    for m in hz["cls"][mode]:
        c = int(hz["cls"][mode][m][i])
        s = hz["score"][mode].get(m)
        out[m] = {"name": MODEL_NAMES[m], "risk_class": c, "risk": config.RISK_CLASSES[c],
                  "score": None if s is None else float(s[i])}
    return out
