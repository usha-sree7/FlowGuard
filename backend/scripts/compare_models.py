"""Evaluate IT2FIS, PSO-tuned IT2FIS, SOM and ANFIS (plus the threshold reference) on identical held-out data,
per forecast horizon, input mode (raw / reliability-fused) and simulated sensor-fault scenario.

    python scripts/compare_models.py bengalas --seeds 3
    python scripts/compare_models.py malwathu --seeds 3

Protocol (leak-free, see docs/EXPERIMENT_PROTOCOL.md):
  * chronological split: train = first 70 % (fit block = first 80 % of it, validation = last 20 %), test = last 30 %;
  * every parameter is fitted / selected on train (fit / validation blocks) only; test is used once, here;
  * faults are injected into a COPY of the test split with fixed seeds; proxy labels come from the ORIGINAL test
    data (class of the observed level h steps ahead), so a fault can never change the answer key;
  * all models see the same (corrupted) input; clean-data confidence intervals use a moving-block bootstrap.

Writes results/comparison_<ds>.json (read by the dashboard), comparison_<ds>_raw.csv and MODEL_COMPARISON_<ds>.md.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floodguard import MODEL_VERSION, config  # noqa: E402
from floodguard.calibration import LEVEL_K, split  # noqa: E402
from floodguard.datasets import DATASETS, load_processed  # noqa: E402
from floodguard.experiment import _clean_json, make_config  # noqa: E402
from floodguard.faults import SCENARIO_DOC, inject  # noqa: E402
from floodguard.forecast import regression_metrics  # noqa: E402
from floodguard.metrics import binary, block_bootstrap_f1, detection, evaluate  # noqa: E402
from floodguard.reliability import flagged  # noqa: E402
from floodguard.suite import MODEL_NAMES, MODELS, MODES, RISK_MODELS, ModelSuite, horizons  # noqa: E402
from floodguard.systems import future_level, proxy_labels  # noqa: E402

SCENARIOS = ("clean", "missing_10", "missing_30", "noise_low", "noise_med", "noise_high", "stuck", "conflict",
             "spikes", "combined")
FAULTED = [s for s in SCENARIOS if s != "clean"]
METRICS = ("f1", "precision", "recall", "fpr", "macro_f1", "accuracy", "kappa", "swe", "missed_alerts", "false_alarms")
A = config.ALERT_CLASS


def risk_row(labels, cls):
    ev = evaluate(labels, cls)
    b, o = ev["binary_alert"], ev["ordinal"]
    return {"f1": b["f1"], "precision": b["precision"], "recall": b["recall"], "fpr": b["fpr"],
            "macro_f1": o["macro_f1"], "accuracy": o["accuracy"], "kappa": o["kappa"],
            "swe": o["severity_weighted_error"], "missed_alerts": b["fn"], "false_alarms": b["fp"],
            "alerts_true": b["tp"] + b["fn"], "alerts_raised": b["tp"] + b["fp"]}, o


def confidence_row(labels, cls, status):
    ok = ~np.isnan(labels)
    t, p, conf = labels[ok].astype(int), cls[ok], status[ok] == "Confident"
    wrong = t != p
    true_alert, pred_alert = t >= A, p >= A
    return {"coverage_confident": float(conf.mean()),
            "error_rate_confident": float(wrong[conf].mean()) if conf.any() else float("nan"),
            "error_rate_not_confident": float(wrong[~conf].mean()) if (~conf).any() else float("nan"),
            "alert_f1_confident": binary(true_alert[conf], pred_alert[conf])["f1"] if conf.any() else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["bengalas", "malwathu", "cwc_chindnar"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--markdown-only", action="store_true",
                    help="rebuild MODEL_COMPARISON_<ds>.md from the saved JSON / CSV without re-running anything")
    a = ap.parse_args()
    if a.markdown_only:
        P = json.loads((config.RESULTS_DIR / f"comparison_{a.dataset}.json").read_text(encoding="utf-8"))
        write_markdown(a.dataset, P, pd.read_csv(config.RESULTS_DIR / f"comparison_{a.dataset}_raw.csv"))
        return
    t_start = time.time()

    df = load_processed(a.dataset, mode="full")
    cfg = make_config(a.dataset)
    train, test = split(df, cfg.train_fraction)
    suite = ModelSuite.fit(train, cfg, progress=lambda m: print("  fitting", m, flush=True))
    pipe, H = suite.pipe, suite.horizons
    labels = {h: proxy_labels(test, pipe.cal, h) for h in H}
    print(f"fitted in {time.time() - t_start:.0f}s; horizons {H}", flush=True)
    block = int(round(24 / cfg.step_hours)) if cfg.step_hours < 24 else 14  # 1 day (sub-daily) or 14 days (daily)

    rows, conf, det, cms, clean, timing = [], [], [], {}, {}, {}
    for k in range(a.seeds):
        seed = config.DEFAULT_SEED + 1000 * k
        for i, scen in enumerate(SCENARIOS):
            if scen == "clean" and k > 0:  # clean data does not depend on the fault seed
                continue
            corrupted, gt = inject(test, scen, pipe.cal, seed=seed + i)
            sh = suite.run_shared(corrupted)
            if scen != "clean":
                any_fault = np.zeros(len(test), bool)
                rel_flag = np.zeros(len(test), bool)
                for ch, m in gt.items():
                    any_fault |= m
                    rel_flag |= flagged(sh["proposed"]["rel"], ch)
                d = detection(any_fault, rel_flag)
                det.append({"seed": seed, "scenario": scen, "flag_rate": float(rel_flag.mean()),
                            "precision": d["precision"], "recall": d["recall"],
                            "f1": 2 * d["precision"] * d["recall"] / (d["precision"] + d["recall"])
                            if d["precision"] and d["recall"] else 0.0, "faulty_share": float(any_fault.mean())})
            for h in H:
                hz = suite.run_horizon(sh, h)
                for mode in MODES:
                    for name in RISK_MODELS:
                        cls = hz["cls"][mode][name]
                        r, o = risk_row(labels[h], cls)
                        rows.append({"seed": seed, "horizon": h, "scenario": scen, "mode": mode, "model": name, **r})
                        if k == 0:
                            cms[f"{h}|{scen}|{mode}|{name}"] = o["confusion_matrix"]
                        if scen == "clean":
                            clean[f"{h}|{mode}|{name}"] = {**r, "per_class_f1": o["per_class_f1"],
                                                           "f1_ci95": block_bootstrap_f1(labels[h], cls, block)}
                        if mode == "fused":
                            conf.append({"seed": seed, "horizon": h, "scenario": scen, "model": name,
                                         **confidence_row(labels[h], cls, sh["proposed"]["status"])})
            if scen == "clean":
                timing = {key: v / len(H) if key.split("|")[0] in ("pso_it2", "som", "anfis") else v
                          for key, v in sh["ms"].items()}
            print(f"  seed {k + 1}/{a.seeds}  {scen}", flush=True)

    # ---- does the dataset support each horizon? (clean data only)
    support, forecast = {}, {}
    lvl_test = test["level"].to_numpy(float)
    for h in H:
        tr_lab = proxy_labels(train, pipe.cal, h)
        cur, fut = np.digitize(lvl_test, pipe.cal.thresholds), labels[h]
        ok = ~np.isnan(fut)
        target = future_level(test, h, cfg.step_hours)
        pred = suite.forecast(test, h)
        support[str(h)] = {
            "label": H[h], "steps": h, "hours": h * cfg.step_hours,
            "train_labelled": int((~np.isnan(tr_lab)).sum()), "test_labelled": int(ok.sum()),
            "test_unlabelled_gap": int((~ok).sum()),
            "train_alerts": int((tr_lab >= A).sum()), "test_alerts": int((fut >= A).sum()),
            "class_changes_share": float((cur[ok] != fut[ok]).mean()),
            "new_alert_onsets": int(((cur[ok] < A) & (fut[ok] >= A)).sum()),
            "level_change_abs_mean_m": float(np.nanmean(np.abs(target - lvl_test))),
        }
        forecast[str(h)] = {"label": H[h], "persistence": regression_metrics(target, lvl_test),
                            "arx": regression_metrics(target, pred)}

    R, C, D = pd.DataFrame(rows), pd.DataFrame(conf), pd.DataFrame(det)

    def agg(frame, keys, cols):
        g = frame.groupby(keys)[cols]
        mean, sd = g.mean(), g.std().fillna(0.0)
        out = {}
        for idx in mean.index:
            key = "|".join(str(v) for v in (idx if isinstance(idx, tuple) else (idx,)))
            out[key] = {c: mean.loc[idx, c] for c in cols} | {f"{c}_sd": sd.loc[idx, c] for c in cols}
        return out

    spec = DATASETS[a.dataset]
    payload = {
        "meta": {"dataset": a.dataset, "dataset_name": spec.name, "model_version": MODEL_VERSION,
                 "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "seeds": [config.DEFAULT_SEED + 1000 * k for k in range(a.seeds)],
                 "train": {"rows": len(train), "first": train["timestamp"].iloc[0] or "index 0",
                           "last": train["timestamp"].iloc[-1] or f"index {len(train) - 1}", **suite.split_info},
                 "test": {"rows": len(test), "first": test["timestamp"].iloc[0] or f"index {len(train)}",
                          "last": test["timestamp"].iloc[-1] or f"index {len(df) - 1}"},
                 "horizons": {str(h): v for h, v in H.items()},
                 "unsupported_horizons": horizons(cfg.step_hours)["unsupported"],
                 "step_hours": cfg.step_hours, "rain_window_hours": cfg.rain_window_steps * cfg.step_hours,
                 "risk_classes": config.RISK_CLASSES, "alert_class": A, "risk_edges": config.RISK_EDGES,
                 "level_k": list(LEVEL_K), "thresholds_m": pipe.cal.thresholds,
                 "label_counts_test": {str(h): {c: int((labels[h] == j).sum()) for j, c in enumerate(config.RISK_CLASSES)}
                                       | {"unlabelled": int(np.isnan(labels[h]).sum())} for h in H},
                 "scenarios": list(SCENARIOS), "scenario_definitions": {s: SCENARIO_DOC[s] for s in SCENARIOS},
                 "models": {m: MODEL_NAMES[m] for m in RISK_MODELS}, "main_models": list(MODELS), "modes": list(MODES),
                 "positive_class": "alert = High or Critical (observed level at the horizon >= train mean + 2 sd)",
                 "bootstrap": {"method": "moving-block bootstrap of the clean test period", "block_steps": block,
                               "resamples": 1000, "interval": "95 % percentile"},
                 "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
                 "elapsed_s": None},
        "risk": agg(R, ["horizon", "scenario", "mode", "model"], list(METRICS)),
        "clean": clean,
        "confusion_seed0": cms,
        "confidence": agg(C, ["horizon", "scenario", "model"], ["coverage_confident", "error_rate_confident",
                                                                "error_rate_not_confident", "alert_f1_confident"]),
        "detection": agg(D, ["scenario"], ["flag_rate", "precision", "recall", "f1", "faulty_share"]),
        "support": support,
        "forecast": forecast,
        "ms_per_obs_clean": timing,
        "fitted": suite.describe(),
    }
    payload["meta"]["elapsed_s"] = round(time.time() - t_start, 1)
    out = config.RESULTS_DIR
    (out / f"comparison_{a.dataset}.json").write_text(json.dumps(_clean_json(payload), indent=1))
    R.to_csv(out / f"comparison_{a.dataset}_raw.csv", index=False)
    write_markdown(a.dataset, payload, R)
    print(f"\nwrote results/comparison_{a.dataset}.json, comparison_{a.dataset}_raw.csv, "
          f"MODEL_COMPARISON_{a.dataset}.md in {payload['meta']['elapsed_s']}s")


def write_markdown(ds, P, R):
    meta = P["meta"]
    fm = lambda v, d=3: "—" if v is None or not np.isfinite(v) else f"{v:.{d}f}"  # noqa: E731
    md = [f"# HydroSentinel AI: model comparison ({meta['dataset_name']})", "",
          f"Generated by `scripts/compare_models.py` (model v{meta['model_version']}, {meta['created']}). "
          f"Train {meta['train']['rows']} rows ({meta['train']['first']} to {meta['train']['last']}), test "
          f"{meta['test']['rows']} rows ({meta['test']['first']} to {meta['test']['last']}); "
          f"{len(meta['seeds'])} fault-placement seeds.", "",
          "**Labels are PROXY labels**: the alert level of the observed water level h steps ahead, using train-split "
          "thresholds " + ", ".join(f"{c} ≥ {t:.3f} m" for c, t in zip(meta['risk_classes'][1:], meta['thresholds_m']))
          + " (mean + 0.5/1/2/3 sd). They are not observed flood records. Positive class for precision / recall / F1 / "
          "FPR: " + meta["positive_class"] + ". The threshold rule is a reference only: the labels are built from it.", ""]
    no_alerts = [s["label"] for s in P["support"].values() if s["test_alerts"] == 0]
    if no_alerts:
        md += [f"> **No alert (High / Critical) occurs in the test period at {', '.join(no_alerts)}.** Alert precision, "
               "recall and F1 are therefore undefined (shown as —); only false alarms, FPR and the 5-level metrics "
               "(macro F1, κ) are meaningful. Do not read these tables as flood-detection skill.", ""]
    for h, hl in meta["horizons"].items():
        md += [f"## {hl} ahead", "", "Clean test data:", "",
               "| Model | Input | Alert F1 [95 % CI] | Precision | Recall | FPR | Macro F1 (5 cl.) | κ | Missed alerts | False alarms |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for mode in meta["modes"]:
            for m in meta["models"]:
                c = P["clean"][f"{h}|{mode}|{m}"]
                md.append(f"| {meta['models'][m]} | {mode} | **{fm(c['f1'])}** [{fm(c['f1_ci95'][0])}, {fm(c['f1_ci95'][1])}] | "
                          f"{fm(c['precision'])} | {fm(c['recall'])} | {fm(c['fpr'], 4)} | {fm(c['macro_f1'])} | "
                          f"{fm(c['kappa'])} | {c['missed_alerts']} | {c['false_alarms']} |")
        sub = R[R.horizon == int(h)]
        f1 = sub.groupby(["mode", "model", "scenario"])["f1"].mean().unstack()
        md += ["", "Alert F1 per simulated fault (mean over seeds):", "",
               "| Model | Input | " + " | ".join(SCENARIOS) + " | Faulted mean |", "|---|---|" + "---|" * (len(SCENARIOS) + 1)]
        for (mode, m), r in f1.iterrows():
            md.append(f"| {meta['models'][m]} | {mode} | " + " | ".join(fm(r[s]) for s in SCENARIOS)
                      + f" | **{fm(r[FAULTED].mean())}** |")
        mf = sub.groupby(["mode", "model", "scenario"])["macro_f1"].mean().unstack()
        md += ["", "5-level macro F1 per simulated fault (mean over seeds; defined even when no alert occurs):", "",
               "| Model | Input | " + " | ".join(SCENARIOS) + " | Faulted mean |", "|---|---|" + "---|" * (len(SCENARIOS) + 1)]
        for (mode, m), r in mf.iterrows():
            md.append(f"| {meta['models'][m]} | {mode} | " + " | ".join(fm(r[s]) for s in SCENARIOS)
                      + f" | **{fm(r[FAULTED].mean())}** |")
        md.append("")
    md += ["## Fault detection by the reliability checks", "",
           "Flag = reliability < 0.5 on any channel, scored against the known injection mask.", "",
           "| Scenario | Faulty share of steps | Flag rate | Precision | Recall | F1 |", "|---|---|---|---|---|---|"]
    for s in FAULTED:
        r = P["detection"][s]
        md.append(f"| {s} | {fm(r['faulty_share'])} | {fm(r['flag_rate'])} | {fm(r['precision'])} | "
                  f"{fm(r['recall'])} | {fm(r['f1'])} |")
    md += ["", "## Horizon support check (clean test data)", "",
           "MAE / RMSE are genuine numerical errors of the forecast water level h steps ahead (not of the risk class).", "",
           "| Horizon | Labelled test rows | Test alerts | Class changes vs now | New alert onsets | Persistence MAE / RMSE [m] | ARX MAE / RMSE [m] |",
           "|---|---|---|---|---|---|---|"]
    for h, s in P["support"].items():
        f = P["forecast"][h]
        md.append(f"| {s['label']} | {s['test_labelled']} | {s['test_alerts']} | {fm(s['class_changes_share'])} | "
                  f"{s['new_alert_onsets']} | {fm(f['persistence']['mae'], 4)} / {fm(f['persistence']['rmse'], 4)} | "
                  f"{fm(f['arx']['mae'], 4)} / {fm(f['arx']['rmse'], 4)} |")
    if meta["unsupported_horizons"]:
        md += ["", "Not supported by the sampling interval: " + ", ".join(meta["unsupported_horizons"]) + "."]
    (config.RESULTS_DIR / f"MODEL_COMPARISON_{ds}.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
