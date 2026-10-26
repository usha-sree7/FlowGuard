"""Compare the original four systems against additional soft-computing models on every fault scenario and rank them.

    python scripts/compare_models.py bengalas --seeds 3
    python scripts/compare_models.py malwathu --seeds 3

Same leak-free protocol as run_experiment.py: chronological split, everything fitted on the clean train
split, faults injected into a copy of the test split, proxy labels from the ORIGINAL test data.
Writes results/MODEL_COMPARISON_<dataset>.md and results/model_comparison_<dataset>.csv.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floodguard import config  # noqa: E402
from floodguard.calibration import split  # noqa: E402
from floodguard.datasets import load_processed  # noqa: E402
from floodguard.experiment import make_config  # noqa: E402
from floodguard.faults import SCENARIOS, inject  # noqa: E402
from floodguard.metrics import evaluate  # noqa: E402
from floodguard.softcomputing import SoftComputingModels  # noqa: E402
from floodguard.systems import SYSTEMS, Pipeline, proxy_labels  # noqa: E402


RISK = ["Normal", "Watch", "Warning", "Critical"]
METRIC_LABELS = {"precision": "Alert precision", "recall": "Alert recall", "f1": "Alert F1", "fpr": "False-positive rate",
                 "macro_f1_4class": "4-class macro F1", "swe": "Severity-weighted error"}


def _describe(name, info):
    if "genes" in info:
        return f"{name} {info['validation_fitness_original']:.3f} -> {info['validation_fitness_tuned']:.3f} (tuned)"
    if "features" in info:
        return f"{name} {info['validation_fitness']:.3f} using {', '.join(info['features'])}"
    return f"{name} selected {info['selected']} {info['validation_fitness']}"


def best_model_report(ds, best, rank, res, test, labels, clean_preds, faulted, info):
    mean = res[res["system"] == best].groupby("scenario")[list(METRIC_LABELS)].mean()
    metrics = {"clean": mean.loc["clean"].to_dict(), "faulted": mean.loc[faulted].mean().to_dict()}
    cm = evaluate(labels, clean_preds[best])["ordinal"]["confusion_matrix"]
    ts = test["timestamp"].fillna("").astype(str)
    t = np.where(ts.str.len() > 0, ts, [f"test row {i}" for i in range(len(test))])
    out = pd.DataFrame({"time": t, "level": test["level"].round(3), "rain": test.filter(like="rain__").mean(axis=1).round(2),
                        "proxy_label": [RISK[int(v)] if v == v else "" for v in labels],
                        best: [RISK[v] for v in clean_preds[best]],
                        "proposed": [RISK[v] for v in clean_preds["proposed"]]})
    out.to_csv(config.RESULTS_DIR / f"best_model_output_{ds}.csv", index=False)
    peak = int(np.nanargmax(test["level"].to_numpy(float)))
    win = out.iloc[max(0, peak - 8):peak + 4]
    sample = [{"time": str(r.time), "level": float(r.level), "rain": float(r.rain), "label": r.proxy_label,
               "best": r[best], "proposed": r.proposed} for _, r in win.iterrows()]
    return {"dataset": ds, "best_model": best, "ranking": rank.index.tolist(), "metrics": metrics,
            "confusion_matrix_clean": cm, "risk_classes": RISK, "sample_output": sample,
            "training_info": info.get(best.removesuffix("+fusion"), {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["bengalas", "malwathu"])
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    df = load_processed(a.dataset, mode="full")
    cfg = make_config(a.dataset)
    train, test = split(df, cfg.train_fraction)
    pipe = Pipeline.fit(train, cfg)
    labels = proxy_labels(test, pipe.cal)
    t0 = time.time()
    sc = SoftComputingModels(config.DEFAULT_SEED).fit(pipe, train, proxy_labels(train, pipe.cal),
                                                      progress=lambda m: print("  fitting", m, flush=True))
    print(f"trained {len(sc.models)} soft-computing models in {time.time() - t0:.1f}s", flush=True)
    for k, v in sc.info.items():
        print(f"  {k}: {v}")

    rows, clean_preds = [], {}
    for k in range(a.seeds):
        seed = config.DEFAULT_SEED + 1000 * k
        for i, scen in enumerate(SCENARIOS):
            corrupted, _ = inject(test, scen, pipe.cal, seed=seed + i)
            preds = {n: o["cls"] for n, o in pipe.run_all(corrupted)["outputs"].items()}
            preds.update(sc.predict_all(pipe, corrupted))
            if k == 0 and scen == "clean":
                clean_preds = preds
            for name, cls in preds.items():
                ev = evaluate(labels, cls)
                b, od = ev["binary_alert"], ev["ordinal"]
                rows.append({"seed": seed, "scenario": scen, "system": name, "f1": b["f1"],
                             "precision": b["precision"], "recall": b["recall"], "fpr": b["fpr"],
                             "macro_f1_4class": od["macro_f1"], "swe": od["severity_weighted_error"]})
            print(f"  seed {k + 1}/{a.seeds}  {scen}", flush=True)

    res = pd.DataFrame(rows)
    out = config.RESULTS_DIR
    res.to_csv(out / f"model_comparison_{a.dataset}.csv", index=False)

    f1 = res.groupby(["system", "scenario"])["f1"].mean().unstack()[list(SCENARIOS)]
    faulted = [s for s in SCENARIOS if s != "clean"]
    rank = pd.DataFrame({
        "clean_f1": f1["clean"],
        "faulted_mean_f1": f1[faulted].mean(axis=1),
        "worst_f1": f1[faulted].min(axis=1),
        "mean_swe": res.groupby("system")["swe"].mean(),
        "mean_macro_f1_4class": res.groupby("system")["macro_f1_4class"].mean(),
    }).sort_values("faulted_mean_f1", ascending=False)
    rank.insert(0, "family", ["original" if s in SYSTEMS else "new+fusion" if s.endswith("+fusion") else "new"
                              for s in rank.index])

    md = [f"# Model comparison: {a.dataset}", "",
          f"{a.seeds} fault-placement seeds; train {len(train)} rows, test {len(test)} rows; proxy labels. "
          "Ranked by mean alert F1 over the 10 faulted scenarios. Family: `original` = the 4 existing systems, "
          "`new` = added soft-computing model on naive inputs, `new+fusion` = same model on the "
          "reliability-weighted fused inputs.", "",
          "Model selection on the validation block (last 20 % of train; fitness = alert F1 + 0.25 x 4-class "
          "macro F1): " + "; ".join(
              _describe(k, v) for k, v in sc.info.items()), "",
          "| Rank | System | Family | Clean F1 | Faulted mean F1 | Worst-case F1 | Mean SWE | 4-class macro F1 |",
          "|---|---|---|---|---|---|---|---|"]
    for i, (s, r) in enumerate(rank.iterrows(), 1):
        md.append(f"| {i} | {s} | {r.family} | {r.clean_f1:.3f} | **{r.faulted_mean_f1:.3f}** | "
                  f"{r.worst_f1:.3f} | {r.mean_swe:.3f} | {r.mean_macro_f1_4class:.3f} |")
    md += ["", "Alert F1 per scenario (mean over seeds):", "",
           "| System | " + " | ".join(SCENARIOS) + " |", "|---|" + "---|" * len(SCENARIOS)]
    for s in rank.index:
        vals = f1.loc[s]
        md.append(f"| {s} | " + " | ".join(
            f"**{v:.3f}**" if np.isclose(v, f1[c].max()) else f"{v:.3f}" for c, v in vals.items()) + " |")
    best = rank.index[0]
    report = best_model_report(a.dataset, best, rank, res, test, labels, clean_preds, faulted, sc.info)
    (out / f"model_comparison_{a.dataset}_best.json").write_text(json.dumps(report, indent=1))
    md += ["", f"## Best model: {best}", "", "Mean over seeds; faulted = mean of the 10 faulted scenarios.", "",
           "| Metric | Clean | Faulted mean |", "|---|---|---|"]
    for m, label in METRIC_LABELS.items():
        md.append(f"| {label} | {report['metrics']['clean'][m]:.4f} | {report['metrics']['faulted'][m]:.4f} |")
    cm = report["confusion_matrix_clean"]
    md += ["", "Confusion matrix on the clean test split (rows = proxy label, columns = prediction):", "",
           "| | " + " | ".join(RISK) + " |", "|---|" + "---|" * 4]
    md += [f"| {RISK[i]} | " + " | ".join(str(v) for v in row) + " |" for i, row in enumerate(cm)]
    md += ["", f"Sample output around the largest test-period flood peak (full output in "
           f"`results/best_model_output_{a.dataset}.csv`):", "",
           "| Time | Level | Rain | Proxy label | " + best + " | proposed |", "|---|---|---|---|---|---|"]
    md += [f"| {r['time']} | {r['level']} | {r['rain']} | {r['label']} | {r['best']} | {r['proposed']} |"
           for r in report["sample_output"]]
    (out / f"MODEL_COMPARISON_{a.dataset}.md").write_text("\n".join(md) + "\n")
    print("\n" + rank.round(3).to_string())
    print(f"\nBEST MODEL: {best}")
    for m, label in METRIC_LABELS.items():
        print(f"  {label:28s} clean {report['metrics']['clean'][m]:.4f}   faulted {report['metrics']['faulted'][m]:.4f}")
    print(f"\nwrote results/MODEL_COMPARISON_{a.dataset}.md, model_comparison_{a.dataset}.csv, "
          f"model_comparison_{a.dataset}_best.json, best_model_output_{a.dataset}.csv")


if __name__ == "__main__":
    main()
