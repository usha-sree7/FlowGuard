"""Compare the original four systems against additional soft-computing models on every fault scenario and rank them.

    python scripts/compare_models.py bengalas --seeds 3
    python scripts/compare_models.py malwathu --seeds 3

Same leak-free protocol as run_experiment.py: chronological split, everything fitted on the clean train
split, faults injected into a copy of the test split, proxy labels from the ORIGINAL test data.
Writes results/MODEL_COMPARISON_<dataset>.md and results/model_comparison_<dataset>.csv.
"""
import argparse
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

    rows = []
    for k in range(a.seeds):
        seed = config.DEFAULT_SEED + 1000 * k
        for i, scen in enumerate(SCENARIOS):
            corrupted, _ = inject(test, scen, pipe.cal, seed=seed + i)
            preds = {n: o["cls"] for n, o in pipe.run_all(corrupted)["outputs"].items()}
            preds.update(sc.predict_all(pipe, corrupted))
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
              f"{k} {v['validation_fitness_original']:.3f} -> {v['validation_fitness_tuned']:.3f} (GA/PSO tuning)"
              if "genes" in v else f"{k} selected {v['selected']} {v['validation_fitness']}"
              for k, v in sc.info.items()), "",
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
    (out / f"MODEL_COMPARISON_{a.dataset}.md").write_text("\n".join(md) + "\n")
    print("\n" + rank.round(3).to_string())
    print(f"\nwrote results/MODEL_COMPARISON_{a.dataset}.md")


if __name__ == "__main__":
    main()
