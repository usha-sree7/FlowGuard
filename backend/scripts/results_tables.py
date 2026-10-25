"""Print Markdown tables from saved results (used to fill docs - numbers are never typed by hand).

    python scripts/results_tables.py <run_id> [--sweep bengalas]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floodguard import config  # noqa: E402
from floodguard.experiment import load_run  # noqa: E402

SYS = ["threshold", "t1_fis", "it2_fis", "proposed"]


def f(v, d=3):
    return "n/a" if v is None else f"{v:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--sweep")
    a = ap.parse_args()
    r = load_run(a.run_id)
    m = r["meta"]
    print(f"Run `{m['run_id']}` - dataset {m['dataset']}, mode {m['mode']}, seed {m['seed']}, model v{m['model_version']}, "
          f"train {m['train']['rows']} rows ({m['train']['first']} -> {m['train']['last']}), test {m['test']['rows']} rows "
          f"({m['test']['first']} -> {m['test']['last']}), proxy-label test counts {m['labels']['test_class_counts']}.\n")
    print("| Scenario | " + " | ".join(f"{s} P / R / F1 / FPR" for s in SYS) + " |")
    print("|---|" + "---|" * len(SYS))
    for scen, res in r["results"].items():
        cells = []
        for s in SYS:
            b = res["systems"][s]["all"]["binary_alert"]
            cells.append(f"{f(b['precision'])} / {f(b['recall'])} / **{f(b['f1'])}** / {f(b['fpr'], 4)}")
        print(f"| {scen} | " + " | ".join(cells) + " |")
    print("\n| Scenario | SWE thr / T1 / IT2 / prop | 4-class acc (prop) | U faulted | U unfaulted | Confident/Uncertain/Insufficient | Level fault det. P / R |")
    print("|---|---|---|---|---|---|---|")
    for scen, res in r["results"].items():
        swe = " / ".join(f(res["systems"][s]["all"]["ordinal"]["severity_weighted_error"]) for s in SYS)
        u = res["uncertainty"]
        sc = u["status_counts"]
        d = res["fault_detection"]["level"]
        det = f"false-flag {f(d['false_flag_rate'], 4)}" if "false_flag_rate" in d else f"{f(d['precision'])} / {f(d['recall'])}"
        print(f"| {scen} | {swe} | {f(res['systems']['proposed']['all']['ordinal']['accuracy'])} | "
              f"{f(u['U_mean_faulted_steps'], 1)} | {f(u['U_mean_unfaulted_steps'], 1)} | "
              f"{sc.get('Confident', 0)}/{sc.get('Uncertain', 0)}/{sc.get('Insufficient data', 0)} | {det} |")
    t = r["results"]["clean"]["ms_per_obs"]
    print("\nProcessing time (ms/observation, clean, vectorised batch): " + ", ".join(f"{s} {t[s]:.4f}" for s in SYS))
    if a.sweep:
        df = pd.read_csv(config.RESULTS_DIR / f"seed_sweep_{a.sweep}.csv")
        g = df.groupby(["scenario", "system"])["f1"].agg(["mean", "std"])
        order = list(dict.fromkeys(df["scenario"]))
        n = df["seed"].nunique()
        print(f"\nSeed sweep ({n} fault-placement seeds), alert F1 mean +/- sd:\n")
        print("| Scenario | " + " | ".join(SYS) + " |")
        print("|---|" + "---|" * len(SYS))
        for sc in order:
            best = max(SYS, key=lambda s: g.loc[(sc, s), "mean"])
            cells = [("**" if s == best else "") + f"{g.loc[(sc, s), 'mean']:.3f} +/- {g.loc[(sc, s), 'std']:.3f}" + ("**" if s == best else "") for s in SYS]
            print(f"| {sc} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
