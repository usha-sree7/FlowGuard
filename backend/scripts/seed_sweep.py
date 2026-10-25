"""Repeat the experiment over several seeds (fault placement changes; data/split/model do not) and report
mean +/- sd of the binary-alert F1 per scenario and system.

    python scripts/seed_sweep.py bengalas --seeds 5
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floodguard import config  # noqa: E402
from floodguard.experiment import run_experiment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["bengalas", "malwathu"])
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    rows = []
    for k in range(a.seeds):
        seed = config.DEFAULT_SEED + 1000 * k
        res = run_experiment(a.dataset, seed=seed, save_inputs=False)
        for scen, r in res["results"].items():
            for name, ev in r["systems"].items():
                b = ev["all"]["binary_alert"]
                rows.append({"seed": seed, "scenario": scen, "system": name, "f1": b["f1"],
                             "precision": b["precision"], "recall": b["recall"], "fpr": b["fpr"],
                             "U_mean": r["uncertainty"]["U_mean"] if name == "proposed" else None,
                             "run_id": res["meta"]["run_id"]})
        print(f"seed {seed} done ({res['meta']['run_id']})", flush=True)
    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / f"seed_sweep_{a.dataset}.csv"
    df.to_csv(out, index=False)
    agg = df.groupby(["scenario", "system"])["f1"].agg(["mean", "std"]).round(3).unstack("system")
    agg.to_csv(config.RESULTS_DIR / f"seed_sweep_{a.dataset}_f1_summary.csv")
    order = list(dict.fromkeys(df["scenario"]))
    print(agg.reindex(order).to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
