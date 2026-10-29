"""Run the fault-injection robustness experiment and print a compact summary.

    python scripts/run_experiment.py bengalas                 # full mode, all scenarios
    python scripts/run_experiment.py bengalas --mode sample   # seeded contiguous 10k-row block
    python scripts/run_experiment.py malwathu --seed 7 --scenarios clean,spikes,missing_30
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floodguard import config  # noqa: E402
from floodguard.experiment import run_experiment  # noqa: E402
from floodguard.faults import SCENARIOS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["bengalas", "malwathu", "cwc_chindnar"])
    ap.add_argument("--mode", choices=["full", "sample"], default="full")
    ap.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    ap.add_argument("--sample-rows", type=int, default=10_000)
    ap.add_argument("--train-fraction", type=float, default=config.TRAIN_FRACTION)
    ap.add_argument("--horizon-steps", type=int, default=None)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--no-save-inputs", action="store_true")
    a = ap.parse_args()
    res = run_experiment(a.dataset, scenarios=a.scenarios.split(","), mode=a.mode, seed=a.seed,
                         sample_rows=a.sample_rows, train_fraction=a.train_fraction,
                         save_inputs=not a.no_save_inputs, horizon_steps=a.horizon_steps,
                         progress=lambda m: print("  ..", m, flush=True))
    m = res["meta"]
    print(f"\nrun {m['run_id']}  rows={m['rows_used']}  test={m['test']['rows']}  labels={m['labels']['test_class_counts']}")
    print(f"{'scenario':20s} {'system':10s} {'P':>6s} {'R':>6s} {'F1':>6s} {'FPR':>7s} {'SWE':>6s}   U_mean  status")
    for scen, r in res["results"].items():
        for name, ev in r["systems"].items():
            b, o = ev["all"]["binary_alert"], ev["all"]["ordinal"]
            fmt = lambda v: "  nan" if v is None else f"{v:6.3f}"  # noqa: E731
            extra = ""
            if name == "proposed":
                u = r["uncertainty"]
                extra = f"   {u['U_mean']:5.1f}  {u['status_counts']}"
            print(f"{scen:20s} {name:10s} {fmt(b['precision'])} {fmt(b['recall'])} {fmt(b['f1'])} "
                  f"{fmt(b['fpr'])} {fmt(o['severity_weighted_error'])}{extra}")
    print(f"\nsaved to results/runs/{m['run_id']}/ (results.json, summary.csv)")


if __name__ == "__main__":
    main()
