# FloodGuard AI

**Reliability-Aware Sensor Fusion Using Interval Type-2 Fuzzy Logic for Flood-Risk Detection** (Soft Computing course project).

```
rainfall (gauge, radar / multi-gauge), water level, derived rise rate
  → per-sensor reliability assessment (missing, spike, stuck, noise, inconsistency, device health)
  → reliability-weighted fusion (point estimates + intervals that keep unresolved uncertainty)
  → Interval Type-2 FIS (FOU, interval firing, 64 rules, Karnik–Mendel type reduction)
  → TWO SEPARATE OUTPUTS:  flood risk  Normal / Watch / Warning / Critical
                           uncertainty status  Confident / Uncertain / Insufficient data  (+ index U)
  → replay dashboard with alarms, model inspection, experiments
```

Real data (audited, see `docs/PAPER_DATASET_AUDIT.md`):

* **Bengalas River, Brazil.** 72,011 × 30-min records, 2019–2023, from the UAE-ML repository of Soares et al. (TGIS 2026).
* **Malwathu Oya, Sri Lanka.** 4,765 daily records, from Rathnayake et al. (PLOS ONE 2023).

**Only proxy labels exist.** These are future level exceedances of statistical thresholds, not observed flood records. Risk scores are not probabilities, and U is not a calibrated error probability.

## Requirements

* Python ≥ 3.11 (tested on 3.13)
* Node ≥ 18 (tested on 24)
* About 200 MB of disk for the data

## Setup and start

```bash
# backend
cd backend
pip install -r requirements.txt
python scripts/import_dataset.py bengalas     # download (41.5 MB) + checksum + validate
python scripts/import_dataset.py malwathu

# frontend (build once; the backend serves it)
cd ../frontend
npm install
npm run build

# run the full app
cd ../backend
python -m uvicorn floodguard.api:app --port 8000
# open http://127.0.0.1:8000
```

For frontend development with hot reload, keep uvicorn running and also run `cd frontend && npm run dev`, then open http://localhost:5173 (`/api` is proxied).

Datasets can also be imported from the UI: **Dataset Explorer → Import dataset**. If the download fails, the UI shows manual-download instructions.

## Hosted demo

Live (read-only): **https://floodguard-ai-chi.vercel.app**. It bundles the processed datasets and the saved experiment summaries; importing datasets and running new experiments are disabled there, so do those locally.

To redeploy after changes (needs the Vercel CLI):

```bash
python deploy/build_vercel.py          # assembles deploy/build/ (gitignored)
cd deploy/build && vercel deploy --prod
```

A full explanation of the project for reviews and interviews is in `docs/FloodGuard_Project_Guide.pdf`.

## Model comparison

Besides the four original systems, four more soft-computing models are compared (`backend/floodguard/softcomputing.py`):
ANFIS, FCM-TSK (fuzzy C-means rules), and the IT2FIS tuned by a genetic algorithm (GA-IT2FIS) or by particle swarm
optimisation (PSO-IT2FIS). Option A (`backend/floodguard/neural.py`) adds a GA + SOM + ANN hybrid, built in three
stages so the effect of each part is measured: ANN, SOM + ANN, GA + SOM + ANN. Every model also runs on the reliability-fused inputs (`+fusion`), 18 in total. Tuning and model selection
use only the last 20 % of the train split.

```bash
cd backend
python scripts/compare_models.py bengalas --seeds 3   # ~6 min; writes results/MODEL_COMPARISON_bengalas.md,
                                                      # model_comparison_bengalas{.csv,_best.json}, best_model_output_bengalas.csv
python scripts/compare_models.py malwathu --seeds 3
```

The **Model Comparison** tab shows the overall best model (metrics, confusion matrix, output around the largest flood peak). You can also select any subset of the 18 models, pick a metric and a scenario, and see the best model, a ranked bar chart, a leaderboard and a scenario heatmap.

## Tests

```bash
cd backend && python -m pytest -q        # 25 tests (dataset/API tests skip if data not imported)
cd frontend && npm run build              # type-check + production build
```

## Experiments

```bash
cd backend
python scripts/run_experiment.py bengalas                  # full mode, 11 fault scenarios, ~25 s
python scripts/run_experiment.py malwathu
python scripts/run_experiment.py bengalas --mode sample    # seeded 10k-row block (demo only)
python scripts/seed_sweep.py bengalas --seeds 5            # repeat over fault-placement seeds
python scripts/results_tables.py <run_id> --sweep bengalas # Markdown tables for docs
```

Outputs go to `results/runs/<run_id>/`:

* `results.json`: all metrics plus metadata (seed, dataset checksums, config, calibration, rule base, model version);
* `summary.csv`;
* `corrupted/` (model inputs);
* `ground_truth/` (injection masks, evaluation only).

## Layout

```
backend/floodguard/
  datasets/        registry (verified metadata), loaders (download, checksum, schema, quality)
  calibration.py   train-split-only parameters (thresholds, MF anchors, FOU, reliability limits)
  reliability.py   per-sensor reliability + flags + unresolved conflict
  fusion.py        reliability-weighted point estimates and interval inputs
  fuzzy/           mf.py (IT2 trapezoids, exact interval firing), km.py (Karnik–Mendel), fis.py (rules, T1/IT2)
  uncertainty.py   Q, D, F → U, bands, status
  systems.py       threshold / Type-1 / IT2 / proposed; proxy labels
  faults.py        fault injection on copies (11 scenarios)
  metrics.py, experiment.py, api.py
backend/scripts/   import_dataset, run_experiment, seed_sweep, results_tables
backend/tests/     25 pytest tests
frontend/          React + TypeScript (Vite): Dashboard, Dataset Explorer, IT2 Model, Experiments
docs/              audit, dataset guide, protocol + results, model, demo guide, research gap, output track, viva Q&A
```

## Headline results (measured; Bengalas, 5-seed mean alert F1; full tables in `docs/EXPERIMENT_PROTOCOL.md`)

| Scenario | Threshold | Type-1 FIS | IT2FIS | Proposed |
|---|---|---|---|---|
| clean | 0.862 | 0.883 | **0.895** | 0.888 |
| spikes | 0.748 | 0.760 | 0.768 | **0.835** |
| missing 30% | 0.836 | 0.839 | **0.845** | 0.825 |
| heavy noise | **0.721** | 0.642 | 0.645 | 0.636 |
| combined | 0.662 | 0.652 | 0.654 | **0.667** (within seed sd) |

The proposed system helps clearly only under spike faults, and it makes degraded data visible: U is 2.4–6.1× higher on faulted steps, and it never reports "Confident" with a missing level. It is not better under noise, missing data, stuck readings or reduced reliability, and shows no benefit on the daily Malwathu data. See the limitations in `docs/EXPERIMENT_PROTOCOL.md` §4.
