# Code guide for beginners

How a reading becomes an alert, and where each step lives. Read the files in this order.

```
NWDP / GitHub file ──► datasets/loaders.py ──► data/processed/<id>.csv
                                                  │
                         calibration.py  ◄────────┘  (train split only: thresholds, fuzzy-set positions)
                                │
     reliability.py + fusion.py (optional "reliability layer": check + fuse sensor readings)
                                │  inputs x = [level, rise, rain]
         ┌──────────────┬───────┴────────┬──────────────┐
     IT2FIS          PSO-IT2FIS         SOM           ANFIS          (suite.py runs all four)
   fuzzy/*.py        tuning.py       patterns.py     anfis.py
         └──────────────┴───────┬────────┴──────────────┘
                     risk 0-100 → Normal / Low / Moderate / High / Critical
                                │
         metrics.py (scores)  alerts.py (reasons)  api.py (website)  scripts/compare_models.py (tables)
```

## Backend (`backend/floodguard/`)

| File | What it does, in plain words |
|---|---|
| `config.py` | Folder paths and fixed settings: the five risk levels, the 0-100 band edges (20 / 40 / 60 / 80), the 70 % train split. |
| `datasets/registry.py` | One "ID card" per dataset: source URL, SHA-256 checksum, location, units, limitations. Add a dataset here first. |
| `datasets/loaders.py` | Downloads the files, checks checksums, reads each dataset's own format and writes one standard table (`level`, `rain__<sensor>`, `rise_rate`). `read_cwc_chindnar_raw` contains every cleaning rule for the Indian data, and each one is counted. |
| `calibration.py` | Learns station-specific numbers **from the training period only**: alert thresholds (mean + 0.5 / 1 / 2 / 3 sd), where the fuzzy sets sit, and limits for the reliability checks. The test period is never touched here, which prevents leakage. |
| `fuzzy/mf.py` | Trapezoid membership functions. An Interval Type-2 set is a pair of trapezoids (lower and upper), and the band between them is the footprint of uncertainty. |
| `fuzzy/fis.py` | The 80 IF-THEN rules (level × rise × rain) and the inference that fires them. |
| `fuzzy/km.py` | Karnik-Mendel type reduction: turns the fired IT2 rules into an interval [yl, yr]. |
| `tuning.py` | **PSO**: 20 particles search 11 numbers (10 shifts of fuzzy-set crossover points + 1 FOU scale) that maximise validation alert F1 + 0.25 × macro F1. The rules themselves are not changed. |
| `patterns.py` | **SOM**: learns a grid of typical river states without labels, then gives each node the risk level with the lowest expected cost over its training labels. Also contains Fuzzy ART (novelty check). |
| `anfis.py` | **ANFIS**: the same 80 rule premises, but each rule outputs a learned linear equation; trained with least squares + gradient steps. |
| `systems.py` | Builds the proxy labels (future level vs thresholds) and the processing pipeline shared by all models. |
| `suite.py` | Fits all four models once on the same training data and runs them on identical inputs. |
| `forecast.py` | Genuine **numerical** forecast of the future water level (persistence and ARX). This is the only place where MAE / RMSE are meaningful. |
| `reliability.py` | Checks every reading (missing, spike, stuck, noisy, out of range, rain sensors disagreeing) using only past data. |
| `fusion.py` | Combines checked readings, weighted by reliability, into the model inputs. |
| `uncertainty.py` | Separate confidence status (Confident / Uncertain / Insufficient data). A missing level can never give "Confident". |
| `faults.py` | Simulates sensor faults on a **copy** of the test data. The labels always come from the original data. |
| `metrics.py` | Precision, recall, F1, macro F1, κ, block-bootstrap confidence intervals. F1 is reported as undefined (—) when a period contains no real alert. |
| `alerts.py` | Turns the model internals at one time step into a human-readable reason ("rule 47 fired: level High AND rising fast …"). |
| `api.py` | FastAPI web server: the URLs the website calls. |

## Scripts (`backend/scripts/`)

| Script | Use |
|---|---|
| `import_dataset.py <id>` | Download + verify + build `data/processed/<id>.csv` and its quality report. |
| `audit_cwc.py <level.csv> <rain.csv>` | Screen all CWC stations in a basin file for a usable level + rain pair. |
| `compare_models.py <id>` | The main experiment: all models × horizons × fault scenarios; writes `results/MODEL_COMPARISON_<id>.md` and the JSON the website reads. |
| `seed_sweep.py`, `run_experiment.py`, `results_tables.py` | Older robustness experiments (kept for reproducibility). |

## Frontend (`frontend/src/`)

| File | What it shows |
|---|---|
| `App.tsx` | Top bar, dataset selector, the four tabs. |
| `pages/Overview.tsx` | Dataset facts, how the system works, the five alert levels, key results. |
| `pages/Assess.tsx` | Replay of real test-period episodes, or your own readings; the alert from each model with its reasons. |
| `pages/Compare.tsx` | Metrics table, charts, MAE / RMSE of the level forecast, confusion matrices, PSO vs untuned. |
| `pages/DataReliability.tsx` | Source, units, coverage, data-quality summary, label method, fault simulator. |
| `charts.tsx`, `api.ts`, `data.ts`, `content.ts` | Small SVG charts, API helper, data hooks and explanatory text. |

## Tests (`backend/tests/`)

`python -m pytest -q` runs 44 tests: fuzzy maths (`test_fuzzy.py`), pipeline and leakage rules
(`test_pipeline.py`), SOM / suite (`test_patterns_and_suite.py`), real imported data and API
(`test_datasets_and_api.py`), and the CWC reader plus the undefined-F1 rule (`test_cwc_and_metrics.py`).

## Feature status

| Feature | Status |
|---|---|
| Bengalas (Brazil, 30 min) and Malwathu (Sri Lanka, daily) import, checksums, quality report | **Working**, tested |
| CWC Chindnar (India, hourly) import from NWDP, cleaning rules, quality report | **Working**, tested |
| Threshold baseline, IT2FIS, PSO-tuned IT2FIS, SOM, ANFIS on identical time-ordered splits | **Working**, results in `results/` |
| Persistence / ARX water-level forecast with MAE / RMSE | **Working** |
| Reliability layer (checks + fusion + confidence status) | **Working** |
| Sensor faults (missing, noise, spikes, stuck, conflict, combined) | **Simulated** (injected into copies of real test data) |
| Flood labels | **Proxy** (statistical threshold on the future level), not observed floods, on all three datasets |
| Alert F1 on Chindnar | **Undefined**: no alert-level event in its test period |
| Official CWC warning / danger levels, rolling-origin evaluation, upstream-station input | **Pending** (future work) |
| Live / real-time data feed | **Not implemented**: historical analysis only |
