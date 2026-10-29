# Changes: v0.7, Indian CWC data, verified literature, honest metrics (2026-10-29, review preparation)

## Data
* **New dataset `cwc_chindnar`**: CWC hourly telemetry at Chindnar (Indravati River, Chhattisgarh) from the National
  Water Data Portal. Hourly level and a rain gauge at identical coordinates. The earlier claim that NWDP offers "daily
  only" data was wrong and is corrected (`docs/INDIAN_DATA_AUDIT.md`, `DATASET_DOCUMENTATION.md` §5).
* Cleaning rules are explicit and counted: off-hour readings moved to the **next** full hour (no look-ahead; the first
  version used pandas `round`, which sends hh:30 to the even hour, sometimes earlier than measured, and was replaced),
  103 impossible levels removed, 13 rain values > 100 mm/h set missing, missing rain hours assumed dry.
* `scripts/audit_cwc.py` screens every CWC station in a basin file (stuck share, outliers, nearest rain gauge).
* NWDP states no licence: raw files are not redistributed and `deploy/build_vercel.py` excludes `cwc_*` data.

## Metrics and reporting
* **Alert F1 is now undefined (NaN, shown as —) when a test period contains no true alert.** Previously it was printed
  as 0.000, which wrongly suggests total failure. Chindnar's test period (Feb 2024 to May 2025) has no level above the
  High threshold, so its alert metrics are undefined and the dashboard says so. Model selection (`patterns._fitness`)
  treats an undefined F1 as 0, so behaviour on Bengalas / Malwathu is unchanged.
* Reports now list water-level **MAE and RMSE** (persistence, ARX) and a **5-level macro F1 per fault** table.
  `compare_models.py --markdown-only` rebuilds a report from saved results.
* Dashboard: data-quality summary on the Dataset tab, MAE / RMSE panel and "no alerts in test period" warning on Compare,
  episode replay falls back to the highest level reached (Moderate on Chindnar), dataset-specific rain units.

## Reproducibility note
Re-running all comparisons on 2026-10-29 (numpy 2.5.4, pandas 3.1.0 instead of numpy 2.3.2, pandas 3.0.6) reproduced
Malwathu exactly and Bengalas exactly **except SOM at 3 h** (clean F1 0.582 → 0.600, inside the confidence interval).
The SOM weights already differ at the 16th digit of the input scaling, which training then amplifies, so this is
floating-point sensitivity, not a code change.

## Documentation
`docs/LITERATURE_REVIEW.md` (six papers checked against Crossref, with corrected details), `docs/CODE_GUIDE.md`
(beginner guide + feature status).

# Changes: v0.6, four-technique dashboard (IT2FIS, PSO, SOM, ANFIS) (2026-10-26, third pass)

The project brief now asks for a comparison of exactly **IT2FIS, SOM, PSO and ANFIS**, five alert levels, and a
four-tab website. v0.5 had removed ANFIS and PSO on instructor advice; they are restored here because the brief
requires them. **Check with your instructor that this reversal is acceptable.**

## Models
* **Compared:** IT2FIS (untuned), PSO-tuned IT2FIS, SOM, ANFIS. The threshold rule stays as a labelled *reference*
  (labels are built from it, so it is not a competitor).
* **PSO** (`tuning.py`, replaces `ga.py`; GA kept as an option): 11 genes = 10 anchor shifts + 1 FOU scale; objective =
  validation alert F1 + 0.25 × 5-level macro F1; 20 particles × 20 iterations, w = 0.7, c1 = c2 = 1.5, velocity clamp,
  untuned IT2FIS as particle 0. Tuned per horizon on the validation block only.
* **ANFIS** (`anfis.py`, restored from the removed `softcomputing.py` and generalised): 80 first-order TSK rules on the
  IT2FIS term grid, hybrid LS + Adam learning, ridge chosen on validation, rule-level explanations.
  **New constraint:** premise learning keeps centres between their neighbours and widths within 0.5–2× the initial
  value. Unconstrained training scrambled the fuzzy sets on Malwathu ("Elevated" moved below "Normal", widths collapsed
  to spikes), so the rule names in explanations were meaningless. This was fixed for interpretability, not for score.
  It lowered ANFIS's Malwathu F1 (0.83 → 0.66, very wide CI); this is reported, not tuned away.
* **Removed from the suite and the UI** (code kept in `patterns.py` / `systems.py`): Type-1 FIS, Fuzzy ART, SOM + ART,
  hybrid, GA-tuned IT2FIS.
* **Reliability layer** (the v0.4 contribution) is kept as an **on/off front end for all four models** (raw vs fused
  inputs), so its effect is measured per model and per fault.

## Five alert levels
* Normal / Low / Moderate / High / Critical from train-split thresholds μ + 0.5 / 1 / 2 / 3 σ. **Alert = High or Critical**
  (≥ μ + 2σ, the paper's flood-threshold rule), so the binary alert definition is unchanged from v0.5.
* Level gets 5 fuzzy terms; the rule base grows from 64 to 80 rules (same template, clipped to 0..4); consequents and
  band edges at 20 / 40 / 60 / 80; SOM cost matrix and metrics generalised to K classes.
* **Monotonicity fix.** With the closer Low/Moderate thresholds, a FOU clipped at the full ramp width made both lower
  MFs vanish at crossovers. The IT2 output then dipped by up to 1.06 points, and a *higher* level gave a *lower* alert
  in 9 of 182 tested curves. The FOU half-shift is now capped at half the ramp width: worst dip 0.09, no reversals.
  A floating-point tie in the trapezoid construction was also fixed.

## Evaluation
* Added: 95 % moving-block-bootstrap CI of alert F1 (block = 1 day / 14 days), Cohen's κ, per-level F1, missed alerts,
  false alarms, fit time and inference time per model; all for raw and fused inputs.
* Horizons are 30 min, 1 h and 3 h (Bengalas) and 1 day (Malwathu). The trivial "current condition" horizon was dropped.
  A per-horizon **support check** was added (labelled rows, class changes vs now, new alert onsets, persistence and
  ARX level-forecast error).
* Fault scenarios: missing 10/30 %, noise low/medium/high, stuck, conflicting rain, spikes, combined; 3 seeds.

## Website (4 tabs)
Overview · Assess Risk (historical replay of real test-period episodes, or what-if input) · Compare Models · Dataset &
Sensor Reliability. Every alert lists the reasons, built from the model's actual internals. The 15 older page files
were removed (still in git history).

## References
Added and Crossref-verified: Jang 1993 (ANFIS), Kennedy & Eberhart 1995 (PSO), Shi & Eberhart 1998 (inertia weight),
Künsch 1989 (block bootstrap), Cohen 1960 (κ). Removed (models no longer compared): Carpenter 1991, Brito da Silva 2019,
Holland 1992.

# Changes: v0.5 audit and website restructure (2026-10-26, second pass)

## Audit findings
* All nine models (A–I) are real implementations with tests. No placeholders were found. Type-1 and IT2 are distinct: IT2 uses FOU bands and exact Karnik–Mendel type reduction.
* Present-condition classification and genuine forecasting are separated. h = 0 is reported only as a consistency check (the threshold model scores 1.0 by construction there). 30 min / 1 h / 3 h use future proxy labels. ARX gives continuous level forecasts.
* **Leakage fix.** At the fit/validation boundary, the labels and ARX targets of the last h fit rows were levels inside the validation block (at most 3 h). These rows are now purged (`suite.py`), and a regression test was added. The test split was never affected. After re-running both comparisons, only the SOM-based rows changed (Bengalas 1 h, clean F1: D 0.722 → 0.737, F 0.720 → 0.734); all conclusions are unchanged.
* **Documentation fix.** Calibration (thresholds, anchors, reliability limits) uses the whole training split, not only the fit block. The docs now say so.
* **Known behaviour, documented.** The spike check can flag a genuine very fast rise, so G/H may under-call such cases (viva Q27).

## Dataset
* Indian sources checked: IMD, CWC, CWC FFS, NWDP, India-WRIS and CAMELS-IND (ESSD 2025). None offers an open, co-located sub-daily level + rain record, so Bengalas is kept and its Brazilian origin is disclosed. CAMELS-IND is recommended for a future daily Indian check.
* New `DATASET_DOCUMENTATION.md`. `references.py` / `REFERENCES.md` list the sources checked, with their findings.

## Website
* Eight pages, matching the brief: Overview, Risk Assessment, Understand the Models, Model Comparison, Sensor-Fault Lab, Forecasting, Dataset & Research, Advanced Details. The old "simple + More details" split (`More.tsx`, `SimpleLearn.tsx`) was removed; old deep links still resolve.
* Light navy/blue theme with a sticky navy header. Inter font. The automatic dark mode was dropped.
* Risk Assessment: historical replay and hypothetical input are clearly separated. Both have model and horizon selectors; replay adds a conflicting-sensor scenario and the recorded level.
* New Sensor-Fault Lab: pick a fault, see the corrupted signal and per-step class strips, the score change per model, and fault detection.
* Model Comparison: winners are named only when they lead by ≥ 0.01 F1. A metric glossary was added.
* Advanced Details: evaluation configuration, class balance and the raw experiment log.

# Changes: FloodGuard AI v0.4 → HydroSentinel AI v0.5

## What was already working (kept unchanged)

* Dataset import with checksum, schema and quality checks for Bengalas (30 min) and Malwathu (daily), with documented provenance (`datasets/`).
* Train-only calibration (`calibration.py`), Type-1 and Interval Type-2 FIS with exact Karnik–Mendel type reduction (`fuzzy/`).
* Per-sensor reliability assessment, reliability-weighted interval fusion, uncertainty index and status (`reliability.py`, `fusion.py`, `uncertainty.py`). This was the old "proposed" system, now **Model G**. The internal key is still `proposed` so stored runs remain readable.
* Fault injection with 11 seeded scenarios (`faults.py`), metrics, the experiment runner and the seed sweep.
* FastAPI + React dashboard with replay, dataset explorer, IT2 model inspector and experiment runner.

## Removed (unnecessary for the research question)

| Removed | Why |
|---|---|
| ANFIS (`softcomputing.py`) | Instructor: unnecessary ANFIS optimisation not required. With proxy labels it only re-learns the threshold rule. |
| FCM-TSK | Another rule-generation method that does not address sensor uncertainty; clearly the weakest model (F1 0.45 under faults). |
| PSO-tuned IT2FIS | Duplicated the GA tuner with the same search space and the same result (0.912 vs 0.910). One optimiser is enough. |
| ANN, SOM + ANN, GA + SOM + ANN (`neural.py`) | Supervised black boxes not asked for. The SOM is kept, but in its proper unsupervised role. |
| `+fusion` variants of every model | Doubled the model count without a separate question; fusion is evaluated once, as Model G. |
| `results/model_comparison_*`, `best_model_output_*` | Outputs of the removed comparison; replaced by `results/comparison_<ds>.json`. |

## Added

* **`patterns.py`**
  * SOM (batch, z-scored inputs). Node → risk class by minimum expected cost under the evaluation's asymmetric cost; grid size chosen on validation.
  * Fuzzy ART (complement coding, fast learning). Vigilance is chosen without labels: the largest ρ whose unfamiliar rate on clean validation data is ≤ 1 %.
  * SOM + ART.
* **`ga.py`**: the GA tuner (moved from the removed `softcomputing.py`), now run per horizon on validation data only. Genes are saved at full precision.
* **`suite.py`**: one object that fits all models (A–I) on train / validation and runs them on identical inputs per horizon. It defines the hybrid (H): risk = G, while confidence is downgraded when the fused pattern is unfamiliar (ART) or the SOM regime differs from the IT2FIS by ≥ 2 classes.
* **Forecast horizons**
  * `proxy_labels(..., horizon_steps)` gives current-condition (h = 0), 30 min, 1 h and 3 h labels; sub-day horizons are reported as unsupported for daily data.
  * `forecast.py` adds persistence and a ridge ARX level forecaster with MAE / RMSE / R².
* **`alerts.py`**: alert messages built from the actual inputs, linguistic terms, fired rules, reliability flags, SOM node, ART familiarity and status.
* **`demo.py` + `/api/assess`**: interactive assessment of hypothetical inputs with six sensor-quality scenarios. The real models run on a documented constructed history. Inputs are validated (HTTP 422 with reasons).
* **`references.py` + `scripts/write_references.py` → `REFERENCES.md`**: every DOI checked against Crossref on 2026-10-26. That check also confirmed Soares et al.'s article number e70254, previously marked unverified.
* **`scripts/compare_models.py` (rewritten)**: 8 risk models × 11 scenarios × all horizons × 3 seeds; ART fault-detection scores against ground truth; G vs H confidence metrics; forecasts.
* **API**: `/api/datasets/{id}/preview`, `/api/learn/{id}`, `/api/assess/{id}` (+ `/options`), `/api/references`; `/api/comparison/{id}` serves the new results; `/api/replay` gained `horizon` and `model` and returns all models plus an explanation.
* **Dashboard**: renamed HydroSentinel AI. New tabs: Overview (architecture diagram), Model Learning Centre, Risk Assessment, Forecast Horizons, Research & Limitations. Dataset & Sources gained a preview, column summary, rain / rise charts and split details. Replay, Comparison and Fault Experiments were reworked for the new models.
* **Tests**: 8 new tests (SOM cost mapping, ART novelty, vigilance budget, SOM + ART fallback, horizon support, suite leakage check, hybrid never changes risk, assess endpoint and validation), the API test extended to the new endpoints, the 3 tests of the removed neural models deleted. 33 in total, all passing.

## Leakage checks added

* `test_suite_ignores_test_data`: SOM codebooks, ART categories and GA genes are identical when only the test part of the series is changed.
* Labels at every horizon use the exact future timestamp (NaN across gaps) and the ORIGINAL data; fault injection never touches them.
* ART vigilance is chosen without labels or simulated faults, so its fault-detection scores are not tuned on the fault types it is tested on.
