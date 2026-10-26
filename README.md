# HydroSentinel AI

**Five-level flood-risk alerts from four soft-computing techniques (IT2FIS, PSO-tuned IT2FIS, SOM, ANFIS), compared on real river data, with sensor-reliability simulation** (Soft Computing course project; previously named *FloodGuard AI*).

```
sensors (water level, rain gauge / radar / several gauges)
  → [optional] reliability layer: check each reading (missing, spike, stuck, noisy, out-of-range, conflict) and fuse
  → 3 inputs: water level, rate of rise, recent rainfall
  → IT2FIS │ PSO-tuned IT2FIS │ SOM │ ANFIS          (identical inputs, run side by side)
  → alert level Normal / Low / Moderate / High / Critical  +  the reasons, built from each model's internals
```

## The four techniques (all implemented here, all evaluated on the same held-out data)

| Technique | What it does | What is learned, from which data | Code |
|---|---|---|---|
| **IT2FIS** | 80 expert IF-THEN rules; Interval Type-2 fuzzy sets; Karnik–Mendel type reduction → risk 0–100 | nothing from labels: set positions from train-split statistics | `fuzzy/`, `calibration.py` |
| **PSO-tuned IT2FIS** | same rules; particle swarm moves the 10 crossover points and scales the FOU | 11 genes per horizon, objective = validation alert F1 + 0.25 × macro F1 | `tuning.py` |
| **SOM** | 2-D map of prototype river states; node → alert level by minimum expected cost | prototypes (fit block), node levels (fit labels), grid size (validation) | `patterns.py` |
| **ANFIS** | 80 first-order TSK rules, Gaussian MFs, hybrid LS + gradient learning | MFs + 320 consequents (fit block), ridge (validation) | `anfis.py` |
| *Threshold rule* | *reference only*: alert level of the current level (the labels are built from it) | thresholds | `systems.py` |

The **reliability layer** (`reliability.py`, `fusion.py`, `uncertainty.py`) can be switched on for any model; it also gives a confidence status (Confident / Uncertain / Insufficient data).
How PSO, SOM and ANFIS work mathematically: `docs/MATHEMATICAL_MODEL.md` §5.

## Data (see `DATASET_DOCUMENTATION.md` and `REFERENCES.md`)

* **Bengalas River, Petrópolis, Brazil** (primary): 72,011 × 30-min records, 2019-01-04 14:30 → 2023-12-31 16:00, station 221342310, from the MIT-licensed UAE-ML repository of Soares et al., *Transactions in GIS* 30(2) e70254 (2026), doi:10.1111/tgis.70254. Water level [m], station rain gauge and a 504-cell weather-radar grid (rain units not documented in the source).
* **Malwathu Oya, Sri Lanka** (generalisation check): 4,765 daily records (2005–2018, no dates in the file) from Rathnayake et al., *PLOS ONE* 18(4) e0282847 (2023).
* **Labels are proxies.** No flood events are recorded. The label is the alert level of the water level *actually observed* h steps later, using train-split thresholds μ + 0.5 / 1 / 2 / 3 σ (Bengalas 0.993 / 1.137 / 1.427 / 1.716 m). **Alert = High or Critical** (≥ μ + 2σ, the paper's flood-threshold rule).
* **Horizons (checked):** Bengalas supports 30 min, 1 h and 3 h. Malwathu is daily, so only 1 day is supported; sub-daily horizons are reported as unsupported.
* Indian sources (IMD, CWC, India-WRIS, NWDP, CAMELS-IND) were checked: none offers open, co-located sub-daily level + rain data.

## Results (held-out test period; full tables in `results/MODEL_COMPARISON_*.md`)

Bengalas, **1 h ahead**, raw inputs, alert F1 with 95 % block-bootstrap interval:

| Model | Clean F1 [95 % CI] | Precision / Recall | Missed / false alarms | Mean F1 over 9 faults | Spikes: layer off → on |
|---|---|---|---|---|---|
| IT2FIS | 0.887 [0.850, 0.915] | 0.866 / 0.910 | 59 / 92 | 0.794 | 0.755 → 0.821 |
| PSO-tuned IT2FIS | 0.894 [0.859, 0.922] | 0.901 / 0.888 | 73 / 64 | 0.791 | 0.756 → 0.809 |
| SOM | 0.737 [0.665, 0.796] | 0.617 / 0.914 | 56 / 371 | 0.497 | 0.534 → 0.626 |
| ANFIS | 0.867 [0.820, 0.902] | 0.830 / 0.907 | 61 / 121 | 0.649 | 0.651 → 0.750 |
| *Threshold (reference)* | *0.862 [0.820, 0.895]* | *0.858 / 0.867* | *87 / 94* | *0.796* | *0.739 → 0.799* |

What the numbers support:

* **IT2FIS, PSO-IT2FIS and ANFIS are statistically tied on clean data** (overlapping intervals). SOM is clearly weaker: it raises many false alarms (371 vs 92 for IT2FIS), consistent with its cost-based node mapping, which deliberately errs towards the higher level when a node holds mixed states.
* **PSO** improves test F1 at every Bengalas horizon (+0.009 / +0.007 / +0.031 at 30 min / 1 h / 3 h), but never beyond the confidence interval. Its main effect is fewer false alarms at the cost of slightly more missed alerts. On Malwathu it raises validation fitness but leaves test F1 unchanged (0.864).
* **Robustness:** the IT2 models degrade least under simulated faults. ANFIS collapses under noise (F1 0.41 at medium, 0.19 at high noise at 1 h); the cause has not been isolated.
* **Reliability layer:** clearly helps against spikes for every model and helps ANFIS under noise. It costs a little on clean data and with missing readings. Its checks detect missing data almost perfectly but catch few stuck (3 %) or conflicting-rain (1 %) faults.
* **Lead time:** skill falls with horizon for every model (IT2FIS 0.935 → 0.887 → 0.664). At 3 h the level class changes on 10 % of steps, versus 2.7 % at 30 min.
* **Daily data (Malwathu, 1 day):** IT2FIS = PSO-IT2FIS 0.864, SOM 0.820, ANFIS 0.661 (very wide interval: only 66 alert days in the test period).

## Requirements and setup

Python ≥ 3.11 (tested 3.13), Node ≥ 18 (tested 24).

```bash
cd backend
pip install -r requirements.txt
python scripts/import_dataset.py bengalas     # download (41.5 MB) + checksum + validate
python scripts/import_dataset.py malwathu
python scripts/compare_models.py malwathu --seeds 3   # ~20 s
python scripts/compare_models.py bengalas --seeds 3   # ~11 min; the website reads these results
python scripts/write_references.py                    # regenerates REFERENCES.md

cd ../frontend
npm install
npm run build

cd ../backend
python -m uvicorn floodguard.api:app --port 8000
# open http://127.0.0.1:8000  (the first Assess Risk load refits the models on train data: ~1-2 min on Bengalas)
```

Dev mode: keep uvicorn running, then `cd frontend && npm run dev` and open http://localhost:5173.

## Website (four tabs)

| Tab | What it shows |
|---|---|
| Overview | What the system does, pipeline diagram, the five alert levels with thresholds in metres, how each technique works (inputs, outputs, what is learned), key results, caveats |
| Assess Risk | **Historical replay** of real test-period flood episodes: level / rain / rise charts, per-step alert strips for every model, the alert each model issues at any chosen step with its reasons, and what actually happened h later. **What-if** inputs with simulated sensor conditions. Horizon and reliability-layer controls on both |
| Compare Models | Metrics table with confidence intervals, fit and inference time; precision / recall / F1 chart; F1 by horizon; per-level F1; confusion matrices; **PSO vs untuned IT2FIS** (objective, convergence, moved crossover points, tuned MF curves) |
| Dataset & Sensor Reliability | Dataset name, DOI, repository and download link, location, timestamps, units, label generation, split; full-period charts of level, rainfall and rise rate; **horizon support check**; **fault simulator** (missing, noisy, stuck, conflicting, spikes, combined) with corrupted signals, per-model strips, and scores with the reliability layer off vs on |

## Tests

```bash
cd backend && python -m pytest -q      # dataset / API tests skip if the data is not imported
cd frontend && npm run build           # type-check + production build
```

## Hosted demo

This version (v0.6) runs read-only at **https://floodguard-ai-chi.vercel.app** (public, no login). To republish after changes:
`python deploy/build_vercel.py && cd deploy/build && vercel deploy --prod` (needs the Vercel CLI).

## Documentation

`DATASET_DOCUMENTATION.md` · `REFERENCES.md` (Crossref-verified) · `docs/CHANGES.md` (what changed and why, incl. disclosed design changes) · `docs/VIVA_QA.md` · `docs/DEMO_SEQUENCE.md` · `docs/MATHEMATICAL_MODEL.md` · `docs/EXPERIMENT_PROTOCOL.md`.
`docs/FloodGuard_Project_Guide.pdf` describes an earlier version and is kept for history only.
