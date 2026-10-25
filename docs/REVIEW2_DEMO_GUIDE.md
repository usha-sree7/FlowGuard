# Review-II Demo Guide (10 minutes)

## Before the panel arrives (once)

```bash
cd backend
pip install -r requirements.txt
python scripts/import_dataset.py bengalas && python scripts/import_dataset.py malwathu
cd ../frontend && npm install && npm run build
cd ../backend && python -m uvicorn floodguard.api:app --port 8000
```

Open http://127.0.0.1:8000. Deep links such as `/?tab=model&ds=bengalas` jump straight to a page.

Fallback if the network fails: the datasets are cached in `data/raw/` with verified checksums, so no download is needed during the demo.

## Script

| Time | Page / action | What to say |
|---|---|---|
| 0:00–1:00 | Title / README | Problem: flood alerts depend on sensors that fail. Contribution: reliability-aware fusion, then an IT2 fuzzy system, with **two separate outputs**: risk and uncertainty status. |
| 1:00–2:30 | **Dataset Explorer** | Real public data, audited. Bengalas: 72,011 × 30-min records, INEA station, MIT licence, checksum. Show the quality table (1,250 gaps, rise rate not computed across gaps). Say plainly that the labels are **proxies** (μ+2σ is the authors' statistical rule), not observed floods. Click **Re-import dataset** to show the live validation. Show Malwathu as the second dataset, with its limitations. |
| 2:30–4:30 | **IT2 Model** | FOU shading: the FOU width is the standard error of the threshold across years. Show the 64-rule grid. In **Try the inference**, widen "± level" and watch [yl, yr] widen while the T1 value stays a single number. KM type reduction is fully implemented and tested against iterative KM. |
| 4:30–7:00 | **Dashboard** | Choose scenario `spikes` and press *Next high-water event*, then *Play*. The raw input spikes; the fused estimate does not. U jumps and the status leaves "Confident", while the risk class is not driven by the spike. Compare the four systems on the same input. Switch to `missing_30`: the status becomes "Uncertain" or "Insufficient data", never "Confident" with a missing level. |
| 7:00–9:00 | **Experiments** | Open the full Bengalas run. F1 table: the proposed system wins clearly on spikes (0.828; 5-seed mean 0.835 vs 0.768). Be honest that it is **not** better under noise, missing data, stuck sensors or reduced reliability, and not on the daily Malwathu data. Show the U-response table (faulted vs unfaulted) and the fault-detection P/R. Optionally click *Run new experiment* (full mode, about 25 s). |
| 9:00–10:00 | `docs/` | Disclosed design revision (v0.3 negative result), the research gap, the article/patent/startup track, and next milestones. |

## Rubric mapping

| Review-II rubric row | Evidence in the project |
|---|---|
| ≥ 50–70% implementation | Working end-to-end system. Loaders for 2 real datasets with download, checksum and validation. Reliability assessment, fusion, IT2FIS (with KM), Type-1 baseline, threshold baseline, uncertainty index and status, fault injection, metrics, experiment runner, REST API, 4-page UI. 24 automated tests. |
| Dataset preparation | `PAPER_DATASET_AUDIT.md`, `DATASET_GUIDE.md`, Dataset Explorer (quality report computed from the files actually read), `data/processed/*.quality.json` |
| Preliminary experimental results | `EXPERIMENT_PROTOCOL.md` §3 (generated tables), `results/runs/*/results.json` + `summary.csv`, `results/seed_sweep_*.csv` |
| Soft-computing justification | `MATHEMATICAL_MODEL.md` §5. Model page (FOU, rules, KM interval). IT2 vs T1 comparison on identical rules. |
| Article / patent / startup track | `OUTPUT_TRACK.md`, `RESEARCH_GAP.md` |
| Honesty / limitations | Negative results reported, design revision disclosed, proxy labels stated everywhere, unverified references flagged |
| Viva readiness | `VIVA_QA.md` |
