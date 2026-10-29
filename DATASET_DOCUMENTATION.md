# HydroSentinel AI: Dataset Documentation

Every value in this document was read from the imported files, their quality reports (`data/processed/<id>.quality.json`), the
dataset registry (`backend/floodguard/datasets/registry.py`) or the stored evaluation metadata (`results/comparison_<id>.json`).
Source pages were checked on **2026-10-26**; Indian sources re-checked on **2026-10-29** (`docs/INDIAN_DATA_AUDIT.md`).

## 1. Decision in one paragraph

HydroSentinel AI keeps the **Bengalas River (Brazil)** dataset as its primary data and the **Malwathu Oya (Sri Lanka)** dataset as a daily
generalisation check. A third dataset, **CWC hourly telemetry at Chindnar (Indravati River, Chhattisgarh, India)** from the National
Water Data Portal, was added on 2026-10-29 (Section 5 and `docs/INDIAN_DATA_AUDIT.md`). It has co-located hourly level and rain, but
also serious quality problems (non-zero-only rain records, implausible values, gaps), no licence statement and **no alert-level event
in its test period**, so it serves as an Indian data-integration and robustness check, not as evidence of flood-detection skill.
Bengalas remains primary: water level and rainfall from the same station at 30-min steps for five years, with a peer-reviewed source
and an open licence. IMD and India-WRIS data are **not** used.

## 2. Primary dataset: Bengalas River (UAE-ML)

| Field | Value |
|---|---|
| Dataset name | Bengalas River watershed, 30-min hydro-meteorological series (`bacia_rio_bengalas_2019_2023_out00h30m.csv`) |
| Published with | Soares, J.A.J.P.; Soares, A.K.S.; Satolo, L.F.; Scofield, G.B.; Stephany, S.; Santos, L.B.L. (2026). *Uncertainty-Aware Machine Learning Models for Flash Flood Prediction*. Transactions in GIS 30(2), e70254. doi:[10.1111/tgis.70254](https://doi.org/10.1111/tgis.70254) |
| Original provider | The paper's authors (repository UAE-ML). Station code 221342310 of INEA (Instituto Estadual do Ambiente, RJ) appears in the repository files. We downloaded from the authors, so the original agency provenance is **not independently verified**. |
| Official source URL | https://github.com/jaqueline-soares/UAE-ML (commit `dfb6ee046667ad7287e2630cc004b2be9aa5347b`) |
| Download link | https://github.com/jaqueline-soares/UAE-ML/raw/main/data/dataset.zip (41,520,920 bytes, SHA-256 `19ef08b7…acef6d4`) |
| Licence | MIT (repository LICENSE, © 2026 the paper authors) |
| Location | Bengalas River watershed, Petrópolis, RJ, Brazil. Station at lon −42.5201, lat −22.2285 (from `HydrologicalStation.dbf`) |
| Collection period | 2019-01-04 14:30 → 2023-12-31 16:00 (first and last timestamps actually present) |
| Sampling frequency | 30 min. 70,760 of the 72,010 consecutive steps are exactly 30 min; 1,250 are longer gaps (largest 699.5 h) |
| Records | 72,011 rows, 0 missing values, 0 duplicate timestamps, 0 out-of-range levels |
| Timezone | Not stated in the repository; treated as naive local time |

### Features

| Source column | Used as | Unit | Notes |
|---|---|---|---|
| `datahora` | `timestamp` | – | |
| `nivel_conselheiro` | `level` | m (inferred) | Water level at the station, 0.50–4.44 m; mean 0.885, sd 0.272 |
| `chuva_conselheiro` | `rain__gauge` | mm per 30 min (**unit not documented**) | Station rain gauge, max 48.4 |
| `precip0 … precip503` | `rain__radar` (mean of the 504 cells) | not documented | Weather-radar grid; a radar→gauge bias factor (2.665) is fitted on the training split inside the model |
| `out00h30m` | not used | m | Verified equal to the next-step level for all 70,760 consecutive pairs; dropped to avoid leaking the target |

## 3. Secondary dataset: Malwathu Oya (generalisation check)

| Field | Value |
|---|---|
| Published with | Rathnayake, N.; Rathnayake, U.; Dang, T.L.; Hoshino, Y. (2023). *Water level prediction using soft computing techniques: A case study in the Malwathu Oya, Sri Lanka*. PLOS ONE 18(4), e0282847. doi:[10.1371/journal.pone.0282847](https://doi.org/10.1371/journal.pone.0282847) |
| Source / download | https://github.com/namalhappy/Malwathu_Oya_RR (commit `b4f6bfea`), file `malwathu_1_day.csv` (201,634 bytes, SHA-256 `3edcc4d6…ec49`) |
| Licence | MIT (© 2023 namalhappy) |
| Location | Malwathu Oya basin, Sri Lanka. Level at Kappachchiya; rain gauges Murunkan, Pavattakulam, Nachchiduva, Vavuniya, Mannar, Anuradhapura |
| Period | 2005–2018 according to the paper. **The file has no dates**, so a day index 0–4764 is used |
| Sampling | Daily, 4,765 reconstructed days, no gaps, no missing values |
| Units | Level in m (metre-scaled target column; the lag column is in cm, ×100). Rain unit ambiguous: the paper says cm, magnitudes look like mm. Kept as-is |
| Limitation | Daily data cannot show flash-flood dynamics; the 30 min / 1 h / 3 h horizons are **not supported** and are reported as such |

## 4. Derived features, labels and splits (both datasets)

* **Rate of rise** `rise_rate = (level_t − level_{t−1}) / Δt` [m/h]. Computed only when the previous row is exactly one nominal step earlier, never across a gap and never interpolated.
* **Recent rainfall** is accumulated over a causal window: 2 h (4 steps) for Bengalas, 2 days for Malwathu.
* **Labels are PROXY labels.** Neither source contains recorded flood events. The label is the class of the **observed level h steps in the future**, using thresholds computed on the training split only:
  Low = μ + 0.5σ, Moderate = μ + 1σ, **High = μ + 2σ** (the statistical "flood threshold" rule in Soares et al.'s code), Critical = μ + 3σ (Low, Moderate and Critical are our extensions). Five alert levels: Normal / Low / Moderate / High / Critical.
  Bengalas thresholds: 0.993 / 1.137 / 1.427 / 1.716 m. Malwathu: 0.357 / 0.572 / 1.002 / 1.433 m.
  Performance against these labels measures agreement with a level-threshold rule. **It does not prove real-world flood-warning accuracy.**
* **Positive class** = alert = High or Critical.
* **Horizon support (checked, `results/MODEL_COMPARISON_<id>.md`).** Bengalas (30-min steps) supports 30 min, 1 h and 3 h: 21,230 / 20,904 / 20,359 labelled test rows, 674 / 653 / 633 alert steps, and the level class changes between now and t + h on 2.7 % / 4.7 % / 10.2 % of steps (46 / 87 / 223 new alert onsets), so the task is not trivial. Malwathu is daily: only 1 day is possible, and 30 min / 1 h / 3 h are reported as unsupported, never interpolated.
* **Class imbalance (test split).** Bengalas, 1 h ahead: Normal 14,672, Low 3,844, Moderate 1,735, High 346, Critical 307 (3.1 % alerts; 700 unlabelled because the future reading falls in a gap). Malwathu, 1 day ahead: 1,290 / 35 / 38 / 12 / 54 (4.6 % alerts).
* **Chronological split, no shuffling.** Train = first 70 % (fit block = its first 80 %, validation = its last 20 %), test = last 30 %.
  Bengalas: fit 2019-01-04 → 2021-10-04 19:00, validation → 2022-07-10 15:00, test 2022-07-10 15:30 → 2023-12-31 16:00.
* **Leakage controls.** Thresholds, fuzzy anchors and reliability limits are calibrated on the whole training split (fit + validation). SOM, ANFIS and ARX are fitted on the fit block only. PSO tuning and model selection (SOM grid, ANFIS ridge) use the validation block. The test split is never used for fitting, tuning or selection. Test labels are computed from the test split alone. The last h fit rows, whose targets lie inside the validation block, are purged from SOM, ANFIS and ARX fitting. Unit tests confirm that changing the test data leaves the SOM, PSO and ANFIS parameters unchanged, and that changing the validation block leaves the ARX coefficients unchanged.

## 5. Indian data sources investigated

| Source | What was found (2026-10-26) | Usable for this project? |
|---|---|---|
| [India Meteorological Department](https://mausam.imd.gov.in/) | Rainfall (station and gridded), daily. No river level | No: no co-located water level |
| [Central Water Commission](https://cwc.gov.in/) | Gauge level is read hourly only in the monsoon (three readings/day otherwise). Validated archives are released through India-WRIS to registered users | Not openly, not continuously sub-daily |
| [CWC Flood Forecasting System](https://ffs.india-water.gov.in/) | Live hydrographs for roughly the last 72 h | No bulk history found |
| [National Water Data Portal](https://nwdp.nwic.gov.in/) | **Corrected 2026-10-29:** also hosts *hourly* CWC telemetry: [river water level](https://nwdp.nwic.gov.in/en/dataset/river-water-level-telemetry-hourly-central-water-commission-cwc) (per basin) and [rainfall](https://nwdp.nwic.gov.in/dataset/rainfall-cwc-telemetry-hourly) (per state), direct CSV download, no licence stated | **Yes: Chindnar integrated** as `cwc_chindnar` (see `docs/INDIAN_DATA_AUDIT.md`) |
| [India-WRIS](https://indiawris.gov.in/wris/) | Script-only application (timed out on the first check); downloads need registration | A level + rain archive could not be verified |
| CAMELS-IND: Mangukiya et al. (2025), *Earth System Science Data* 17, 461–491, doi:[10.5194/essd-17-461-2025](https://doi.org/10.5194/essd-17-461-2025); data doi:[10.5281/zenodo.14005378](https://doi.org/10.5281/zenodo.14005378) | Verified open Indian dataset: 472 catchments, 1980–2020, **daily** streamflow and meteorological forcings, no water level | Not for sub-daily horizons. **Best candidate for a future daily Indian check** (a discharge-threshold proxy, like Malwathu) |

An open-source downloader, HydroSwift (https://hydroswift.readthedocs.io/), claims scripted access to India-WRIS and CWC water level
and rainfall. Its documentation does not state the temporal resolution or licence, and it could not be installed and tested in this
session. Someone with India-WRIS registration should test it before anyone claims an Indian sub-daily dataset is available.

## 6. Preservation and transformation log

1. Download the original file to `data/raw/<id>/`. It is never modified.
2. Verify SHA-256 against the pinned value; a mismatch aborts.
3. Read the schema and fail loudly if an expected column is missing.
4. Check reader assumptions (Bengalas: `out00h30m` = next level; Malwathu: lag columns = previous row).
5. Standardise to `t_hours, timestamp, level, rain__*, step_ok, rise_rate` in `data/processed/<id>.csv`. Bengalas / Malwathu: **no rows dropped, no imputation, no outlier removal.** CWC Chindnar: documented cleaning rules (impossible levels removed, rain > 100 mm/h set missing, missing rain hours assumed dry), each counted in the quality report.
6. Read the processed file back, validate it, and write the quality report.
7. Simulated faults are applied only to an in-memory copy of the test split. Ground-truth masks are stored separately and labels always come from the original data.

Import with `python backend/scripts/import_dataset.py bengalas` (and `malwathu`, `cwc_chindnar`). See `docs/DATASET_GUIDE.md` for details.

## 7. Limitations

* Proxy labels only; no observed flood impacts.
* Rain units and timezone are undocumented for Bengalas. The rain unit is ambiguous and there are no dates for Malwathu.
* One level sensor per site, so level faults cannot be cross-checked against a redundant level gauge.
* The sensor faults are simulated. Neither dataset labels real sensor failures.
* Indian data (Chindnar) is a single station whose test period has no alert-level event, so no Indian flood-detection skill is claimed (Section 5).
