# Paper and Dataset Audit

Audit date: **2026-10-25**. Everything below was checked against the files and pages that were actually opened. Anything not verified is labelled **UNVERIFIED**.

Method: both repositories were cloned (`git clone --depth 1`). The data files were opened with pandas and checked column by column. The importer (`backend/floodguard/datasets/loaders.py`) repeats the key checks on every import and stores the results in `data/processed/<id>.quality.json`.

---

## 1. Primary: Soares et al., UAE-ML (Bengalas River, Petrópolis, Brazil)

| Item | Verified finding |
|---|---|
| Paper | "Uncertainty-Aware Machine Learning Models for Flash Flood Prediction". Authors per repo README: J.A.J.P. Soares, A.K.S. Soares, L.F. Satolo, G.B. Scofield, S. Stephany, L.B.L. Santos. |
| Venue | *Transactions in GIS* 30(2), April 2026: confirmed only through a MetaTOC listing. DOI 10.1111/tgis.70254 redirects to Wiley, which returned **HTTP 403**. **Full text and article number e70254 UNVERIFIED.** |
| Repository | https://github.com/jaqueline-soares/UAE-ML, commit `dfb6ee046667ad7287e2630cc004b2be9aa5347b` (2026-05-18) |
| Licence | MIT (repo `LICENSE`, © 2026 the authors). Local caching and redistribution are permitted with the notice. |
| Download | `https://github.com/jaqueline-soares/UAE-ML/raw/main/data/dataset.zip`: 41,520,920 bytes, SHA-256 `19ef08b7…acef6d4`. Automatic download **succeeded**. |
| Archive contents | 12 CSVs `bacia_rio_bengalas_2019_2023_outXXhYYm.csv`, one per lead time from 00h30m to 06h00m (≈144–149 MB each). Shapefiles: watershed, hydrological station, radar grid. |
| File used | `bacia_rio_bengalas_2019_2023_out00h30m.csv`: **72,011 rows × 508 columns** |
| Columns | `datahora` (timestamp), `chuva_conselheiro`, `nivel_conselheiro`, `precip0`…`precip503` (504 columns), `out00h30m` |
| Period | 2019-01-04 14:30:00 → 2023-12-31 16:00:00. Timezone **not stated**. |
| Resolution | 30 min. 70,760 of the 72,010 steps are exactly 30 min; **1,250 gaps**, the largest 699.5 h. |
| Quality | 0 missing values, 0 duplicate timestamps, 0 negative rain values. Level ranges 0.50–4.44 (mean 0.885). |
| Rows per year | 2019: 13,395 · 2020: 15,053 · 2021: 15,764 · 2022: 13,988 · 2023: 13,811 |
| Claim ">73,000 tuples" | **Not confirmed.** The 30-min file has 72,011 rows and the 6-h file has 69,356. |
| `out00h30m` | Verified to equal `nivel_conselheiro` at t + 30 min for **100%** of the 70,760 consecutive pairs. It is the authors' prediction target, not an independent sensor, so it is **not used** as an input. |
| `precip0..503` | 504 weather-radar grid cells. `WeatherRadarCoverage_Grid.dbf` has exactly 504 records (fields id, xmin, xmax, ymin, ymax; ≈0.01° cells, WGS84). The columns are **not** time lags; a lag-structure test was negative. |
| Station | `HydrologicalStation.dbf`: network **INEA**, code 221342310, lon −42.5201, lat −22.2285 |
| Units | Level is treated as metres. This is inferred from the values and from the authors comparing it with a "flood threshold". Units of `chuva_conselheiro` and `precip*` are **not documented in the repo (UNVERIFIED)**. Radar-mean vs gauge correlation is 0.445; the radar mean is about 2.7× smaller in sum, so a mean-field bias factor is fitted on the train split. |
| Flood labels | **No observed flood labels.** The authors' `code/uncertainty-analysis/main.py` defines `threshold_flood_ds = mean + 2·std` of `nivel_conselheiro`, computed on the **full** dataset (≈1.43 m). |
| Authors' split | `HOLDOUT_SLICE = 0.9` (chronological: first 90% nested CV, last 10% holdout); peak window `2023-10-27 08:00` – `2023-11-04 23:59`. |

**Fit for rainfall + water-level fusion: good.** There is one level sensor, one rain gauge, and an independent areal rainfall source (radar), all at 30 min. True soil-moisture or flow data are **not** present, so FloodGuard does not use those variables.

**Proxy-label feasibility.** A future level exceeding the authors' statistical threshold can be computed without leakage. FloodGuard recomputes μ and σ on the **train split only**: Warning = 1.427 m (train) vs 1.43 m (authors, full data). The proxy is *exceedance of a statistical level*, not an observed flood.

## 2. Secondary: Malwathu Oya (Sri Lanka)

| Item | Verified finding |
|---|---|
| Paper | Rathnayake, N.; Rathnayake, U.; Dang, T.L.; Hoshino, Y. "Water level prediction using soft computing techniques: A case study in the Malwathu Oya, Sri Lanka", *PLOS ONE* 18(4): e0282847, 2023 (verified on journals.plos.org). |
| Paper says | Daily data 2005–2018 (years only); rainfall at Murunkan, Pavattakulam, Nachchiduva, Vavuniya, Mannar, Anuradhapura **in centimetres**; water level at Kappachchiya **in metres**; 4,765 samples; 70/30 split; ANFIS-based ensemble with XGBoost/LightGBM/CatBoost; **no numeric flood/alert levels**; data source agency not stated. |
| Repository | https://github.com/namalhappy/Malwathu_Oya_RR, commit `b4f6bfea42de79f6267cf02352cfc7cff01a4541`. MIT licence (© 2023 namalhappy). |
| Files | `malwathu_1_day.csv` (4,764 × 14), `malwathu_2_day.csv` (4,763 data rows × 21), `malwathu_3_day.csv` (4,762 × 28), plus XGBoost/LightGBM/CatBoost scripts. The scripts reference `train_3.csv` / `test_3.csv`, which are **not in the repo**. |
| Download | Raw GitHub URL for `malwathu_1_day.csv` (201,634 bytes, SHA-256 `3edcc4d6…ec49`). **Succeeded.** |
| Structure | **No timestamp column.** Each row holds day *t−1* (`<Station>-1`) and day *t* values. For every rain gauge, `<Station>-1` on row *i+1* equals `<Station>` on row *i* (verified for 100% of rows). Reconstructing the series gives **4,765 days**, matching the paper. |
| Level scaling | `Kappachichiya-1` is exactly **100×** `Kappachichiya` from the previous row (median ratio 100.0). The loader uses the metre-scaled target column. |
| Units issue | The paper says rain is in cm, but maxima reach 417.97 per day, which suggests mm. **UNVERIFIED.** Values are kept as-is; the fuzzy partitions are data-quantile based, so the unit does not change the model. |
| Duplicates | 117 identical raw rows (dry days). These are not duplicate timestamps and were not removed. |
| Flood labels | None. FloodGuard uses the same proxy rule (train μ + kσ). |

**Fit:** usable as a **second real dataset for a generalisation check** (multi-gauge rain plus a level series), at a **daily** resolution that cannot capture flash-flood dynamics. It also serves as a method reference: the paper is a soft-computing (ANFIS ensemble) study.

## 3. Other candidates (audited, not integrated)

| Candidate | Finding | Status |
|---|---|---|
| FlowDB, arXiv:2012.11154 | Godfried et al. 2020: hourly river flow and precipitation plus a flash-flood event subset. The arXiv page has **no dataset download link**; the dataset licence is not stated. | Cite only. Not integrated. |
| "LamaH-CE, arXiv:2601.09336" | **Reference mismatch.** 2601.09336 is Bertoli et al. 2026, "A Hybrid Machine Learning Framework for Improved Short-Term Peak-Flow Forecasting", which *uses* LamaH-CE (857 catchments, 6-h). It is not the LamaH-CE data paper. | **Correct the reference.** Not integrated. |
| "ANNEX benchmark, arXiv:2001.06414 App. C" | **Reference mismatch / unverified.** 2001.06414 is Okuno, Ikeuchi & Aihara 2020, "Practical data-driven flood forecasting based on dynamical systems theory: Case studies from Japan". The abstract page does not mention an "ANNEX" benchmark or a download link. Appendix C was **not checked**. | Manual verification needed. |
| Generic Kaggle/Mendeley "flood prediction" tables | Excluded by design (often synthetic, with no sensors or time axis). | Not used. |

## 4. Standard-field mapping summary

| Standard field | Bengalas | Malwathu |
|---|---|---|
| `level` [m] | `nivel_conselheiro` | `Kappachichiya` (metre-scaled) |
| `rain__*` | `rain__gauge` = `chuva_conselheiro`; `rain__radar` = mean(`precip0..503`) | `rain__<6 stations>` |
| `rise_rate` [m/h] | Δlevel / 0.5 h, only across exact 30-min steps | Δlevel / 24 h |
| time | `datahora` → `t_hours` | relative day index × 24 |
| soil moisture / flow | **not available** | **not available** |

## 5. Items still requiring manual verification

1. TGIS full text: the reported period, the ">73,000 tuples" claim, the units of rain and radar, and the timezone.
2. Malwathu rain unit (cm vs mm) and the data-provider agency.
3. LamaH-CE data paper and download (use the ESSD paper, not arXiv:2601.09336). A rainfall source would be needed if it were ever used.
4. The "ANNEX" benchmark: confirm whether arXiv:2001.06414 Appendix C exists and contains downloadable data.
