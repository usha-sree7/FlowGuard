# Indian data audit (CWC telemetry via NWDP, IMD)

Audit date: **2026-10-29**. Every number below was computed from files actually downloaded on that date
(scripts: `backend/scripts/audit_cwc.py`, importer `backend/floodguard/datasets/loaders.py`).
This report **replaces** the earlier statement in `DATASET_DOCUMENTATION.md` §5 that NWDP offers "daily only" data:
hourly CWC telemetry files *are* openly downloadable.

## 1. Sources checked

| Source | What is actually there | Usable here? |
|---|---|---|
| [NWDP: Rainfall (Telemetry - Hourly), CWC](https://nwdp.nwic.gov.in/dataset/rainfall-cwc-telemetry-hourly) | One CSV per state and period (1991-2020, 2021-2025, 2026-2030). Direct HTTP download, no login. Column `Telemetry Hourly Rainfall (mm)`. | **Yes**, with the caveats in §3 |
| [NWDP: River Water Level (Telemetry - Hourly), CWC](https://nwdp.nwic.gov.in/en/dataset/river-water-level-telemetry-hourly-central-water-commission-cwc) | One CSV per river basin and period (1961-1990, 1991-2020, 2021-2025, 2026-2030). Direct download, no login. Column `River Water Level Telemetry Hourly (meter)` plus gauge-zero RL. | **Yes**, with the caveats in §3 |
| [IMD](https://mausam.imd.gov.in/) | Forecasts, warnings, rainfall summaries, an API (api.imd.gov.in) and a Data Supply Portal (dsp.imdpune.gov.in). Historical station data is supplied on request. No river levels. | Not for this project: no open bulk station download, no water level |

Licence: the NWDP dataset pages state **no open licence** (footer "© NWIC, All Rights Reserved"; the National Water
Data Policy applies). The project therefore does **not** redistribute the raw files: the importer downloads them from
NWDP and checks their SHA-256.

## 2. Files downloaded and their schema

| File | Size | SHA-256 (first 16) | Rows |
|---|---|---|---|
| `rwl_tel_hr_cwc_006_2021_2025.csv` (Godavari basin, level) | 185,715,692 B | `675a55fc63aab786` | 1,055,737 (66 stations) |
| `rainfall_tel_hr_cwc_cg_2021_2025.csv` (Chhattisgarh, rain) | 5,850,926 B | `6338787c5beb798e` | 40,151 |
| `rainfall_tel_hr_cwc_ap_2021_2025.csv`, `_ts_`, `_mh_` (checked, not used) | | | |
| `rwl_tele_hr_cwc_006_1991_2020.csv` (checked, not used) | 79,926,803 B | `61ee620221c9b84b` | Chindnar: 8,288 rows, Jun 2015 to Dec 2020 |
| `rwl_tel_hr_cwc_006_2026_2030.csv` (checked) | 2,748,268 B | | no Chindnar rows |

Columns (both files): `Station, Agency, State, District, ..., River, Basin, Tributary, ..., Latitude, Longitude,
Data Acquisition Time` (format `DD-MM-YYYY HH:MM`, timezone not stated) and the value column. Level files also carry
`RL_of_zeroGauge` and `Is_DischargeDataAvailable`.

## 3. Problems found in the real data

| Problem | Evidence |
|---|---|
| **Rain file stores non-zero hours only** | Minimum value 0.1 mm, zero share 0.000. A missing hour cannot be told apart from a dry hour. |
| **Implausible rain values** | Almost every gauge peaks near 500 mm/h; Polavaram repeats 142.5 mm/h for many consecutive hours; Chindnar records 342 mm in one hour in December 2023. |
| **Stuck / flat level sensors** | 99 %+ of readings identical at Ghugus, Purna, Mancherial, Degloor, Kiwaibalenga; Koida 76 %; Ambabal 82 %. |
| **Regulated sites** | Polavaram (dam under construction, pond ~33 m), Dowlaiswaram (barrage pond, 12.99 m floor), Kaleswaram (level baseline drops from about 101.7 m to 96 m after late 2023). |
| **Corrupted level values** | Polavaram: values ending in .517 up to 1,018 m; Chindnar: 103 values above 350 m RL (up to 909 m). |
| **Irregular timing** | 486 Chindnar level readings at hh:30 or other off-hour minutes; 1,944 gaps after import, largest 1,316 h. |
| **Weak local rain signal** | Correlation of past-24 h rain with next-24 h level rise: Chindnar 0.08, Kaleswaram 0.09, Bhadrachalam 0.02, Jagdalpur -0.13. A single point gauge explains little at these large rivers. |

## 4. Station choice

`audit_cwc.py` lists every level station with its nearest rain gauge. Nearly all Godavari telemetry level stations
have a rain gauge at **identical coordinates**, so spatial compatibility is not the problem; data quality is.
Chindnar (Indravati River, Dantewada, Chhattisgarh; lat 19.0831, lon 81.3000; gauge zero 327.15 m) was chosen because:

* level and rain gauge share the same coordinates (checked again by the importer, which refuses mismatches);
* natural, unregulated monsoon rises of 2 to 10 m in every year 2021 to 2024;
* few flat runs (most common value 11.7 % of readings, the dry-season floor at about 330.4 m);
* 22,847 hourly level readings and 1,441 rain records in 2021-2025.

Jagdalpur has fewer flat readings (1.6 %) but monsoon ranges of only 0.1-1.1 m in 2022-2023, so it carries less
flood signal. Bhadrachalam (the classic Godavari flood gauge) sits on a 39.43 m floor 43 % of the time.

## 5. Preprocessing applied (in `read_cwc_chindnar_raw`, counts stored in `data/processed/cwc_chindnar.quality.json`)

1. Keep rows with `Station == "Chindnar"` from both files; check that level and rain coordinates match.
2. Move off-hour readings to the **next** full hour (never earlier than measured, so no look-ahead); average
   readings that land on the same hour (166 level rows merged).
3. Remove level readings outside 327.15-350 m RL (below gauge zero or impossible): **103 removed**.
4. Rain: hours with a level reading but no rain record → **0 mm (assumption)**; rain above 100 mm/h → missing
   (**13 values**); rain records at hours without a level reading are dropped (43).
5. Compute `rise_rate` only across exactly one hour; never across gaps. No interpolation.

Result: **22,578 hourly rows, 2021-01-01 00:00 to 2025-05-17 11:00**, 1,944 gaps, 8 missing rain values.
Rain totals on kept hours: about 2,000-2,550 mm in the full years 2022-2024.

## 6. Labels and what the results can and cannot show

There are **no observed flood events and no CWC warning / danger levels** in the files, so the same proxy rule as
for the other datasets is used (train-split mean + 0.5 / 1 / 2 / 3 sd; Alert = above mean + 2 sd = 337.8 m).

**Finding:** the chronological test period (last 30 %, Feb 2024 to May 2025) contains **no** hour above the High
threshold: the 2024 monsoon peaked at 336.9 m. Alert precision, recall and F1 are therefore **undefined** on this
dataset and are shown as "—". Extending the record with the 2015-2020 file (tested after the first run) does not
change this: the test period would still contain no alert. The Chindnar results are therefore reported as a
**data-integration and robustness check** (false alarms, 5-level macro F1, κ, water-level MAE / RMSE), **not** as
flood-detection skill.

## 7. Future work

* Obtain official CWC warning / danger levels for Chindnar to replace the statistical proxy.
* Use rolling-origin evaluation (each monsoon in turn as test) so that alert metrics become defined.
* Add an upstream level station (e.g. Jagdalpur) as a routing input; local point rain is weakly informative here.
* Ask NWIC whether a missing rain hour means "dry" or "no data".
