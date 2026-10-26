# Dataset Guide

## Importing

```bash
cd backend
python scripts/import_dataset.py bengalas    # downloads 41.5 MB from the UAE-ML GitHub repo
python scripts/import_dataset.py malwathu    # downloads 0.2 MB
```

You can also open the **Dataset Explorer** tab and click **Import dataset**.

Each import runs these steps in order:

1. **Download** the original file to `data/raw/<id>/`. If the file is already there with the correct checksum, the download is skipped.
2. **Verify SHA-256** against the value pinned in `backend/floodguard/datasets/registry.py`. A mismatch aborts the import.
3. **Read the actual schema.** The importer fails loudly if any expected column is missing. It never guesses.
4. **Validate reader assumptions:**
   - Bengalas: `out00h30m` must equal the next-step level.
   - Malwathu: the lag columns must match the previous row.
5. **Standardise** the data to the processed schema and write `data/processed/<id>.csv`. No rows are dropped, no values are imputed, and no outliers are removed.
6. **Read the processed copy back and validate it.** Only after this does the dataset count as *loaded*, and `<id>.quality.json` is written.

If the download fails, the API and the UI show manual instructions: the exact URLs, target paths and checksums. To import files you placed yourself:

```bash
python scripts/import_dataset.py <id> --no-download
```

## Stores

| Store | Location | Purpose |
|---|---|---|
| Original | `data/raw/<id>/` | Byte-identical download. Never modified. |
| Processed | `data/processed/<id>.csv` | Standardised copy: `t_hours, timestamp, level, rain__*, step_ok, rise_rate` |
| Quality report | `data/processed/<id>.quality.json` | Row counts, gaps, missing values, invalid values, statistics, reader checks, checksums |
| Corrupted inputs | `results/runs/<run>/corrupted/<scenario>.csv.gz` | Model input only |
| Injection ground truth | `results/runs/<run>/ground_truth/<scenario>.csv.gz` | Evaluation only |

## Quality report (actual values)

| | Bengalas | Malwathu |
|---|---|---|
| Rows / valid rows | 72,011 / 72,011 | 4,765 / 4,765 |
| Span | 2019-01-04 14:30 → 2023-12-31 16:00 | day 0 → day 4764 (no dates in source) |
| Nominal step | 0.5 h | 24 h |
| Gaps (> 1 step) | 1,250 (largest 699.5 h) | 0 |
| Duplicate timestamps | 0 | n/a |
| Missing values | 0 | 0 |
| Level outside the configured plausibility range | 0 (range 0–10 m) | 0 (range 0–15 m) |
| Rise rate computable | 70,760 rows | 4,764 rows |

The plausibility ranges are configured sanity bounds. They are **not** flood thresholds.

## Derived variables

* `rise_rate = (level_t − level_{t−1}) / step` is computed **only** when the previous row is exactly one nominal step earlier. Across the 1,250 Bengalas gaps it is NaN; it is never interpolated.
* `rain__radar` (Bengalas) is the arithmetic mean of the 504 radar cells. A multiplicative radar→gauge bias factor is fitted on the training split only (Σgauge / Σradar = 2.665). It is applied inside the model, not in the processed file.

## Labels: what they are and are not

| Type | Used? | Definition |
|---|---|---|
| True observed flood labels | **Not available** in either source | none |
| **Proxy labels** | Yes (evaluation) | Class of the **observed level `horizon` steps in the future**, from the ORIGINAL data, using thresholds computed on the train split: Low μ+0.5σ, Moderate μ+σ, High μ+2σ, Critical μ+3σ. Horizons: 30 min, 1 h, 3 h for Bengalas; 1 day for Malwathu. |
| Synthetic demo labels | Not used | none |

Threshold sources:

* **High = μ+2σ** is the statistical "flood threshold" rule in Soares et al.'s `uncertainty-analysis/main.py`. The authors computed it on the full data; HydroSentinel AI uses the train split only.
* **Low = μ+0.5σ**, **Moderate = μ+1σ** and **Critical = μ+3σ** are project extensions. They are not from either paper. Alert = High or Critical.
* Bengalas thresholds: 1.137 / 1.427 / 1.716 m. Malwathu thresholds: 0.572 / 1.002 / 1.433 m.

**Circularity check.** The threshold baseline applies the same thresholds to the *current* level, so evaluating it on the *future* level is a persistence test, not a self-consistency test. All systems predict a future quantity they never see.

## Sample vs full mode

* `mode="full"` uses all rows.
* `mode="sample"` uses a **seeded contiguous block** (default 10,000 rows) so time order is kept.

**Caveat (measured):** in a sample, the thresholds and anchors are recalibrated on a short training window. For seed 42 that window was Jul–Dec 2019, which gave a μ+2σ threshold of 1.15 m vs 1.43 m on the full train split. Sample results are therefore **not comparable** to full-mode results. Use sample mode only for quick demos.

## Limitations

* Proxy labels only. Results say nothing about observed flood damage.
* Bengalas rain units and timezone are undocumented. Malwathu rain unit is ambiguous (cm vs mm) and has no dates.
* There is a single level sensor per site, so level faults cannot be cross-checked against a redundant level gauge.
* Malwathu is daily, which is too coarse for flash-flood dynamics. It has only 66 positive (≥ High) test labels.
