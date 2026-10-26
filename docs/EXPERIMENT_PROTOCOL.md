# Experiment Protocol and Preliminary Results

> **Current protocol (v0.6): `backend/scripts/compare_models.py`.** Section 1 below still applies for data, split, fitting, labels and
> fault injection, with these changes:
> * **Models.** IT2FIS, PSO-tuned IT2FIS, SOM and ANFIS, plus the threshold rule as a reference. Each runs on raw inputs and on the
>   reliability-fused inputs (reliability layer on/off).
> * **Validation block.** The training split has a fit block (first 80 %) and a validation block (last 20 %). SOM grid, ANFIS ridge and the
>   PSO genes are chosen on the validation block. Fit rows whose horizon target lies in the validation block are purged.
> * **Labels.** 5 levels (Normal / Low / Moderate / High / Critical) from train-split μ + 0.5 / 1 / 2 / 3 σ; alert = High or Critical.
> * **Horizons.** 30 min, 1 h and 3 h (Bengalas); 1 day (Malwathu). Unsupported horizons are reported, not approximated.
> * **Scenarios.** clean + missing_10, missing_30, noise_low / med / high, stuck, conflict, spikes, combined; 3 fault seeds
>   (20261025, 20262025, 20263025). Clean data is evaluated once (it does not depend on the seed).
> * **Metrics.** Alert precision / recall / F1 / FPR, missed alerts, false alarms; 5-level macro F1, accuracy, Cohen's κ, severity-weighted
>   error; 95 % moving-block-bootstrap interval of clean alert F1 (1,000 resamples, block 1 day / 14 days); fit and inference time.
> * **Model selection rule for claims.** A model is called better only when its F1 interval does not overlap the other's.
>
> Results: `results/MODEL_COMPARISON_<dataset>.md` and `results/comparison_<dataset>.json`. Sections 1–4 below describe the original v0.4
> four-system experiment (`run_experiment.py`), kept for history.

## 1. Protocol

**Data.** The processed copy of the original data is used; its SHA-256 is recorded in every `results.json`.

**Split.** Chronological. The first 70% of rows are the train split and the last 30% the test split. Nothing is shuffled.

* Bengalas: train 2019-01-04 → 2022-07-10 (50,407 rows); test 2022-07-10 → 2023-12-31 (21,604 rows).
* Malwathu: train days 0–3334; test days 3335–4764.

**Fitting.** Every data-derived parameter is estimated on the **clean train split only**:

* thresholds;
* MF anchors and FOU widths;
* the radar bias factor;
* reliability-check thresholds;
* the F normalisation.

A unit test (`test_calibration_uses_train_only`) checks this: tampering with the test data leaves the calibration unchanged.

**Labels.** Proxy labels are computed from the **original** test data: the class of the observed level 1 h ahead (Bengalas) or 1 day ahead (Malwathu). See `DATASET_GUIDE.md`.

**Fault injection.** Faults are applied to a **copy** of the test split (`floodguard/faults.py`). Three stores are kept separate: the original data, the corrupted input (model only), and the injection ground truth (evaluation only).

| Scenario | Definition |
|---|---|
| clean | no corruption |
| noise_low / med / high | additive Gaussian noise on all sensors: level sd = 0.10 / 0.25 / 0.50 × train level sd; rain sd = same factor × rain scale |
| missing_10 / missing_30 | 10% / 30% of every sensor's readings missing at random |
| spikes | 2% of level readings ± U(4,8)·σ; 1% of each rain sensor + U(2,5)·rain scale |
| stuck | level and first rain sensor frozen in random segments (~10% of steps; 12 h for Bengalas, 5 days for Malwathu) |
| conflict | first rain sensor reads 0 in ~15% of segments ("clogged") and shows spurious rain at 1% of dry steps |
| reduced_reliability | level device-health metadata drops to 0.4 in ~20% of segments, plus noise of 0.25σ there (simulated metadata) |
| combined | noise_med + missing_10 + spikes + stuck |

**Systems.** All four see identical inputs:

1. Threshold on the current level.
2. Type-1 FIS.
3. IT2FIS with crisp inputs.
4. Reliability assessment, then fusion, then IT2FIS (proposed).

Systems 1–3 use conventional naive handling: missing values are forward-filled and rain is a plain mean.

**Metrics.**

* Binary alert (class ≥ Warning): precision, recall, F1, FPR, FNR.
* 4-class: confusion matrix, accuracy, macro-F1, and a severity-weighted error. The error costs 2 per class step for under-warning and 1 per step for over-warning; these costs are a provisional choice.
* Fault detection: reliability < 0.5 compared with the injection mask.
* Uncertainty response: mean U on faulted vs unfaulted steps, and the status distribution.
* Processing time: ms per observation, vectorised batch on a laptop CPU.

**Reproducibility.**

* Seeds are recorded: `DEFAULT_SEED = 20261025`, and scenario *i* uses seed + *i*.
* The dataset checksums, the full calibration, the rule base, the configuration, and the model version are written to `results.json`.
* Commands:

```bash
cd backend
python scripts/run_experiment.py bengalas            # ~25 s
python scripts/run_experiment.py malwathu            # ~1 s
python scripts/seed_sweep.py bengalas --seeds 5      # 5 fault placements
python scripts/results_tables.py <run_id> --sweep bengalas   # Markdown tables below
```

## 2. Design history (disclosed)

**v0.3 (negative result).** The risk class was taken from the KM midpoint of the **interval** inputs. Because risk is bounded below, widening an input interval mostly pushes [yl, yr] upwards, so input uncertainty was silently converted into higher risk. Measured on Bengalas: clean F1 0.864 (FPR 0.0073); missing_30 F1 **0.281** (FPR **0.150**); spikes F1 0.663. This is exactly the risk/uncertainty conflation the project must avoid. Run kept as `results/runs/20261025-232540-bengalas-full-s20261025-v0.3-midpoint-decision/`; tables are in `results/TABLES_bengalas_v0.3_negative.md`.

**v0.4 (current).** The risk class comes from the IT2FIS applied to reliability-weighted fused **point** estimates. The interval inputs are propagated separately to give a *plausible risk range* [yl, yr], which feeds only the uncertainty index. Two further changes were made: the persistence prior became a causal median of trustworthy readings, and the missing-level growth scale became the q95 step.

**Caveat:** this revision was made **after** seeing v0.3 test-split results. The v0.4 numbers are therefore not a fully blind evaluation. A future protocol should make design decisions on a validation block carved from the training period.

**v0.6 (disclosed).** Three v0.6 design changes were made during development, after some v0.6 test numbers had been printed:
* **FOU cap (ε ≤ δ/2).** Triggered by a monotonicity check of the fitted IT2FIS (a higher level gave a lower alert), not by a score.
  Malwathu test results of the uncapped version had been seen at that point.
* **ANFIS premise constraints.** Triggered by meaningless rule names in the ANFIS explanations (fuzzy sets out of order). The change
  *lowered* ANFIS's Malwathu test F1 (0.83 → 0.66) and was kept anyway, because interpretability was the reason for the change.
* **Trapezoid rounding fix.** A crash, not a modelling choice.
No hyper-parameter was chosen by test score. The final Bengalas comparison was run once, after all three changes.

## 3. Results (v0.4, measured)

The tables below are generated by `scripts/results_tables.py` from the saved JSON. "SWE" is the severity-weighted error. Labels are proxies.

### 3.1 Bengalas (primary, 30-min)

Run `20261025-232735-bengalas-full-s20261025` - dataset bengalas, mode full, seed 20261025, model v0.4.0, train 50407 rows (2019-01-04 14:30:00 -> 2022-07-10 15:00:00), test 21604 rows (2022-07-10 15:30:00 -> 2023-12-31 16:00:00), proxy-label test counts {'Normal': 18516, 'Watch': 1735, 'Warning': 346, 'Critical': 307}.

| Scenario | threshold P / R / F1 / FPR | t1_fis P / R / F1 / FPR | it2_fis P / R / F1 / FPR | proposed P / R / F1 / FPR |
|---|---|---|---|---|
| clean | 0.858 / 0.867 / **0.862** / 0.0046 | 0.863 / 0.905 / **0.883** / 0.0046 | 0.876 / 0.916 / **0.895** / 0.0042 | 0.879 / 0.897 / **0.888** / 0.0040 |
| noise_low | 0.861 / 0.864 / **0.862** / 0.0045 | 0.860 / 0.862 / **0.861** / 0.0045 | 0.860 / 0.865 / **0.863** / 0.0045 | 0.855 / 0.847 / **0.851** / 0.0046 |
| noise_med | 0.793 / 0.832 / **0.812** / 0.0070 | 0.771 / 0.832 / **0.800** / 0.0080 | 0.779 / 0.838 / **0.807** / 0.0077 | 0.785 / 0.799 / **0.792** / 0.0071 |
| noise_high | 0.649 / 0.815 / **0.722** / 0.0142 | 0.527 / 0.822 / **0.642** / 0.0238 | 0.534 / 0.827 / **0.649** / 0.0233 | 0.653 / 0.640 / **0.647** / 0.0110 |
| missing_10 | 0.849 / 0.851 / **0.850** / 0.0049 | 0.855 / 0.888 / **0.872** / 0.0048 | 0.866 / 0.894 / **0.880** / 0.0044 | 0.865 / 0.876 / **0.871** / 0.0044 |
| missing_30 | 0.836 / 0.844 / **0.840** / 0.0053 | 0.839 / 0.848 / **0.844** / 0.0052 | 0.840 / 0.858 / **0.848** / 0.0053 | 0.820 / 0.812 / **0.816** / 0.0057 |
| spikes | 0.648 / 0.861 / **0.739** / 0.0151 | 0.648 / 0.900 / **0.753** / 0.0158 | 0.657 / 0.910 / **0.763** / 0.0153 | 0.800 / 0.858 / **0.828** / 0.0069 |
| stuck | 0.787 / 0.830 / **0.808** / 0.0073 | 0.793 / 0.864 / **0.827** / 0.0073 | 0.806 / 0.874 / **0.839** / 0.0068 | 0.808 / 0.855 / **0.830** / 0.0066 |
| conflict | 0.858 / 0.867 / **0.862** / 0.0046 | 0.865 / 0.902 / **0.883** / 0.0045 | 0.878 / 0.913 / **0.895** / 0.0041 | 0.881 / 0.894 / **0.888** / 0.0039 |
| reduced_reliability | 0.837 / 0.862 / **0.849** / 0.0054 | 0.841 / 0.890 / **0.865** / 0.0054 | 0.852 / 0.899 / **0.875** / 0.0050 | 0.838 / 0.853 / **0.845** / 0.0053 |
| combined | 0.594 / 0.721 / **0.651** / 0.0159 | 0.569 / 0.724 / **0.637** / 0.0177 | 0.575 / 0.727 / **0.642** / 0.0173 | 0.621 / 0.658 / **0.639** / 0.0129 |

| Scenario | SWE thr / T1 / IT2 / prop | 4-class acc (prop) | U faulted | U unfaulted | Confident/Uncertain/Insufficient | Level fault det. P / R |
|---|---|---|---|---|---|---|
| clean | 0.042 / 0.038 / 0.039 / 0.042 | 0.976 | n/a | 4.2 | 20906/698/0 | false-flag 0.0025 |
| noise_low | 0.052 / 0.061 / 0.064 / 0.067 | 0.961 | 5.1 | n/a | 20771/833/0 | 1.000 / 0.003 |
| noise_med | 0.090 / 0.102 / 0.103 / 0.103 | 0.938 | 11.5 | n/a | 18064/3540/0 | 1.000 / 0.068 |
| noise_high | 0.176 / 0.194 / 0.192 / 0.142 | 0.912 | 26.4 | n/a | 9994/11610/0 | 1.000 / 0.610 |
| missing_10 | 0.044 / 0.041 / 0.042 / 0.046 | 0.973 | 35.7 | 7.0 | 13873/7694/37 | 0.984 / 1.000 |
| missing_30 | 0.051 / 0.050 / 0.052 / 0.060 | 0.967 | 47.7 | 14.7 | 5000/15508/1096 | 0.996 / 1.000 |
| spikes | 0.074 / 0.080 / 0.081 / 0.060 | 0.963 | 30.4 | 7.4 | 19205/2399/0 | 0.135 / 0.934 |
| stuck | 0.055 / 0.051 / 0.051 / 0.055 | 0.969 | 4.2 | 4.2 | 20846/758/0 | 0.312 / 0.010 |
| conflict | 0.042 / 0.038 / 0.039 / 0.042 | 0.976 | 9.9 | 4.2 | 20889/715/0 | 0.000 / n/a |
| reduced_reliability | 0.051 / 0.054 / 0.055 / 0.056 | 0.967 | 26.2 | 4.3 | 19198/2406/0 | 0.983 / 1.000 |
| combined | 0.145 / 0.164 / 0.161 / 0.146 | 0.917 | 22.3 | n/a | 11420/9875/309 | 1.000 / 0.236 |

Processing time (ms/observation, clean, vectorised batch): threshold 0.0001, t1_fis 0.0012, it2_fis 0.0112, proposed 0.0234

Seed sweep (5 fault-placement seeds), alert F1 mean +/- sd:

| Scenario | threshold | t1_fis | it2_fis | proposed |
|---|---|---|---|---|
| clean | 0.862 +/- 0.000 | 0.883 +/- 0.000 | **0.895 +/- 0.000** | 0.888 +/- 0.000 |
| noise_low | 0.859 +/- 0.003 | 0.862 +/- 0.004 | **0.866 +/- 0.005** | 0.855 +/- 0.006 |
| noise_med | **0.819 +/- 0.006** | 0.804 +/- 0.007 | 0.808 +/- 0.008 | 0.796 +/- 0.005 |
| noise_high | **0.721 +/- 0.005** | 0.642 +/- 0.009 | 0.645 +/- 0.008 | 0.636 +/- 0.019 |
| missing_10 | 0.855 +/- 0.003 | 0.871 +/- 0.003 | **0.883 +/- 0.002** | 0.869 +/- 0.006 |
| missing_30 | 0.836 +/- 0.007 | 0.839 +/- 0.005 | **0.845 +/- 0.006** | 0.825 +/- 0.008 |
| spikes | 0.748 +/- 0.015 | 0.760 +/- 0.012 | 0.768 +/- 0.014 | **0.835 +/- 0.011** |
| stuck | 0.818 +/- 0.008 | 0.837 +/- 0.008 | **0.848 +/- 0.009** | 0.836 +/- 0.009 |
| conflict | 0.862 +/- 0.000 | 0.884 +/- 0.001 | **0.895 +/- 0.001** | 0.888 +/- 0.001 |
| reduced_reliability | 0.851 +/- 0.002 | 0.865 +/- 0.005 | **0.875 +/- 0.004** | 0.818 +/- 0.022 |
| combined | 0.662 +/- 0.022 | 0.652 +/- 0.021 | 0.654 +/- 0.019 | **0.667 +/- 0.036** |

### 3.2 Malwathu (secondary, daily) - generalisation check

Run `20261025-232801-malwathu-full-s20261025` - dataset malwathu, mode full, seed 20261025, model v0.4.0, train 3335 rows (index 0 -> index 3334), test 1430 rows (index 3335 -> index 4764), proxy-label test counts {'Normal': 1325, 'Watch': 38, 'Warning': 12, 'Critical': 54}.

| Scenario | threshold P / R / F1 / FPR | t1_fis P / R / F1 / FPR | it2_fis P / R / F1 / FPR | proposed P / R / F1 / FPR |
|---|---|---|---|---|
| clean | 0.848 / 0.848 / **0.848** / 0.0073 | 0.862 / 0.848 / **0.855** / 0.0066 | 0.864 / 0.864 / **0.864** / 0.0066 | 0.851 / 0.864 / **0.857** / 0.0073 |
| noise_low | 0.838 / 0.864 / **0.851** / 0.0081 | 0.851 / 0.864 / **0.857** / 0.0073 | 0.866 / 0.879 / **0.872** / 0.0066 | 0.853 / 0.879 / **0.866** / 0.0073 |
| noise_med | 0.843 / 0.894 / **0.868** / 0.0081 | 0.868 / 0.894 / **0.881** / 0.0066 | 0.870 / 0.909 / **0.889** / 0.0066 | 0.857 / 0.909 / **0.882** / 0.0073 |
| noise_high | 0.857 / 0.909 / **0.882** / 0.0073 | 0.882 / 0.909 / **0.896** / 0.0059 | 0.896 / 0.909 / **0.902** / 0.0051 | 0.882 / 0.909 / **0.896** / 0.0059 |
| missing_10 | 0.812 / 0.848 / **0.830** / 0.0095 | 0.824 / 0.848 / **0.836** / 0.0088 | 0.826 / 0.864 / **0.844** / 0.0088 | 0.831 / 0.818 / **0.824** / 0.0081 |
| missing_30 | 0.779 / 0.803 / **0.791** / 0.0110 | 0.779 / 0.803 / **0.791** / 0.0110 | 0.779 / 0.803 / **0.791** / 0.0110 | 0.746 / 0.803 / **0.774** / 0.0132 |
| spikes | 0.764 / 0.833 / **0.797** / 0.0125 | 0.786 / 0.833 / **0.809** / 0.0110 | 0.789 / 0.848 / **0.818** / 0.0110 | 0.743 / 0.833 / **0.786** / 0.0139 |
| stuck | 0.800 / 0.848 / **0.824** / 0.0103 | 0.812 / 0.848 / **0.830** / 0.0095 | 0.814 / 0.864 / **0.838** / 0.0095 | 0.792 / 0.864 / **0.826** / 0.0110 |
| conflict | 0.848 / 0.848 / **0.848** / 0.0073 | 0.862 / 0.848 / **0.855** / 0.0066 | 0.864 / 0.864 / **0.864** / 0.0066 | 0.851 / 0.864 / **0.857** / 0.0073 |
| reduced_reliability | 0.848 / 0.848 / **0.848** / 0.0073 | 0.862 / 0.848 / **0.855** / 0.0066 | 0.864 / 0.864 / **0.864** / 0.0066 | 0.851 / 0.864 / **0.857** / 0.0073 |
| combined | 0.674 / 0.879 / **0.763** / 0.0205 | 0.699 / 0.879 / **0.779** / 0.0183 | 0.694 / 0.894 / **0.781** / 0.0191 | 0.652 / 0.879 / **0.748** / 0.0227 |

| Scenario | SWE thr / T1 / IT2 / prop | 4-class acc (prop) | U faulted | U unfaulted | Confident/Uncertain/Insufficient | Level fault det. P / R |
|---|---|---|---|---|---|---|
| clean | 0.069 / 0.071 / 0.071 / 0.071 | 0.962 | n/a | 1.9 | 1397/33/0 | false-flag 0.0000 |
| noise_low | 0.066 / 0.072 / 0.071 / 0.071 | 0.961 | 2.0 | n/a | 1398/32/0 | n/a / 0.000 |
| noise_med | 0.064 / 0.064 / 0.062 / 0.062 | 0.965 | 2.2 | n/a | 1396/34/0 | n/a / 0.000 |
| noise_high | 0.087 / 0.083 / 0.078 / 0.082 | 0.948 | 3.8 | n/a | 1374/56/0 | 1.000 / 0.003 |
| missing_10 | 0.072 / 0.074 / 0.074 / 0.085 | 0.958 | 19.9 | 4.5 | 618/812/0 | 0.993 / 1.000 |
| missing_30 | 0.085 / 0.084 / 0.087 / 0.095 | 0.955 | 33.9 | 7.6 | 79/1312/39 | 0.998 / 1.000 |
| spikes | 0.088 / 0.088 / 0.088 / 0.094 | 0.955 | 5.7 | 2.6 | 1381/49/0 | 0.077 / 0.036 |
| stuck | 0.076 / 0.077 / 0.077 / 0.083 | 0.957 | 1.5 | 2.2 | 1394/36/0 | 0.600 / 0.022 |
| conflict | 0.069 / 0.071 / 0.071 / 0.071 | 0.962 | 5.3 | 1.8 | 1397/33/0 | n/a / n/a |
| reduced_reliability | 0.069 / 0.071 / 0.071 / 0.074 | 0.958 | 19.2 | 1.8 | 1361/69/0 | 1.000 / 1.000 |
| combined | 0.099 / 0.099 / 0.095 / 0.111 | 0.945 | 13.7 | n/a | 620/808/2 | 1.000 / 0.126 |

Processing time (ms/observation, clean, vectorised batch): threshold 0.0010, t1_fis 0.0018, it2_fis 0.0088, proposed 0.0212

Seed sweep (5 fault-placement seeds), alert F1 mean +/- sd:

| Scenario | threshold | t1_fis | it2_fis | proposed |
|---|---|---|---|---|
| clean | 0.848 +/- 0.000 | 0.855 +/- 0.000 | **0.864 +/- 0.000** | 0.857 +/- 0.000 |
| noise_low | 0.852 +/- 0.009 | 0.865 +/- 0.013 | **0.871 +/- 0.005** | 0.865 +/- 0.005 |
| noise_med | 0.853 +/- 0.012 | 0.864 +/- 0.014 | **0.869 +/- 0.015** | 0.864 +/- 0.013 |
| noise_high | 0.848 +/- 0.021 | 0.849 +/- 0.028 | **0.853 +/- 0.028** | 0.847 +/- 0.029 |
| missing_10 | 0.833 +/- 0.010 | 0.842 +/- 0.008 | **0.849 +/- 0.011** | 0.836 +/- 0.008 |
| missing_30 | 0.795 +/- 0.016 | 0.797 +/- 0.014 | **0.804 +/- 0.019** | 0.782 +/- 0.017 |
| spikes | 0.773 +/- 0.026 | 0.780 +/- 0.027 | **0.788 +/- 0.028** | 0.766 +/- 0.033 |
| stuck | 0.821 +/- 0.027 | 0.827 +/- 0.027 | **0.836 +/- 0.027** | 0.828 +/- 0.029 |
| conflict | 0.848 +/- 0.000 | 0.855 +/- 0.000 | **0.865 +/- 0.004** | 0.859 +/- 0.004 |
| reduced_reliability | 0.853 +/- 0.007 | 0.859 +/- 0.011 | **0.865 +/- 0.005** | 0.835 +/- 0.020 |
| combined | 0.743 +/- 0.022 | 0.749 +/- 0.023 | **0.757 +/- 0.026** | 0.752 +/- 0.019 |

## 4. Interpretation (preliminary, honest)

**What the evidence supports**

* **Spikes (Bengalas).** Reliability-aware fusion clearly improves alerting under spike faults. Alert F1 is 0.835 ± 0.011 against 0.768 ± 0.014 for the best baseline, over 5 seeds. FPR roughly halves (0.0069 vs 0.0151). The spike detector reaches 93% recall.
* **Uncertainty status reacts to data problems that risk alone would hide.** On Bengalas, mean U on faulted vs unfaulted steps is:

  | Scenario | U faulted | U unfaulted |
  |---|---|---|
  | missing_30 | 47.7 | 14.7 |
  | spikes | 30.4 | 7.4 |
  | reduced_reliability | 26.2 | 4.3 |
  | conflict | 9.9 | 4.2 |

  Over 21,604 test steps × 11 scenarios × 2 datasets, a missing level never produced a "Confident" status.
* **IT2 vs Type-1 on the same rules.** IT2FIS with crisp inputs scores slightly but consistently higher F1 than the Type-1 FIS on the clean data of both datasets: 0.895 vs 0.883 on Bengalas and 0.864 vs 0.855 on Malwathu. It is also higher in most fault scenarios.
* **Rule-based fuzzy systems with a rise term vs a level-only threshold.** On clean Bengalas data both FIS variants beat the threshold baseline (F1 0.883–0.895 vs 0.862), at the 1-hour proxy horizon.

**What the evidence does NOT support (yet)**

* **No overall robustness advantage.** The proposed system is **below** the crisp IT2FIS on clean data (0.888 vs 0.895) and under noise, missing data, stuck readings and reduced reliability. Its recall drops because the fused estimate lags genuine rises; for example, noise_high recall is 0.640 vs 0.827. Under heavy noise the plain threshold is best, because every FIS uses the noise-sensitive rise rate.
* **Reduced-reliability metadata hurts** (0.818 vs 0.875). Down-weighting a still-informative level sensor towards a persistence prior delays warnings.
* **Stuck sensors are essentially undetected:** level-flag recall is 1%.
* **The clogged rain gauge cannot be attributed with only two rain sources.** It is carried only as unresolved conflict, by design.
* **No benefit on the daily Malwathu data.** The proposed system is at or slightly below the crisp IT2FIS in every scenario. Daily level jumps are large, so the calibrated spike threshold (3.58 m) exceeds most injected spikes. The positives are few (66), so the confidence intervals are wide.
* **Spike-detector precision is low (0.135).** The roughness window keeps flagging the steps after a spike.

**Next steps suggested by these results**

* Make the persistence prior trend-aware, e.g. a reliability-weighted Kalman-style level estimate, to reduce lag.
* Separate spike handling from noise handling.
* Add a stuck detector that compares the level with rain-driven expectations.
* Use a validation block for design decisions.
* Report alert lead time, not just per-step classification.
