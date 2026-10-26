# Model comparison: malwathu

3 fault-placement seeds; train 3335 rows, test 1430 rows; proxy labels. Ranked by mean alert F1 over the 10 faulted scenarios. Family: `original` = the 4 existing systems, `new` = added soft-computing model on naive inputs, `new+fusion` = same model on the reliability-weighted fused inputs.

Model selection on the validation block (last 20 % of train; fitness = alert F1 + 0.25 x 4-class macro F1): anfis selected 100.0 {'0.01': 0.488, '1.0': 0.607, '10.0': 0.938, '100.0': 0.944}; fcm_tsk selected 16 {'8': 0.686, '16': 0.797, '24': 0.47, '32': 0.4}; ga_it2 0.953 -> 1.003 (tuned); pso_it2 0.953 -> 1.003 (tuned); ann 0.911 using level, rise, rain; som_ann 0.917 using level, rise, rain; ga_som_ann 1.032 using level, rain, level_lag2, level_lag4, level_max8, rain_lag1

| Rank | System | Family | Clean F1 | Faulted mean F1 | Worst-case F1 | Mean SWE | 4-class macro F1 |
|---|---|---|---|---|---|---|---|
| 1 | som_ann | new | 0.909 | **0.866** | 0.774 | 0.087 | 0.580 |
| 2 | som_ann+fusion | new+fusion | 0.902 | **0.850** | 0.763 | 0.090 | 0.578 |
| 3 | pso_it2 | new | 0.866 | **0.838** | 0.758 | 0.082 | 0.571 |
| 4 | it2_fis | original | 0.864 | **0.838** | 0.766 | 0.080 | 0.607 |
| 5 | ga_it2 | new | 0.859 | **0.835** | 0.761 | 0.081 | 0.629 |
| 6 | ann | new | 0.857 | **0.834** | 0.754 | 0.102 | 0.556 |
| 7 | t1_fis | original | 0.855 | **0.832** | 0.759 | 0.080 | 0.615 |
| 8 | threshold | original | 0.848 | **0.824** | 0.754 | 0.079 | 0.622 |
| 9 | proposed | original | 0.857 | **0.823** | 0.749 | 0.083 | 0.603 |
| 10 | pso_it2+fusion | new+fusion | 0.859 | **0.822** | 0.740 | 0.085 | 0.569 |
| 11 | ga_it2+fusion | new+fusion | 0.853 | **0.819** | 0.745 | 0.084 | 0.622 |
| 12 | ann+fusion | new+fusion | 0.851 | **0.818** | 0.733 | 0.104 | 0.554 |
| 13 | ga_som_ann | new | 0.842 | **0.815** | 0.754 | 0.086 | 0.582 |
| 14 | anfis | new | 0.843 | **0.811** | 0.766 | 0.122 | 0.526 |
| 15 | ga_som_ann+fusion | new+fusion | 0.836 | **0.802** | 0.746 | 0.088 | 0.579 |
| 16 | anfis+fusion | new+fusion | 0.831 | **0.797** | 0.748 | 0.126 | 0.512 |
| 17 | fcm_tsk+fusion | new+fusion | 0.767 | **0.711** | 0.614 | 0.230 | 0.472 |
| 18 | fcm_tsk | new | 0.761 | **0.700** | 0.590 | 0.243 | 0.462 |

Alert F1 per scenario (mean over seeds):

| System | clean | noise_low | noise_med | noise_high | missing_10 | missing_30 | spikes | stuck | conflict | reduced_reliability | combined |
|---|---|---|---|---|---|---|---|---|---|---|---|
| som_ann | **0.909** | **0.909** | 0.860 | 0.840 | **0.893** | **0.849** | **0.825** | **0.889** | **0.909** | **0.909** | **0.774** |
| som_ann+fusion | 0.902 | 0.902 | 0.857 | 0.836 | 0.866 | 0.814 | 0.807 | 0.881 | 0.902 | 0.873 | 0.763 |
| pso_it2 | 0.866 | 0.867 | 0.862 | 0.870 | 0.857 | 0.793 | 0.785 | 0.849 | 0.868 | 0.871 | 0.758 |
| it2_fis | 0.864 | 0.874 | 0.867 | 0.862 | 0.853 | 0.793 | 0.786 | 0.844 | 0.864 | 0.868 | 0.766 |
| ga_it2 | 0.859 | 0.864 | 0.862 | **0.871** | 0.853 | 0.797 | 0.778 | 0.842 | 0.862 | 0.856 | 0.761 |
| ann | 0.857 | 0.854 | **0.876** | 0.834 | 0.871 | 0.815 | 0.774 | 0.841 | 0.857 | 0.864 | 0.754 |
| t1_fis | 0.855 | 0.866 | 0.861 | 0.862 | 0.844 | 0.792 | 0.777 | 0.835 | 0.855 | 0.864 | 0.759 |
| threshold | 0.848 | 0.855 | 0.851 | 0.855 | 0.838 | 0.788 | 0.770 | 0.829 | 0.848 | 0.856 | 0.754 |
| proposed | 0.857 | 0.867 | 0.863 | 0.858 | 0.836 | 0.771 | 0.764 | 0.836 | 0.857 | 0.832 | 0.749 |
| pso_it2+fusion | 0.859 | 0.862 | 0.858 | 0.863 | 0.830 | 0.764 | 0.767 | 0.840 | 0.862 | 0.837 | 0.740 |
| ga_it2+fusion | 0.853 | 0.857 | 0.858 | 0.865 | 0.824 | 0.769 | 0.761 | 0.834 | 0.855 | 0.827 | 0.745 |
| ann+fusion | 0.851 | 0.848 | 0.870 | 0.835 | 0.839 | 0.782 | 0.761 | 0.833 | 0.851 | 0.829 | 0.733 |
| ga_som_ann | 0.842 | 0.842 | 0.846 | 0.816 | 0.834 | 0.779 | 0.761 | 0.828 | 0.844 | 0.841 | 0.754 |
| anfis | 0.843 | 0.841 | 0.829 | 0.778 | 0.825 | 0.783 | 0.793 | 0.819 | 0.843 | 0.836 | 0.766 |
| ga_som_ann+fusion | 0.836 | 0.842 | 0.849 | 0.821 | 0.810 | 0.749 | 0.749 | 0.815 | 0.838 | 0.801 | 0.746 |
| anfis+fusion | 0.831 | 0.827 | 0.820 | 0.775 | 0.800 | 0.769 | 0.772 | 0.810 | 0.831 | 0.818 | 0.748 |
| fcm_tsk+fusion | 0.767 | 0.739 | 0.725 | 0.614 | 0.745 | 0.715 | 0.684 | 0.742 | 0.763 | 0.754 | 0.628 |
| fcm_tsk | 0.761 | 0.730 | 0.711 | 0.590 | 0.731 | 0.697 | 0.682 | 0.733 | 0.756 | 0.737 | 0.630 |

## Best model: som_ann

Mean over seeds; faulted = mean of the 10 faulted scenarios.

| Metric | Clean | Faulted mean |
|---|---|---|
| Alert precision | 0.9091 | 0.8393 |
| Alert recall | 0.9091 | 0.8975 |
| Alert F1 | 0.9091 | 0.8658 |
| False-positive rate | 0.0044 | 0.0087 |
| 4-class macro F1 | 0.6062 | 0.5779 |
| Severity-weighted error | 0.0756 | 0.0883 |

Confusion matrix on the clean test split (rows = proxy label, columns = prediction):

| | Normal | Watch | Warning | Critical |
|---|---|---|---|---|
| Normal | 1293 | 32 | 0 | 0 |
| Watch | 5 | 27 | 5 | 1 |
| Warning | 0 | 3 | 4 | 5 |
| Critical | 0 | 3 | 18 | 33 |

Sample output around the largest test-period flood peak (full output in `results/best_model_output_malwathu.csv`):

| Time | Level | Rain | Proxy label | som_ann | proposed |
|---|---|---|---|---|---|
| test row 556 | 0.213 | 14.77 | Normal | Normal | Normal |
| test row 557 | 0.153 | 10.7 | Normal | Normal | Normal |
| test row 558 | 0.144 | 0.0 | Normal | Normal | Normal |
| test row 559 | 0.11 | 17.73 | Watch | Normal | Normal |
| test row 560 | 0.839 | 8.72 | Normal | Watch | Watch |
| test row 561 | 0.411 | 152.47 | Critical | Warning | Watch |
| test row 562 | 2.93 | 131.0 | Critical | Critical | Critical |
| test row 563 | 4.86 | 25.37 | Critical | Critical | Critical |
| test row 564 | 5.787 | 0.08 | Critical | Critical | Critical |
| test row 565 | 4.944 | 1.18 | Critical | Warning | Critical |
| test row 566 | 4.477 | 0.03 | Critical | Warning | Critical |
| test row 567 | 2.296 | 0.0 | Watch | Warning | Critical |
