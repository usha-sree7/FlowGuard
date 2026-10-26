# Model comparison: bengalas

3 fault-placement seeds; train 50407 rows, test 21604 rows; proxy labels. Ranked by mean alert F1 over the 10 faulted scenarios. Family: `original` = the 4 existing systems, `new` = added soft-computing model on naive inputs, `new+fusion` = same model on the reliability-weighted fused inputs.

Model selection on the validation block (last 20 % of train; fitness = alert F1 + 0.25 x 4-class macro F1): anfis selected 10.0 {'0.01': 1.151, '1.0': 1.15, '10.0': 1.152, '100.0': 1.137}; fcm_tsk selected 32 {'8': 0.912, '16': 1.025, '24': 1.009, '32': 1.044}; ga_it2 1.146 -> 1.165 (tuned); pso_it2 1.146 -> 1.166 (tuned); ann 1.158 using level, rise, rain; som_ann 1.165 using level, rise, rain; ga_som_ann 1.156 using level, rise, level_lag4, level_max8, rain_lag1, rain_mean4

| Rank | System | Family | Clean F1 | Faulted mean F1 | Worst-case F1 | Mean SWE | 4-class macro F1 |
|---|---|---|---|---|---|---|---|
| 1 | pso_it2 | new | 0.912 | **0.819** | 0.642 | 0.080 | 0.787 |
| 2 | ga_it2 | new | 0.910 | **0.817** | 0.638 | 0.080 | 0.785 |
| 3 | pso_it2+fusion | new+fusion | 0.904 | **0.808** | 0.634 | 0.075 | 0.779 |
| 4 | it2_fis | original | 0.895 | **0.808** | 0.645 | 0.079 | 0.777 |
| 5 | ga_it2+fusion | new+fusion | 0.901 | **0.807** | 0.632 | 0.075 | 0.777 |
| 6 | threshold | original | 0.862 | **0.802** | 0.654 | 0.074 | 0.777 |
| 7 | t1_fis | original | 0.883 | **0.801** | 0.643 | 0.079 | 0.771 |
| 8 | proposed | original | 0.888 | **0.800** | 0.628 | 0.074 | 0.769 |
| 9 | anfis+fusion | new+fusion | 0.857 | **0.744** | 0.494 | 0.124 | 0.715 |
| 10 | anfis | new | 0.877 | **0.741** | 0.305 | 0.146 | 0.723 |
| 11 | som_ann+fusion | new+fusion | 0.872 | **0.733** | 0.449 | 0.137 | 0.701 |
| 12 | ann+fusion | new+fusion | 0.847 | **0.731** | 0.452 | 0.129 | 0.706 |
| 13 | som_ann | new | 0.885 | **0.707** | 0.241 | 0.168 | 0.699 |
| 14 | ann | new | 0.856 | **0.706** | 0.242 | 0.157 | 0.706 |
| 15 | ga_som_ann+fusion | new+fusion | 0.825 | **0.680** | 0.360 | 0.154 | 0.678 |
| 16 | ga_som_ann | new | 0.840 | **0.665** | 0.216 | 0.180 | 0.678 |
| 17 | fcm_tsk+fusion | new+fusion | 0.693 | **0.481** | 0.139 | 0.559 | 0.463 |
| 18 | fcm_tsk | new | 0.696 | **0.448** | 0.089 | 0.606 | 0.455 |

Alert F1 per scenario (mean over seeds):

| System | clean | noise_low | noise_med | noise_high | missing_10 | missing_30 | spikes | stuck | conflict | reduced_reliability | combined |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pso_it2 | **0.912** | **0.867** | 0.809 | 0.678 | **0.897** | **0.852** | 0.776 | **0.866** | **0.912** | **0.888** | 0.642 |
| ga_it2 | 0.910 | 0.865 | 0.808 | 0.677 | 0.897 | 0.851 | 0.774 | 0.865 | 0.910 | 0.886 | 0.638 |
| pso_it2+fusion | 0.904 | 0.854 | 0.793 | 0.634 | 0.888 | 0.833 | **0.846** | 0.853 | 0.903 | 0.831 | 0.649 |
| it2_fis | 0.895 | 0.865 | 0.811 | 0.649 | 0.882 | 0.844 | 0.767 | 0.848 | 0.895 | 0.875 | 0.645 |
| ga_it2+fusion | 0.901 | 0.852 | 0.791 | 0.632 | 0.887 | 0.834 | 0.845 | 0.852 | 0.901 | 0.828 | 0.647 |
| threshold | 0.862 | 0.860 | **0.820** | **0.722** | 0.855 | 0.836 | 0.746 | 0.817 | 0.862 | 0.850 | **0.654** |
| t1_fis | 0.883 | 0.861 | 0.806 | 0.648 | 0.872 | 0.839 | 0.758 | 0.836 | 0.884 | 0.864 | 0.643 |
| proposed | 0.888 | 0.853 | 0.797 | 0.628 | 0.873 | 0.823 | 0.832 | 0.836 | 0.888 | 0.820 | 0.647 |
| anfis+fusion | 0.857 | 0.820 | 0.657 | 0.494 | 0.850 | 0.817 | 0.779 | 0.820 | 0.858 | 0.801 | 0.545 |
| anfis | 0.877 | 0.841 | 0.620 | 0.305 | 0.864 | 0.838 | 0.835 | 0.845 | 0.878 | 0.821 | 0.562 |
| som_ann+fusion | 0.872 | 0.784 | 0.624 | 0.449 | 0.858 | 0.818 | 0.773 | 0.827 | 0.874 | 0.805 | 0.522 |
| ann+fusion | 0.847 | 0.793 | 0.651 | 0.452 | 0.838 | 0.808 | 0.768 | 0.809 | 0.851 | 0.791 | 0.546 |
| som_ann | 0.885 | 0.801 | 0.569 | 0.241 | 0.871 | 0.826 | 0.724 | 0.843 | 0.887 | 0.821 | 0.486 |
| ann | 0.856 | 0.805 | 0.606 | 0.242 | 0.846 | 0.814 | 0.731 | 0.821 | 0.860 | 0.821 | 0.518 |
| ga_som_ann+fusion | 0.825 | 0.749 | 0.516 | 0.360 | 0.816 | 0.789 | 0.732 | 0.794 | 0.822 | 0.766 | 0.461 |
| ga_som_ann | 0.840 | 0.764 | 0.477 | 0.216 | 0.831 | 0.801 | 0.692 | 0.812 | 0.837 | 0.791 | 0.427 |
| fcm_tsk+fusion | 0.693 | 0.413 | 0.154 | 0.139 | 0.694 | 0.688 | 0.570 | 0.672 | 0.695 | 0.616 | 0.166 |
| fcm_tsk | 0.696 | 0.413 | 0.129 | 0.089 | 0.689 | 0.676 | 0.587 | 0.676 | 0.697 | 0.379 | 0.142 |

## Best model: pso_it2

Mean over seeds; faulted = mean of the 10 faulted scenarios.

| Metric | Clean | Faulted mean |
|---|---|---|
| Alert precision | 0.9245 | 0.8012 |
| Alert recall | 0.9005 | 0.8451 |
| Alert F1 | 0.9123 | 0.8188 |
| False-positive rate | 0.0024 | 0.0076 |
| 4-class macro F1 | 0.8774 | 0.7783 |
| Severity-weighted error | 0.0375 | 0.0839 |

Confusion matrix on the clean test split (rows = proxy label, columns = prediction):

| | Normal | Watch | Warning | Critical |
|---|---|---|---|---|
| Normal | 18422 | 91 | 3 | 0 |
| Watch | 185 | 1505 | 43 | 2 |
| Warning | 7 | 46 | 253 | 40 |
| Critical | 6 | 6 | 25 | 270 |

Sample output around the largest test-period flood peak (full output in `results/best_model_output_bengalas.csv`):

| Time | Level | Rain | Proxy label | pso_it2 | proposed |
|---|---|---|---|---|---|
| 2022-12-07 20:00:00 | 1.17 | 3.03 | Critical | Normal | Watch |
| 2022-12-07 20:30:00 | 1.4 | 6.63 | Critical | Critical | Critical |
| 2022-12-07 21:00:00 | 1.88 | 2.12 | Critical | Critical | Critical |
| 2022-12-07 21:30:00 | 2.44 | 0.61 | Critical | Critical | Critical |
| 2022-12-07 22:00:00 | 2.6 | 2.48 | Critical | Critical | Critical |
| 2022-12-07 22:30:00 | 2.77 | 3.76 | Critical | Critical | Critical |
| 2022-12-07 23:00:00 | 2.84 | 1.19 | Critical | Critical | Critical |
| 2022-12-07 23:30:00 | 2.95 | 0.73 | Critical | Critical | Critical |
| 2022-12-08 00:00:00 | 2.96 | 0.0 | Critical | Critical | Critical |
| 2022-12-08 00:30:00 | 2.87 | 0.0 | Critical | Critical | Critical |
| 2022-12-08 01:00:00 | 2.76 | 0.0 | Critical | Critical | Critical |
| 2022-12-08 01:30:00 | 2.64 | 0.0 | Critical | Critical | Critical |
