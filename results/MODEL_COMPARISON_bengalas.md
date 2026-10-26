# Model comparison: bengalas

3 fault-placement seeds; train 50407 rows, test 21604 rows; proxy labels. Ranked by mean alert F1 over the 10 faulted scenarios. Family: `original` = the 4 existing systems, `new` = added soft-computing model on naive inputs, `new+fusion` = same model on the reliability-weighted fused inputs.

Model selection on the validation block (last 20 % of train; fitness = alert F1 + 0.25 x 4-class macro F1): anfis selected 10.0 {'0.01': 1.151, '1.0': 1.15, '10.0': 1.152, '100.0': 1.137}; fcm_tsk selected 32 {'8': 0.912, '16': 1.025, '24': 1.009, '32': 1.044}; ga_it2 1.146 -> 1.165 (GA/PSO tuning); pso_it2 1.146 -> 1.166 (GA/PSO tuning)

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
| 11 | fcm_tsk+fusion | new+fusion | 0.693 | **0.481** | 0.139 | 0.559 | 0.463 |
| 12 | fcm_tsk | new | 0.696 | **0.448** | 0.089 | 0.606 | 0.455 |

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
| fcm_tsk+fusion | 0.693 | 0.413 | 0.154 | 0.139 | 0.694 | 0.688 | 0.570 | 0.672 | 0.695 | 0.616 | 0.166 |
| fcm_tsk | 0.696 | 0.413 | 0.129 | 0.089 | 0.689 | 0.676 | 0.587 | 0.676 | 0.697 | 0.379 | 0.142 |
