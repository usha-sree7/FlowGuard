# Model comparison: malwathu

3 fault-placement seeds; train 3335 rows, test 1430 rows; proxy labels. Ranked by mean alert F1 over the 10 faulted scenarios. Family: `original` = the 4 existing systems, `new` = added soft-computing model on naive inputs, `new+fusion` = same model on the reliability-weighted fused inputs.

Model selection on the validation block (last 20 % of train; fitness = alert F1 + 0.25 x 4-class macro F1): anfis selected 100.0 {'0.01': np.float64(0.488), '1.0': np.float64(0.607), '10.0': np.float64(0.938), '100.0': np.float64(0.944)}; fcm_tsk selected 16 {'8': np.float64(0.686), '16': np.float64(0.797), '24': np.float64(0.47), '32': np.float64(0.4)}; ga_it2 0.953 -> 1.003 (GA/PSO tuning); pso_it2 0.953 -> 1.003 (GA/PSO tuning)

| Rank | System | Family | Clean F1 | Faulted mean F1 | Worst-case F1 | Mean SWE | 4-class macro F1 |
|---|---|---|---|---|---|---|---|
| 1 | pso_it2 | new | 0.866 | **0.838** | 0.758 | 0.082 | 0.571 |
| 2 | it2_fis | original | 0.864 | **0.838** | 0.766 | 0.080 | 0.607 |
| 3 | ga_it2 | new | 0.859 | **0.835** | 0.761 | 0.081 | 0.629 |
| 4 | t1_fis | original | 0.855 | **0.832** | 0.759 | 0.080 | 0.615 |
| 5 | threshold | original | 0.848 | **0.824** | 0.754 | 0.079 | 0.622 |
| 6 | proposed | original | 0.857 | **0.823** | 0.749 | 0.083 | 0.603 |
| 7 | pso_it2+fusion | new+fusion | 0.859 | **0.822** | 0.740 | 0.085 | 0.569 |
| 8 | ga_it2+fusion | new+fusion | 0.853 | **0.819** | 0.745 | 0.084 | 0.622 |
| 9 | anfis | new | 0.843 | **0.811** | 0.766 | 0.122 | 0.526 |
| 10 | anfis+fusion | new+fusion | 0.831 | **0.797** | 0.748 | 0.126 | 0.512 |
| 11 | fcm_tsk+fusion | new+fusion | 0.767 | **0.711** | 0.614 | 0.230 | 0.472 |
| 12 | fcm_tsk | new | 0.761 | **0.700** | 0.590 | 0.243 | 0.462 |

Alert F1 per scenario (mean over seeds):

| System | clean | noise_low | noise_med | noise_high | missing_10 | missing_30 | spikes | stuck | conflict | reduced_reliability | combined |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pso_it2 | **0.866** | 0.867 | 0.862 | 0.870 | **0.857** | 0.793 | 0.785 | **0.849** | **0.868** | **0.871** | 0.758 |
| it2_fis | 0.864 | **0.874** | **0.867** | 0.862 | 0.853 | 0.793 | 0.786 | 0.844 | 0.864 | 0.868 | 0.766 |
| ga_it2 | 0.859 | 0.864 | 0.862 | **0.871** | 0.853 | **0.797** | 0.778 | 0.842 | 0.862 | 0.856 | 0.761 |
| t1_fis | 0.855 | 0.866 | 0.861 | 0.862 | 0.844 | 0.792 | 0.777 | 0.835 | 0.855 | 0.864 | 0.759 |
| threshold | 0.848 | 0.855 | 0.851 | 0.855 | 0.838 | 0.788 | 0.770 | 0.829 | 0.848 | 0.856 | 0.754 |
| proposed | 0.857 | 0.867 | 0.863 | 0.858 | 0.836 | 0.771 | 0.764 | 0.836 | 0.857 | 0.832 | 0.749 |
| pso_it2+fusion | 0.859 | 0.862 | 0.858 | 0.863 | 0.830 | 0.764 | 0.767 | 0.840 | 0.862 | 0.837 | 0.740 |
| ga_it2+fusion | 0.853 | 0.857 | 0.858 | 0.865 | 0.824 | 0.769 | 0.761 | 0.834 | 0.855 | 0.827 | 0.745 |
| anfis | 0.843 | 0.841 | 0.829 | 0.778 | 0.825 | 0.783 | **0.793** | 0.819 | 0.843 | 0.836 | **0.766** |
| anfis+fusion | 0.831 | 0.827 | 0.820 | 0.775 | 0.800 | 0.769 | 0.772 | 0.810 | 0.831 | 0.818 | 0.748 |
| fcm_tsk+fusion | 0.767 | 0.739 | 0.725 | 0.614 | 0.745 | 0.715 | 0.684 | 0.742 | 0.763 | 0.754 | 0.628 |
| fcm_tsk | 0.761 | 0.730 | 0.711 | 0.590 | 0.731 | 0.697 | 0.682 | 0.733 | 0.756 | 0.737 | 0.630 |
