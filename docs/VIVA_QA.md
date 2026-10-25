# Viva: likely questions and answers

**1. What is the core soft-computing contribution?**
Reliability-aware evidence fusion feeding an Interval Type-2 fuzzy inference system, which outputs flood risk and, separately, an uncertainty / data-sufficiency status. The IT2FIS includes:
* LMF/UMF trapezoids with a footprint of uncertainty;
* interval firing strengths;
* a 64-rule base;
* full Karnik–Mendel center-of-sets type reduction;
* midpoint defuzzification.

**2. Is your type reduction really KM, or a shortcut?**
It is real KM:
* `km_iterative` is the classical iterative algorithm.
* `km_exact` evaluates every switch point with cumulative sums, vectorised over time. KM is proven to converge to the optimal switch point, so the two must agree.
* The test suite checks agreement to 1e-9 on 1,500 random cases.

**3. Where does the footprint of uncertainty come from? Did you just pick it?**
Each FOU half-width is the standard error of the station-specific anchor across calendar years of training data (365-day blocks for Malwathu). For Bengalas, the μ+2σ level threshold has a standard error of 0.121 m. The FOU therefore represents uncertainty about where the linguistic boundary lies.

**4. Why IT2 rather than Type-1?**
Because the boundaries are uncertain estimates, and the interval output gives a principled model-uncertainty channel. Measured with identical rules: F1 0.895 for IT2 vs 0.883 for T1 on clean Bengalas data, and 0.864 vs 0.855 on Malwathu. The gain is small, and we say so.

**5. Why not ANFIS or deep learning?**
There are no observed flood labels; only proxies built from level thresholds exist. A trained model would learn the proxy rule, which is circular. Events are rare (653 positive test steps). The rule base is explainable. ANFIS, ML and PSO baselines were **not run**, so we make no superiority claim against them; they are listed as future work.

**6. What are your labels? Aren't you validating on your own rules?**
They are proxy labels: the class of the observed level 1 h ahead (Bengalas) or 1 day ahead (Malwathu), from the **original** data, using train-split μ + kσ thresholds. μ+2σ is the authors' own "flood threshold" rule in the UAE-ML code; μ±1σ and μ+3σ are user-configured. Every system predicts a future quantity it does not see, and the threshold baseline acts as a persistence test. These are not observed flood impacts.

**7. How did you prevent leakage?**
* The split is chronological (70/30).
* All parameters are fitted on the clean train split: thresholds, anchors, FOU, radar bias, reliability thresholds and the F normalisation. A unit test confirms that tampering with test data leaves the calibration unchanged.
* Faults are injected into copies.
* Labels come from the original test data.
* Injection ground truth is used only for evaluation.

**8. Does the proposed system beat the baselines?**
Only partially.
* **Clear win: spikes.** 5-seed F1 0.835 ± 0.011 vs 0.768 ± 0.014 for the best baseline.
* **Combined faults:** equal within noise.
* **Worse:** clean data (0.888 vs 0.895), noise, missing data, stuck readings, reduced reliability, and every scenario on the daily dataset.

**9. Why is it worse under noise?**
The fused estimate shrinks noisy readings towards a persistence prior, which lags genuine rises; recall drops from 0.827 to 0.640 under heavy noise. All FIS variants also use the rise rate, and differencing amplifies noise, which is why the plain threshold wins under heavy noise.

**10. You changed the design after seeing results — isn't that cheating?**
We disclose it. v0.3 decided risk from the KM midpoint of interval inputs, which converted uncertainty into higher risk (F1 0.281 at 30% missing). v0.4 decouples risk from uncertainty. Because the change was made after seeing test results, v0.4 numbers are not a fully blind evaluation. The next protocol will use a validation block.

**11. What does the uncertainty index mean? Is it a probability?**
No. U = 100·(0.4Q + 0.3D + 0.3F), where Q is sensor-quality deficit, D is data-sufficiency deficit and F is the normalised KM interval width. The weights are provisional. U is a ranking index, not a calibrated error probability. The risk score is not a flood probability either.

**12. Can the system say "Confident" when the level sensor is down?**
No. "Confident" requires D = 0 and U < 25. A missing level makes D > 0, and after 2 steps the status becomes "Insufficient data". This is tested, and counted in every experiment: zero occurrences.

**13. How are faulty sensors detected?**
* Missing or out-of-range readings.
* Spikes: a jump vs the recent median, rain-aware.
* Stuck readings: a flat level during unusually heavy rain.
* Noise: roughness of second differences.
* Rain-gauge inconsistency vs the other gauges.
* Device-health metadata.

Measured detection on Bengalas: spikes recall 0.93 but precision 0.14; stuck recall only 0.01, a known weakness; missing 1.00.

**14. What if the two rain sources disagree?**
With only two sources (gauge and radar) you cannot tell which is wrong. We do not penalise one arbitrarily; the disagreement is recorded as unresolved conflict that widens the input intervals and raises U. With three or more gauges (Malwathu), the outlier is down-weighted.

**15. Are the datasets real? How did you verify them?**
Yes.
* Both were downloaded from the authors' GitHub repositories (MIT licence) and SHA-256 pinned.
* Bengalas: 72,011 rows × 508 columns; we verified the radar cell count against the shapefile and that the target column equals the next-step level.
* Malwathu: we reconstructed 4,765 days from the lag-structured rows, matching the paper.

Discrepancies are documented: ">73,000 tuples" is not confirmed, and the rain units are unclear.

**16. Why does the radar need bias adjustment?**
The radar areal mean sums to about 2.7× less than the station gauge; the units are not documented. A mean-field bias factor (Σgauge / Σradar) is fitted on the training split only. This is standard hydrological practice.

**17. Why are soil moisture and flow not used?**
Neither dataset contains them. We do not invent variables.

**18. How fast is it?**
About 0.01–0.03 ms per observation for the full proposed pipeline (vectorised batch, laptop CPU), against 0.0001 ms for the threshold baseline. A full 11-scenario Bengalas experiment takes about 25 s.

**19. Is the dashboard live?**
It is a **replay** of the held-out historical test split, labelled as such. The pipeline is causal (it uses only past data at each step), so the same code could run on a live feed. No live feed is connected.

**20. What are the main limitations and next milestones?**

Limitations:
* proxy labels only;
* a single level sensor per site;
* no advantage under noise or missing data yet;
* stuck faults are not detected;
* the daily dataset shows no benefit;
* the weights and thresholds are provisional;
* no systematic literature review yet.

Next milestones:
* a trend-aware (Kalman-style) fused level estimate;
* a better stuck detector;
* a validation-block protocol;
* event-level and lead-time metrics;
* an optional ANFIS baseline;
* PSO tuning of MF / U weights on validation data;
* one more catchment with observed flood records.
