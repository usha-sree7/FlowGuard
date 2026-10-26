# HydroSentinel AI: viva preparation

Numbers are from `results/MODEL_COMPARISON_bengalas.md` / `_malwathu.md` (model v0.6.0, held-out test split, 3 fault seeds). Re-run `scripts/compare_models.py` before the viva and quote the file, not memory.

## A. Explain the project in 60 seconds

"River-level sensors and rain gauges give three inputs: water level, rate of rise and recent rainfall. Four soft-computing techniques turn them into a five-level flood alert (Normal, Low, Moderate, High, Critical). The first is an Interval Type-2 fuzzy system with 80 expert rules. The second is the same system with its fuzzy sets tuned by particle swarm optimisation. The third is a Self-Organising Map that matches the situation to learned prototype river states. The fourth is ANFIS, a neuro-fuzzy network that learns its rules from data. Every alert comes with the reasons the model actually used. We evaluate all four on a held-out test period at 30 min, 1 h and 3 h ahead, with confidence intervals, and again under simulated sensor faults, with and without a reliability layer that checks and fuses the readings. The honest result: the two IT2 systems and ANFIS are statistically tied on clean data. PSO gives small, non-significant gains. ANFIS is fragile under noise and SOM is clearly weaker. The reliability layer helps against spikes but not everywhere."

## B. Data

**1. Where does the data come from?**
* **Bengalas River, Petrópolis, Brazil.** 72,011 records at 30-min steps (2019-01-04 14:30 to 2023-12-31 16:00), from the public UAE-ML repository (MIT licence) of Soares et al., *Transactions in GIS* 30(2) e70254, 2026, doi:10.1111/tgis.70254. Station code 221342310 (INEA) appears in the files, but we downloaded from the authors, not from INEA.
* **Malwathu Oya, Sri Lanka.** 4,765 daily values from Rathnayake et al., *PLOS ONE* 2023.
* Both files are SHA-256 pinned. The website's Dataset tab shows the DOI, repository, location, timestamps and units.

**2. Is any of it Indian data (IMD, CWC, India-WRIS)?**
No. We checked IMD, CWC, the CWC flood-forecast site, the National Water Data Portal, India-WRIS and CAMELS-IND. None offers an open, co-located river-level and rainfall record at sub-daily resolution, so the Brazilian data is used and disclosed. CAMELS-IND (daily) is the recommended next step.

**3. What exactly is the label?**
A **proxy**. It is the alert level of the water level *actually observed* h steps later, compared with train-split thresholds: Low = μ + 0.5σ, Moderate = μ + σ, High = μ + 2σ, Critical = μ + 3σ (Bengalas 0.99 / 1.14 / 1.43 / 1.72 m). μ + 2σ is the "flood threshold" in the Soares et al. code; the others are our extensions. Alert = High or Critical. These are not recorded floods, so every score means "agreement with this rule".

**4. Why five levels and not two?**
The brief asks for five, and the extra levels make the ordinal error visible: macro F1, κ and the severity-weighted error penalise calling Critical "Low" more than calling it "High". The binary alert metrics stay comparable with earlier versions because the alert line (μ + 2σ) did not move.

**5. Does the data support 30 min, 1 h and 3 h?**
We checked before offering them (Dataset tab → horizon support):
* **Bengalas.** The 30-min steps give 1, 2 and 6 steps for the three horizons. There are 20,000+ labelled test rows per horizon and 633–674 alert steps. The level class changes between now and t + h on 2.7 %, 4.7 % and 10.2 % of steps, so the task is not trivial.
* **Malwathu.** The data is daily, so only "1 day" exists. The sub-daily horizons are shown as unsupported, not interpolated.

**6. How do you avoid leakage?**
* **Chronological split.** Train = first 70 %, made of a fit block (first 80 %) and a validation block (last 20 %). Test = last 30 %, used once.
* **Fitting.** Thresholds, fuzzy anchors and reliability limits come from the training split. SOM, ANFIS and the ARX forecaster are fitted on the fit block. The SOM grid, the ANFIS ridge and the PSO genes are chosen on validation.
* **Purging.** The last h fit rows, whose future target lies in validation, are not used for fitting.
* **Faults.** Faults are injected into a copy of the test data. Labels always come from the original data.
* **Tested.** Changing only the test data leaves SOM, PSO genes and ANFIS parameters identical (unit test).

## C. Models

**7. Why exactly these four?**
They are the brief's techniques, and together they span the soft-computing families:
* **IT2FIS:** expert knowledge plus explicit uncertainty in the fuzzy sets.
* **PSO:** evolutionary optimisation of that expert system.
* **SOM:** unsupervised neural pattern recognition.
* **ANFIS:** neuro-fuzzy learning from data.

All four see identical inputs, so differences come from the technique. The threshold rule is shown only as a reference, because the labels are built from it.

**8. Type-1 vs Type-2 in one sentence?**
Type-1 says "1.40 m is 0.6 High". Type-2 says "1.40 m is between 0.45 and 0.75 High, because we are not sure where High begins". The band (FOU) comes from how much each boundary varies between training years, and Karnik–Mendel type reduction turns it into an output interval [yl, yr] whose midpoint is the risk score.

**9. What exactly does PSO optimise?**
An 11-number vector:
* **Shifts.** One shift per crossover point (4 level, 3 rise, 3 rain), each at most 40 % of the gap to its neighbour, so the order of the sets is kept.
* **FOU scale.** One scale factor for the footprint of uncertainty (0–3×).

The rules and the inference are unchanged. Objective: validation alert F1 + 0.25 × 5-level macro F1, on the last 20 % of the training period. Settings: 20 particles × 20 iterations, inertia 0.7, c1 = c2 = 1.5, velocity clamp. Particle 0 is the untuned IT2FIS, so PSO can only win if validation says so.

**10. Did PSO help?**
A little, and not significantly:
* **Bengalas.** Test F1 rose at every horizon: 0.935 → 0.944 (30 min), 0.887 → 0.894 (1 h), 0.664 → 0.695 (3 h). Every pair of 95 % intervals overlaps.
* **What changed.** It mainly cut false alarms (1 h: 92 → 64) and missed a few more alerts (59 → 73).
* **Malwathu.** Validation fitness rose (0.931 → 0.969), but test F1 stayed at 0.864. This is a textbook sign of tuning to the validation block.

**11. How does the SOM produce an alert level?**
The SOM is unsupervised: it arranges (level, rise, rain) prototypes on a grid. Afterwards each node gets the level with the lowest expected cost over the training states it won. Missing a level costs 2 and a false step up costs 1, so rare High states are not outvoted by Normal. The grid size is chosen on validation (8 × 8 at Bengalas, 1 h).

**12. Why is the SOM weaker?**
It is a step function, with one level per node. Its cost-based mapping deliberately errs upward when a node holds mixed states, which shows as high recall (0.914) with many false alarms (371 vs 92 for IT2FIS at 1 h). It is also sensitive to noise, because a noisy reading lands on a different prototype.

**13. How does ANFIS learn, and what did you have to fix?**
ANFIS has 80 first-order TSK rules on the same term grid. Each epoch fits the 320 consequent parameters by weighted least squares, then takes one gradient step on the Gaussian MF centres and widths.

Unconstrained, the gradient steps scrambled the fuzzy sets on Malwathu: "Elevated" moved below "Normal" and widths collapsed to spikes. The explanations then named the wrong terms. We constrain each centre to stay between its neighbours and each width within 0.5–2× its start. This lowered ANFIS's Malwathu F1 (0.83 → 0.66, very wide interval). We kept it, because an explanation that names the wrong term is worse than a lower score. The change is disclosed in CHANGES and EXPERIMENT_PROTOCOL.

**14. Why is ANFIS so fragile under noise?**
Measured: F1 0.867 clean, 0.413 under medium noise and 0.186 under high noise at 1 h, versus 0.799 / 0.631 for IT2FIS. We have not isolated the cause. The reliability layer partly helps (0.413 → 0.508).

**15. How are alerts explained?**
Every reason is computed from the model at that step:
* **IT2FIS / PSO:** the inputs and their linguistic terms, the position against the thresholds, the strongest fired rule with its interval firing strength, the type-reduced interval and its band. PSO also reports which crossover it moved most.
* **SOM:** the matching prototype and how many training states on that node later became each level.
* **ANFIS:** the most active rules with their learned outputs, and the weighted sum.
* **All models:** sensor flags with their reliability, and the confidence status when the reliability layer is on.

**16. What does the reliability layer do?**
Each reading gets a reliability r from causal checks with limits taken from clean training data: missing, spike (jump > 1.5 × the 99.9 % training step), stuck, noisy, out-of-range, and rain-sensor disagreement. The fused level is r · x + (1 − r) · (median of the last trustworthy readings). Rain is a reliability-weighted mean. The models then run on these estimates. It also yields an uncertainty index U (not a probability) and a status. A missing essential input can never be "Confident" (tested).

## D. Results

**17. Which model is best?**
It depends on the criterion. The intervals tell you when not to choose:
* **Clean data, 1 h:** PSO-IT2FIS 0.894, IT2FIS 0.887, ANFIS 0.867. All three are tied (intervals overlap). SOM at 0.737 is clearly lower.
* **Under faults:** the IT2 models are most robust (mean F1 0.79 over 9 faults vs 0.65 for ANFIS and 0.50 for SOM).
* **Fewest false alarms:** PSO-IT2FIS.
* **Daily data:** IT2FIS = PSO-IT2FIS.

A model is only called better when its interval does not overlap the other's.

**18. Why does the threshold reference score so well?**
The labels are built from the same thresholds, so "the level now" is a strong guess for "the level in 30 min". The IT2 models beat it at every Bengalas horizon, but within the intervals. That is why it is a reference and not a competitor.

**19. Does the reliability layer help?**
Sometimes:
* **Spikes:** it clearly helps every model (IT2FIS 0.755 → 0.821, ANFIS 0.651 → 0.750 at 1 h). It also helps ANFIS and SOM under noise.
* **Clean data and missing readings:** it costs a little (IT2FIS clean 0.887 → 0.880).
* **Detection:** its checks detect missing data almost perfectly, but catch only 3 % of stuck readings and 1 % of conflicting-rain faults. That is a real weakness, and it is shown on the website.

**20. What happens as the horizon grows?**
Skill falls for every model (IT2FIS 0.935 / 0.887 / 0.664 at 30 min / 1 h / 3 h), because current readings say less about the level 3 h ahead. A simple linear ARX level forecaster does not beat persistence on MAE, so better forecasting needs more than these three inputs.

**21. Is the 0–100 score a probability?**
No. It is a severity index. The uncertainty index U is a ranking index with provisional weights. Neither is calibrated.

**22. What do the results NOT prove?**
* **Real floods:** the labels are proxies, so nothing shows the system would warn of a real flood.
* **Real failures:** the faults are simulated.
* **Generality:** only two catchments, with few large events (Malwathu has only 66 alert days in its test period).
* **PSO:** it is not shown to be better than the untuned IT2FIS.

**23. What is your contribution, if none of the techniques is new?**
* **Like-for-like comparison:** four soft-computing techniques on identical inputs, labels and splits.
* **Honest statistics:** leak-free evaluation with block-bootstrap intervals, 9 seeded sensor faults and three horizons.
* **Measured reliability layer:** an on/off reliability layer whose effect is measured per model and per fault.
* **Faithful explanations:** each alert's explanation comes from the model's own internals. Two problems were found and fixed this way: a monotonicity defect in the IT2 FOU and scrambled ANFIS fuzzy sets.

## E. Practical

**24. Is the dashboard live?**
No. Historical replay uses the held-out test period, and what-if mode runs the real models on a short constructed history, labelled as hypothetical. It is not an official warning service.

**25. How fast is it?**
Measured on a laptop CPU, in ms per 30-min step:
* **Inference:** SOM 0.001, ANFIS 0.006, IT2FIS 0.024, PSO-IT2FIS 0.027. The reliability layer adds 0.05.
* **Fitting (Bengalas, per horizon):** IT2FIS calibration 2 s, SOM about 4 s, ANFIS about 65 s, PSO about 70–80 s.

**26. Can a genuine, very fast rise be mistaken for a spike?**
Yes. The spike check flags jumps larger than 1.5 × the 99.9 % training step. If rain supports the rise, the reliability only drops to 0.6, but the fused level then lags the true level. This is a measured, documented limitation.

**27. You said in an earlier review that ANFIS and PSO were removed. Why are they back?**
The current brief asks for these four techniques. They were rebuilt, adapted to five levels, constrained for interpretability (ANFIS), and evaluated with the same protocol as the IT2FIS.
