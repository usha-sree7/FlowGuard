# Research Gap (preliminary wording)

> Status: **preliminary**. No systematic prior-art review has been done yet. Nothing below claims global novelty. The wording reflects only the gap *relative to the works we actually checked*.

## Problem

Flood early-warning decisions at a gauged catchment combine rainfall and water-level evidence that is **itself unreliable**: gauges drop out, spike, freeze, or disagree with radar. Most data-driven flood predictors studied here:

* take inputs as given;
* produce one deterministic number;
* when they do quantify uncertainty, treat it as a property of the model output rather than of each sensor's evidence.

As a result, operators cannot tell a confident "Normal" from a "Normal" produced because the level sensor went silent.

## Gap statement

Among the works checked (table below), we did not find a design that does all of the following together:

1. **Per-sensor, per-time-step reliability assessment** (missing, spike, stuck, noise, cross-sensor inconsistency, device health), feeding
2. **Reliability-weighted evidence fusion that keeps unresolved uncertainty as intervals** rather than discarding it, feeding
3. **An Interval Type-2 fuzzy inference system** whose KM interval output is used as a model-uncertainty term, with
4. **Flood risk and an uncertainty / data-sufficiency status reported as two separate outputs**, with a hard rule that missing essential inputs can never yield "Confident".

Soares et al. (2026) quantify the effect of *input perturbations on ML forecasts* via Monte Carlo noise (`N_PERTURBATIONS = 10,000` in their code). They do not perform per-sensor reliability assessment or fusion. Rathnayake et al. (2023) use soft computing (an ANFIS ensemble) for daily water-level prediction, without a sensor-reliability layer.

**Research question (preliminary).** Does reliability-aware interval fusion in front of an IT2FIS keep flood-alert quality more stable under realistic sensor faults than threshold, Type-1 and plain IT2 baselines? Does it expose degraded data through a separate uncertainty status?

**Current answer (from our experiments, `EXPERIMENT_PROTOCOL.md`).** Partially.

* Yes for **spikes**: F1 0.835 vs 0.768.
* Yes for **making degraded data visible**: U is 2.4–6.1× higher on faulted than unfaulted steps, and the status is never Confident with a missing level.
* **No** for noise, missing data, stuck sensors and reduced reliability, where alert quality was equal or lower.
* **No benefit** on daily data.

The gap is therefore real as a design gap, but the performance benefit is not yet demonstrated beyond spike faults.

## Literature table (verified metadata only)

| # | Work | Verified via | Relevance | Notes |
|---|---|---|---|---|
| 1 | Soares, J.A.J.P. et al., "Uncertainty-Aware Machine Learning Models for Flash Flood Prediction", *Transactions in GIS* 30(2), 2026, doi:10.1111/tgis.70254 | Repo README (title, authors), MetaTOC listing (vol/issue/date); full text blocked (HTTP 403) | Primary dataset; input-perturbation uncertainty for ML flash-flood forecasting | Article number e70254 not verified |
| 2 | Rathnayake, N.; Rathnayake, U.; Dang, T.L.; Hoshino, Y., "Water level prediction using soft computing techniques: A case study in the Malwathu Oya, Sri Lanka", *PLOS ONE* 18(4): e0282847, 2023 | journals.plos.org article page | Secondary dataset; ANFIS-ensemble soft-computing baseline | Daily, 2005–2018 |
| 3 | Godfried, I.; Mahajan, K.; Wang, M.; Li, K.; Tiwari, P., "FlowDB: a large scale precipitation, river, and flash flood dataset", arXiv:2012.11154, 2020 | arXiv abstract page | Candidate benchmark | No download link on the abstract page |
| 4 | Bertoli, G.; Schroeter, K.; Arcucci, R.; Caporali, E.; et al., "A Hybrid Machine Learning Framework for Improved Short-Term Peak-Flow Forecasting", arXiv:2601.09336, 2026 | arXiv abstract page | Uses LamaH-CE (857 catchments, 6-h) | Not the LamaH-CE data paper |
| 5 | Okuno, S.; Ikeuchi, K.; Aihara, K., "Practical data-driven flood forecasting based on dynamical systems theory: Case studies from Japan", arXiv:2001.06414, 2020 | arXiv abstract page | Data-driven forecasting of warning-level exceedance | "ANNEX benchmark" not confirmed |
| 6 | Karnik, N.N.; Mendel, J.M., "Centroid of a type-2 fuzzy set", *Information Sciences* 132, 2001 | **From memory — verify before citing** | KM type reduction used here | |
| 7 | Mendel, J.M.; John, R.I.B., "Type-2 fuzzy sets made simple", *IEEE Trans. Fuzzy Systems* 10(2), 2002 | **From memory — verify before citing** | IT2 FOU / embedded-set framework | |

## To do before any novelty claim

* Run a systematic search (Scopus / IEEE Xplore / Web of Science) on combinations of: "interval type-2" AND (flood OR "early warning") AND (sensor reliability OR "data quality" OR fusion).
* Check IT2-based sensor-fusion work outside hydrology, e.g. structural health monitoring and industrial fault-tolerant fusion.
* Check operational early-warning QA/QC practice, such as the WMO guidance on automated QC of hydrometric data. Many reliability checks used here are standard QC; the contribution is in the coupling, not the individual checks.
