# Literature review and research gap

Bibliographic details of all six papers were checked against **Crossref** on 2026-10-29. Findings are taken from
the published abstracts (Crossref or Semantic Scholar). Where only metadata could be accessed, this is stated, and
no finding is claimed for that paper.

## 1. The six papers

| # | Verified reference | What the paper does / finds (from its abstract) | How it supports our method | Limitation for our purpose |
|---|---|---|---|---|
| 1 | Kambalimath, S. & Deka, P.C. (2020). A basic review of fuzzy logic applications in hydrology and water resources. *Applied Water Science* 10(8), 191. [doi:10.1007/s13201-020-01276-2](https://doi.org/10.1007/s13201-020-01276-2) | Review. Hydrological problems involve imprecision and vagueness that fuzzy logic handles well; hybrid-fuzzy models generally work better than pure fuzzy models. | Justifies a fuzzy rule base for flood risk, and comparing a pure fuzzy system (IT2FIS) with hybrids (PSO-IT2FIS, ANFIS). | A brief review: no quantitative benchmark, and Type-2 fuzzy sets and sensor faults are not its focus. |
| 2 | Nayak, P.C., Sudheer, K.P., Rangan, D.M. & Ramasastri, K.S. (2005). Short-term flood forecasting with a neurofuzzy model. *Water Resources Research* 41(4). [doi:10.1029/2004WR003562](https://doi.org/10.1029/2004WR003562) | Neuro-fuzzy (ANFIS-type, hybrid backpropagation + least squares) rainfall-runoff model for the Kolar basin, **India**. At 1 h ahead it forecast 47.95 % of flows within 1 % relative error, against 36.96 % (ANN) and 18.89 % (fuzzy). It stayed better up to 6 h ahead. | Direct precedent for our ANFIS (same hybrid learning) and for short hourly horizons; reports RMSE, as we do for the genuine level forecast. | Forecasts flow (a regression target) with clean data; it does not issue alert classes or test sensor faults. |
| 3 | Khanesar, M.A. & Branson, D.T. (2021). Prediction interval identification using interval type-2 fuzzy logic systems: Lake water level prediction using remote sensing data. *IEEE Sensors Journal* 21(12), 13815-13827. [doi:10.1109/JSEN.2021.3067841](https://doi.org/10.1109/JSEN.2021.3067841) | IT2 fuzzy system with support-vector regression that outputs a value **and** lower / upper prediction bounds. A penalty term controls the interval width, giving narrower yet inclusive intervals. Applied to satellite-derived **lake** (Urmia) water level. | Supports using the IT2 output interval as an uncertainty signal (our KM interval [yl, yr] and uncertainty index). | **Lake** level from remote sensing, not river flood alerts; the intervals are learned by quadratic programming, unlike our rule-based IT2FIS. |
| 4 | Nanda, T., Sahoo, B. & Chatterjee, C. (2017). Enhancing the applicability of Kohonen Self-Organizing Map (KSOM) estimator for gap-filling in hydrometeorological timeseries data. *Journal of Hydrology* 549, 133-147. [doi:10.1016/j.jhydrol.2017.03.072](https://doi.org/10.1016/j.jhydrol.2017.03.072) | KSOM gap-filling for streamflow, rainfall, evapotranspiration and temperature at 30 Indian stations. The best number of map units depends on the coefficient of variation of the data, and the estimator behaves differently for different variables (from published highlights; full abstract withheld by the publisher). | Supports SOMs as a tool for **Indian** hydrological series and choosing the grid size from data (we select it on validation data). | Uses SOM for **gap-filling** (estimation), not risk classification. Our SOM needs an explicit cluster → risk mapping (§2). |
| 5 | Nayak, P.C. (2010). Explaining internal behavior in a fuzzy if-then rule-based flood-forecasting model. *Journal of Hydrologic Engineering* 15(1), 20-28. [doi:10.1061/(ASCE)HE.1943-5584.0000146](https://doi.org/10.1061/(ASCE)HE.1943-5584.0000146) | **Metadata verified only**: the abstract is withheld by the publisher and the page returned HTTP 403. The title states the topic: explaining the internal behaviour of a fuzzy rule-based flood-forecasting model. | Motivates our rule-level explanations (which rules fired, which inputs drove the risk). | Findings not verified; **do not quote results from this paper** without reading the full text. |
| 6 | Mai, D.S., Dang, T.H. & Ngo, L.T. (2021). Optimization of interval type-2 fuzzy system using the PSO technique for predictive problems. *Journal of Information and Telecommunication* 5(2), 197-213 (online 2020). [doi:10.1080/24751839.2020.1833141](https://doi.org/10.1080/24751839.2020.1833141) | IT2 fuzzy systems are sensitive to their parameters. PSO tunes the Gaussian membership-function parameters and beats random forest, SVM, ANN, ANFIS and an FCM-initialised IT2 system on forest-fire area and wine-quality prediction. | Direct precedent for PSO-tuned IT2FIS and for comparing it with the untuned system. | Non-hydrological benchmarks. We tune only 10 crossover shifts and 1 FOU scale (not every parameter) to limit overfitting. |

Corrections to the brief: paper 3 predicts **lake** (not river) water level; paper 5 is by **P.C. Nayak alone**
(2010); paper 6 is by **Mai, Dang & Ngo**, published in volume 5(2), 2021.

## 2. How the papers map onto our methodology

| Our component | Supported by | Design choice it implies |
|---|---|---|
| IT2FIS with 80 rules | 1, 3, 5 | Readable rules; output interval used as uncertainty |
| PSO-tuned IT2FIS | 6 | PSO optimises **10 crossover shifts + 1 FOU scale**. Loss = −(validation alert F1 + 0.25 × 5-level macro F1) on the last 20 % of the training period; the untuned IT2FIS is particle 0. Same splits as the untuned model. |
| ANFIS | 2 | Hybrid least-squares + gradient learning, hourly horizon |
| SOM | 4 | Unsupervised grid. **Cluster → risk mapping is fixed before testing**: each node gets the risk level with the lowest expected cost over its fit-block labels (missing a level costs 2 per step, a false step up costs 1). Grid size is chosen on validation. |
| Threshold baseline | – | Reference rule from which the proxy labels are built |

## 3. Research gap (stated carefully)

The checked papers each cover one piece: fuzzy rule-based or neuro-fuzzy forecasting on clean data (1, 2, 5),
IT2 prediction intervals (3), SOMs for gap-filling (4) and PSO tuning of IT2 systems on non-hydrological data (6).
**Among these six papers**, none:

1. compares IT2FIS, PSO-tuned IT2FIS, SOM and ANFIS **on identical inputs and the same time-ordered splits**;
2. evaluates them under **simulated sensor faults** (missing, noise, spikes, stuck, conflicting rain);
3. reports alert levels together with a **separate data-reliability status**;
4. is tested on **Indian hourly CWC telemetry**, with its real data-quality problems documented.

This is a gap **relative to the works we checked**, not a claim of global novelty: no systematic database search
(Scopus / Web of Science) has been done. The combination is an engineering contribution, and our own results show
where it does **not** help (see `results/MODEL_COMPARISON_*.md`).

## 4. Datasets cited elsewhere

The Bengalas (Soares et al., 2026) and Malwathu Oya (Rathnayake et al., 2023) dataset papers are documented in
`REFERENCES.md` and `docs/PAPER_DATASET_AUDIT.md`; the CWC / NWDP data in `docs/INDIAN_DATA_AUDIT.md`.
