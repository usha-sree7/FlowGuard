# Output Track: Article / Patent (IPR) / Startup

## (a) Research article skeleton

**Working title.** Reliability-Aware Sensor Fusion with Interval Type-2 Fuzzy Inference for Flood-Risk Detection with Decoupled Uncertainty Status

**Abstract (draft; numbers are measured values from `EXPERIMENT_PROTOCOL.md`)**

> Flood warnings at gauged catchments rely on rainfall and water-level sensors that fail in practice. We propose a pipeline that:
>
> * assesses the reliability of each sensor reading;
> * fuses readings by reliability while keeping unresolved uncertainty as intervals;
> * infers flood risk with an Interval Type-2 fuzzy system using Karnik–Mendel type reduction;
> * reports risk separately from an uncertainty / data-sufficiency status.
>
> On 30-min data from the Bengalas River, Brazil (72,011 records, 2019–2023, public UAE-ML dataset), with proxy labels (1-h-ahead exceedance of statistical level thresholds) and 11 fault scenarios, the method:
>
> * raised alert F1 under spike faults from 0.768 to 0.835 (5 seeds) and halved the false-positive rate;
> * kept a "Confident" status impossible whenever the level was missing;
> * raised the uncertainty index on faulted steps by a factor of 2.4–6.1.
>
> It did not improve alert quality under noise, missing data, stuck sensors or reduced reliability. It showed no benefit on a daily dataset (Malwathu Oya, Sri Lanka). On clean data, IT2 inference outperformed a Type-1 system with identical rules (F1 0.895 vs 0.883).

**Sections**

1. Introduction: unreliable sensors in flood early warning, and the risk/uncertainty conflation problem.
2. Related work: see `RESEARCH_GAP.md`. A systematic review is still pending.
3. Data: two public datasets, audited (`PAPER_DATASET_AUDIT.md`), and the proxy-label definition.
4. Method: reliability assessment; fusion; IT2FIS (FOU from anchor standard error, exact interval firing, KM); uncertainty index; status (`MATHEMATICAL_MODEL.md`).
5. Experimental protocol: chronological split, leak-free calibration, three data stores, 11 scenarios, 4 systems.
6. Results: Tables 3.1 / 3.2, the seed sweep, U response, and fault-detection P/R. All numbers are already measured.
7. Discussion: the honest negative results (lag under noise, stuck faults not detected, no daily benefit) and the disclosed design revision (v0.3 → v0.4).
8. Limitations and future work.

**Results placeholders still to fill (not yet measured):**

* alert lead time;
* event-level (not step-level) metrics;
* validation-block design protocol;
* ANFIS baseline;
* additional catchments.

**Candidate venues (to check scope, indexing and fees; not verified):**

* *Applied Soft Computing*
* *Journal of Hydroinformatics*
* *Natural Hazards*
* *Environmental Modelling & Software*
* *IEEE Access*
* IEEE FUZZ-IEEE (conference)
* national soft-computing conferences

## (b) Patent / IPR notes (UNVERIFIED; not legal advice)

**Candidate inventive aspect.** A flood-warning method in which per-sensor reliability scores do three things:

1. weight a fused best estimate used for the risk decision;
2. set the width of interval-valued inputs to an Interval Type-2 fuzzy inference system, whose type-reduced interval width is then used **only** in an uncertainty index and never to change the risk class;
3. combine with a data-sufficiency rule that forbids a "confident" status when an essential input is missing.

**Draft claim ideas (unverified, for discussion with an IPR cell):**

1. A method comprising: computing a reliability value per sensor reading from missing / spike / stuck / noise / consistency checks; computing a reliability-weighted point estimate and an interval per input; evaluating an IT2 fuzzy system on the point estimates to output a risk class; evaluating the same system on the intervals to output a risk range; and computing an uncertainty index from reliability, data-sufficiency and the risk-range width.
2. The method of 1, wherein the footprint of uncertainty of each membership function is set from the inter-period standard error of a station-specific threshold estimate.
3. The method of 1, wherein a status output is restricted from a "confident" state when any essential input is missing.
4. The method of 1, wherein disagreement between two sensors that cannot be attributed to either sensor widens the fused input intervals instead of reducing either sensor's reliability.

**Prior-art search checklist (not yet done):**

* [ ] Google Patents / Espacenet / WIPO Patentscope with the keywords "type-2 fuzzy" + flood / water level / early warning; "sensor reliability" + fusion + flood; "data quality" + "warning level".
* [ ] Indian Patent Office (InPASS) search.
* [ ] CPC classes G08B 21/10 (alarms for floods), G01F 23 (level measurement), G06N 7/02 (fuzzy logic).
* [ ] Commercial hydromet QA/QC products; many reliability checks are standard practice.
* [ ] Check whether course disclosure, public demos or a GitHub release affect patentability (novelty / grace period rules in the target jurisdiction).

**Assessment.** The individual parts (QC checks, IT2FIS, KM) are well known. Any patentable aspect would sit in the specific coupling. The current evidence of technical effect is limited to spike faults and uncertainty visibility. A software copyright or registration plus an open publication may be the more realistic IPR route.

## (c) Startup / product angle (brief)

* **Product.** A "trust layer" for low-cost flood-monitoring networks run by municipalities, campuses and industrial sites. It ingests existing gauge and radar feeds and outputs (i) a risk level and (ii) a data-trust status, plus maintenance alerts such as "level sensor likely stuck" or "rain gauge disagrees with radar".
* **Why this angle.** The clearest measured value is not better prediction; it is **making bad data visible** and suppressing spike-driven false alarms. That matches an O&M and maintenance value proposition.
* **Risks.** The product is safety-critical, so liability, certification and integration with official warning authorities must be handled. Validation against observed flood impacts is needed (currently only proxies). The market is dominated by public agencies.
* **Next validation steps.** Interview 3–5 operators (e.g. municipal disaster-management cells). Obtain one real deployment dataset with maintenance logs to measure fault-detection value on real faults.
