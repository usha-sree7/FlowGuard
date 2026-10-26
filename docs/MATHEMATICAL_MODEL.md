# Mathematical Model

Code references: `backend/floodguard/{reliability,fusion,uncertainty,systems,calibration,tuning,patterns,anfis,suite}.py` and `backend/floodguard/fuzzy/{mf,km,fis}.py`.

```
sensors ──► reliability r_k(t) ──► reliability-weighted fusion ──┬─► point estimates ──► IT2FIS ──► FLOOD RISK (class, score)
 (level, rain gauges/radar,         (+ flags, conflict)           │
  derived rise rate)                                              └─► interval inputs ──► IT2FIS ──► risk range [yl, yr]
                                                                                                      │
                                     Q (quality)  D (sufficiency)  F (range width) ──────────────────► UNCERTAINTY index U, STATUS
```

Flood risk, sensor reliability, model uncertainty and data sufficiency are four separate quantities:

* **Risk** = score/class.
* **Reliability** = r_k per sensor.
* **Model/inference uncertainty** = F, the width of the KM range.
* **Data sufficiency** = D.

U combines Q, D and F for display, but none of them changes the risk class.

## 1. Reliability assessment

Reliability is assessed per sensor *k* and per step *t*, causally (data ≤ t only):

$$ r_k(t) = h_k(t)\prod_j \bigl(1-\pi_{kj}(t)\bigr),\qquad r_k(t)=0 \text{ if missing or outside the plausibility range} $$

where *h_k* is the device-health prior (1 by default; in experiments it is simulated metadata).

Level penalties π:

* **Spike:** |x_t − median(previous 4 valid)| > J, with J = 1.5 × q₉₉.₉|Δlevel|, evaluated only across exact steps. π = 0.9, or 0.4 if recent rain supports a real rise.
* **Stuck:** the reading is flat for ≥ 3 h while more rain fell than in 99.5% of flat runs in training. π = 0.85.
* **Noise:** the rolling mean of |Δ²x| exceeds its train q₉₉. π = min(0.7, 0.5·(ratio − 1)).

Rain penalties π:

* **Stuck:** an identical non-zero value repeats beyond the train q₉₉.₉ run length. π = 0.7.
* **Inconsistency:** z = |x_k − median(others)| / (1 + median) / q₉₉(train disagreement).
  * With ≥ 3 sensors, π = min(0.8, 0.8·(z − 1)).
  * With 2 sensors, both get π ≤ 0.3 and the rest is recorded as **unresolved conflict** C, because the disagreement cannot be attributed to either sensor.

Physical conflict C also includes "level rising fast while no rain is observed anywhere".

## 2. Reliability-weighted fusion

Let p_t be the causal median of the last 4 trustworthy level readings (r ≥ 0.5), s₀ the q₉₅ |Δlevel| step, and k the number of steps since the last trustworthy reading.

**Level**

$$ c_t = r\,x_t + (1-r)\,p_t,\qquad h_t=(1-r)\max(|x_t-p_t|,s_0)+C_t s_0 $$

If the reading is missing: c_t = p_t and h_t = 2 s₀ √k.

**Rise**

$$ \dot c_t=(c_t-c_{t-1})/\Delta t,\qquad \text{half-width}=(h_t+h_{t-1})/\Delta t $$

While the level is missing, the last known rise is carried for up to 2 steps. Across a timestamp gap the rise is unknown and spans the whole universe.

**Rain**

Rain uses bias-adjusted sensors x_k with weights r_k:

* weighted mean m;
* weighted dispersion σ_w;
* per-step interval [(m − σ_w)⁺·max r, m + σ_w + (1 − max r)·s_rain + C·s_rain];
* both the centre and the interval are accumulated over the rain window (2 h or 2 days).

If all rain sensors are missing, the interval is [0, max].

## 3. Interval Type-2 fuzzy inference

### Membership functions

Level has 5 terms and rise and rain have 4 each, in a shoulder–trapezoid–…–shoulder partition. The crossovers are at station-specific anchors a₁ < a₂ < …:

| Input | Terms | Anchors (fitted on the train split) |
|---|---|---|
| level | Normal, AboveNormal, Elevated, High, VeryHigh | μ + 0.5σ, μ + σ, μ + 2σ, μ + 3σ (the alert-level thresholds) |
| rise | Falling, Steady, Rising, RisingFast | q₀₁, q₉₉, q₉₉.₉ |
| rain accumulation | None, Light, Moderate, Heavy | wet-period q₇₅, q₉₅, q₉₉ |

Each crossover has a linear ramp of half-width δ_k = ¼ × the smaller neighbouring gap.

**Footprint of uncertainty (FOU).** The UMF ramps are shifted outward by ε_k and the LMF ramps inward by ε_k:

* ε_k is the **standard error of anchor k across calendar years** of training data (365-day blocks for Malwathu), clipped to **δ_k / 2**.
* The FOU therefore encodes how uncertain the station-specific boundary itself is.
* Why δ/2 and not δ: with ε = δ the lower MFs of both neighbouring terms reach 0 exactly at the crossover, every rule's lower firing vanishes there, the KM interval jumps open, and the IT2 midpoint stops being monotone in the level. On Bengalas (where μ + 0.5σ and μ + σ are only 0.14 m apart) this gave dips of up to 1.06 points that **lowered the alert level for a higher water level** in 9 of 182 tested rise/rain curves. With the δ/2 cap the worst dip is 0.09 points and no level reversal occurs.
* The embedded Type-1 MF is the unshifted midline. Tests verify LMF ≤ T1 ≤ UMF and that the T1 MFs form a partition of unity.

### Interval inputs

For an unknown value known to lie in [a, b], the firing bounds are

$$ \underline\mu = \inf_{x\in[a,b]}\underline\mu_A(x),\qquad \overline\mu=\sup_{x\in[a,b]}\overline\mu_A(x) $$

* Both are **exact** closed forms for quasi-concave (trapezoidal) MFs: the infimum is at an endpoint, and the supremum is the height if [a, b] touches the plateau.
* This is a bounded-error interpretation, not the standard non-singleton fuzzifier, which would use the supremum for both bounds.
* A singleton input reduces to [LMF(x), UMF(x)].

### Rules

The rule base is a complete 5 × 4 × 4 = 80-rule grid generated by an explicit expert template, `rule_severity` in `fis.py`:

* The base severity is the level term index (level term k ↔ alert level k).
* Add +1 if rise = RisingFast, or if Rising with rain ≥ Moderate.
* Add another +1 if rain = Heavy with rise ≥ Rising and level ≥ AboveNormal.
* Subtract 1 for Falling with no rain at AboveNormal / Elevated / High (receding).
* Heavy rain at Normal level gives at least Low.
* The result is clipped to 0..4.

| level \ rise → | Falling | Steady | Rising | RisingFast |
|---|---|---|---|---|
| Normal | N, N, N, L | N, N, N, L | N, N, L, L | L, L, L, L |
| AboveNormal | N, L, L, L | L, L, L, L | L, L, M, H | M, M, M, H |
| Elevated | L, M, M, M | M, M, M, M | M, M, H, C | H, H, H, C |
| High | M, H, H, H | H, H, H, H | H, H, C, C | C, C, C, C |
| VeryHigh | C, C, C, C | C, C, C, C | C, C, C, C | C, C, C, C |

Each cell lists rain = None, Light, Moderate, Heavy. N = Normal, L = Low, M = Moderate, H = High, C = Critical. The table is generated from the code.

### Firing

Product t-norm: $\underline f_i = \prod_j \underline\mu_{ij}$ and $\overline f_i = \prod_j \overline\mu_{ij}$.

### Consequents

Interval centroids on a 0–100 severity scale: Normal [5, 15], Low [25, 35], Moderate [45, 55], High [65, 75], Critical [85, 95].

### Type reduction

Type reduction is **center-of-sets with Karnik–Mendel, fully implemented**:

$$ y_l=\min_{f_i\in[\underline f_i,\overline f_i]}\frac{\sum f_i c^l_i}{\sum f_i},\qquad y_r=\max_{f_i\in[\underline f_i,\overline f_i]}\frac{\sum f_i c^r_i}{\sum f_i} $$

* `km_iterative` is the classical KM algorithm.
* `km_exact` evaluates all switch points with cumulative sums, vectorised over time.
* The two agree to 1e-9 on 1,500 random cases in the test suite, plus 2,000 in an initial check.

### Defuzzification

y = (y_l + y_r)/2. Alert-level edges are at 20 / 40 / 60 / 80 (fixed, not tuned).

### Type-1 baseline

Same rules, the embedded T1 MFs, crisp inputs, and a COS weighted average with the consequent midpoints.

### Properties verified by tests

* The T1 output lies inside [y_l, y_r].
* A wider input interval never narrows [y_l, y_r].
* T1 is monotone in level.
* IT2 is monotone in level up to < 0.1 point. Exact monotonicity is **not** guaranteed for KM-based IT2 systems; the FOU cap above keeps the violation far below one band (20 points).

## 4. Decision outputs

* **Flood risk** = IT2FIS on the fused **point** estimates (singleton inputs), giving a score and one of Normal / Low / Moderate / High / Critical.
* **Plausible risk range** = [min(y_l), max(y_r)] over the interval-input IT2FIS and the point IT2FIS.
* **Uncertainty index**

$$ U = 100\,\frac{w_Q Q + w_D D + w_F F}{\sum_{\text{available}} w},\quad (w_Q,w_D,w_F)=(0.40,0.30,0.30)\ \text{provisional} $$

The components are:

* Q = 1 − (0.6 r_level + 0.4 mean r_rain).
* D = level term (0.5 if missing, 0.3 if missing for ≤ 2 steps) + 0.2·[rise unknown while level present] + 0.3·(fraction of rain sensors missing).
* F = clip((y_r − y_l − w₀)/(50 − w₀), 0, 1), where w₀ is the median clean training width.

Bands: Low < 25 ≤ Moderate < 50 ≤ High < 75 ≤ Very High.

**Status:**

* **Insufficient data** if the level has been unusable for more than 2 steps, or D ≥ 0.6.
* **Confident** only if D = 0 **and** U < 25.
* **Uncertain** otherwise.

A missing essential input can therefore never yield "Confident". This is tested, and checked in every experiment: the count is 0 everywhere.

U is a ranking index. It is **not** a calibrated probability of error, and the risk score is **not** a probability of flooding.

## 5. The four compared techniques

All four use the same three inputs x = (level, rise, rain), either raw or reliability-fused (Sections 1–2), and output one of the five alert levels. Data use: calibration (thresholds, anchors, FOU) on the whole training split; learned parameters on its first 80 % (fit block); hyper-parameters and PSO genes on its last 20 % (validation block). The test split is used once, for evaluation. Fit rows whose horizon-h target lies in the validation block are purged.

### 5.1 IT2FIS (untuned)

Sections 3–4 above. Nothing is trained on labels; one IT2FIS serves all horizons.

### 5.2 PSO-tuned IT2FIS (`tuning.py`)

Decision vector g ∈ ℝ¹¹: one shift s_k ∈ [−1, 1] per anchor (4 level + 3 rise + 3 rain), applied as a_k' = a_k + s_k · 0.4 · min(gap_left, gap_right), so anchors stay ordered; and one FOU scale φ ∈ [0, 3] with ε_k' = φ ε_k (then capped at δ_k'/2).

Objective (maximised, per horizon, on the validation block): J(g) = F1_alert + 0.25 · macroF1₅.

PSO (Kennedy & Eberhart 1995; inertia weight, Shi & Eberhart 1998), 20 particles, 20 iterations:

$$ v \leftarrow \mathrm{clip}\big(w v + c_1 r_1 (p_\text{best} - x) + c_2 r_2 (g_\text{best} - x),\ \pm v_\max\big),\quad x \leftarrow \mathrm{clip}(x + v) $$

with w = 0.7, c₁ = c₂ = 1.5, r₁, r₂ ~ U(0, 1), v_max = 0.2 × range. Particle 0 is the untuned IT2FIS (s = 0, φ = 1), so J(tuned) ≥ J(untuned) on validation by construction; the test period decides whether the gain is real.

### 5.3 SOM (`patterns.py`)

Batch Kohonen SOM on z-scored inputs (fit block), Gaussian neighbourhood shrinking from half the map to 0.5 node over 25 rounds:

$$ w_k = \frac{\sum_t h_{k,b(t)} z_t}{\sum_t h_{k,b(t)}},\qquad b(t) = \arg\min_k \lVert z_t - w_k \rVert $$

Node → alert level by minimum expected cost under the evaluation's ordinal cost C(t, p) = p − t if p ≥ t else 2(t − p): class_k = argmin_p Σ_t n_{k,t} C(t, p). Grid size ∈ {4², 6², 8², 10²} chosen on validation.

### 5.4 ANFIS (`anfis.py`, Jang 1993)

First-order TSK with Gaussian MFs on the same 5 × 4 × 4 term grid (80 rules):

$$ w_r = \prod_j \exp\!\Big(-\tfrac{(z_j - c_{j,r})^2}{2\sigma_{j,r}^2}\Big),\quad \bar w_r = \frac{w_r}{\sum w},\quad y = \sum_r \bar w_r\,(p_{r0} + p_{r1} z_1 + p_{r2} z_2 + p_{r3} z_3) $$

z = inputs scaled to ≈ [0, 1] (0.1 % / 99.9 % quantiles of the fit block). Target = band centre of the proxy class (10, 30, 50, 70, 90), samples weighted by 1/√(class frequency). Hybrid learning, 40 epochs: consequents p by weighted ridge least squares (ridge ∈ {0.01, 1, 100} chosen on validation), then one Adam step on (c, σ). y is clipped to [0, 100] and banded with the fixed edges.

### 5.5 Reference: threshold rule

Alert level of the current level. The labels are built from the same thresholds, so this is a ceiling-like reference at short horizons on clean data, not a competitor.

## 6. Evaluation

* Alert = level class ≥ High. Precision, recall, F1, FPR on that binary question; 5-level macro F1, accuracy, Cohen's κ and the severity-weighted error on the ordinal question.
* 95 % interval of alert F1: moving-block bootstrap (Künsch 1989) of the clean test period, 1,000 resamples, block = 1 day (48 steps) for 30-min data and 14 days for daily data.
* Sensor faults: 9 seeded scenarios (missing 10 / 30 %, noise low / medium / high, stuck, conflicting rain, spikes, combined) × 3 seeds, injected into a copy of the test split; labels always from the original data.
