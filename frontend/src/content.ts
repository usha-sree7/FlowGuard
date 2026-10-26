// Explanatory text for the dashboard. Every NUMBER shown next to these texts comes from the backend
// (/api/comparison, /api/model, /api/replay, /api/assess), never from this file.

export interface ModelDoc {
  id: string
  name: string
  family: string
  idea: string
  inputs: string
  output: string
  steps: string[]
  learned: string
  strengths: string[]
  limits: string[]
}

export const MODEL_DOCS: ModelDoc[] = [
  {
    id: 'it2_fis', name: 'IT2FIS', family: 'Interval Type-2 fuzzy inference (expert rules)',
    idea: 'Flood experts reason in words: "the river is high, rising fast, with heavy rain". The IT2FIS turns such statements into 80 IF-THEN rules. Each word is a fuzzy set whose boundary is itself uncertain, drawn as a band (the footprint of uncertainty) instead of a single curve.',
    inputs: 'Water level [m], rate of rise [m/h], rainfall accumulated over the recent window.',
    output: 'A risk interval [yl, yr] on a 0–100 scale; its midpoint gives the alert level (bands 0–20 Normal, 20–40 Low, 40–60 Moderate, 60–80 High, 80–100 Critical).',
    steps: [
      'Fuzzify: each input gets a lower and an upper membership grade in every term (level: Normal, AboveNormal, Elevated, High, VeryHigh; rise: Falling, Steady, Rising, RisingFast; rain: None, Light, Moderate, Heavy).',
      'Fire the 80 rules: firing strength = product of the three memberships, giving an interval [lower, upper] per rule.',
      'Karnik–Mendel type reduction combines the rule consequents into the interval [yl, yr].',
      'Defuzzify: midpoint (yl + yr) / 2 → alert band.',
    ],
    learned: 'Nothing is trained on labels. The crossover points come from training-period statistics (level: mean + 0.5/1/2/3 sd; rise and rain: quantiles). The band widths come from how much those points vary from year to year.',
    strengths: ['Fully transparent: every alert can be traced to named rules.', 'Uses rise and rain, so it can warn before the level crosses a line.', 'Uncertainty about where "High" begins is represented explicitly.'],
    limits: ['The rules are hand-written: they are not optimised for this river.', 'A faulty reading is used as-is unless the reliability layer is switched on.'],
  },
  {
    id: 'pso_it2', name: 'PSO-tuned IT2FIS', family: 'IT2FIS + particle swarm optimisation',
    idea: 'Same rules, same inference, but the positions of the fuzzy sets are tuned. A swarm of 20 candidate settings moves through the search space. Each particle is pulled towards its own best setting and the best setting found by the swarm.',
    inputs: 'Same as the IT2FIS.',
    output: 'Same as the IT2FIS, with the tuned fuzzy sets. Tuned separately for each forecast horizon.',
    steps: [
      'Decision vector (11 numbers): a shift for each of the 10 crossover points (4 level, 3 rise, 3 rain; at most 40 % of the gap to the neighbouring point, so the order is kept) and 1 scale for the footprint of uncertainty (0 to 3×).',
      'Objective: alert F1 + 0.25 × 5-level macro F1 on the validation block (last 20 % of the training period). The test period is never used.',
      'PSO: 20 particles × 20 iterations, inertia 0.7, c1 = c2 = 1.5, velocity limited to 20 % of each range, fixed seed.',
      'Particle 0 starts at the untuned IT2FIS, so PSO can only win if it beats the untuned system on validation.',
    ],
    learned: 'The 11 numbers above, per horizon, on validation data only. The rules are not changed.',
    strengths: ['Keeps the readable rules of the IT2FIS.', 'Directly optimises the alert quality we report.', 'The comparison with the untuned IT2FIS isolates exactly what optimisation adds.'],
    limits: ['A better validation score can still overfit: only the test period shows whether it carries over.', 'Tuning cost: about 420 full IT2 evaluations per horizon.'],
  },
  {
    id: 'som', name: 'SOM', family: 'Self-Organising Map (unsupervised neural network)',
    idea: 'A grid of prototype "river states" is learned without labels: similar states land on neighbouring nodes. Each node is then given the alert level that was most costly to miss among the training states that matched it.',
    inputs: 'Same three inputs, standardised (z-scores from the fit block).',
    output: 'The best-matching node, and that node\'s alert level.',
    steps: [
      'Batch SOM training on the fit block: each prototype moves towards the states it wins, and its grid neighbours move with it (the neighbourhood shrinks over 25 rounds).',
      'Node → alert level: the level with the lowest expected cost over the node\'s training labels. Missing a level costs 2 per step and a false step up costs 1, so rare High/Critical states are not outvoted.',
      'Grid size (4×4, 6×6, 8×8 or 10×10) is chosen on the validation block.',
      'At run time: find the closest prototype and report its level.',
    ],
    learned: 'Prototype vectors (fit block, no labels), node → level mapping (fit-block labels), grid size (validation).',
    strengths: ['Shows which typical hydrological situations exist in the data.', 'Explanation by example: "this looks like prototype 23, whose training states became High".'],
    limits: ['Output is a step function: one level per node, with no smooth score.', 'Rare extremes share nodes with milder states, which limits precision.'],
  },
  {
    id: 'anfis', name: 'ANFIS', family: 'Adaptive neuro-fuzzy inference system (Jang, 1993)',
    idea: 'A fuzzy system that is trained like a neural network. It uses the same 80 rule premises, but each rule\'s output is a learned linear equation of the inputs, and the fuzzy sets themselves are adjusted during training.',
    inputs: 'Same three inputs (scaled to about 0–1).',
    output: 'A 0–100 risk score (weighted sum of the 80 rule outputs) → alert band.',
    steps: [
      'Layer 1: Gaussian membership functions (5 level, 4 rise, 4 rain), initialised at the calibrated crossover points.',
      'Layers 2–3: rule firing = product of memberships, normalised over all 80 rules.',
      'Layer 4: each rule outputs p0 + p1·level + p2·rise + p3·rain (320 learned parameters).',
      'Hybrid learning, 40 epochs: least squares for the rule outputs, then one gradient (Adam) step on the membership centres and widths. Target = centre of the true level\'s band; rare levels are up-weighted.',
    ],
    learned: 'Membership centres and widths plus 320 consequent parameters (fit block). The regularisation strength is chosen on the validation block.',
    strengths: ['Learns the input–risk relationship from data, not from hand-written rules.', 'Still interpretable: each alert can be traced to the most active rules and their learned outputs.'],
    limits: ['Needs labels, and these are proxy labels (see Dataset).', 'Rule outputs are linear equations, which are harder to read than "THEN risk is High".'],
  },
]

export const REFERENCE_DOC = 'Threshold rule (reference): the alert level of the current water level. The proxy labels are built from the same thresholds, so on clean data at short horizons this rule is almost perfect by construction. It is shown only to put the four techniques in context; it is not a competitor.'

export const ALERT_LEVELS = [
  { name: 'Normal', rule: 'below mean + 0.5 sd', meaning: 'River in its usual range.' },
  { name: 'Low', rule: 'mean + 0.5 sd to mean + 1 sd', meaning: 'Slightly above normal.' },
  { name: 'Moderate', rule: 'mean + 1 sd to mean + 2 sd', meaning: 'Clearly elevated; be prepared.' },
  { name: 'High', rule: 'mean + 2 sd to mean + 3 sd', meaning: 'Above the statistical flood threshold (Soares et al. rule). Counts as an ALERT.' },
  { name: 'Critical', rule: 'above mean + 3 sd', meaning: 'Far above the flood threshold. Counts as an ALERT.' },
]

export const METRIC_HELP: [string, string][] = [
  ['Alert', 'High or Critical. Precision, recall, F1 and FPR treat "alert vs no alert" as a yes/no question.'],
  ['Precision', 'Of the alerts raised, the share that were correct (few false alarms → high precision).'],
  ['Recall', 'Of the real alert situations, the share that were caught (few missed floods → high recall).'],
  ['F1', 'Harmonic mean of precision and recall: high only when both are high. Main ranking metric.'],
  ['95 % CI', 'Range of F1 under a moving-block bootstrap of the test period. Whole days are resampled together, because neighbouring 30-min steps are not independent. Overlapping intervals mean the difference is not established.'],
  ['FPR', 'False-positive rate: share of non-alert steps on which an alert was raised.'],
  ['Macro F1', 'F1 of each of the 5 levels, averaged with equal weight, so the rare levels count as much as Normal.'],
  ['κ (kappa)', 'Agreement on the 5 levels beyond what chance would give (1 = perfect, 0 = chance).'],
  ['Severity-weighted error', 'Average cost per step: missing a level costs 2 per level, a false step up costs 1.'],
  ['Missed / false', 'Count of alert steps missed and alarms raised without an alert, on the test period.'],
]
