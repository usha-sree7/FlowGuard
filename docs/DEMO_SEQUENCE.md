# HydroSentinel AI: demonstration sequence (about 10 minutes)

Start the server (`cd backend && python -m uvicorn floodguard.api:app --port 8000`), open http://127.0.0.1:8000 and select **Bengalas River**.
Open *Assess Risk* once before the review: the first load fits the models (about a minute on Bengalas).

1. **Overview (1.5 min).** Read the hero line. Walk the pipeline diagram: sensors → optional reliability layer → 3 inputs → the four models
   in parallel → alert with reasons. Show the five-level ladder with the thresholds in metres, and say that High = mean + 2 sd is the
   paper's flood threshold. Click through the four technique cards. Each card states its inputs, its output, what is learned and from
   which data.
2. **Assess Risk, historical replay (3 min).** Take the largest episode, horizon 1 h, layer off. The slider starts h steps *before* the
   river crosses High, i.e. when a 1-h-ahead alert should already be raised. Compare the five alert cards with "what actually happened".
   Open the IT2FIS explanation (fired rules, interval, band), then the SOM one (prototype, labels seen on the node), then the ANFIS one
   (most active rules and their learned outputs). Switch the sensor condition to *Spikes*, then turn the reliability layer **on** and show
   how the level the models use changes.
3. **Assess Risk, what-if (1 min).** Move level and rain. Choose *Level spike* and toggle the reliability layer. With it off, the
   threshold rule jumps to Critical. With it on, the spike is flagged and down-weighted.
4. **Compare Models (2.5 min).** Explain alert = High or Critical. Read the metrics table, including the confidence intervals: a model is only
   called better when its interval does not overlap the other's. Switch horizon 30 min → 1 h → 3 h: skill drops with lead time. Scroll
   to **PSO vs untuned IT2FIS**: what PSO optimises (11 genes, validation objective), the convergence curve, the moved crossover points,
   and whether the test gain is significant.
5. **Dataset & Sensor Reliability (2 min).** Name, DOI and repository link, location, timestamps, units, label generation (proxy
   labels!), and the chronological split. Show the horizon support table: why Malwathu (daily) cannot do 30 min / 1 h / 3 h. Pick a fault
   (missing, noisy, stuck, conflicting): first the corrupted signal and the per-model strips, then the score table with the reliability
   layer off vs on. Point out where the layer does **not** help.
6. **Close (30 s).** Limitations: proxy labels, simulated faults, one primary catchment. Next step: a daily Indian check with CAMELS-IND.
