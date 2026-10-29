import { useState } from 'react'
import { fmt, MODEL_COLOR, MODELS, RISK_COLOR } from '../api'
import { ALERT_LEVELS, MODEL_DOCS, REFERENCE_DOC } from '../content'
import { clean, defaultHorizon, separated, useComparison, useDatasetSpec } from '../data'

// Plain-language 4-step workflow shown on the home screen. Technical detail (rules, learning, maths) stays in
// "Technical details" below and in the Assess Risk / Compare Models tabs.
const WORKFLOW = [
  { n: 1, title: 'Sensors', text: 'Water level and rainfall readings from the river gauges.' },
  { n: 2, title: 'Reliability check', text: 'An optional layer flags missing, noisy, spiking or stuck readings and fuses the trustworthy ones.' },
  { n: 3, title: 'Four techniques', text: 'IT2FIS, PSO-tuned IT2FIS, SOM and ANFIS analyse the same three inputs side by side.' },
  { n: 4, title: 'Final alert', text: 'One primary alert — IT2FIS on the fused readings — with its reasons and confidence status.' },
]

export default function Overview({ dataset, go }: { dataset: string; go: (t: string) => void }) {
  const { d, err } = useComparison(dataset)
  const spec = useDatasetSpec(dataset)
  const [open, setOpen] = useState<string>('it2_fis')
  const h = d ? defaultHorizon(d) : ''
  const q = spec?.quality
  const step = spec ? (spec.step_hours < 1 ? `${spec.step_hours * 60} min` : spec.step_hours === 24 ? 'daily' : `${spec.step_hours} h`) : ''

  const findings: string[] = []
  const lc = d?.meta.label_counts_test[h]
  if (d && lc.High + lc.Critical === 0) {
    // No alert in the test period: alert F1 is undefined, so report the 5-level macro F1 and false alarms instead.
    const rows = MODELS.map((m) => ({ m, c: clean(d, h, 'raw', m) })).sort((a, b) => b.c.macro_f1 - a.c.macro_f1)
    findings.push(`The test period contains no High or Critical level, so alert precision, recall and F1 are undefined: this dataset cannot show flood-detection skill.`)
    findings.push(`5-level agreement (macro F1), ${d.meta.horizons[h]} ahead: ${rows.map(({ m, c }) => `${d.meta.models[m]} ${fmt(c.macro_f1)}`).join(', ')}. False alarms: ${rows.map(({ m, c }) => `${d.meta.models[m]} ${c.false_alarms}`).join(', ')}.`)
    const faulted = d.meta.scenarios.filter((s: string) => s !== 'clean')
    const mean = (mode: string) => MODELS.reduce((a, m) => a + faulted.reduce((b: number, s: string) => b + d.risk[`${h}|${s}|${mode}|${m}`].macro_f1, 0) / faulted.length, 0) / MODELS.length
    findings.push(`Under simulated sensor faults, the average macro F1 of the four models is ${fmt(mean('raw'))} on raw inputs and ${fmt(mean('fused'))} with the reliability layer.`)
  } else if (d) {
    const rows = MODELS.map((m) => ({ m, c: clean(d, h, 'raw', m) })).sort((a, b) => b.c.f1 - a.c.f1)
    const top = rows[0], second = rows[1]
    findings.push(separated(top.c, second.c)
      ? `On clean test data, ${d.meta.horizons[h]} ahead, ${d.meta.models[top.m]} has the highest alert F1 (${fmt(top.c.f1)}), and its 95 % interval does not overlap the runner-up (${d.meta.models[second.m]}, ${fmt(second.c.f1)}).`
      : `On clean test data, ${d.meta.horizons[h]} ahead, ${d.meta.models[top.m]} has the highest alert F1 (${fmt(top.c.f1)}), but its 95 % interval overlaps ${d.meta.models[second.m]} (${fmt(second.c.f1)}), so the two are statistically tied.`)
    const it2 = clean(d, h, 'raw', 'it2_fis'), pso = clean(d, h, 'raw', 'pso_it2')
    findings.push(`PSO tuning changed test F1 from ${fmt(it2.f1)} (untuned IT2FIS) to ${fmt(pso.f1)} (${pso.f1 >= it2.f1 ? '+' : ''}${fmt(pso.f1 - it2.f1)})${separated(it2, pso) ? '' : '; the intervals overlap, so the gain is not established'}.`)
    const faulted = d.meta.scenarios.filter((s: string) => s !== 'clean')
    const mean = (mode: string) => MODELS.reduce((a, m) => a + faulted.reduce((b: number, s: string) => b + d.risk[`${h}|${s}|${mode}|${m}`].f1, 0) / faulted.length, 0) / MODELS.length
    findings.push(`Under simulated sensor faults, the average F1 of the four models is ${fmt(mean('raw'))} on raw inputs and ${fmt(mean('fused'))} with the reliability layer.`)
  }

  return (
    <div className="stack">
      <section className="hero">
        <span className="mode-tag">Historical analysis · research prototype · not a live warning service</span>
        <h2>Flash floods give little warning. HydroSentinel AI turns river sensors into one clear risk alert.</h2>
        <p>It reads river water level and rainfall, derives the rate of rise, and issues a five-level flood-risk alert with a plain-language
          reason — then checks, on real held-out river data, how good and how robust that alert is.</p>
        <div className="row">
          <button className="big-btn" onClick={() => go('assess')}>Assess risk →</button>
          <button className="big-btn btn-ghost on-dark" onClick={() => go('compare')}>Compare models</button>
          <button className="big-btn btn-ghost on-dark" onClick={() => go('data')}>Dataset &amp; sensors</button>
        </div>
      </section>

      <div className="facts">
        <div className="fact"><span>Dataset</span><b>{spec ? spec.name.split(' - ')[0] : '…'}</b></div>
        <div className="fact"><span>Period</span><b>{q ? (q.first ? `${q.first.slice(0, 10)} → ${q.last.slice(0, 10)}` : '2005–2018 (years only)') : '…'}</b></div>
        <div className="fact"><span>Sampling</span><b>{step}{q ? ` · ${q.rows.toLocaleString()} records` : ''}</b></div>
        <div className="fact"><span>Source</span><b>{!spec ? '…' : spec.doi
          ? <a href={`https://doi.org/${spec.doi}`} target="_blank" rel="noreferrer">{spec.paper_ref.split(';')[0].split(',')[0]} et al.</a>
          : <a href={spec.repo} target="_blank" rel="noreferrer">CWC via NWDP (India)</a>}</b></div>
        <div className="fact"><span>Labels</span><b>Proxy: future level vs thresholds</b></div>
      </div>

      <div className="panel">
        <h2>How it works</h2>
        <div className="steps">
          {WORKFLOW.map((s) => (
            <div key={s.n} className="step">
              <span className="n">{s.n}</span>
              <b>{s.title}</b>
              <span>{s.text}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="panel callout">
        <b>One primary alert, four techniques compared.</b>
        <p className="small" style={{ margin: '6px 0 10px' }}>
          The dashboard leads with a single recommended alert — <b>IT2FIS on reliability-fused sensor inputs</b> — shown prominently on
          Assess Risk with its reasons and a confidence status (Confident / Uncertain / Insufficient data). PSO-tuned IT2FIS, SOM and ANFIS
          run alongside it and are evaluated as comparison results; they are never averaged or combined into that alert.
        </p>
        <button className="big-btn" onClick={() => go('assess')}>See the final alert →</button>
      </div>

      <div className="panel">
        <h2>The five alert levels</h2>
        <div className="ladder">
          {ALERT_LEVELS.map((a, i) => (
            <div key={a.name} style={{ background: RISK_COLOR[i] }}>
              <b>{a.name}</b>
              {d ? (i === 0 ? `< ${fmt(d.meta.thresholds_m[0], 2)} m` : `≥ ${fmt(d.meta.thresholds_m[i - 1], 2)} m`) : a.rule}
              <div>{a.rule}</div>
            </div>
          ))}
        </div>
        <p className="note" style={{ marginTop: 8 }}>Thresholds are computed from the training period only (mean and standard deviation of the water level).
          High = mean + 2 sd is the statistical "flood threshold" used in the source paper's code; the other levels are our extensions. <b>Alert = High or Critical.</b></p>
      </div>

      <div className="panel">
        <h2>Key results</h2>
        {err && <p className="error small">No stored evaluation for this dataset ({err}). Run <code>python scripts/compare_models.py {dataset}</code>.</p>}
        {d && <ul className="findings">{findings.map((f, i) => <li key={i}>{f}</li>)}</ul>}
        {d && <p className="muted small">Test period {d.meta.test.first} → {d.meta.test.last} ({d.meta.test.rows.toLocaleString()} steps); models fitted on {d.meta.train.first} → {d.meta.train.last}. Results generated {d.meta.created.replace('T', ' ')}.</p>}
      </div>

      <details className="panel">
        <summary>Technical details: how each of the four techniques works</summary>
        <div className="chips" role="tablist" style={{ marginTop: 10 }}>
          {MODEL_DOCS.map((m) => <button key={m.id} role="tab" aria-selected={open === m.id} className={`chip ${open === m.id ? 'on' : ''}`} onClick={() => setOpen(m.id)}>{m.name}</button>)}
        </div>
        {MODEL_DOCS.filter((m) => m.id === open).map((m) => {
          const c = d && clean(d, h, 'raw', m.id)
          return (
            <div key={m.id} className="card mdoc" style={{ borderLeftColor: MODEL_COLOR[m.id], marginTop: 12 }}>
              <div className="row-between"><h3 style={{ margin: 0 }}>{m.name} <span className="muted small">· {m.family}</span></h3>
                {c && <span className="small">Test alert F1 ({d.meta.horizons[h]}): <b>{fmt(c.f1)}</b> <span className="ci">[{fmt(c.f1_ci95[0])}, {fmt(c.f1_ci95[1])}]</span></span>}</div>
              <p>{m.idea}</p>
              <div className="grid2">
                <div><b>Inputs.</b> {m.inputs}<br /><b>Output.</b> {m.output}<br /><b>What is learned, from what.</b> {m.learned}</div>
                <div><b>How it decides</b><ol>{m.steps.map((s, i) => <li key={i}>{s}</li>)}</ol></div>
              </div>
              <div className="grid2">
                <div><b>Strengths</b><ul className="small">{m.strengths.map((s, i) => <li key={i}>{s}</li>)}</ul></div>
                <div><b>Limitations</b><ul className="small">{m.limits.map((s, i) => <li key={i}>{s}</li>)}</ul></div>
              </div>
            </div>
          )
        })}
        <p className="muted small" style={{ marginTop: 8 }}>{REFERENCE_DOC}</p>
      </details>

      <div className="panel callout warn">
        <b>Read before interpreting any number.</b>
        <ul className="small" style={{ marginBottom: 0 }}>
          <li><b>Proxy labels.</b> The data contain no recorded flood events. The "true" level is the class of the water level actually observed h steps later. Scores measure agreement with that rule, not real-world warning accuracy.</li>
          <li><b>Simulated faults.</b> Sensor faults are injected into a copy of the test data with fixed seeds. They are not recorded sensor failures.</li>
          <li><b>Historical replay.</b> Nothing here is a live feed or an official warning.</li>
        </ul>
      </div>
    </div>
  )
}
