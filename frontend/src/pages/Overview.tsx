import { useState } from 'react'
import { fmt, MODEL_COLOR, MODELS, RISK_COLOR } from '../api'
import { ALERT_LEVELS, MODEL_DOCS, REFERENCE_DOC } from '../content'
import { clean, defaultHorizon, separated, useComparison, useDatasetSpec } from '../data'

// Processing pipeline: sensors -> (optional) reliability layer -> 3 inputs -> 4 models in parallel -> alert + reasons.
function Pipeline() {
  const box = (x: number, y: number, w: number, h: number, t: string, s: string, stroke = 'var(--accent)', dash = false) => (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={8} fill="var(--bg)" stroke={stroke} strokeDasharray={dash ? '5 4' : undefined} />
      <text x={x + 10} y={y + 21} fill="var(--fg)" fontSize={13} fontWeight={600}>{t}</text>
      <text x={x + 10} y={y + 39} fill="var(--muted)" fontSize={10.5}>{s}</text>
    </g>
  )
  const arrow = (x1: number, y1: number, x2: number, y2: number) => <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--muted)" strokeWidth={1.4} markerEnd="url(#arr)" />
  const ms = MODEL_DOCS.map((m, i) => ({ ...m, y: 8 + i * 50 }))
  return (
    <figure className="chart">
      <svg viewBox="0 0 760 212" role="img" aria-label="HydroSentinel AI pipeline">
        <defs>
          <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--muted)" />
          </marker>
        </defs>
        {box(4, 78, 120, 52, 'Sensors', 'level + rain gauges')}
        {box(150, 78, 130, 52, 'Reliability layer', 'optional: check + fuse', 'var(--risk2)', true)}
        {box(306, 78, 118, 52, '3 inputs', 'level · rise · rain')}
        {ms.map((m) => (
          <g key={m.id}>
            <rect x={452} y={m.y} width={150} height={40} rx={8} fill="var(--bg)" stroke={MODEL_COLOR[m.id]} strokeWidth={1.6} />
            <text x={462} y={m.y + 25} fill="var(--fg)" fontSize={13} fontWeight={600}>{m.name}</text>
            {arrow(424, 104, 450, m.y + 20)}
            {arrow(602, m.y + 20, 628, 104)}
          </g>
        ))}
        {box(630, 78, 126, 52, 'Alert + reasons', 'Normal … Critical')}
        {arrow(124, 104, 148, 104)}
        {arrow(280, 104, 304, 104)}
      </svg>
      <figcaption className="muted small">All four models receive identical inputs. The reliability layer (dashed) can be switched on or off on every page to show its effect.</figcaption>
    </figure>
  )
}

export default function Overview({ dataset, go }: { dataset: string; go: (t: string) => void }) {
  const { d, err } = useComparison(dataset)
  const spec = useDatasetSpec(dataset)
  const [open, setOpen] = useState<string>('it2_fis')
  const h = d ? defaultHorizon(d) : ''
  const q = spec?.quality
  const step = spec ? (spec.step_hours < 1 ? `${spec.step_hours * 60} min` : spec.step_hours === 24 ? 'daily' : `${spec.step_hours} h`) : ''

  const findings: string[] = []
  if (d) {
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
        <h2>Four soft-computing techniques for flood-risk alerts, compared on the same river data</h2>
        <p>HydroSentinel AI reads river water level and rainfall, derives the rate of rise, and lets four techniques (IT2FIS, PSO-tuned IT2FIS,
          SOM and ANFIS) each issue a five-level flood-risk alert with a reason. It then checks how good and how robust each alert is on a held-out test period.</p>
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
        <div className="fact"><span>Source</span><b>{spec ? <a href={`https://doi.org/${spec.doi}`} target="_blank" rel="noreferrer">{spec.paper_ref.split(';')[0].split(',')[0]} et al.</a> : '…'}</b></div>
        <div className="fact"><span>Labels</span><b>Proxy: future level vs thresholds</b></div>
      </div>

      <div className="panel">
        <h2>How it works</h2>
        <Pipeline />
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
        <h2>The four techniques</h2>
        <div className="chips" role="tablist">
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
      </div>

      <div className="panel">
        <h2>Key results</h2>
        {err && <p className="error small">No stored evaluation for this dataset ({err}). Run <code>python scripts/compare_models.py {dataset}</code>.</p>}
        {d && <ul>{findings.map((f, i) => <li key={i}>{f}</li>)}</ul>}
        {d && <p className="muted small">Test period {d.meta.test.first} → {d.meta.test.last} ({d.meta.test.rows.toLocaleString()} steps); models fitted on {d.meta.train.first} → {d.meta.train.last}. Results generated {d.meta.created.replace('T', ' ')}.</p>}
      </div>

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
