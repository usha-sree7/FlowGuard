import { useEffect, useState } from 'react'
import { api, fmt, RISK_COLOR } from '../api'
import { LineChart } from '../charts'

const TERM_COLORS = ['var(--risk0)', 'var(--risk1)', 'var(--risk2)', 'var(--risk3)']
const UNITS: Record<string, string> = { level: 'm', rise: 'm/h', rain: 'accumulated rain' }

function MFPlot({ name, c }: { name: string; c: any }) {
  return (
    <div className="panel">
      <h3>{name} <span className="muted small">[{UNITS[name]}]</span></h3>
      <LineChart height={170} yMin={0} yMax={1.05}
        bands={c.terms.map((t: any, i: number) => ({ lo: t.lower, hi: t.upper, color: TERM_COLORS[i], label: `${t.term} FOU` }))}
        series={c.terms.map((t: any, i: number) => ({ values: t.type1, color: TERM_COLORS[i], label: `${t.term} (Type-1)`, dash: true, width: 1.2 }))}
        xLabels={[fmt(c.x[0], 2), fmt(c.x[c.x.length - 1], 2)]} />
    </div>
  )
}

export default function Model({ dataset }: { dataset: string }) {
  const [m, setM] = useState<any>(null)
  const [err, setErr] = useState('')
  const [inp, setInp] = useState({ level: 1.3, rise: 0.1, rain: 3, lw: 0, rw: 0, pw: 0 })
  const [res, setRes] = useState<any>(null)

  useEffect(() => {
    setM(null)
    api(`/model/${dataset}`).then((r) => {
      setM(r)
      const a = r.calibration.anchors
      setInp({ level: a.level[1], rise: a.rise[1], rain: a.rain[1], lw: 0, rw: 0, pw: 0 })
      setErr('')
    }).catch((e) => setErr(e.message))
  }, [dataset])

  useEffect(() => {
    if (!m) return
    const body = {
      level: [inp.level - inp.lw, inp.level + inp.lw],
      rise: [inp.rise - inp.rw, inp.rise + inp.rw],
      rain: [Math.max(0, inp.rain - inp.pw), inp.rain + inp.pw],
    }
    api(`/infer/${dataset}`, { method: 'POST', body: JSON.stringify(body) }).then(setRes).catch(() => {})
  }, [inp, m, dataset])

  if (err) return <div className="panel error">{err}</div>
  if (!m) return <div className="panel">Loading model…</div>
  const cal = m.calibration
  const slider = (key: keyof typeof inp, min: number, max: number, label: string) => (
    <label className="slider">{label}: <b className="mono">{fmt(inp[key], 3)}</b>
      <input type="range" min={min} max={max} step={(max - min) / 400} value={inp[key]}
        onChange={(e) => setInp({ ...inp, [key]: Number(e.target.value) })} />
    </label>
  )
  const a = cal.anchors

  return (
    <div className="stack">
      <div className="panel">
        <h2>Interval Type-2 fuzzy inference system <span className="muted small">model v{m.model_version}</span></h2>
        <p>Trapezoidal IT2 membership functions with a footprint of uncertainty (shaded: between lower and upper MF);
          the dashed line is the embedded Type-1 MF used by the Type-1 baseline. Crossovers are station-specific and
          estimated on the chronological training split only. FOU half-widths are the standard error of each anchor across
          {cal.cfg.step_hours < 24 ? ' calendar years' : ' 365-day blocks'} of training data.</p>
        <ul className="small">
          <li>Rule base: {m.fis.rules.length} rules (complete 4×4×4 grid), product t-norm</li>
          <li>Type reduction: {m.fis.type_reduction}</li>
          <li>Defuzzification: {m.fis.defuzzification}</li>
          <li>Level thresholds: {cal.thresholds.map((t: number) => fmt(t, 3)).join(' / ')} m — {cal.threshold_source.Warning}</li>
        </ul>
      </div>
      <div className="grid3">
        {Object.entries(m.curves).map(([name, c]) => <MFPlot key={name} name={name} c={c} />)}
      </div>

      <div className="panel">
        <h2>Try the inference</h2>
        <p className="muted small">Give each input a centre and an uncertainty half-width. Width 0 = crisp reading. Wider inputs produce a wider KM interval [yl, yr].</p>
        <div className="grid3">
          <div>{slider('level', 0, a.level[2] * 1.4, 'level')}{slider('lw', 0, (a.level[2] - a.level[0]), '± level')}</div>
          <div>{slider('rise', a.rise[0] * 2, a.rise[2] * 1.5, 'rise')}{slider('rw', 0, a.rise[2], '± rise')}</div>
          <div>{slider('rain', 0, a.rain[2] * 1.5, 'rain')}{slider('pw', 0, a.rain[2], '± rain')}</div>
        </div>
        {res && (
          <div className="cards">
            <div className="card"><h3>IT2FIS</h3>
              <div className="big" style={{ color: RISK_COLOR[['Normal', 'Watch', 'Warning', 'Critical'].indexOf(res.class_it2)] }}>{res.class_it2}</div>
              <div className="mono">y = {fmt(res.it2.y, 2)} · [yl, yr] = [{fmt(res.it2.yl, 2)}, {fmt(res.it2.yr, 2)}]</div>
            </div>
            <div className="card"><h3>Type-1 FIS (centres)</h3><div className="mono big">{fmt(res.t1.y, 2)}</div></div>
            <div className="card wide"><h3>Strongest rules (upper firing)</h3>
              <table className="mini"><tbody>{res.top_rules.map((r: any, i: number) => (
                <tr key={i}><td>IF level {r.if.level} AND rise {r.if.rise} AND rain {r.if.rain}</td><td>→ {r.then}</td>
                  <td className="mono">[{r.firing[0]}, {r.firing[1]}]</td></tr>))}</tbody></table>
            </div>
          </div>
        )}
      </div>

      <div className="panel">
        <h2>Rule base</h2>
        <div className="rules">
          {m.fis.rules.map((r: any, i: number) => (
            <span key={i} className="rule" style={{ borderLeftColor: RISK_COLOR[m.output_terms.indexOf(r.then)] }}>
              {r.if.level} · {r.if.rise} · {r.if.rain} → <b>{r.then}</b>
            </span>
          ))}
        </div>
      </div>

      <div className="panel">
        <h2>Calibrated parameters (training split only)</h2>
        <pre className="small">{JSON.stringify({ anchors: cal.anchors, fou_eps: cal.fou_eps, rain_bias: cal.rain_bias, reliability: cal.rel, F_normalisation: cal.width_ref }, null, 1)}</pre>
      </div>
    </div>
  )
}
