import { useEffect, useMemo, useState } from 'react'
import { api, fmt } from '../api'

type Model = {
  id: string; name: string; family: string; description: string
  per_scenario: Record<string, Record<string, number>>
}
type Data = { dataset: string; seeds: number; scenarios: string[]; metrics: string[]; models: Model[] }

const METRICS: Record<string, { label: string; lower: boolean }> = {
  f1: { label: 'Alert F1 (≥ Warning)', lower: false },
  precision: { label: 'Precision', lower: false },
  recall: { label: 'Recall', lower: false },
  fpr: { label: 'False-positive rate', lower: true },
  macro_f1_4class: { label: '4-class macro F1', lower: false },
  swe: { label: 'Severity-weighted error', lower: true },
}
const FAMILIES: [string, string][] = [['original', 'Original systems'], ['new', 'New soft-computing models'], ['new+fusion', 'New models + reliability fusion']]
const DEFAULT = ['proposed', 'it2_fis', 'pso_it2', 'pso_it2+fusion', 'ga_it2', 'anfis', 'fcm_tsk']

type Row = { m: Model; clean: number; faulted: number; worst: number; sel: number }

function summarise(m: Model, scenarios: string[], metric: string, scen: string): Row {
  const lower = METRICS[metric].lower
  const v = (s: string) => m.per_scenario[s][metric]
  const faulted = scenarios.filter((s) => s !== 'clean').map(v)
  return {
    m, clean: v('clean'), sel: scen === 'faulted' ? faulted.reduce((a, b) => a + b, 0) / faulted.length : v(scen),
    faulted: faulted.reduce((a, b) => a + b, 0) / faulted.length,
    worst: lower ? Math.max(...faulted) : Math.min(...faulted),
  }
}

function RankBars({ rows, metric, scenLabel }: { rows: Row[]; metric: string; scenLabel: string }) {
  const lower = METRICS[metric].lower
  const vals = rows.map((r) => r.sel)
  const max = Math.max(...vals, 1e-9)
  return (
    <figure className="chart" role="img" aria-label={`${METRICS[metric].label}, ${scenLabel}, by model`}>
      <div className="hbars">
        {rows.map((r, i) => (
          <div key={r.m.id} className="hbar-row" title={`${r.m.name}: ${fmt(r.sel)} (${scenLabel})`}>
            <span className="hbar-label">{r.m.name}</span>
            <span className="hbar-track">
              <span className={i === 0 ? 'hbar best' : 'hbar'} style={{ width: `${Math.max(0.5, (r.sel / max) * 100)}%` }} />
            </span>
            <span className="hbar-value">{fmt(r.sel)}</span>
          </div>
        ))}
      </div>
      <figcaption className="muted small">Sorted best first ({lower ? 'lower' : 'higher'} is better). Hover a bar for its value.</figcaption>
    </figure>
  )
}

function Heatmap({ rows, scenarios, metric }: { rows: Row[]; scenarios: string[]; metric: string }) {
  const lower = METRICS[metric].lower
  const all = rows.flatMap((r) => scenarios.map((s) => r.m.per_scenario[s][metric]))
  const lo = Math.min(...all), hi = Math.max(...all)
  const best: Record<string, number> = Object.fromEntries(scenarios.map((s) => {
    const col = rows.map((r) => r.m.per_scenario[s][metric])
    return [s, lower ? Math.min(...col) : Math.max(...col)]
  }))
  // single-hue sequential: stronger = better
  const shade = (v: number) => {
    const t = hi > lo ? (v - lo) / (hi - lo) : 1
    return Math.round(8 + 62 * (lower ? 1 - t : t))
  }
  return (
    <table className="results heat">
      <thead><tr><th>Model</th>{scenarios.map((s) => <th key={s}>{s}</th>)}</tr></thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.m.id}>
            <td>{r.m.name}</td>
            {scenarios.map((s) => {
              const v = r.m.per_scenario[s][metric]
              const p = shade(v)
              return (
                <td key={s} className={v === best[s] ? 'best' : ''} title={`${r.m.name} · ${s}: ${fmt(v)} ± ${fmt(r.m.per_scenario[s][`${metric}_sd`])}`}
                  style={{ background: `color-mix(in srgb, var(--accent) ${p}%, var(--panel))`, color: p > 45 ? 'var(--on-accent)' : undefined }}>
                  {fmt(v)}
                </td>
              )
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export default function Compare({ dataset }: { dataset: string }) {
  const [data, setData] = useState<Data | null>(null)
  const [err, setErr] = useState('')
  const [picked, setPicked] = useState<string[]>(DEFAULT)
  const [metric, setMetric] = useState('f1')
  const [scen, setScen] = useState('faulted')

  useEffect(() => {
    setData(null); setErr('')
    api(`/comparison/${dataset}`).then(setData).catch((e) => setErr(e.message))
  }, [dataset])

  const rows = useMemo(() => {
    if (!data) return []
    const lower = METRICS[metric].lower
    return data.models.filter((m) => picked.includes(m.id))
      .map((m) => summarise(m, data.scenarios, metric, scen))
      .sort((a, b) => (lower ? a.sel - b.sel : b.sel - a.sel))
  }, [data, picked, metric, scen])

  if (err) return <div className="error">{err}</div>
  if (!data) return <div className="panel">Loading comparison…</div>

  const toggle = (id: string) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]))
  const scenLabel = scen === 'faulted' ? 'mean of faulted scenarios' : scen
  const winner = rows[0]

  return (
    <div className="stack">
      <div className="panel">
        <div className="row-between">
          <h3>Select models to compare</h3>
          <div className="row">
            <button onClick={() => setPicked(data.models.map((m) => m.id))}>All</button>
            <button onClick={() => setPicked(data.models.filter((m) => m.family === 'original').map((m) => m.id))}>Original only</button>
            <button onClick={() => setPicked(DEFAULT)}>Default</button>
            <button onClick={() => setPicked([])}>None</button>
          </div>
        </div>
        <div className="grid3">
          {FAMILIES.map(([fam, label]) => (
            <fieldset key={fam} className="model-group">
              <legend>{label}</legend>
              {data.models.filter((m) => m.family === fam).map((m) => (
                <label key={m.id} className="check" title={m.description}>
                  <input type="checkbox" checked={picked.includes(m.id)} onChange={() => toggle(m.id)} /> {m.name}
                </label>
              ))}
            </fieldset>
          ))}
        </div>
      </div>

      <div className="panel toolbar">
        <label>Metric
          <select value={metric} onChange={(e) => setMetric(e.target.value)}>
            {Object.entries(METRICS).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
          </select>
        </label>
        <label>Scenario
          <select value={scen} onChange={(e) => setScen(e.target.value)}>
            <option value="faulted">All faulted (mean)</option>
            {data.scenarios.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <span className="sep" />
        <span className="muted small">Mean over {data.seeds} fault-placement seeds · proxy labels · same test split for every model</span>
      </div>

      {rows.length === 0 ? <div className="panel">Select at least one model.</div> : (
        <>
          <div className="cards">
            <div className="card">
              <div className="muted small">Best model ({scenLabel})</div>
              <div className="big">{winner.m.name}</div>
              <div className="muted small">{METRICS[metric].label}: <b>{fmt(winner.sel)}</b></div>
            </div>
            <div className="card">
              <div className="muted small">Models compared</div>
              <div className="big">{rows.length}</div>
              <div className="muted small">of {data.models.length} available</div>
            </div>
            <div className="card">
              <div className="muted small">About the winner</div>
              <p className="small" style={{ margin: '4px 0 0' }}>{winner.m.description}</p>
            </div>
          </div>

          <div className="panel">
            <h3>{METRICS[metric].label}: {scenLabel}</h3>
            <RankBars rows={rows} metric={metric} scenLabel={scenLabel} />
          </div>

          <div className="panel">
            <h3>Leaderboard</h3>
            <table className="results">
              <thead><tr><th>#</th><th className="name">Model</th><th className="name">Family</th><th>Clean</th><th>Faulted mean</th><th>Worst case</th><th>{scen === 'faulted' ? 'Selected' : scen}</th></tr></thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={r.m.id}>
                    <td>{i + 1}</td><td className="name" title={r.m.description}>{r.m.name}</td><td className="name">{r.m.family}</td>
                    <td>{fmt(r.clean)}</td><td>{fmt(r.faulted)}</td><td>{fmt(r.worst)}</td>
                    <td className={i === 0 ? 'best' : ''}>{fmt(r.sel)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="muted small">Metric: {METRICS[metric].label}. Worst case = weakest faulted scenario.</p>
          </div>

          <div className="panel">
            <h3>Every scenario: {METRICS[metric].label}</h3>
            <Heatmap rows={rows} scenarios={data.scenarios} metric={metric} />
            <p className="muted small">Stronger blue = better. Bold = best in that scenario. Hover a cell for mean ± sd across seeds.</p>
          </div>
        </>
      )}
    </div>
  )
}
