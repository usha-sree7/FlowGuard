import { useCallback, useEffect, useState } from 'react'
import { api, fmt } from '../api'

const SYSTEMS = ['threshold', 't1_fis', 'it2_fis', 'proposed']
const NAMES: Record<string, string> = { threshold: 'Threshold', t1_fis: 'Type-1 FIS', it2_fis: 'IT2FIS', proposed: 'Reliability fusion + IT2FIS' }

function ResultTable({ res, metric }: { res: any; metric: string }) {
  const get = (ev: any) => {
    const b = ev.all.binary_alert
    const o = ev.all.ordinal
    return ({ f1: b.f1, precision: b.precision, recall: b.recall, fpr: b.fpr, swe: o.severity_weighted_error, macro_f1: o.macro_f1 } as any)[metric]
  }
  const lowerBetter = metric === 'fpr' || metric === 'swe'
  return (
    <table className="results">
      <thead><tr><th>Scenario</th>{SYSTEMS.map((s) => <th key={s}>{NAMES[s]}</th>)}</tr></thead>
      <tbody>
        {Object.entries(res.results).map(([scen, r]: any) => {
          const vals = SYSTEMS.map((s) => get(r.systems[s]))
          const best = lowerBetter ? Math.min(...vals.filter((v) => v !== null)) : Math.max(...vals.filter((v) => v !== null))
          return (
            <tr key={scen}><td>{scen}</td>{vals.map((v, i) => <td key={i} className={v === best ? 'best' : ''}>{fmt(v)}</td>)}</tr>
          )
        })}
      </tbody>
    </table>
  )
}

export default function Experiments({ dataset }: { dataset: string }) {
  const [runs, setRuns] = useState<any[]>([])
  const [sel, setSel] = useState('')
  const [res, setRes] = useState<any>(null)
  const [metric, setMetric] = useState('f1')
  const [job, setJob] = useState<any>(null)
  const [mode, setMode] = useState('full')
  const [seed, setSeed] = useState(20261025)
  const [err, setErr] = useState('')

  const refresh = useCallback(() => api('/experiments').then((r) => {
    const mine = r.runs.filter((x: any) => x.dataset === dataset)
    setRuns(mine)
    // Default to the newest FULL-mode run of the current design (sample runs recalibrate on a short window).
    const full = mine.filter((x: any) => x.mode === 'full' && !x.run_id.includes('v0.3'))
    const headline = full.find((x: any) => x.seed === 20261025) ?? full[0]
    setSel((s) => (mine.some((x: any) => x.run_id === s) ? s : headline?.run_id ?? mine[0]?.run_id ?? ''))
  }), [dataset])
  useEffect(() => { refresh() }, [refresh])
  useEffect(() => { if (sel) api(`/experiments/${sel}`).then(setRes) }, [sel])

  useEffect(() => {
    if (job?.state !== 'running') return
    const t = window.setInterval(async () => {
      const r = await api('/experiments')
      const j = r.jobs[job.job]
      if (j) setJob({ ...j, job: job.job })
      if (j?.state === 'done') { await refresh(); setSel(j.run_id) }
    }, 1000)
    return () => window.clearInterval(t)
  }, [job, refresh])

  const start = async () => {
    try {
      setErr('')
      setJob(await api('/experiments', { method: 'POST', body: JSON.stringify({ dataset, mode, seed }) }))
    } catch (e: any) { setErr(e.message) }
  }

  const meta = res?.meta
  return (
    <div className="stack">
      <div className="panel toolbar">
        <label>Run
          <select value={sel} onChange={(e) => setSel(e.target.value)}>
            {runs.map((r) => <option key={r.run_id} value={r.run_id}>{r.run_id}</option>)}
          </select>
        </label>
        <label>Metric
          <select value={metric} onChange={(e) => setMetric(e.target.value)}>
            <option value="f1">Alert F1 (≥ Warning)</option><option value="precision">Precision</option>
            <option value="recall">Recall</option><option value="fpr">False-positive rate</option>
            <option value="swe">Severity-weighted error</option><option value="macro_f1">4-class macro F1</option>
          </select>
        </label>
        <span className="sep" />
        <label>Mode<select value={mode} onChange={(e) => setMode(e.target.value)}><option value="sample">sample (10k rows)</option><option value="full">full</option></select></label>
        <label>Seed<input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} style={{ width: 110 }} /></label>
        <button onClick={start} disabled={job?.state === 'running'}>{job?.state === 'running' ? `Running: ${job.message}` : 'Run new experiment'}</button>
      </div>
      {err && <div className="error">{err}</div>}
      {job?.state === 'failed' && <div className="error">Experiment failed: {job.message}</div>}
      {!res && <div className="panel">No experiment run yet for this dataset.</div>}
      {res && (
        <>
          <div className="panel">
            <h2>{meta.run_id}</h2>
            {meta.run_id.includes('v0.3') && <p className="error">Superseded design (v0.3, risk decided by the KM midpoint of interval inputs). Kept as a documented negative result.</p>}
            <table className="kv"><tbody>
              <tr><th>Dataset / mode</th><td>{meta.dataset} / {meta.mode} · {meta.rows_used.toLocaleString()} rows · seed {meta.seed} · model v{meta.model_version}</td></tr>
              <tr><th>Chronological split</th><td>train {meta.train.rows.toLocaleString()} ({meta.train.first} → {meta.train.last}) · test {meta.test.rows.toLocaleString()} ({meta.test.first} → {meta.test.last})</td></tr>
              <tr><th>Labels</th><td>{meta.labels.kind}; horizon {meta.labels.horizon_hours} h; test counts {Object.entries(meta.labels.test_class_counts).map(([k, v]) => `${k} ${v}`).join(', ')}</td></tr>
              <tr><th>Processed data SHA-256</th><td className="mono small">{meta.processed_sha256}</td></tr>
            </tbody></table>
          </div>
          <div className="panel">
            <h3>Flood-risk performance on identical corrupted inputs (best per row highlighted)</h3>
            <ResultTable res={res} metric={metric} />
            <p className="muted small">Labels are proxies (future observed level vs. statistical thresholds), not observed flood impacts.</p>
          </div>
          <div className="grid2">
            <div className="panel">
              <h3>Uncertainty-index response (proposed)</h3>
              <table className="results"><thead><tr><th>Scenario</th><th>U faulted</th><th>U unfaulted</th><th>Confident / Uncertain / Insufficient</th><th>Confident while level missing</th></tr></thead>
                <tbody>{Object.entries(res.results).map(([s, r]: any) => {
                  const u = r.uncertainty
                  const c = u.status_counts
                  return <tr key={s}><td>{s}</td><td>{fmt(u.U_mean_faulted_steps, 1)}</td><td>{fmt(u.U_mean_unfaulted_steps, 1)}</td>
                    <td>{c.Confident ?? 0} / {c.Uncertain ?? 0} / {c['Insufficient data'] ?? 0}</td><td>{u.confident_while_level_missing}</td></tr>
                })}</tbody></table>
            </div>
            <div className="panel">
              <h3>Fault detection (reliability &lt; 0.5) — level sensor</h3>
              <table className="results"><thead><tr><th>Scenario</th><th>Injected</th><th>Flagged</th><th>Precision</th><th>Recall</th></tr></thead>
                <tbody>{Object.entries(res.results).map(([s, r]: any) => {
                  const d = r.fault_detection.level
                  return d.false_flag_rate !== undefined
                    ? <tr key={s}><td>{s}</td><td colSpan={4}>false-flag rate on clean data: {fmt(d.false_flag_rate, 4)}</td></tr>
                    : <tr key={s}><td>{s}</td><td>{d.injected}</td><td>{d.flagged}</td><td>{fmt(d.precision)}</td><td>{fmt(d.recall)}</td></tr>
                })}</tbody></table>
            </div>
          </div>
          <div className="panel">
            <h3>Processing time (ms per observation, clean scenario)</h3>
            <p className="mono">{SYSTEMS.map((s) => `${NAMES[s]}: ${fmt(res.results.clean?.ms_per_obs?.[s], 4)}`).join(' · ')}</p>
          </div>
        </>
      )}
    </div>
  )
}
