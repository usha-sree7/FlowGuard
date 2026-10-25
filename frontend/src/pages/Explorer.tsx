import { useCallback, useEffect, useState } from 'react'
import { api, fmt } from '../api'
import { LineChart } from '../charts'

function Series({ id }: { id: string }) {
  const [s, setS] = useState<any>(null)
  useEffect(() => { api(`/datasets/${id}/series`).then(setS).catch(() => setS(null)) }, [id])
  if (!s) return null
  const split = Math.floor(s.train_end_index / s.step)
  const marks = s.index.map((_: number, i: number) => i >= split)
  return (
    <LineChart height={170} yLabel="level"
      series={[{ values: s.level, color: 'var(--accent)', label: `water level [m] (every ${s.step}th of ${s.n} rows)`, width: 1 }]}
      marks={marks} markLabel="chronological test split (last 30%)"
      xLabels={s.timestamp.map((t: string, i: number) => t || `day ${s.index[i]}`)} />
  )
}

function Quality({ q }: { q: any }) {
  const rows: [string, any][] = [
    ['Valid records loaded', q.valid_rows.toLocaleString()],
    ['Rows', q.rows.toLocaleString()],
    ['First / last', `${q.first} → ${q.last}`],
    ['Nominal step', `${q.nominal_step_hours} h (${q.steps_at_nominal.toLocaleString()} steps)`],
    ['Timestamp gaps', `${q.gaps} (largest ${fmt(q.largest_gap_hours, 1)} h)`],
    ['Duplicate timestamps', q.duplicate_timestamps],
    ['Missing values', Object.entries(q.missing_values).map(([k, v]) => `${k}: ${v}`).join(', ')],
    ['Level outside plausibility range', q.invalid_level_outside_bounds],
    ['Negative rain', Object.values(q.invalid_negative_rain).reduce((a: number, b: any) => a + b, 0)],
    ['Rise rate computable', `${q.rise_rate_available.toLocaleString()} rows (not across gaps)`],
    ['Level range [m]', `${fmt(q.stats.level.min, 2)} – ${fmt(q.stats.level.max, 2)} (mean ${fmt(q.stats.level.mean, 3)})`],
    ['Raw SHA-256', Object.entries(q.raw_sha256 ?? {}).map(([k, v]: any) => `${k}: ${v.slice(0, 16)}…`).join(', ')],
    ['Imported', q.imported_at],
  ]
  return (
    <table className="kv"><tbody>{rows.map(([k, v]) => <tr key={k}><th>{k}</th><td>{String(v)}</td></tr>)}</tbody></table>
  )
}

export default function Explorer({ onChanged }: { onChanged: () => void }) {
  const [info, setInfo] = useState<any>(null)
  const [jobs, setJobs] = useState<Record<string, any>>({})
  const refresh = useCallback(() => api('/datasets').then(setInfo).catch(() => {}), [])
  useEffect(() => { refresh() }, [refresh])

  useEffect(() => {
    const running = Object.entries(jobs).filter(([, j]) => j?.state === 'running')
    if (!running.length) return
    const t = window.setInterval(async () => {
      for (const [id] of running) {
        const j = await api(`/datasets/${id}/import`)
        setJobs((p) => ({ ...p, [id]: j }))
        if (j.state !== 'running') { refresh(); onChanged() }
      }
    }, 1000)
    return () => window.clearInterval(t)
  }, [jobs, refresh, onChanged])

  const start = async (id: string) => {
    const j = await api(`/datasets/${id}/import`, { method: 'POST' })
    setJobs((p) => ({ ...p, [id]: j }))
  }

  if (!info) return <div className="panel">Loading datasets…</div>
  return (
    <div className="stack">
      {info.datasets.map((d: any) => {
        const job = jobs[d.id] ?? d.import_job
        return (
          <section className="panel" key={d.id}>
            <div className="row-between">
              <h2>{d.name}</h2>
              <div className="row">
                <span className={`tag ${d.loaded ? 'ok' : 'warn'}`}>{d.loaded ? 'Loaded & validated' : 'Not imported'}</span>
                <button onClick={() => start(d.id)} disabled={job?.state === 'running'}>
                  {job?.state === 'running' ? 'Importing…' : d.loaded ? 'Re-import dataset' : 'Import dataset'}
                </button>
              </div>
            </div>
            {job?.state === 'running' && <p className="note">{job.message}</p>}
            {job?.state === 'failed' && (
              <div className="error"><b>Import failed:</b> {job.message}<pre>{job.manual}</pre></div>
            )}
            <table className="kv"><tbody>
              <tr><th>Paper</th><td><i>{d.paper_title}</i><br />{d.paper_ref}</td></tr>
              <tr><th>DOI</th><td><a href={`https://doi.org/${d.doi}`} target="_blank" rel="noreferrer">{d.doi}</a></td></tr>
              <tr><th>Repository</th><td><a href={d.repo} target="_blank" rel="noreferrer">{d.repo}</a> @ <code>{d.repo_commit.slice(0, 10)}</code></td></tr>
              <tr><th>Licence</th><td>{d.licence}</td></tr>
              <tr><th>Location</th><td>{d.location}</td></tr>
              <tr><th>Period</th><td>{d.period_claimed}</td></tr>
              <tr><th>Resolution</th><td>{d.resolution}</td></tr>
              <tr><th>Rain sensors</th><td>{d.rain_sensors.join(', ')}</td></tr>
              <tr><th>Labels</th><td><span className="tag warn">{d.label_info.true_flood_labels ? 'observed' : 'PROXY labels only'}</span> {d.label_info.description}</td></tr>
            </tbody></table>
            <details><summary>Column mapping (original → standard field)</summary>
              <table className="kv"><tbody>{Object.entries(d.column_mapping).map(([k, v]: any) => <tr key={k}><th><code>{k}</code></th><td>{v}</td></tr>)}</tbody></table>
            </details>
            {d.loaded && d.quality && (
              <>
                <h3>Data quality (computed from the files actually read)</h3>
                <Quality q={d.quality} />
                <details><summary>Processing steps</summary><ol>{d.quality.processing_steps.map((s: string) => <li key={s}>{s}</li>)}</ol></details>
                <Series id={d.id} />
              </>
            )}
            <h3>Limitations</h3>
            <ul>{d.limitations.map((l: string) => <li key={l}>{l}</li>)}</ul>
            {d.unverified.length > 0 && (<><h3>Needs manual verification</h3><ul>{d.unverified.map((l: string) => <li key={l}>{l}</li>)}</ul></>)}
          </section>
        )
      })}
      <section className="panel">
        <h2>Audited but not integrated</h2>
        <table className="kv"><tbody>{info.not_integrated.map((x: any) => (
          <tr key={x.name}><th>{x.name}</th><td>{x.ref}<br /><span className="muted">{x.status}</span></td></tr>
        ))}</tbody></table>
      </section>
    </div>
  )
}
