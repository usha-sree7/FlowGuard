import { useEffect, useRef, useState } from 'react'
import { api, fmt, RISK, RISK_COLOR } from '../api'
import { FinalAlert } from '../Alert'
import { LineChart } from '../charts'

async function upload(file: File): Promise<any> {
  const fd = new FormData()
  fd.append('file', file)
  const res = await fetch('/api/lab/upload', { method: 'POST', body: fd })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.detail ?? res.statusText)
  return body
}

type Step = 'upload' | 'map' | 'results'
type Mapping = { timestamp: string | null; level: string; rain: string[]; step_hours: number | null }

function MappingForm({ p, onValidated }: { p: any; onValidated: (m: Mapping, r: any) => void }) {
  const [ts, setTs] = useState<string>(p.suggested_mapping.timestamp ?? '')
  const [level, setLevel] = useState<string>(p.suggested_mapping.level ?? '')
  const [rain, setRain] = useState<string[]>(p.suggested_mapping.rain ?? [])
  const [stepHours, setStepHours] = useState<string>('')
  const [busy, setBusy] = useState(false)
  const [res, setRes] = useState<any>(null)

  const toggleRain = (c: string) => setRain((r) => (r.includes(c) ? r.filter((x) => x !== c) : [...r, c]))
  const validate = () => {
    setBusy(true)
    const m: Mapping = { timestamp: ts || null, level, rain, step_hours: stepHours ? Number(stepHours) : null }
    api('/lab/map', { method: 'POST', body: JSON.stringify({ csv_text: p.csv_text, ...m }) })
      .then((r) => { setRes(r); setBusy(false); if (r.ready) onValidated(m, r) })
      .catch((e) => { setRes({ ready: false, errors: [e.message], warnings: [] }); setBusy(false) })
  }

  return (
    <div className="stack">
      <div className="panel">
        <h3>Preview: {p.filename}</h3>
        <p className="note">{p.rows.toLocaleString()} rows, {p.columns.length} columns. {p.duplicate_rows > 0 && `${p.duplicate_rows} fully duplicate rows found. `}
          {!p.enough_rows && <span className="error">Only {p.rows} rows — at least {p.min_rows_required} are needed to calibrate anything.</span>}</p>
        <div className="table-scroll">
          <table className="results">
            <thead><tr><th>Column</th><th>Missing</th><th>Numeric</th><th>Date/time</th><th>Sample</th></tr></thead>
            <tbody>{p.columns.map((c: string) => {
              const prof = p.column_profiles[c]
              return <tr key={c}><td>{c}</td><td>{prof.missing}</td><td>{Math.round(prof.numeric_rate * 100)}%</td>
                <td>{Math.round(prof.datetime_rate * 100)}%</td><td className="mono small">{prof.sample.filter((v: any) => v !== null).join(' · ')}</td></tr>
            })}</tbody>
          </table>
        </div>
      </div>

      <div className="panel">
        <h3>Map your columns</h3>
        <p className="note">Columns are suggested from their names and content — confirm or correct them. Nothing is analysed yet.</p>
        <div className="grid2">
          <label className="field">Timestamp column (leave blank to use a fixed interval instead)
            <select value={ts} onChange={(e) => setTs(e.target.value)}>
              <option value="">(none — fixed interval)</option>
              {p.columns.map((c: string) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
          <label className="field">Sampling interval in hours{ts ? ' (optional cross-check)' : ' (required — no timestamp mapped)'}
            <input type="number" min="0" step="any" value={stepHours} onChange={(e) => setStepHours(e.target.value)} placeholder={ts ? 'auto-detected from timestamps' : 'e.g. 1'} />
          </label>
          <label className="field">Water level column (required)
            <select value={level} onChange={(e) => setLevel(e.target.value)}>
              <option value="">(choose a column)</option>
              {p.columns.map((c: string) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
          <div className="field">Rainfall column(s) (at least one required)
            <div className="chips">
              {p.columns.map((c: string) => <button key={c} type="button" className={`chip small ${rain.includes(c) ? 'on' : ''}`} onClick={() => toggleRain(c)}>{c}</button>)}
            </div>
          </div>
        </div>
        <button style={{ marginTop: 12 }} disabled={!level || !p.enough_rows || busy} onClick={validate}>{busy ? 'Validating…' : 'Validate'}</button>
      </div>

      {res && (
        <div className={`panel ${res.ready ? '' : 'error'}`}>
          {res.errors?.length > 0 && <><b>Cannot proceed:</b><ul className="small">{res.errors.map((e: string, i: number) => <li key={i}>{e}</li>)}</ul></>}
          {res.warnings?.length > 0 && <><b>Warnings (data was NOT silently dropped beyond what is listed here):</b><ul className="small">{res.warnings.map((w: string, i: number) => <li key={i}>{w}</li>)}</ul></>}
          {res.ready && <p className="small" style={{ marginBottom: 0 }}>Validated: {res.rows.toLocaleString()} standardised rows. Continue below to run the analysis.</p>}
        </div>
      )}
    </div>
  )
}

function Results({ csvText, filename, mapping, report }: { csvText: string; filename: string; mapping: Mapping; report: any }) {
  const [h, setH] = useState(Object.keys(report.horizons)[0] ?? '1')
  const [r, setR] = useState<any>(null)
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)
  const latest = useRef(0)
  const run = () => {
    const token = ++latest.current
    setBusy(true); setErr('')
    api('/lab/analyze', { method: 'POST', body: JSON.stringify({ csv_text: csvText, filename, ...mapping, horizon_steps: Number(h) }) })
      .then((x) => { if (latest.current === token) { setR(x); setBusy(false) } })
      .catch((e) => { if (latest.current === token) { setErr(e.message); setBusy(false) } })
  }
  useEffect(() => { run() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="stack">
      <div className="panel">
        <h3>Run analysis</h3>
        <div className="controls">
          <label className="ctl">Forecast horizon
            <div className="chips">{Object.entries(report.horizons).map(([k, n]: any) => <button key={k} className={`chip small ${h === k ? 'on' : ''}`} onClick={() => setH(k)}>{n}</button>)}</div>
          </label>
          <button disabled={busy} onClick={run}>{busy ? 'Running…' : 'Run analysis'}</button>
        </div>
      </div>

      {err && <div className="panel error">{err}</div>}
      {r && (
        <>
          <div className="panel callout">
            <b>Only the primary model was run.</b>
            <p className="small" style={{ margin: '4px 0 0' }}>{r.models_not_run_reason}</p>
          </div>

          <div className="panel callout">
            <b>This alert is for ONE reading, not a summary of your whole file.</b>
            <p className="small" style={{ margin: '4px 0 0' }}>Most recent test-period row: <b>{r.primary.timestamp}</b> (row {r.primary.row_index} of {r.primary.test_rows} in the test split).
              Scroll down to see the full series and where this point falls in it.</p>
          </div>

          <FinalAlert e={r.primary} />

          <div className="panel">
            <h3>What was done</h3>
            <table className="kv"><tbody>
              <tr><th>Rows</th><td>{r.split.total_rows.toLocaleString()} total → {r.split.train_rows.toLocaleString()} train / {r.split.test_rows.toLocaleString()} test (chronological, no shuffling)</td></tr>
              <tr><th>Horizon</th><td>{r.horizon_label} ahead, {r.rain_window_hours} h rainfall accumulation</td></tr>
              <tr><th>Level thresholds (train split)</th><td>{r.thresholds.map((t: number, i: number) => `${RISK[i + 1]} ≥ ${fmt(t, 3)}`).join(' · ')} (train mean {fmt(r.train_level_mean, 3)}, sd {fmt(r.train_level_sd, 3)})</td></tr>
              <tr><th>Level bounds used by the reliability checks</th><td>[{fmt(r.level_bounds[0], 2)}, {fmt(r.level_bounds[1], 2)}] — data-derived from this upload, NOT a verified physical limit</td></tr>
            </tbody></table>
            <p className="muted small" style={{ marginTop: 8 }}>{r.thresholds_note}</p>
          </div>

          <div className="panel">
            <h3>Evaluation on the held-out test split</h3>
            {!r.has_alert_in_test ? (
              <p className="note">{r.metrics_note}</p>
            ) : (
              <div className="table-scroll">
                <table className="results">
                  <thead><tr><th>Inputs</th><th>Alert F1</th><th>Precision</th><th>Recall</th><th>Macro F1 (5-level)</th><th>κ</th></tr></thead>
                  <tbody>
                    <tr><td>Raw</td><td>{fmt(r.metrics.raw.binary_alert.f1)}</td><td>{fmt(r.metrics.raw.binary_alert.precision)}</td><td>{fmt(r.metrics.raw.binary_alert.recall)}</td><td>{fmt(r.metrics.raw.ordinal.macro_f1)}</td><td>{fmt(r.metrics.raw.ordinal.kappa)}</td></tr>
                    <tr><td>Reliability-fused</td><td>{fmt(r.metrics.fused.binary_alert.f1)}</td><td>{fmt(r.metrics.fused.binary_alert.precision)}</td><td>{fmt(r.metrics.fused.binary_alert.recall)}</td><td>{fmt(r.metrics.fused.ordinal.macro_f1)}</td><td>{fmt(r.metrics.fused.ordinal.kappa)}</td></tr>
                  </tbody>
                </table>
              </div>
            )}
            <p className="muted small" style={{ marginTop: 8 }}>Labels are proxy labels (the level actually observed {r.horizon_label} later vs. thresholds from this upload's own training split) — not observed flood events. No confidence interval is computed here (the curated datasets' bootstrap CI needs more test data than most uploads will have).</p>
          </div>

          <div className="panel">
            <h3>Water level over the full uploaded series</h3>
            <LineChart height={200} yLabel="level" xLabels={[r.series.timestamp[0] || 'start', r.series.timestamp[r.series.timestamp.length - 1] || 'end']}
              marks={r.series.level.map((_: any, i: number) => i >= r.series.train_end_index)} markLabel="test period"
              series={[{ values: r.series.level, color: 'var(--accent)', label: 'water level', width: 1.2 }]}
              hlines={r.thresholds.map((t: number, i: number) => ({ y: t, label: RISK[i + 1], color: RISK_COLOR[i + 1] }))} />
            <LineChart height={130} yLabel="rain" xLabels={[r.series.timestamp[0] || 'start', r.series.timestamp[r.series.timestamp.length - 1] || 'end']}
              series={Object.entries(r.series.rain).map(([k, v]: any, i) => ({ values: v, color: ['var(--rain)', 'var(--m-pso)'][i % 2], label: k, width: 1 }))} />
          </div>

          <div className="panel callout warn">
            <b>{r.disclaimer}</b>
          </div>
        </>
      )}
    </div>
  )
}

export default function DatasetLab() {
  const [maxMb, setMaxMb] = useState<number | null>(null)
  const [step, setStep] = useState<Step>('upload')
  const [preview, setPreview] = useState<any>(null)
  const [mapping, setMapping] = useState<Mapping | null>(null)
  const [report, setReport] = useState<any>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  useEffect(() => { api('/health').then((h) => setMaxMb(h.lab_max_upload_mb)).catch(() => setMaxMb(null)) }, [])

  const onFile = (f: File | undefined) => {
    if (!f) return
    setErr(''); setBusy(true)
    upload(f).then((p) => { setPreview(p); setStep('map'); setBusy(false) })
      .catch((e) => { setErr(e.message); setBusy(false) })
  }

  const reset = () => { setStep('upload'); setPreview(null); setMapping(null); setReport(null); setErr('') }

  return (
    <div className="stack">
      <div className="panel">
        <h2>Dataset Lab</h2>
        <p className="note">Upload your own river-level + rainfall CSV, map its columns, validate it, and run the primary flood-risk
          model (IT2FIS, reliability-fused) on it — using the same calibration and fuzzy-inference code as the built-in datasets.
          Nothing is written to this server's disk: each request carries its own data and nothing is saved between requests, so an
          upload can never overwrite the Bengalas, Malwathu or Chindnar datasets or their results.</p>
        <p className="muted small">Workflow: Upload → Preview → Map columns → Validate → Run analysis → View results. CSV only,
          up to {maxMb ?? '…'} MB on this server. Only IT2FIS runs here — PSO-tuned IT2FIS, SOM and ANFIS need iterative
          training and are not run for ad-hoc uploads (see the note after analysis).</p>
        <div className="row" style={{ marginTop: 10 }}>
          <label className="big-btn" style={{ cursor: 'pointer' }}>
            {busy ? 'Uploading…' : 'Choose CSV file'}
            <input type="file" accept=".csv" style={{ display: 'none' }} disabled={busy}
              onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = '' }} />
          </label>
          {step !== 'upload' && <button className="btn-ghost" onClick={reset}>Start over with a new file</button>}
        </div>
        {err && <div className="error" style={{ marginTop: 10 }}>{err}</div>}
      </div>

      {step === 'map' && preview && <MappingForm p={preview} onValidated={(m, r) => { setMapping(m); setReport(r); setStep('results') }} />}
      {step === 'results' && report && mapping && <Results csvText={preview.csv_text} filename={preview.filename} mapping={mapping} report={report} />}
    </div>
  )
}
