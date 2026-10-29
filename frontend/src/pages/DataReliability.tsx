import { useEffect, useState } from 'react'
import { api, fmt, MODEL_COLOR, MODELS, RISK, RISK_COLOR } from '../api'
import { GroupedBars, LineChart, Strip } from '../charts'
import { defaultHorizon, useComparison, useDatasetSpec } from '../data'

const FAULTS: [string, string, [string, string][]][] = [
  ['missing', 'Missing readings', [['missing_10', '10 %'], ['missing_30', '30 %']]],
  ['noise', 'Noisy readings', [['noise_low', 'low'], ['noise_med', 'medium'], ['noise_high', 'high']]],
  ['stuck', 'Stuck sensor', [['stuck', '']]],
  ['conflict', 'Conflicting rain sensors', [['conflict', '']]],
  ['spikes', 'Spikes', [['spikes', '']]],
  ['combined', 'Several at once', [['combined', '']]],
]

function DatasetCard({ dataset, d }: { dataset: string; d: any }) {
  const spec = useDatasetSpec(dataset)
  const [s, setS] = useState<any>(null)
  useEffect(() => { api(`/datasets/${dataset}/series?max_points=1200`).then(setS).catch(() => setS(null)) }, [dataset])
  if (!spec) return <div className="panel">Loading dataset description…</div>
  const q = spec.quality
  const meta = d?.meta
  const xl = s ? [s.timestamp[0] || 'day 0', s.timestamp[s.timestamp.length - 1] || `day ${s.n - 1}`] : []
  const trainMark = s ? s.index.map((i: number) => i >= s.train_end_index) : undefined
  return (
    <>
      <div className="panel">
        <h2>{spec.name}</h2>
        <table className="kv"><tbody>
          <tr><th>{spec.doi ? 'Published with' : 'Publisher'}</th><td>{spec.paper_ref}. <i>{spec.paper_title}</i>.{spec.doi && <> <a href={`https://doi.org/${spec.doi}`} target="_blank" rel="noreferrer">doi:{spec.doi}</a></>}</td></tr>
          <tr><th>Source / download</th><td><a href={spec.repo} target="_blank" rel="noreferrer">{spec.repo}</a> ({spec.doi ? <>commit <code>{spec.repo_commit.slice(0, 10)}</code></> : spec.repo_commit}) · {spec.files.length > 1 ? 'files' : 'file'} {spec.files.map((f: any, i: number) => <span key={f.name}>{i > 0 && ', '}<a href={f.url}>{f.name}</a></span>)}, SHA-256 checked on import · {spec.licence}</td></tr>
          <tr><th>Location</th><td>{spec.location}</td></tr>
          <tr><th>Timestamps</th><td>{spec.period_claimed}{spec.step_hours >= 24 ? '' : '. Timezone not stated in the source; treated as local time.'}</td></tr>
          <tr><th>Sampling</th><td>{spec.resolution}{q ? ` · ${q.rows.toLocaleString()} records` : ''}</td></tr>
          <tr><th>Variables and units</th><td><ul className="small" style={{ margin: 0, paddingLeft: 18 }}>
            {Object.entries(spec.column_mapping).filter(([k]) => k !== 'source_file').map(([k, v]: any) => <li key={k}><code>{k}</code>: {v}</li>)}
            <li><code>rise_rate</code> (derived): level change per hour over exactly one step [m/h], never computed across a gap.</li>
            {meta && <li><code>rain</code> model input (derived): rain accumulated over the last {meta.rain_window_hours} h.</li>}
          </ul></td></tr>
          {q && <tr><th>Data quality</th><td><ul className="small" style={{ margin: 0, paddingLeft: 18 }}>
            <li>{q.rows.toLocaleString()} records, {q.steps_at_nominal.toLocaleString()} exactly one step apart; {q.gaps.toLocaleString()} gaps (largest {fmt(q.largest_gap_hours, 1)} h); {q.duplicate_timestamps} duplicate timestamps.</li>
            <li>Missing values after import: {Object.entries(q.missing_values).map(([k, v]: any) => `${k} ${v}`).join(', ')}.</li>
            {q.reader_checks?.level_removed_outside_bounds !== undefined && <li>Removed as implausible: {q.reader_checks.level_removed_outside_bounds} level readings outside {q.reader_checks.level_bounds_m.join('–')} m; {q.reader_checks.rain_records_above_limit_set_missing} rain readings above {q.reader_checks.rain_limit_mm_per_h} mm/h set to missing.</li>}
            {q.reader_checks?.rain_hours_assumed_zero !== undefined && <li>{q.reader_checks.rain_hours_assumed_zero.toLocaleString()} of {q.rows.toLocaleString()} hours have no rain record and are assumed dry (0 mm): the source lists non-zero rain only.</li>}
          </ul></td></tr>}
          <tr><th>Label generation</th><td>
            <b>No recorded flood events exist in this dataset</b>, so labels are <b>proxy labels</b>. The label for time t at horizon h is the alert level of the water level <i>actually observed</i> at t + h,
            using thresholds computed from the training period only{meta ? `: ${RISK.slice(1).map((r, i) => `${r} ≥ ${fmt(meta.thresholds_m[i], 3)} m (mean + ${meta.level_k[i]} sd)`).join(', ')}` : ''}.
            {' '}{spec.label_info.description}
          </td></tr>
          {meta && <tr><th>Chronological split</th><td>Fit {meta.train.first} → {meta.train.fit_last} ({meta.train.fit_rows.toLocaleString()} rows) · validation → {meta.train.last} ({meta.train.validation_rows.toLocaleString()}) · <b>test {meta.test.first} → {meta.test.last}</b> ({meta.test.rows.toLocaleString()}). No shuffling.</td></tr>}
          <tr><th>Known limitations</th><td><ul className="small" style={{ margin: 0, paddingLeft: 18 }}>{spec.limitations.map((l: string) => <li key={l}>{l}</li>)}</ul></td></tr>
        </tbody></table>
      </div>

      {s && (
        <div className="panel">
          <h3>Recorded parameters, full period (shaded = test period)</h3>
          <LineChart height={200} yLabel="water level [m]" xLabels={xl} marks={trainMark} markLabel="test period"
            series={[{ values: s.level, color: 'var(--accent)', label: 'water level [m]', width: 1.2 }]}
            hlines={meta ? meta.thresholds_m.map((t: number, i: number) => ({ y: t, label: RISK[i + 1], color: RISK_COLOR[i + 1] })) : []} />
          <LineChart height={140} yLabel="rain" xLabels={xl}
            series={Object.entries(s.rain).slice(0, 3).map(([k, v]: any, i) => ({ values: v, color: ['var(--rain)', 'var(--m-pso)', 'var(--risk2)'][i], label: `rain ${k} (max per plotted point)`, width: 1 }))} />
          <LineChart height={140} yLabel="rise [m/h]" xLabels={xl} series={[{ values: s.rise, color: 'var(--m-anfis)', label: 'rate of rise [m/h] (max per plotted point)', width: 1 }]} />
          <p className="muted small">Down-sampled to {s.level.length} points (1 of every {s.step} records){Object.keys(s.rain).length > 3 ? `; rain shows the first 3 of ${Object.keys(s.rain).length} gauges` : ''}; rain and rise show the maximum inside each plotted interval so short peaks stay visible.</p>
        </div>
      )}
    </>
  )
}

function HorizonCheck({ d }: { d: any }) {
  const step = d.meta.step_hours
  return (
    <div className="panel">
      <h2>Does the data support 30-min, 1-h and 3-h forecasting?</h2>
      <p className="note">Checked before any horizon is offered. A horizon is usable only if (1) it is a whole number of sampling steps,
        (2) enough future readings exist to label it, and (3) the river actually changes over that time, so the forecast is not trivial.</p>
      <table className="results" style={{ marginTop: 8 }}>
        <thead><tr><th>Horizon</th><th>Steps of {step < 1 ? `${step * 60} min` : `${step} h`}</th><th>Labelled train / test rows</th><th>Test alerts</th><th>Steps whose level class changes</th><th>New alert onsets</th><th>Mean |level change|</th><th>Persistence MAE</th><th>ARX MAE</th><th>Verdict</th></tr></thead>
        <tbody>
          {Object.entries(d.support).map(([h, s]: any) => (
            <tr key={h}><td>{s.label}</td><td>{s.steps}</td><td>{s.train_labelled.toLocaleString()} / {s.test_labelled.toLocaleString()}</td><td>{s.test_alerts}</td>
              <td>{fmt(s.class_changes_share * 100, 1)} %</td><td>{s.new_alert_onsets}</td><td>{fmt(s.level_change_abs_mean_m, 3)} m</td>
              <td>{fmt(d.forecast[h].persistence.mae, 4)} m</td><td>{fmt(d.forecast[h].arx.mae, 4)} m</td><td className="best">Supported</td></tr>
          ))}
          {d.meta.unsupported_horizons.map((u: string) => (
            <tr key={u}><td>{u}</td><td>—</td><td colSpan={7} className="muted">Not a whole number of {step} h steps: the data have no readings at that spacing, so it cannot be labelled or evaluated.</td><td style={{ color: 'var(--risk4)' }}>Not supported</td></tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">"Steps whose level class changes": share of test steps where the level class h ahead differs from now (the part a forecast must get right).
        Persistence = "the level stays as it is"; ARX = a linear auto-regressive level forecaster fitted on the fit block. Both are continuous water-level forecasts, shown as a sanity check of predictability.</p>
    </div>
  )
}

function FaultLab({ dataset, d }: { dataset: string; d: any }) {
  const [fam, setFam] = useState('missing')
  const [scen, setScen] = useState('missing_30')
  const [h, setH] = useState(defaultHorizon(d))
  const [r, setR] = useState<any>(null)
  const [eps, setEps] = useState<any[]>([])
  const meta = d.meta
  useEffect(() => { api(`/episodes/${dataset}?limit=1`).then((x) => setEps(x.episodes)).catch(() => setEps([])) }, [dataset])
  const W = meta.step_hours >= 24 ? 60 : 144
  useEffect(() => {
    if (!eps.length) return
    let live = true
    const off = Math.max(0, eps[0].start - Math.round(W * 0.6))
    Promise.all(['raw', 'fused'].map((m) => api(`/replay/${dataset}?scenario=${scen}&offset=${off}&limit=${W}&horizon=${h}&mode=${m}`)))
      .then(([a, b]) => live && setR({ raw: a, fused: b })).catch(() => live && setR(null))
    return () => { live = false }
  }, [dataset, scen, h, eps, W])
  const variants = FAULTS.find((f) => f[0] === fam)![2]
  const f1 = (s: string, mode: string, m: string) => d.risk[`${h}|${s}|${mode}|${m}`]
  const det = d.detection[scen]
  const faulted = meta.scenarios.filter((s: string) => s !== 'clean')

  return (
    <>
      <div className="panel">
        <span className="demo-tag">SIMULATED FAULTS · injected into a copy of the test data with fixed seeds · labels always come from the original data</span>
        <h2>Sensor reliability: what happens when sensors fail?</h2>
        <p className="note">Pick a fault. All four models receive exactly the same corrupted readings. With the reliability layer <b>off</b>, the models use the readings as they are.
          With it <b>on</b>, every reading is first checked (missing, spike, stuck, noisy, out-of-range, disagreement between rain sensors), down-weighted by its reliability and fused.</p>
        <div className="chips" style={{ marginTop: 10 }}>
          {FAULTS.map(([k, n, v]) => <button key={k} className={`chip ${fam === k ? 'on' : ''}`} onClick={() => { setFam(k); setScen(v[v.length - 1][0]) }}>{n}</button>)}
        </div>
        <div className="controls" style={{ marginTop: 10 }}>
          {variants.length > 1 && <label className="ctl">Severity<div className="chips">{variants.map(([k, n]) => <button key={k} className={`chip small ${scen === k ? 'on' : ''}`} onClick={() => setScen(k)}>{n}</button>)}</div></label>}
          <label className="ctl">Horizon<div className="chips">{Object.entries(meta.horizons).map(([k, n]: any) => <button key={k} className={`chip small ${h === k ? 'on' : ''}`} onClick={() => setH(k)}>{n}</button>)}</div></label>
        </div>
        <p className="small" style={{ marginTop: 8 }}><b>Definition:</b> {meta.scenario_definitions[scen]}</p>
      </div>

      <div className="panel">
        <h3>What the models see around the largest test-period flood</h3>
        {!r ? <p className="muted">Loading…</p> : (
          <>
            <LineChart height={220} xLabels={r.raw.timestamp} marks={r.raw.injected_fault} markLabel="injected fault"
              series={[
                { values: r.raw.level_original, color: 'var(--fg)', label: 'recorded level (truth)', width: 1.2, dash: true },
                { values: r.raw.level_used, color: 'var(--risk3)', label: 'raw input (layer off)', width: 1.3 },
                { values: r.fused.level_used, color: 'var(--accent)', label: 'fused input (layer on)', width: 2 },
              ]}
              hlines={r.raw.thresholds.slice(2).map((t: number, i: number) => ({ y: t, label: RISK[i + 3], color: RISK_COLOR[i + 3] }))} />
            <div className="strip-wrap">
              <Strip name={`Actual level ${r.raw.horizon_label} later`} values={r.raw.proxy_label} colors={RISK_COLOR} />
              {MODELS.map((m) => (
                <div key={m} className="stack" style={{ gap: 2 }}>
                  <Strip name={`${meta.models[m]} · off`} values={r.raw.models[m]} colors={RISK_COLOR} />
                  <Strip name={`${meta.models[m]} · on`} values={r.fused.models[m]} colors={RISK_COLOR} />
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      <div className="panel">
        <h3>Effect on test-period scores ({meta.horizons[h]} ahead, mean ± sd over {meta.seeds.length} seeds)</h3>
        <div className="table-scroll">
          <table className="results">
            <thead><tr><th>Model</th><th>F1 clean</th><th>F1 with fault · layer off</th><th>Change</th><th>F1 with fault · layer on</th><th>Change</th><th>Missed alerts off / on</th><th>False alarms off / on</th></tr></thead>
            <tbody>{MODELS.map((m) => {
              const c = f1('clean', 'raw', m), a = f1(scen, 'raw', m), b = f1(scen, 'fused', m)
              const delta = (x: number) => <span style={{ color: x < -0.005 ? 'var(--risk4)' : x > 0.005 ? 'var(--risk0)' : 'var(--muted)' }}>{x >= 0 ? '+' : ''}{fmt(x)}</span>
              return (
                <tr key={m}><td><i className="dot" style={{ background: MODEL_COLOR[m] }} />{meta.models[m]}</td><td>{fmt(c.f1)}</td>
                  <td>{fmt(a.f1)} <span className="ci">± {fmt(a.f1_sd)}</span></td><td>{delta(a.f1 - c.f1)}</td>
                  <td>{fmt(b.f1)} <span className="ci">± {fmt(b.f1_sd)}</span></td><td>{delta(b.f1 - c.f1)}</td>
                  <td>{fmt(a.missed_alerts, 0)} / {fmt(b.missed_alerts, 0)}</td><td>{fmt(a.false_alarms, 0)} / {fmt(b.false_alarms, 0)}</td></tr>
              )
            })}</tbody>
          </table>
        </div>
        {det && <p className="small">The reliability checks flagged {fmt(det.flag_rate * 100, 1)} % of steps; {fmt(det.faulty_share * 100, 1)} % actually carried an injected fault
          (precision {fmt(det.precision)}, recall {fmt(det.recall)}).{scen.startsWith('noise') ? ' Low recall on small Gaussian noise is expected: noise within the normal variability of the river cannot be told apart from it.' : ''}</p>}
        <p className="muted small">The reliability layer does not help in every case (compare the two columns per fault). One known reason: its spike check can mistake a genuine, very fast rise for a fault and down-weight it.</p>
      </div>

      <div className="panel">
        <h3>All faults at a glance: alert F1 per model ({meta.horizons[h]})</h3>
        <div className="grid2">
          {['raw', 'fused'].map((mode) => (
            <div key={mode}>
              <h4>Reliability layer {mode === 'raw' ? 'off' : 'on'}</h4>
              <GroupedBars groups={['clean', ...faulted].map((s) => ({ label: s, values: MODELS.map((m) => f1(s, mode, m).f1) }))}
                series={MODELS.map((m) => ({ label: meta.models[m], color: MODEL_COLOR[m] }))} />
            </div>
          ))}
        </div>
      </div>
    </>
  )
}

export default function DataReliability({ dataset }: { dataset: string }) {
  const { d, err } = useComparison(dataset)
  return (
    <div className="stack">
      <DatasetCard dataset={dataset} d={d} />
      {err && <div className="panel error">No stored evaluation: {err}</div>}
      {d && <HorizonCheck d={d} />}
      {d && <FaultLab dataset={dataset} d={d} />}
    </div>
  )
}
