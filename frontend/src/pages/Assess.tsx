import { useEffect, useState } from 'react'
import { api, fmt, ALL_MODELS, RISK, RISK_COLOR, MODEL_COLOR } from '../api'
import AlertCard, { AlertGrid, FinalAlert, LevelChip } from '../Alert'
import { LineChart, Strip } from '../charts'

const CONDITIONS: [string, string][] = [['clean', 'Sensors OK'], ['missing_30', 'Missing readings'], ['noise_med', 'Noisy'],
  ['stuck', 'Stuck'], ['conflict', 'Conflicting rain'], ['spikes', 'Spikes']]
const WHATIF: [string, string][] = [['normal', 'Sensors OK'], ['missing_level', 'Level missing'], ['noisy', 'Noisy'],
  ['stuck_level', 'Level stuck'], ['rain_conflict', 'Conflicting rain'], ['spike', 'Level spike']]

function Controls({ horizons, h, setH, mode, setMode, unsupported }: any) {
  return (
    <>
      <label className="ctl">Forecast horizon
        <div className="chips">{Object.entries(horizons).map(([k, n]: any) => <button key={k} className={`chip small ${h === k ? 'on' : ''}`} onClick={() => setH(k)}>{n}</button>)}</div>
        {unsupported?.length > 0 && <span className="small">Not available on this dataset: {unsupported.join(', ')} (sampling interval too coarse)</span>}
      </label>
      <label className="ctl">Reliability layer
        <div className="toggle">
          <button className={mode === 'raw' ? 'on' : ''} onClick={() => setMode('raw')}>Off (raw readings)</button>
          <button className={mode === 'fused' ? 'on' : ''} onClick={() => setMode('fused')}>On (checked + fused)</button>
        </div>
      </label>
    </>
  )
}

function Replay({ dataset, opts }: { dataset: string; opts: any }) {
  const [eps, setEps] = useState<any[] | null>(null)
  const [epClass, setEpClass] = useState<string | null>(null)
  const [ep, setEp] = useState(0)
  const [h, setH] = useState(String(opts.default_horizon))
  const [mode, setMode] = useState('raw')
  const [cond, setCond] = useState('clean')
  const [pos, setPos] = useState<number | null>(null)
  const [r, setR] = useState<any>(null)
  const [sel, setSel] = useState('it2_fis')
  const [err, setErr] = useState('')
  const daily = (opts.step_hours ?? 0) >= 24
  const W = daily ? 60 : 96
  useEffect(() => { api(`/episodes/${dataset}`).then((x) => { setEps(x.episodes); setEpClass(x.episode_class) }).catch((e) => setErr(e.message)) }, [dataset])
  const e = eps?.[ep]
  const offset = e ? Math.max(0, e.start - Math.round(W * 0.7)) : 0
  // default step: h steps before the episode starts, i.e. when an h-ahead alert should already be raised
  const at = pos ?? (e ? Math.max(offset, e.start - Number(h)) : 0)
  useEffect(() => { setPos(null) }, [ep, h])
  useEffect(() => {
    if (!e) return
    let live = true
    api(`/replay/${dataset}?scenario=${cond}&offset=${offset}&limit=${W}&horizon=${h}&mode=${mode}&at=${at}`)
      .then((x) => { if (live) { setR(x); setErr('') } })
      .catch((x) => live && setErr(x.message))
    return () => { live = false }
  }, [dataset, cond, offset, W, h, mode, at, e])

  if (err) return <div className="panel error">{err}</div>
  if (eps === null) return <div className="panel">Loading test-period episodes…</div>
  if (!e) return <div className="panel">The test period never rises above Normal, so there is no episode to replay. Switch to "What-if input" above.</div>
  const th = r?.thresholds ?? opts.thresholds
  const cur = r ? r.at - r.offset : 0
  return (
    <div className="stack">
      <div className="panel">
        <span className="demo-tag replay">HISTORICAL REPLAY · held-out test period · real recorded data</span>
        <div className="controls" style={{ marginTop: 10 }}>
          <label className="ctl">{epClass === 'High' ? 'High-water episode' : `${epClass}-level episode (no High level in this test period)`} (test period, largest first)
            <select value={ep} onChange={(x) => setEp(Number(x.target.value))} style={{ marginLeft: 0 }}>
              {eps.map((x, i) => <option key={i} value={i}>{x.timestamp} · peak {fmt(x.peak_level, 2)} m ({x.peak_class})</option>)}
            </select>
          </label>
          <Controls horizons={opts.horizons} h={h} setH={setH} mode={mode} setMode={setMode} unsupported={opts.unsupported_horizons} />
          <label className="ctl">Sensor condition (simulated)
            <select value={cond} onChange={(x) => setCond(x.target.value)} style={{ marginLeft: 0 }}>
              {CONDITIONS.map(([k, n]) => <option key={k} value={k}>{n}</option>)}
            </select>
          </label>
        </div>
      </div>

      {r && (
        <div className="panel">
          <div className="row-between"><h3>{r.timestamp[0]} → {r.timestamp[r.timestamp.length - 1]}</h3>
            <span className="small">Assessed step: <b>{r.timestamp[cur]}</b></span></div>
          <input type="range" min={0} max={r.timestamp.length - 1} value={cur} style={{ width: '100%' }} aria-label="assessed time step"
            onChange={(x) => setPos(r.offset + Number(x.target.value))} />
          <LineChart height={210} yLabel="water level [m]" cursor={cur} xLabels={r.timestamp}
            series={[
              { values: r.level_original, color: 'var(--fg)', label: 'recorded level [m]', width: 1.2, dash: cond !== 'clean' },
              ...(cond !== 'clean' ? [{ values: r.level_sensor, color: 'var(--risk3)', label: 'faulty sensor reading', width: 1.3 }] : []),
              { values: r.level_used, color: 'var(--accent)', label: `level the models used (${mode})`, width: 2 },
            ]}
            hlines={th.map((t: number, i: number) => ({ y: t, label: RISK[i + 1], color: RISK_COLOR[i + 1] })).slice(1)}
            marks={cond !== 'clean' ? r.injected_fault : undefined} markLabel="injected fault" />
          <div className="grid2">
            <LineChart height={130} yLabel="rain" cursor={cur} series={[{ values: r.rain_used, color: 'var(--rain)', label: `rain, last ${opts.rain_window_hours} h (model input)` }]} />
            <LineChart height={130} yLabel="rise [m/h]" cursor={cur} series={[{ values: r.rise_used, color: 'var(--m-pso)', label: 'rate of rise [m/h] (model input)' }]} />
          </div>
          <div className="strip-wrap">
            <Strip name={`Actual level ${r.horizon_label} later`} values={r.proxy_label} colors={RISK_COLOR} />
            {ALL_MODELS.map((m) => <Strip key={m} name={r.model_names[m]} values={r.models[m]} colors={RISK_COLOR} />)}
          </div>
          <p className="muted small">Each strip shows the alert level per step (green Normal → red Critical). Model strips are aligned with the time the alert is issued, and the top strip with the level actually observed {r.horizon_label} later.</p>
        </div>
      )}

      {r && (
        <>
          <FinalAlert e={r.primary} />
          <div className="panel">
            <div className="row-between">
              <h3>Model comparison — alerts issued at {r.timestamp[cur]} for {r.horizon_label} ahead</h3>
              <span className="small">What actually happened: level {fmt(r.outcome.future_level, 2)} m → {r.outcome.proxy_label === null ? 'no reading (gap)' : <LevelChip c={r.outcome.proxy_label} />}</span>
            </div>
            <p className="muted small">Results from the evaluation study (see Compare Models). Shown for comparison only; they do not change the final alert above.</p>
            <AlertGrid ex={r.explanations} selected={sel} onSelect={setSel} />
          </div>
          <AlertCard e={r.explanations[sel]} />
        </>
      )}
    </div>
  )
}

function WhatIf({ dataset, opts }: { dataset: string; opts: any }) {
  const th = opts.thresholds
  const [v, setV] = useState({ level: Number((th[2] - 0.02).toFixed(2)), rise: 0.1, rain: Number(opts.anchors.rain[1].toFixed(1)) })
  const [h, setH] = useState(String(opts.default_horizon))
  const [mode, setMode] = useState('raw')
  const [scen, setScen] = useState('normal')
  const [r, setR] = useState<any>(null)
  const [sel, setSel] = useState('it2_fis')
  const [err, setErr] = useState('')
  useEffect(() => {
    let live = true
    const t = setTimeout(() => {
      api(`/assess/${dataset}`, { method: 'POST', body: JSON.stringify({ ...v, horizon: Number(h), scenario: scen, mode }) })
        .then((x) => { if (live) { setR(x); setErr('') } }).catch((x) => live && setErr(x.message))
    }, 250)
    return () => { live = false; clearTimeout(t) }
  }, [dataset, v, h, mode, scen])
  const field = (k: 'level' | 'rise' | 'rain', label: string, unit: string, min: number, max: number, step: number) => (
    <label className="bigslider">
      <div className="top"><span>{label}</span><span className="val">{fmt(v[k], k === 'rise' ? 3 : 2)} <span className="muted small">{unit}</span></span></div>
      <input type="range" min={min} max={max} step={step} value={v[k]} onChange={(x) => setV({ ...v, [k]: Number(x.target.value) })} />
    </label>
  )
  const lvlMax = Math.min(opts.bounds.level[1], th[3] * 1.4)
  return (
    <div className="stack">
      <div className="panel">
        <span className="demo-tag">WHAT-IF · hypothetical readings · the real fitted models run on a constructed short history</span>
        <div className="grid2" style={{ marginTop: 10 }}>
          <div>
            {field('level', 'Water level', 'm', 0, Number(lvlMax.toFixed(2)), 0.01)}
            {field('rise', 'Rate of rise', 'm/h', Math.max(opts.bounds.rise[0], -0.5), Math.min(opts.bounds.rise[1], 1.5), 0.005)}
            {field('rain', `Rainfall, last ${opts.rain_window_hours} h`, opts.units.rain.includes('(mm') ? '(mm)' : '(gauge units)', 0, Number((opts.anchors.rain[2] * 2).toFixed(0)), 0.1)}
          </div>
          <div className="controls" style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
            <Controls horizons={opts.horizons} h={h} setH={setH} mode={mode} setMode={setMode} unsupported={opts.unsupported_horizons} />
            <label className="ctl">Sensor condition
              <div className="chips">{WHATIF.map(([k, n]) => <button key={k} className={`chip small ${scen === k ? 'on' : ''}`} onClick={() => setScen(k)}>{n}</button>)}</div>
              <span className="small">{opts.scenarios[scen]}</span>
            </label>
          </div>
        </div>
        <p className="muted small">Thresholds: {th.map((t: number, i: number) => `${RISK[i + 1]} ≥ ${fmt(t, 2)} m`).join(' · ')}. Rain unit is not documented in the source data.</p>
      </div>
      {err && <div className="panel error">{err}</div>}
      {r && (
        <>
          <FinalAlert e={r.primary} demo />
          <div className="panel">
            <h3>Model comparison for {r.horizon_label} ahead</h3>
            <p className="muted small">Results from the evaluation study (see Compare Models). Shown for comparison only; they do not change the final alert above.</p>
            <AlertGrid ex={r.explanations} selected={sel} onSelect={setSel} />
            {scen !== 'normal' && (
              <LineChart height={150} yLabel="level" series={[
                { values: r.history.level_true, color: 'var(--fg)', label: 'true level (constructed)', dash: true, width: 1.2 },
                { values: r.history.level_sensor, color: 'var(--risk3)', label: 'what the sensor reports' },
                { values: r.history.level_fused, color: MODEL_COLOR.it2_fis, label: 'reliability-fused estimate' },
              ]} />
            )}
          </div>
          <AlertCard e={r.explanations[sel]} demo />
        </>
      )}
    </div>
  )
}

export default function Assess({ dataset }: { dataset: string }) {
  const [kind, setKind] = useState<'replay' | 'whatif'>('replay')
  const [opts, setOpts] = useState<any>(null)
  const [err, setErr] = useState('')
  useEffect(() => { api(`/assess/${dataset}/options`).then(setOpts).catch((e) => setErr(e.message)) }, [dataset])
  if (err) return <div className="panel error">{err}</div>
  if (!opts) return <div className="panel">Loading the fitted models (first load can take a minute)…</div>
  return (
    <div className="stack">
      <div className="segmented">
        <button className={kind === 'replay' ? 'on' : ''} onClick={() => setKind('replay')}><b>Historical replay</b><span>Real test-period data: see each model's alert and what actually happened next.</span></button>
        <button className={kind === 'whatif' ? 'on' : ''} onClick={() => setKind('whatif')}><b>What-if input</b><span>Set level, rise and rain yourself and watch the four models respond.</span></button>
      </div>
      {kind === 'replay' ? <Replay dataset={dataset} opts={opts} /> : <WhatIf dataset={dataset} opts={opts} />}
    </div>
  )
}
