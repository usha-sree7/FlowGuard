import { useCallback, useEffect, useRef, useState } from 'react'
import { api, fmt, RISK, RISK_COLOR, STATUS_COLOR } from '../api'
import { LineChart, Meter } from '../charts'

const WINDOW = 96

interface Scenario { id: string; description: string }

export default function Dashboard({ dataset }: { dataset: string }) {
  const [scenarios, setScenarios] = useState<Scenario[]>([])
  const [scenario, setScenario] = useState('spikes')
  const [cursor, setCursor] = useState(WINDOW)
  const [n, setN] = useState(0)
  const [data, setData] = useState<any>(null)
  const [playing, setPlaying] = useState(false)
  const [err, setErr] = useState('')
  const [events, setEvents] = useState<number[]>([])
  const timer = useRef<number | null>(null)

  useEffect(() => { api('/scenarios').then((r) => setScenarios(r.scenarios)).catch(() => {}) }, [])

  const load = useCallback(async (c: number) => {
    try {
      const off = Math.max(0, c - WINDOW + 1)
      const r = await api(`/replay/${dataset}?scenario=${scenario}&offset=${off}&limit=${WINDOW}`)
      setData(r)
      setN(r.n)
      setEvents(r.event_index.filter((v: number | null) => v !== null))
      setErr('')
    } catch (e: any) {
      setErr(e.message)
      setPlaying(false)
    }
  }, [dataset, scenario])

  useEffect(() => { load(cursor) }, [cursor, load])

  useEffect(() => {
    if (!playing) return
    timer.current = window.setInterval(() => setCursor((c) => (n && c >= n - 1 ? c : c + 1)), 600)
    return () => { if (timer.current) window.clearInterval(timer.current) }
  }, [playing, n])

  const jumpToEvent = () => {
    const next = events.find((e) => e > cursor + 4) ?? events[0]
    if (next !== undefined) setCursor(Math.min(n - 1, next + 6))
  }

  if (err) return <div className="panel error">Replay unavailable: {err}</div>
  if (!data) return <div className="panel">Loading replay…</div>

  const last = data.risk_score.length - 1
  const cls: number = data.risk_class[last] ?? 0
  const status: string = data.status[last]
  const U: number | null = data.U[last]
  const label = data.proxy_label[last]
  const th: number[] = data.thresholds
  const alert = cls >= 2
  const sensors = Object.keys(data.r_rain)
  const flagNames = Object.entries(data.flags).filter(([, v]: any) => v[last]).map(([k]) => k)

  return (
    <div className="stack">
      <div className="toolbar panel">
        <label>Fault scenario
          <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
            {scenarios.map((s) => <option key={s.id} value={s.id} title={s.description}>{s.id}</option>)}
          </select>
        </label>
        <button onClick={() => setPlaying((p) => !p)}>{playing ? 'Pause' : 'Play'}</button>
        <button onClick={() => setCursor((c) => Math.max(WINDOW - 1, c - 1))}>◀ Step</button>
        <button onClick={() => setCursor((c) => Math.min(n - 1, c + 1))}>Step ▶</button>
        <button onClick={jumpToEvent} title="Jump to the next period whose proxy label is Warning or higher">Next high-water event</button>
        <input type="range" min={WINDOW - 1} max={Math.max(n - 1, WINDOW)} value={cursor}
          onChange={(e) => setCursor(Number(e.target.value))} aria-label="Replay position" />
        <span className="muted mono">{data.timestamp[last]} · step {cursor + 1}/{n}</span>
      </div>
      <p className="note">{data.note} Scenario: {scenarios.find((s) => s.id === scenario)?.description}</p>

      {alert && (
        <div className="alarm" style={{ borderColor: RISK_COLOR[cls] }}>
          <strong style={{ color: RISK_COLOR[cls] }}>{RISK[cls].toUpperCase()}</strong> — flood-risk score {fmt(data.risk_score[last], 1)}
          {' '}(plausible range {fmt(data.risk_yl[last], 1)}–{fmt(data.risk_yr[last], 1)}). Decision support status: <b>{status}</b>.
        </div>
      )}

      <div className="cards">
        <div className="card">
          <h3>Flood risk</h3>
          <div className="big" style={{ color: RISK_COLOR[cls] }}>{RISK[cls]}</div>
          <div className="muted">score {fmt(data.risk_score[last], 1)} / 100 · range [{fmt(data.risk_yl[last], 1)}, {fmt(data.risk_yr[last], 1)}]</div>
          <div className="muted small">Score is a fuzzy severity index, not a probability.</div>
        </div>
        <div className="card">
          <h3>Uncertainty status</h3>
          <div className="big" style={{ color: STATUS_COLOR[status] }}>{status}</div>
          <div className="muted">U = {fmt(U, 1)} ({data.U_band[last]}) — not an error probability</div>
          <Meter label="Q quality" value={data.Q[last]} color="var(--st-unc)" />
          <Meter label="D data" value={data.D[last]} color="var(--st-ins)" />
          <Meter label="F model" value={data.F[last]} color="var(--accent)" />
        </div>
        <div className="card">
          <h3>Sensor reliability</h3>
          <Meter label="level" value={data.r_level[last]} color="var(--accent)" />
          {sensors.map((s) => <Meter key={s} label={`rain ${s}`} value={data.r_rain[s][last]} color="var(--accent)" />)}
          <div className="muted small">Flags: {flagNames.length ? flagNames.join(', ') : 'none'}
            {data.conflict[last] > 0 && ` · unresolved conflict ${fmt(data.conflict[last], 2)}`}</div>
        </div>
        <div className="card">
          <h3>Comparison (same input)</h3>
          <table className="mini">
            <tbody>
              <tr><td>Threshold</td><td>{RISK[data.baselines.threshold[last]]}</td></tr>
              <tr><td>Type-1 FIS</td><td>{RISK[data.baselines.t1_fis[last]]}</td></tr>
              <tr><td>IT2FIS (no fusion)</td><td>{RISK[data.baselines.it2_fis[last]]}</td></tr>
              <tr><td><b>Proposed</b></td><td><b>{RISK[cls]}</b></td></tr>
              <tr className="muted"><td>Proxy label (future level, eval only)</td><td>{label === null ? 'n/a' : RISK[label]}</td></tr>
            </tbody>
          </table>
          {data.injected_fault[last] && <div className="tag warn">fault injected at this step (ground truth, hidden from models)</div>}
        </div>
      </div>

      <div className="panel">
        <h3>Water level [m]</h3>
        <LineChart
          height={220}
          series={[
            { values: data.level_input, color: 'var(--muted-line)', label: 'sensor input (possibly corrupted)', width: 1.2 },
            { values: data.level_fused, color: 'var(--accent)', label: 'fused estimate' },
            { values: data.level_original, color: 'var(--fg)', label: 'original (eval only)', dash: true, width: 1 },
          ]}
          bands={[{ lo: data.level_lo, hi: data.level_hi, color: 'var(--accent)', label: 'fused interval' }]}
          hlines={[{ y: th[0], label: 'Watch', color: RISK_COLOR[1] }, { y: th[1], label: 'Warning', color: RISK_COLOR[2] }, { y: th[2], label: 'Critical', color: RISK_COLOR[3] }]}
          marks={data.injected_fault} markLabel="injected fault"
          cursor={last} xLabels={data.timestamp}
        />
      </div>
      <div className="grid2">
        <div className="panel">
          <h3>Risk score and plausible range</h3>
          <LineChart height={180} yMin={0} yMax={100}
            series={[{ values: data.risk_score, color: 'var(--risk2)', label: 'risk score (proposed)' }]}
            bands={[{ lo: data.risk_yl, hi: data.risk_yr, color: 'var(--risk2)', label: 'KM range [yl, yr]' }]}
            hlines={[{ y: 25, label: 'Watch', color: RISK_COLOR[1] }, { y: 50, label: 'Warning', color: RISK_COLOR[2] }, { y: 75, label: 'Critical', color: RISK_COLOR[3] }]}
            cursor={last} />
        </div>
        <div className="panel">
          <h3>Uncertainty index U</h3>
          <LineChart height={180} yMin={0} yMax={100}
            series={[{ values: data.U, color: 'var(--st-unc)', label: 'U (0–100)' }]}
            hlines={[{ y: 25, label: 'Confident below', color: 'var(--st-conf)' }]}
            marks={data.injected_fault} markLabel="injected fault" cursor={last} />
        </div>
      </div>
      <div className="panel">
        <h3>Rainfall (fused, accumulated over the rain window)</h3>
        <LineChart height={150}
          series={[{ values: data.rain_point, color: 'var(--rain)', label: 'fused rain' }]}
          bands={[{ lo: data.rain_lo, hi: data.rain_hi, color: 'var(--rain)', label: 'fused interval' }]}
          cursor={last} />
      </div>
    </div>
  )
}
