import { ALL_MODELS, fmt, RISK, RISK_COLOR, STATUS_COLOR } from './api'
import { Meter } from './charts'

export const LevelChip = ({ c }: { c: number }) => <span className="level-chip" style={{ background: RISK_COLOR[c] }}>{RISK[c]}</span>

// One card per model with its alert level; clicking a card selects whose explanation is shown below.
export function AlertGrid({ ex, selected, onSelect }: { ex: Record<string, any>; selected: string; onSelect: (m: string) => void }) {
  return (
    <div className="alert-grid">
      {ALL_MODELS.map((m) => {
        const e = ex[m]
        if (!e) return null
        return (
          <button key={m} className={`acard ${selected === m ? 'on' : ''}`} style={{ borderTopColor: RISK_COLOR[e.risk_class] }}
            onClick={() => onSelect(m)} aria-pressed={selected === m}>
            <div className="who">{e.model_name}</div>
            <div className="lvl" style={{ color: RISK_COLOR[e.risk_class] }}>{e.risk}{e.alert ? ' · ALERT' : ''}</div>
            <div className="muted small">{e.score !== null ? `score ${fmt(e.score, 1)} / 100` : m === 'som' ? `SOM node ${e.detail.som.node}` : 'current-level rule'}</div>
          </button>
        )
      })}
    </div>
  )
}

// Renders an explanation produced by the backend (floodguard/alerts.py) from the actual model outputs.
export default function AlertCard({ e, demo }: { e: any; demo?: boolean }) {
  const c = RISK_COLOR[e.risk_class]
  const d = e.detail
  return (
    <div className="stack">
      <div className="alert-box" style={{ borderColor: c }}>
        {demo && <span className="demo-tag">HYPOTHETICAL INPUT · not an evaluated prediction · not an official warning</span>}
        <div className="big" style={{ color: c }}>{e.headline}</div>
        <p style={{ margin: '2px 0 6px' }}><b>{e.advice}</b></p>
        <h4 style={{ margin: '6px 0 0' }}>Why this alert level</h4>
        <ol className="reasons">{e.reasons.map((r: string, i: number) => <li key={i}>{r}</li>)}</ol>
      </div>
      <div className="cards">
        {(d.rules || d.anfis_rules) && (
          <div className="card wide">
            <h3>{d.rules ? 'Rules that fired (interval firing strength)' : 'Most active ANFIS rules'}</h3>
            <table className="mini"><tbody>
              {d.rules && d.rules.map((r: any, i: number) => (
                <tr key={i}><td>IF level {r.if.level} AND rise {r.if.rise} AND rain {r.if.rain}</td><td>→ {r.then}</td><td className="mono">[{fmt(r.firing[0], 3)}, {fmt(r.firing[1], 3)}]</td></tr>
              ))}
              {d.anfis_rules && d.anfis_rules.map((r: any, i: number) => (
                <tr key={i}><td>IF level {r.if.level} AND rise {r.if.rise} AND rain {r.if.rain}</td><td className="mono">weight {fmt(r.weight, 3)}</td><td className="mono">output {fmt(r.rule_output, 1)}</td></tr>
              ))}
            </tbody></table>
            {d.interval && <div className="muted small">Type-reduced interval [{fmt(d.interval[0], 1)}, {fmt(d.interval[1], 1)}] → midpoint {fmt(e.score, 1)}</div>}
          </div>
        )}
        {d.som && (
          <div className="card wide">
            <h3>Closest SOM prototype: node {d.som.node} of a {d.som.grid[0]}×{d.som.grid[1]} map</h3>
            <table className="mini"><tbody>
              <tr><td>prototype</td><td className="mono">level {fmt(d.som.prototype.level, 2)} m · rise {fmt(d.som.prototype.rise, 3)} m/h · rain {fmt(d.som.prototype.rain, 1)}</td></tr>
              <tr><td>training states on this node, by later level</td><td>{d.som.label_counts.map((n: number, i: number) => <span key={i} className={`pill r${i}`} style={{ marginRight: 4 }}>{RISK[i]} {n}</span>)}</td></tr>
            </tbody></table>
          </div>
        )}
        <div className="card">
          <h3>Inputs used ({e.mode === 'fused' ? 'reliability-fused' : 'raw'})</h3>
          <table className="mini"><tbody>
            {Object.entries(e.terms).map(([k, t]: any) => (
              <tr key={k}><td>{k}</td><td className="mono">{fmt(e.inputs_used[k], 3)}</td><td>{t.term} ({fmt(t.membership, 2)})</td></tr>
            ))}
          </tbody></table>
        </div>
        <div className="card">
          <h3>Sensor reliability</h3>
          {e.sensors.map((s: any) => (
            <div key={s.sensor}>
              <Meter label={s.sensor} value={s.reliability} color={s.reliability < 0.5 ? 'var(--risk4)' : 'var(--accent)'} />
              {s.flags.length > 0 && <div className="muted small">flags: {s.flags.join(', ')}</div>}
            </div>
          ))}
          {e.status && <div className="small" style={{ marginTop: 6 }}>Confidence: <b style={{ color: STATUS_COLOR[e.status] }}>{e.status}</b> (U = {fmt(e.U, 0)} / 100)</div>}
          {!e.status && <div className="muted small" style={{ marginTop: 6 }}>Raw mode: the model does not see these reliability values.</div>}
        </div>
      </div>
    </div>
  )
}
