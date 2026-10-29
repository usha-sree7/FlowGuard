import { useEffect, useState } from 'react'
import { api, fmt, ALL_MODELS, MODEL_COLOR, MODELS, RISK, RISK_COLOR } from '../api'
import { GroupedBars, LineChart } from '../charts'
import { METRIC_HELP } from '../content'
import { clean, defaultHorizon, separated, useComparison } from '../data'

function Confusion({ cm, title }: { cm: number[][]; title: string }) {
  const max = Math.max(...cm.flat(), 1)
  return (
    <div>
      <h4>{title}</h4>
      <table className="cm">
        <thead><tr><th title="true level (rows) vs predicted (columns)">t \ p</th>{RISK.map((r) => <th key={r}>{r.slice(0, 3)}</th>)}</tr></thead>
        <tbody>{cm.map((row, i) => (
          <tr key={i}><th>{RISK[i].slice(0, 3)}</th>{row.map((v, j) => (
            <td key={j} title={`true ${RISK[i]}, predicted ${RISK[j]}: ${v}`}
              style={{ background: v ? `color-mix(in srgb, ${i === j ? 'var(--accent)' : 'var(--risk4)'} ${Math.round(12 + 70 * Math.sqrt(v / max))}%, transparent)` : undefined, fontWeight: i === j ? 700 : 400 }}>{v}</td>
          ))}</tr>
        ))}</tbody>
      </table>
    </div>
  )
}

function PsoPanel({ d, h, dataset }: { d: any; h: string; dataset: string }) {
  const p = d.fitted.pso_it2[h]
  const it2 = clean(d, h, 'raw', 'it2_fis'), pso = clean(d, h, 'raw', 'pso_it2')
  const [m, setM] = useState<any>(null)
  useEffect(() => { api(`/model/${dataset}?horizon=${h}`).then(setM).catch(() => setM(null)) }, [dataset, h])
  const inputs = ['level', 'rise', 'rain']
  return (
    <div className="panel">
      <h2>PSO-tuned vs untuned IT2FIS ({d.meta.horizons[h]})</h2>
      <p className="note">Both use the same 80 rules and the same inference. PSO only moves the fuzzy-set crossover points and scales the footprint of uncertainty.
        It maximises <b>{p.objective}</b> on the validation block (the last 20 % of the training period), never on the test period.</p>
      <div className="grid2" style={{ marginTop: 10 }}>
        <div>
          <table className="results">
            <thead><tr><th></th><th>Untuned IT2FIS</th><th>PSO-tuned</th><th>Change</th></tr></thead>
            <tbody>
              <tr><td>Validation objective (used for tuning)</td><td>{fmt(p.validation_fitness_untuned)}</td><td>{fmt(p.validation_fitness_tuned)}</td><td>{fmt(p.validation_fitness_tuned - p.validation_fitness_untuned)}</td></tr>
              <tr><td>Test alert F1</td><td>{fmt(it2.f1)}</td><td>{fmt(pso.f1)}</td><td>{pso.f1 == null || it2.f1 == null ? '—' : (pso.f1 >= it2.f1 ? '+' : '') + fmt(pso.f1 - it2.f1)}</td></tr>
              <tr><td>Test precision / recall</td><td>{fmt(it2.precision)} / {fmt(it2.recall)}</td><td>{fmt(pso.precision)} / {fmt(pso.recall)}</td><td></td></tr>
              <tr><td>Test macro F1 (5 levels)</td><td>{fmt(it2.macro_f1)}</td><td>{fmt(pso.macro_f1)}</td><td>{fmt(pso.macro_f1 - it2.macro_f1)}</td></tr>
              <tr><td>Missed alerts / false alarms</td><td>{it2.missed_alerts} / {it2.false_alarms}</td><td>{pso.missed_alerts} / {pso.false_alarms}</td><td></td></tr>
            </tbody>
          </table>
          <p className="small">{separated(it2, pso)
            ? 'The 95 % intervals of test F1 do not overlap: the difference is supported by the data.'
            : `The 95 % intervals of test F1 overlap ([${fmt(it2.f1_ci95[0])}, ${fmt(it2.f1_ci95[1])}] vs [${fmt(pso.f1_ci95[0])}, ${fmt(pso.f1_ci95[1])}]): the tuning gain is not statistically established.`}</p>
          <h4>Swarm convergence (best validation objective per iteration)</h4>
          <LineChart height={150} series={[{ values: p.best_fitness_per_iteration, color: MODEL_COLOR.pso_it2, label: `global best (${p.population} particles × ${p.iterations} iterations)` }]} />
        </div>
        <div>
          <h4>Crossover points moved by PSO</h4>
          <table className="results">
            <thead><tr><th>Input</th><th>Original</th><th>Tuned</th></tr></thead>
            <tbody>{inputs.map((n) => p.original_anchors[n].map((o: number, k: number) => (
              <tr key={n + k}><td>{n} #{k + 1}</td><td>{fmt(o, 3)}</td><td>{fmt(p.tuned_anchors[n][k], 3)}</td></tr>
            )))}
              <tr><td>FOU scale</td><td>1.000</td><td>{fmt(p.fou_scale, 3)}</td></tr>
            </tbody>
          </table>
          {m?.tuned_curves && (
            <>
              <h4>Level membership functions (Type-1 midline): untuned solid, PSO dashed</h4>
              <LineChart height={160} xLabels={[fmt(m.curves.level.x[0], 2) + ' m', fmt(m.curves.level.x[m.curves.level.x.length - 1], 2) + ' m']}
                series={[...m.curves.level.terms.map((t: any, i: number) => ({ values: t.type1, color: RISK_COLOR[i], label: t.term, width: 1.6 })),
                  ...m.tuned_curves.level.terms.map((t: any, i: number) => ({ values: t.type1, color: RISK_COLOR[i], label: `${t.term} (PSO)`, dash: true, width: 1.3 }))]} />
              <p className="muted small">Both curves are drawn on the untuned x-range, so the shift of each crossover is visible directly.</p>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default function Compare({ dataset }: { dataset: string }) {
  const { d, err } = useComparison(dataset)
  const [h, setH] = useState('')
  const [mode, setMode] = useState('raw')
  useEffect(() => { if (d) setH(defaultHorizon(d)) }, [d])
  if (err) return <div className="panel error">No stored evaluation: {err}</div>
  if (!d || !h) return <div className="panel">Loading evaluation results…</div>
  const meta = d.meta
  const rows = ALL_MODELS.map((m) => ({ m, c: clean(d, h, mode, m) }))
  const best = Math.max(...MODELS.map((m) => clean(d, h, mode, m).f1))
  const fit = d.fitted.fit_seconds
  const ms = d.ms_per_obs_clean
  const lc = meta.label_counts_test[h]
  const horizonsList = Object.keys(meta.horizons)

  return (
    <div className="stack">
      <div className="panel">
        <h2>Compare the four models</h2>
        <p className="note">All results are computed on the held-out test period ({meta.test.first} → {meta.test.last}, {meta.test.rows.toLocaleString()} steps), which no model saw during fitting or tuning.
          Test labels at {meta.horizons[h]}: {RISK.map((r) => `${r} ${lc[r]}`).join(' · ')} ({lc.unlabelled} steps without a future reading).</p>
        <div className="controls" style={{ marginTop: 10 }}>
          <label className="ctl">Forecast horizon
            <div className="chips">{horizonsList.map((k) => <button key={k} className={`chip small ${h === k ? 'on' : ''}`} onClick={() => setH(k)}>{meta.horizons[k]}</button>)}</div>
          </label>
          <label className="ctl">Reliability layer
            <div className="toggle">
              <button className={mode === 'raw' ? 'on' : ''} onClick={() => setMode('raw')}>Off (raw)</button>
              <button className={mode === 'fused' ? 'on' : ''} onClick={() => setMode('fused')}>On (fused)</button>
            </div>
          </label>
          {meta.unsupported_horizons.length > 0 && <span className="small muted">{meta.unsupported_horizons.join(', ')}: not possible at this sampling interval.</span>}
        </div>
        {lc.High + lc.Critical === 0 && <div className="error" style={{ marginTop: 10 }}>
          <b>No High or Critical level occurs in this test period.</b> Alert precision, recall and F1 are undefined (shown as —), so this dataset
          cannot show flood-detection skill. Only false alarms, FPR, the 5-level metrics (macro F1, κ) and the water-level MAE / RMSE below are meaningful.</div>}
      </div>

      <div className="panel">
        <h3>Evaluation metrics, clean sensors</h3>
        <div className="table-scroll">
          <table className="results">
            <thead><tr><th>Model</th><th>Alert F1 [95 % CI]</th><th>Precision</th><th>Recall</th><th>FPR</th><th>Macro F1</th><th>κ</th><th>Sev. error</th><th>Missed</th><th>False alarms</th><th>Fit time</th><th>ms / step</th></tr></thead>
            <tbody>{rows.map(({ m, c }) => (
              <tr key={m} className={m === 'threshold' ? 'ref-row' : ''}>
                <td><i className="dot" style={{ background: MODEL_COLOR[m] }} />{meta.models[m]}</td>
                <td className={m !== 'threshold' && c.f1 === best ? 'best' : ''}>{fmt(c.f1)} <span className="ci">[{fmt(c.f1_ci95[0])}, {fmt(c.f1_ci95[1])}]</span></td>
                <td>{fmt(c.precision)}</td><td>{fmt(c.recall)}</td><td>{fmt(c.fpr, 4)}</td><td>{fmt(c.macro_f1)}</td><td>{fmt(c.kappa)}</td>
                <td>{fmt(c.swe)}</td><td>{c.missed_alerts}</td><td>{c.false_alarms}</td>
                <td>{m === 'threshold' ? '—' : fit[m][h] == null ? '—' : `${fmt(fit[m][h], 1)} s`}</td>
                <td>{m === 'threshold' ? '—' : fmt(ms[`${m}|${mode}`], 4)}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
        <p className="muted small">Bold = highest F1 among the four techniques. The threshold row (italic) is a reference: the labels are built from it.
          Fit time for IT2FIS is only the statistical calibration (it has no training); PSO includes the swarm search; SOM and ANFIS include their validation selection.
          ms / step excludes the reliability layer ({fmt(ms.reliability_layer, 4)} ms / step when on).</p>
      </div>

      {lc.High + lc.Critical > 0 && <div className="grid2">
        <div className="panel">
          <h3>Precision, recall and F1</h3>
          <GroupedBars groups={MODELS.map((m) => { const c = clean(d, h, mode, m); return { label: meta.models[m], values: [c.precision, c.recall, c.f1], lo: [null, null, c.f1_ci95[0]], hi: [null, null, c.f1_ci95[1]] } })}
            series={[{ label: 'precision', color: 'var(--muted-line)' }, { label: 'recall', color: 'var(--rain)' }, { label: 'F1 (whisker = 95 % CI)', color: 'var(--accent)' }]} />
        </div>
        <div className="panel">
          <h3>Alert F1 by horizon</h3>
          <GroupedBars groups={horizonsList.map((k) => ({ label: meta.horizons[k], values: MODELS.map((m) => clean(d, k, mode, m).f1) }))}
            series={MODELS.map((m) => ({ label: meta.models[m], color: MODEL_COLOR[m] }))} />
          <p className="muted small">Skill falls with lead time for every model: the further ahead, the less the current readings say about the future level.</p>
        </div>
      </div>}

      <div className="panel">
        <h3>Numerical water-level forecast (genuine regression errors)</h3>
        <table className="results">
          <thead><tr><th>Horizon</th><th>Persistence MAE</th><th>Persistence RMSE</th><th>ARX MAE</th><th>ARX RMSE</th><th>Test points</th></tr></thead>
          <tbody>{horizonsList.map((k) => { const f = d.forecast[k]; return (
            <tr key={k}><td>{meta.horizons[k]}</td><td>{fmt(f.persistence.mae, 4)} m</td><td>{fmt(f.persistence.rmse, 4)} m</td>
              <td className={f.arx.mae < f.persistence.mae ? 'best' : ''}>{fmt(f.arx.mae, 4)} m</td><td>{fmt(f.arx.rmse, 4)} m</td><td>{f.arx.n.toLocaleString()}</td></tr>) })}</tbody>
        </table>
        <p className="muted small">MAE and RMSE apply only to models that predict a number: here the future water level. Persistence = "the level stays as it is" (no-skill reference);
          ARX = linear auto-regressive forecaster with rain, fitted on the fit block. The four soft-computing models output a risk class, so they are scored with classification metrics above, not MAE / RMSE.</p>
      </div>

      <div className="panel">
        <h3>F1 for each alert level</h3>
        <table className="results">
          <thead><tr><th>Model</th>{RISK.map((r, i) => <th key={r} style={{ color: RISK_COLOR[i] }}>{r}</th>)}</tr></thead>
          <tbody>{rows.map(({ m, c }) => (
            <tr key={m} className={m === 'threshold' ? 'ref-row' : ''}><td>{meta.models[m]}</td>{c.per_class_f1.map((v: number | null, i: number) => <td key={i}>{fmt(v)}</td>)}</tr>
          ))}</tbody>
        </table>
        <h3 style={{ marginTop: 14 }}>Confusion matrices (rows = actual level, columns = predicted)</h3>
        <div className="grid4">{MODELS.map((m) => <Confusion key={m} cm={d.confusion_seed0[`${h}|clean|${mode}|${m}`]} title={meta.models[m]} />)}</div>
      </div>

      <PsoPanel d={d} h={h} dataset={dataset} />

      <details className="panel">
        <summary>What the metrics mean</summary>
        <table className="kv"><tbody>{METRIC_HELP.map(([k, v]) => <tr key={k}><th>{k}</th><td>{v}</td></tr>)}</tbody></table>
      </details>
    </div>
  )
}
