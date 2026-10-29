import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import Assess from './pages/Assess'
import Compare from './pages/Compare'
import DataReliability from './pages/DataReliability'
import DatasetLab from './pages/DatasetLab'
import Overview from './pages/Overview'

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'assess', label: 'Assess Risk' },
  { id: 'compare', label: 'Compare Models' },
  { id: 'data', label: 'Dataset & Sensor Reliability' },
  { id: 'lab', label: 'Dataset Lab' },
] as const

type Tab = (typeof TABS)[number]['id']
// deep links from earlier versions (?tab=faults etc.) open the matching page
const ALIAS: Record<string, Tab> = {
  home: 'overview', learn: 'overview', check: 'assess', replay: 'assess', results: 'compare', advanced: 'compare',
  experiments: 'data', faults: 'data', forecast: 'data', horizons: 'data', explorer: 'data', references: 'data',
}

function ImportPanel({ dataset, onDone }: { dataset: string; onDone: () => void }) {
  const [job, setJob] = useState<any>(null)
  useEffect(() => {
    if (job?.state !== 'running') return
    const t = setInterval(() => api(`/datasets/${dataset}/import`).then((j) => { setJob(j); if (j.state === 'done') onDone() }), 1500)
    return () => clearInterval(t)
  }, [job, dataset, onDone])
  return (
    <div className="panel">
      <p>This dataset has not been imported yet. Importing downloads the original file(s) from the source listed in the dataset documentation, checks the SHA-256 and builds the processed series.</p>
      <button disabled={job?.state === 'running'} onClick={() => api(`/datasets/${dataset}/import`, { method: 'POST' }).then(setJob).catch((e) => setJob({ state: 'failed', message: e.message }))}>
        {job?.state === 'running' ? `Importing… ${job.message ?? ''}` : 'Import dataset'}</button>
      {job?.state === 'failed' && <div className="error" style={{ marginTop: 8 }}>{job.message}{job.manual && <pre>{job.manual}</pre>}</div>}
    </div>
  )
}

export default function App() {
  const params = new URLSearchParams(window.location.search)
  const stored = (k: string) => { try { return localStorage.getItem(k) } catch { return null } }
  const valid = (t: string | null): Tab | null => (t && TABS.some((x) => x.id === t) ? (t as Tab) : t && ALIAS[t] ? ALIAS[t] : null)
  const [tab, setTab] = useState<Tab>(() => valid(params.get('tab')) || valid(stored('hs-tab4')) || 'overview')
  const [dataset, setDataset] = useState(() => params.get('ds') || stored('hs-ds') || 'bengalas')
  const [loaded, setLoaded] = useState<Record<string, boolean>>({})
  const [backend, setBackend] = useState<'ok' | 'down' | '?'>('?')

  const refresh = useCallback(() => {
    api('/datasets')
      .then((r) => { setBackend('ok'); setLoaded(Object.fromEntries(r.datasets.map((d: any) => [d.id, d.loaded]))) })
      .catch(() => setBackend('down'))
  }, [])
  useEffect(() => { refresh() }, [refresh])
  useEffect(() => { try { localStorage.setItem('hs-tab4', tab); localStorage.setItem('hs-ds', dataset) } catch { /* storage blocked */ } }, [tab, dataset])
  useEffect(() => { window.scrollTo(0, 0) }, [tab])

  const go = (t: string) => setTab(valid(t) || 'overview')

  return (
    <>
      <header className="topbar">
        <div className="topbar-in">
          <div className="brand">
            <svg className="logo" viewBox="0 0 32 32" aria-hidden>
              <path d="M16 3 L27 8 V16 C27 22.5 22 27.5 16 29 C10 27.5 5 22.5 5 16 V8 Z" fill="none" stroke="currentColor" strokeWidth="2" />
              <path d="M9 17 q3.5 -3 7 0 t7 0 M9 21.5 q3.5 -3 7 0 t7 0" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
            <div>
              <h1>HydroSentinel AI</h1>
              <p>Flood-risk alerts with IT2FIS · PSO · SOM · ANFIS · research prototype</p>
            </div>
          </div>
          <label className="ds">Dataset
            <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
              <option value="bengalas">Bengalas River, Brazil · 30 min</option>
              <option value="malwathu">Malwathu Oya, Sri Lanka · daily</option>
            </select>
          </label>
        </div>
        <nav className="topnav" aria-label="Sections">
          {TABS.map((t) => <button key={t.id} className={tab === t.id ? 'active' : ''} aria-current={tab === t.id ? 'page' : undefined} onClick={() => setTab(t.id)}>{t.label}</button>)}
        </nav>
      </header>
      <main className="app">
        {backend === 'down' && <div className="error panel">Backend not reachable. Start it with <code>python -m uvicorn floodguard.api:app --port 8000</code> from <code>backend/</code>.</div>}
        {backend === '?' && <div className="panel muted">Connecting to the analysis server…</div>}
        {backend === 'ok' && tab === 'lab' && <DatasetLab />}
        {backend === 'ok' && tab !== 'lab' && !loaded[dataset] && <ImportPanel dataset={dataset} onDone={refresh} />}
        {backend === 'ok' && tab !== 'lab' && loaded[dataset] && (
          <>
            {tab === 'overview' && <Overview key={dataset} dataset={dataset} go={go} />}
            {tab === 'assess' && <Assess key={dataset} dataset={dataset} />}
            {tab === 'compare' && <Compare key={dataset} dataset={dataset} />}
            {tab === 'data' && <DataReliability key={dataset} dataset={dataset} />}
          </>
        )}
        <footer>HydroSentinel AI is a student research prototype analysing <b>historical</b> data. It is not an official flood warning service.
          Alert levels are proxy labels (river level above statistical thresholds), and every sensor fault shown is simulated.</footer>
      </main>
    </>
  )
}
