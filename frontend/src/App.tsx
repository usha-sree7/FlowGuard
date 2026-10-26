import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import Dashboard from './pages/Dashboard'
import Compare from './pages/Compare'
import Experiments from './pages/Experiments'
import Explorer from './pages/Explorer'
import Model from './pages/Model'

const TABS = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'explorer', label: 'Dataset Explorer' },
  { id: 'model', label: 'IT2 Model' },
  { id: 'experiments', label: 'Experiments' },
  { id: 'compare', label: 'Model Comparison' },
] as const

type Tab = (typeof TABS)[number]['id']

export default function App() {
  const params = new URLSearchParams(window.location.search) // ?tab=model&ds=malwathu deep links for demos
  const stored = (k: string) => { try { return localStorage.getItem(k) } catch { return null } }
  const [tab, setTab] = useState<Tab>(() => (params.get('tab') as Tab) || (stored('fg-tab') as Tab) || 'dashboard')
  const [dataset, setDataset] = useState(() => params.get('ds') || stored('fg-ds') || 'bengalas')
  const [loaded, setLoaded] = useState<Record<string, boolean>>({})
  const [backend, setBackend] = useState<'ok' | 'down' | '?'>('?')

  const refresh = useCallback(() => {
    api('/datasets')
      .then((r) => {
        setBackend('ok')
        setLoaded(Object.fromEntries(r.datasets.map((d: any) => [d.id, d.loaded])))
      })
      .catch(() => setBackend('down'))
  }, [])
  useEffect(() => { refresh() }, [refresh])
  useEffect(() => { try { localStorage.setItem('fg-tab', tab); localStorage.setItem('fg-ds', dataset) } catch { /* storage blocked */ } }, [tab, dataset])

  const needsData = tab !== 'explorer' && backend === 'ok' && !loaded[dataset]

  return (
    <div className="app">
      <header>
        <div className="brand">
          <span className="logo" aria-hidden>≋</span>
          <div>
            <h1>FloodGuard AI</h1>
            <p>Reliability-aware sensor fusion · Interval Type-2 fuzzy flood-risk detection</p>
          </div>
        </div>
        <label className="ds">Dataset
          <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
            <option value="bengalas">Bengalas River, BR (30 min)</option>
            <option value="malwathu">Malwathu Oya, LK (daily)</option>
          </select>
        </label>
      </header>
      <nav>
        {TABS.map((t) => (
          <button key={t.id} className={tab === t.id ? 'active' : ''} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </nav>
      <main>
        {backend === 'down' && <div className="error panel">Backend not reachable at /api. Start it with <code>uvicorn floodguard.api:app --port 8000</code> from <code>backend/</code>.</div>}
        {needsData ? (
          <div className="panel">
            Dataset <b>{dataset}</b> has not been imported and validated yet.{' '}
            <button onClick={() => setTab('explorer')}>Open Dataset Explorer to import it</button>
          </div>
        ) : (
          <>
            {tab === 'dashboard' && <Dashboard key={dataset} dataset={dataset} />}
            {tab === 'explorer' && <Explorer onChanged={refresh} />}
            {tab === 'model' && <Model dataset={dataset} />}
            {tab === 'experiments' && <Experiments dataset={dataset} />}
            {tab === 'compare' && <Compare key={dataset} dataset={dataset} />}
          </>
        )}
      </main>
      <footer>Research prototype for coursework. Risk scores are fuzzy severity indices, not probabilities; the uncertainty index is not a calibrated error probability; labels are proxies. Not an operational warning service.</footer>
    </div>
  )
}
