export async function api<T = any>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json()
}

export const RISK = ['Normal', 'Low', 'Moderate', 'High', 'Critical'] as const
export const RISK_COLOR = ['var(--risk0)', 'var(--risk1)', 'var(--risk2)', 'var(--risk3)', 'var(--risk4)']
export const ALERT_CLASS = 3
export const STATUS_COLOR: Record<string, string> = {
  Confident: 'var(--st-conf)',
  Uncertain: 'var(--st-unc)',
  'Insufficient data': 'var(--st-ins)',
}
// the four compared techniques, then the reference rule
export const MODELS = ['it2_fis', 'pso_it2', 'som', 'anfis'] as const
export const ALL_MODELS = [...MODELS, 'threshold'] as const
export const MODEL_COLOR: Record<string, string> = {
  it2_fis: 'var(--m-it2)', pso_it2: 'var(--m-pso)', som: 'var(--m-som)', anfis: 'var(--m-anfis)', threshold: 'var(--muted-line)',
}

export const fmt = (v: number | null | undefined, d = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d)
