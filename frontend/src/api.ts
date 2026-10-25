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

export const RISK = ['Normal', 'Watch', 'Warning', 'Critical'] as const
export const RISK_COLOR = ['var(--risk0)', 'var(--risk1)', 'var(--risk2)', 'var(--risk3)']
export const STATUS_COLOR: Record<string, string> = {
  Confident: 'var(--st-conf)',
  Uncertain: 'var(--st-unc)',
  'Insufficient data': 'var(--st-ins)',
}

export const fmt = (v: number | null | undefined, d = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d)
