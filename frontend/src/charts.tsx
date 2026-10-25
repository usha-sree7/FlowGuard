import { useMemo } from 'react'

type Num = number | null

export interface Series {
  values: Num[]
  color: string
  label: string
  dash?: boolean
  width?: number
}

export interface Band {
  lo: Num[]
  hi: Num[]
  color: string
  label: string
}

interface LineChartProps {
  series: Series[]
  bands?: Band[]
  hlines?: { y: number; label: string; color: string }[]
  marks?: boolean[] // shaded columns (e.g. injected faults)
  markLabel?: string
  height?: number
  yLabel?: string
  yMin?: number
  yMax?: number
  cursor?: number
  xLabels?: string[]
}

const W = 760

function path(vals: Num[], x: (i: number) => number, y: (v: number) => number) {
  let d = ''
  let pen = false
  vals.forEach((v, i) => {
    if (v === null || !Number.isFinite(v)) {
      pen = false
      return
    }
    d += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`
    pen = true
  })
  return d
}

export function LineChart({
  series, bands = [], hlines = [], marks, markLabel, height = 200, yLabel, yMin, yMax, cursor, xLabels,
}: LineChartProps) {
  const pad = { l: 46, r: 12, t: 10, b: 22 }
  const n = Math.max(...series.map((s) => s.values.length), ...bands.map((b) => b.lo.length), 1)
  const { lo, hi } = useMemo(() => {
    const all: number[] = []
    series.forEach((s) => s.values.forEach((v) => v !== null && Number.isFinite(v) && all.push(v)))
    bands.forEach((b) => [...b.lo, ...b.hi].forEach((v) => v !== null && Number.isFinite(v) && all.push(v)))
    hlines.forEach((h) => all.push(h.y))
    let a = yMin ?? Math.min(...all)
    let b = yMax ?? Math.max(...all)
    if (!Number.isFinite(a) || !Number.isFinite(b)) [a, b] = [0, 1]
    if (a === b) [a, b] = [a - 1, b + 1]
    const m = (b - a) * 0.06
    return { lo: yMin ?? a - m, hi: yMax ?? b + m }
  }, [series, bands, hlines, yMin, yMax])
  const x = (i: number) => pad.l + (i / Math.max(n - 1, 1)) * (W - pad.l - pad.r)
  const y = (v: number) => pad.t + (1 - (v - lo) / (hi - lo)) * (height - pad.t - pad.b)
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => lo + f * (hi - lo))
  const colW = (W - pad.l - pad.r) / Math.max(n - 1, 1)

  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label={yLabel}>
        {ticks.map((t, k) => (
          <g key={k}>
            <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} className="grid" />
            <text x={pad.l - 6} y={y(t) + 4} textAnchor="end" className="tick">
              {Math.abs(t) >= 100 ? t.toFixed(0) : t.toFixed(2)}
            </text>
          </g>
        ))}
        {marks?.map((m, i) =>
          m ? <rect key={i} x={x(i) - colW / 2} width={Math.max(colW, 1)} y={pad.t} height={height - pad.t - pad.b} className="mark" /> : null,
        )}
        {bands.map((b, k) => {
          const up = path(b.hi, x, y)
          const pts: string[] = []
          b.hi.forEach((v, i) => v !== null && Number.isFinite(v) && pts.push(`${x(i)},${y(v)}`))
          const back: string[] = []
          b.lo.forEach((v, i) => v !== null && Number.isFinite(v) && back.push(`${x(i)},${y(v)}`))
          return up ? <polygon key={k} points={[...pts, ...back.reverse()].join(' ')} fill={b.color} opacity={0.22} /> : null
        })}
        {hlines.map((h, k) => (
          <g key={k}>
            <line x1={pad.l} x2={W - pad.r} y1={y(h.y)} y2={y(h.y)} stroke={h.color} strokeDasharray="4 4" strokeWidth={1} />
            <text x={W - pad.r - 2} y={y(h.y) - 3} textAnchor="end" className="tick" fill={h.color}>{h.label}</text>
          </g>
        ))}
        {series.map((s, k) => (
          <path key={k} d={path(s.values, x, y)} fill="none" stroke={s.color} strokeWidth={s.width ?? 1.8}
            strokeDasharray={s.dash ? '5 4' : undefined} strokeLinejoin="round" strokeLinecap="round" />
        ))}
        {cursor !== undefined && <line x1={x(cursor)} x2={x(cursor)} y1={pad.t} y2={height - pad.b} className="cursor" />}
        {xLabels && n > 1 && (
          <>
            <text x={pad.l} y={height - 6} className="tick">{xLabels[0]}</text>
            <text x={W - pad.r} y={height - 6} className="tick" textAnchor="end">{xLabels[xLabels.length - 1]}</text>
          </>
        )}
      </svg>
      <figcaption className="legend">
        {series.map((s) => (
          <span key={s.label}><i style={{ background: s.color, opacity: s.dash ? 0.6 : 1 }} />{s.label}</span>
        ))}
        {bands.map((b) => (
          <span key={b.label}><i style={{ background: b.color, opacity: 0.35, height: 8 }} />{b.label}</span>
        ))}
        {marks && markLabel && <span><i className="mark-swatch" />{markLabel}</span>}
      </figcaption>
    </figure>
  )
}

export function Meter({ value, max = 1, color, label }: { value: number | null; max?: number; color: string; label: string }) {
  const f = value === null || !Number.isFinite(value) ? 0 : Math.max(0, Math.min(1, value / max))
  return (
    <div className="meter" title={`${label}: ${value === null ? '—' : value.toFixed(2)}`}>
      <span className="meter-label">{label}</span>
      <span className="meter-track"><span className="meter-fill" style={{ width: `${f * 100}%`, background: color }} /></span>
      <span className="meter-val">{value === null || !Number.isFinite(value) ? '—' : value.toFixed(2)}</span>
    </div>
  )
}
