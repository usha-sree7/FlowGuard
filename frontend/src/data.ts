import { useEffect, useState } from 'react'
import { api } from './api'

// Stored evaluation results (results/comparison_<ds>.json, written by scripts/compare_models.py).
export function useComparison(dataset: string) {
  const [d, setD] = useState<any>(null)
  const [err, setErr] = useState('')
  useEffect(() => {
    let live = true
    setD(null)
    setErr('')
    api(`/comparison/${dataset}`).then((x) => live && setD(x)).catch((e) => live && setErr(e.message))
    return () => { live = false }
  }, [dataset])
  return { d, err }
}

export function useDatasetSpec(dataset: string) {
  const [spec, setSpec] = useState<any>(null)
  useEffect(() => {
    api('/datasets').then((r) => setSpec(r.datasets.find((x: any) => x.id === dataset))).catch(() => setSpec(null))
  }, [dataset])
  return spec
}

// Default horizon: 1 h on 30-min data, otherwise the first supported one.
export const defaultHorizon = (d: any) => (d.meta.horizons['2'] ? '2' : Object.keys(d.meta.horizons)[0])

// Clean-test metrics of one model.
export const clean = (d: any, h: string, mode: string, m: string) => d.clean[`${h}|${mode}|${m}`]

// Two F1 values are only called different when their bootstrap intervals do not overlap.
export const separated = (a: any, b: any) => a.f1_ci95[0] > b.f1_ci95[1] || b.f1_ci95[0] > a.f1_ci95[1]
