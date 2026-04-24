import { useState } from 'react'
import { api } from '../api/client'
import type { PolicyInputs, PolicyRecommendation } from '../types'

const STORAGE_KEY = 'calibrai_policy'

function loadSaved(): PolicyInputs | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

export function usePolicy() {
  const saved = loadSaved()

  const [inputs, setInputs] = useState<PolicyInputs>(
    saved ?? { industry: 'Banking', costPerViolation: 5000, weeklyVolume: 10000 },
  )
  const [recommendation, setRecommendation] = useState<PolicyRecommendation | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const calculate = async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await api.getPolicy(
        inputs.industry,
        inputs.costPerViolation,
        inputs.weeklyVolume,
      ) as PolicyRecommendation
      setRecommendation(result)
      localStorage.setItem(STORAGE_KEY, JSON.stringify(inputs))
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  const update = (partial: Partial<PolicyInputs>) => {
    setInputs(s => ({ ...s, ...partial }))
    setRecommendation(null)
  }

  return { inputs, update, recommendation, calculate, loading, error }
}
