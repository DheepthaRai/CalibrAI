import { useCallback, useRef, useState } from 'react'
import { api } from '../api/client'
import type {
  CalibrationState,
  CalibrationSummary,
  QueryResult,
} from '../types'

const INITIAL: CalibrationState = {
  runId: null,
  status: 'idle',
  progress: 0,
  completedQueries: 0,
  totalQueries: 0,
  queries: [],
  levelStats: {},
  recommendedLevel: null,
  recommendationText: '',
  error: null,
}

export function useCalibration() {
  const [state, setState] = useState<CalibrationState>(INITIAL)
  const wsRef = useRef<WebSocket | null>(null)

  const reset = useCallback(() => {
    wsRef.current?.close()
    wsRef.current = null
    setState(INITIAL)
  }, [])

  const start = useCallback(
    async (params: {
      industry: string
      wave_size: number
      cost_per_violation: number
      weekly_volume: number
      base_url: string
      llama_model: string
      deepseek_model: string
    }) => {
      reset()
      setState(s => ({ ...s, status: 'running', error: null }))

      let runId: string
      try {
        const res = await api.startCalibration(params)
        runId = res.run_id
        setState(s => ({
          ...s,
          runId,
          totalQueries: params.wave_size,
        }))
      } catch (e) {
        setState(s => ({ ...s, status: 'error', error: String(e) }))
        return
      }

      // Connect WebSocket via Vite proxy
      const wsUrl = `ws://${window.location.host}/ws/calibration/${runId}`
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onmessage = (evt) => {
        const msg = JSON.parse(evt.data)

        switch (msg.type) {
          case 'heartbeat':
          case 'status':
            // keep-alive / informational — no state update needed
            break

          case 'queries_ready':
            setState(s => ({ ...s, totalQueries: msg.count }))
            break

          case 'query_result': {
            const qr: QueryResult = {
              qid: msg.qid,
              query: msg.query,
              is_attack: msg.is_attack,
              results: msg.results,
            }
            const stats: CalibrationSummary = {}
            for (const [k, v] of Object.entries(msg.level_stats as Record<string, any>)) {
              stats[Number(k)] = v as any
            }
            setState(s => ({
              ...s,
              completedQueries: msg.completed,
              progress: msg.pct,
              queries: [...s.queries, qr],
              levelStats: stats,
            }))
            break
          }

          case 'complete':
            setState(s => ({
              ...s,
              status: 'complete',
              progress: 100,
              levelStats: msg.summary,
              recommendedLevel: msg.recommended_level,
              recommendationText: msg.recommendation_text,
            }))
            break

          case 'error':
            setState(s => ({ ...s, status: 'error', error: msg.message }))
            break
        }
      }

      ws.onerror = () => {
        setState(s => ({
          ...s,
          status: 'error',
          error: 'WebSocket connection failed. Is the backend running?',
        }))
      }

      ws.onclose = () => {
        wsRef.current = null
      }
    },
    [reset],
  )

  return { state, start, reset }
}
