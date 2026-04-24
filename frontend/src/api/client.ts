const BASE = '/api'

export async function fetchJson<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${body}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  getPolicy: (industry: string, cost: number, volume: number) =>
    fetchJson('/policy', {
      method: 'POST',
      body: JSON.stringify({ industry, cost_per_violation: cost, weekly_volume: volume }),
    }),

  startCalibration: (payload: {
    industry: string
    wave_size: number
    cost_per_violation: number
    weekly_volume: number
    base_url: string
    llama_model: string
    deepseek_model: string
  }) =>
    fetchJson<{ run_id: string; status: string; total_tests: number }>(
      '/calibration/start',
      { method: 'POST', body: JSON.stringify(payload) },
    ),

  getCalibrationSummary: (runId: string) =>
    fetchJson(`/calibration/${runId}/summary`),

  getStatus: (baseUrl?: string) =>
    fetchJson(`/status${baseUrl ? `?base_url=${encodeURIComponent(baseUrl)}` : ''}`),

  getAudit: (runId?: string, page = 1, pageSize = 100) => {
    const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) })
    if (runId) params.set('run_id', runId)
    return fetchJson(`/audit?${params}`)
  },

  exportAuditCsv: (runId?: string) => {
    const params = runId ? `?run_id=${encodeURIComponent(runId)}` : ''
    return fetch(`${BASE}/audit/export${params}`)
  },

  listInspectorQueries: (runId: string) =>
    fetchJson(`/inspector/${runId}`),

  inspectQuery: (runId: string, qid: number) =>
    fetchJson(`/inspector/${runId}/${qid}`),
}
