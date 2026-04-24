import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { OllamaStatus } from '../types'

interface Props {
  baseUrl: string
  llamaModel?: string
  deepseekModel?: string
}

export default function ModelStatus({ baseUrl, llamaModel = 'llama3.1:latest', deepseekModel = 'deepseek-r1:7b' }: Props) {
  const [status, setStatus] = useState<OllamaStatus | null>(null)
  const [loading, setLoading] = useState(false)

  const refresh = async () => {
    setLoading(true)
    try {
      const s = await api.getStatus(baseUrl) as OllamaStatus
      setStatus(s)
    } catch {
      setStatus({ ollama_online: false, llama_available: false, deepseek_available: false, models: [] })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 30_000)
    return () => clearInterval(id)
  }, [baseUrl])

  const Row = ({
    label,
    ok,
    note,
  }: {
    label: string
    ok: boolean | null
    note?: string
  }) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '5px 0' }}>
      <span className={`dot ${ok === null ? 'dot-yellow' : ok ? 'dot-green' : 'dot-red'}`} />
      <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', flex: 1 }}>{label}</span>
      {note && <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{note}</span>}
    </div>
  )

  return (
    <div>
      <div className="section-label">Model Status</div>

      {loading && !status && (
        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Checking...</span>
      )}

      {status && (
        <div>
          <Row
            label="Ollama"
            ok={status.ollama_online}
            note={status.ollama_online ? '' : 'offline'}
          />
          {status.ollama_online ? (
            <>
              <Row
                label={llamaModel}
                ok={status.llama_available}
                note={status.llama_available ? 'query gen' : 'not pulled'}
              />
              <Row
                label={deepseekModel}
                ok={status.deepseek_available}
                note={
                  status.deepseek_available
                    ? 'safety test'
                    : status.llama_available
                    ? '→ fallback: LLaMA'
                    : 'not pulled'
                }
              />
            </>
          ) : (
            <div
              style={{
                marginTop: 8,
                padding: '8px 10px',
                background: 'var(--red-dim)',
                borderRadius: 'var(--radius)',
                fontSize: '0.78rem',
                color: 'var(--red)',
                lineHeight: 1.5,
              }}
            >
              Ollama is offline. Start Ollama to run calibration.
              <br />
              UI navigation is still available.
            </div>
          )}

          {status.models.length > 0 && (
            <details style={{ marginTop: 8 }}>
              <summary
                style={{ fontSize: '0.75rem', color: 'var(--text-muted)', cursor: 'pointer' }}
              >
                {status.models.length} model{status.models.length !== 1 ? 's' : ''} loaded
              </summary>
              <ul
                style={{
                  margin: '6px 0 0 0',
                  padding: '0 0 0 16px',
                  fontSize: '0.72rem',
                  color: 'var(--text-muted)',
                  lineHeight: 1.8,
                }}
              >
                {status.models.map(m => (
                  <li key={m}>{m}</li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}

      <button
        className="btn btn-ghost btn-sm"
        style={{ marginTop: 10, width: '100%' }}
        onClick={refresh}
        disabled={loading}
      >
        {loading ? 'Checking…' : '↻ Refresh'}
      </button>
    </div>
  )
}
