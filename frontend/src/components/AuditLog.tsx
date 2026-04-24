import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { AuditRow } from '../types'
import { LEVEL_NAMES } from '../types'

interface Props {
  runId: string | null
}

export default function AuditLog({ runId }: Props) {
  const [rows, setRows] = useState<AuditRow[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    if (!runId) return
    setLoading(true)
    api.getAudit(runId, 1, 250)
      .then((res: any) => {
        setRows(res.items ?? [])
        setTotal(res.total ?? 0)
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [runId])

  const exportCsv = async () => {
    const res = await api.exportAuditCsv(runId ?? undefined)
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `calibrai_audit_${runId ?? 'all'}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (!runId) return null

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <h2>Governance Audit Log</h2>
          {total > 0 && (
            <span className="badge badge-ghost">{total} rows</span>
          )}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-ghost btn-sm" onClick={exportCsv} disabled={rows.length === 0}>
            ⬇ Export CSV
          </button>
          <button className="btn btn-ghost btn-sm" onClick={() => setExpanded(v => !v)}>
            {expanded ? '▲ Collapse' : '▼ Expand'}
          </button>
        </div>
      </div>

      {loading && (
        <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Loading audit log…</div>
      )}

      {!loading && rows.length === 0 && (
        <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', padding: '10px 0' }}>
          No audit records yet.
        </div>
      )}

      {!loading && rows.length > 0 && (
        <div style={{
          overflowX: 'auto',
          maxHeight: expanded ? 'none' : 280,
          overflowY: expanded ? 'visible' : 'auto',
        }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.78rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)' }}>
                {['Timestamp', 'Query', 'Type', 'Risk', 'Level', 'Outcome', 'Justification'].map(h => (
                  <th key={h} style={{
                    padding: '6px 10px',
                    textAlign: 'left',
                    color: 'var(--text-muted)',
                    fontWeight: 600,
                    fontSize: '0.68rem',
                    textTransform: 'uppercase',
                    letterSpacing: '0.06em',
                    whiteSpace: 'nowrap',
                    position: 'sticky',
                    top: 0,
                    background: 'var(--bg-card)',
                  }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={row.id} style={{
                  borderBottom: '1px solid var(--border)',
                  background: i % 2 === 0 ? 'transparent' : 'var(--bg-elevated)',
                }}>
                  <td style={{ padding: '5px 10px', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                    {new Date(row.timestamp).toLocaleTimeString()}
                  </td>
                  <td style={{ padding: '5px 10px', maxWidth: 220 }}>
                    <div className="truncate">{row.query_text}</div>
                  </td>
                  <td style={{ padding: '5px 10px' }}>
                    <span className={`badge ${row.is_attack ? 'badge-red' : 'badge-green'}`}>
                      {row.is_attack ? 'Attack' : 'Benign'}
                    </span>
                  </td>
                  <td style={{ padding: '5px 10px', fontFamily: 'var(--font-mono)', color: riskColor(row.risk_score) }}>
                    {row.risk_score.toFixed(2)}
                  </td>
                  <td style={{ padding: '5px 10px', color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
                    L{row.safety_level} · {LEVEL_NAMES[row.safety_level]}
                  </td>
                  <td style={{ padding: '5px 10px' }}>
                    <span className={`badge ${row.outcome === 'BLOCKED' ? 'badge-red' : 'badge-green'}`}>
                      {row.outcome}
                    </span>
                  </td>
                  <td style={{ padding: '5px 10px', color: 'var(--text-muted)', maxWidth: 240 }}>
                    <div className="truncate">{row.cost_justification}</div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function riskColor(score: number): string {
  if (score >= 0.8) return 'var(--red)'
  if (score >= 0.4) return 'var(--yellow)'
  return 'var(--green)'
}
