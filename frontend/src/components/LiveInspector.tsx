import { useState } from 'react'
import type { CalibrationState, QueryResult } from '../types'
import { LEVEL_NAMES } from '../types'

interface Props {
  calibration: CalibrationState
}

export default function LiveInspector({ calibration }: Props) {
  const [selectedQid, setSelectedQid] = useState<number | null>(null)
  const queries = calibration.queries

  const selected: QueryResult | undefined = selectedQid !== null
    ? queries.find(q => q.qid === selectedQid)
    : undefined

  if (queries.length === 0) {
    return (
      <div className="card">
        <h2 style={{ marginBottom: 8 }}>Live Inspector</h2>
        <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', padding: '20px 0' }}>
          No queries yet. Run a calibration wave to inspect individual results.
        </div>
      </div>
    )
  }

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div>
        <h2 style={{ marginBottom: 4 }}>Live Inspector</h2>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          Select any query to see how each of the 5 safety levels handled it
        </div>
      </div>

      {/* Query selector */}
      <select
        className="select"
        value={selectedQid ?? ''}
        onChange={e => setSelectedQid(Number(e.target.value))}
      >
        <option value="">Select a query…</option>
        {queries.map(q => (
          <option key={q.qid} value={q.qid}>
            [{q.is_attack ? 'ATTACK' : 'BENIGN'}] {q.query.slice(0, 70)}
            {q.query.length > 70 ? '…' : ''}
          </option>
        ))}
      </select>

      {selected && (
        <>
          {/* Query text + type */}
          <div style={{
            background: 'var(--bg-elevated)',
            borderRadius: 'var(--radius)',
            padding: '10px 14px',
            display: 'flex',
            gap: 10,
            alignItems: 'flex-start',
          }}>
            <span className={`badge ${selected.is_attack ? 'badge-red' : 'badge-green'}`} style={{ flexShrink: 0, marginTop: 1 }}>
              {selected.is_attack ? 'ATTACK' : 'BENIGN'}
            </span>
            <span style={{ fontSize: '0.85rem', lineHeight: 1.5 }}>{selected.query}</span>
          </div>

          {/* 5-level grid */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(5, 1fr)',
            gap: 8,
          }}>
            {[1, 2, 3, 4, 5].map(level => {
              const res = selected.results[level]
              const blocked = res?.blocked ?? false
              const bg = blocked ? 'var(--red-dim)' : 'var(--green-dim)'
              const borderColor = blocked ? 'var(--red)' : 'var(--green)'
              const textColor = blocked ? 'var(--red)' : 'var(--green)'

              return (
                <div key={level} style={{
                  background: bg,
                  border: `1px solid ${borderColor}`,
                  borderRadius: 'var(--radius)',
                  padding: '10px 8px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 6,
                }}>
                  <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
                    Level {level}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>
                    {LEVEL_NAMES[level]}
                  </div>
                  <div style={{
                    fontWeight: 700,
                    fontSize: '0.85rem',
                    color: textColor,
                    letterSpacing: '0.04em',
                  }}>
                    {blocked ? '⊘ BLOCKED' : '✓ ALLOWED'}
                  </div>
                  {res?.response && (
                    <div style={{
                      fontSize: '0.68rem',
                      color: 'var(--text-muted)',
                      lineHeight: 1.4,
                      overflow: 'hidden',
                      display: '-webkit-box',
                      WebkitLineClamp: 3,
                      WebkitBoxOrient: 'vertical' as const,
                    }}>
                      {res.response}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </>
      )}
    </div>
  )
}
