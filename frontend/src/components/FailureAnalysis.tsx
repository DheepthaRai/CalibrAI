import type { CalibrationState, QueryResult } from '../types'
import { LEVEL_NAMES } from '../types'

interface Props {
  calibration: CalibrationState
  costPerViolation: number
  selectedLevel: number
}

export default function FailureAnalysis({ calibration, costPerViolation, selectedLevel }: Props) {
  const queries = calibration.queries

  const fps: QueryResult[] = queries.filter(
    q => !q.is_attack && q.results[selectedLevel]?.blocked,
  )
  const fns: QueryResult[] = queries.filter(
    q => q.is_attack && !q.results[selectedLevel]?.blocked,
  )

  const fpCost = costPerViolation * 0.02    // FP = 2% of violation cost
  const fnCost = costPerViolation           // FN = full violation cost

  const isEmpty = queries.length === 0

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h2>Failure Analysis</h2>
        <span className="badge badge-ghost" style={{ fontSize: '0.72rem' }}>
          Level {selectedLevel} — {LEVEL_NAMES[selectedLevel]}
        </span>
      </div>

      {isEmpty ? (
        <div style={{
          textAlign: 'center',
          padding: '30px 0',
          color: 'var(--text-muted)',
          fontSize: '0.85rem',
        }}>
          No calibration data yet. Run a wave to see failure analysis.
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {/* False Positives panel */}
          <Panel
            title="False Positives"
            subtitle="Legitimate queries blocked"
            count={fps.length}
            color="var(--red)"
            costLabel={`Est. $${(fps.length * fpCost).toFixed(2)} user-friction cost`}
            empty="No false positives at this level"
          >
            {fps.slice(0, 8).map((q, i) => (
              <QueryRow key={i} query={q.query} type="fp" response={q.results[selectedLevel]?.response} />
            ))}
            {fps.length > 8 && (
              <div style={{ fontSize: '0.73rem', color: 'var(--text-muted)', paddingTop: 4 }}>
                + {fps.length - 8} more
              </div>
            )}
          </Panel>

          {/* False Negatives panel */}
          <Panel
            title="False Negatives"
            subtitle="Attacks that slipped through"
            count={fns.length}
            color="var(--yellow)"
            costLabel={`Est. $${(fns.length * fnCost).toFixed(2)} regulatory exposure`}
            empty="No false negatives at this level"
          >
            {fns.slice(0, 8).map((q, i) => (
              <QueryRow key={i} query={q.query} type="fn" response={q.results[selectedLevel]?.response} />
            ))}
            {fns.length > 8 && (
              <div style={{ fontSize: '0.73rem', color: 'var(--text-muted)', paddingTop: 4 }}>
                + {fns.length - 8} more
              </div>
            )}
          </Panel>
        </div>
      )}
    </div>
  )
}

function Panel({
  title, subtitle, count, color, costLabel, empty, children,
}: {
  title: string
  subtitle: string
  count: number
  color: string
  costLabel: string
  empty: string
  children: React.ReactNode
}) {
  return (
    <div style={{
      background: 'var(--bg-elevated)',
      border: `1px solid var(--border)`,
      borderTop: `3px solid ${color}`,
      borderRadius: 'var(--radius)',
      padding: '14px 16px',
      display: 'flex',
      flexDirection: 'column',
      gap: 10,
      minHeight: 200,
    }}>
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2 }}>
          <h3 style={{ color }}>{title}</h3>
          <span className="badge" style={{
            background: `${color}22`,
            color,
            fontSize: '0.72rem',
          }}>
            {count}
          </span>
        </div>
        <div style={{ fontSize: '0.73rem', color: 'var(--text-muted)' }}>{subtitle}</div>
        <div style={{
          marginTop: 6,
          fontSize: '0.72rem',
          color,
          background: `${color}18`,
          padding: '3px 8px',
          borderRadius: 4,
          display: 'inline-block',
        }}>
          {costLabel}
        </div>
      </div>
      <div style={{ flex: 1, overflowY: 'auto', maxHeight: 260, display: 'flex', flexDirection: 'column', gap: 6 }}>
        {count === 0 ? (
          <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', padding: '8px 0' }}>
            ✓ {empty}
          </div>
        ) : children}
      </div>
    </div>
  )
}

function QueryRow({ query, type, response }: { query: string; type: 'fp' | 'fn'; response?: string }) {
  const borderColor = type === 'fp' ? 'var(--red)' : 'var(--yellow)'
  return (
    <div style={{
      background: 'var(--bg-card)',
      borderLeft: `2px solid ${borderColor}`,
      borderRadius: '0 4px 4px 0',
      padding: '6px 10px',
      fontSize: '0.78rem',
    }}>
      <div style={{ color: 'var(--text-primary)', marginBottom: 2 }} className="truncate">
        {query}
      </div>
      {response && (
        <div className="truncate" style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>
          {response.slice(0, 80)}…
        </div>
      )}
    </div>
  )
}
