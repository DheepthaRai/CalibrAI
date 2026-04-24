import { useNavigate } from 'react-router-dom'
import { usePolicy } from '../hooks/usePolicy'
import { INDUSTRIES, LEVEL_NAMES } from '../types'

const LEVEL_COLORS: Record<number, string> = {
  1: 'var(--red)',
  2: 'var(--yellow)',
  3: 'var(--green)',
  4: 'var(--accent)',
  5: 'var(--purple)',
}

export default function Onboarding() {
  const navigate = useNavigate()
  const { inputs, update, recommendation, calculate, loading, error } = usePolicy()

  const handleContinue = () => {
    navigate('/dashboard', { state: { policy: recommendation, inputs } })
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: 'var(--bg-base)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: 32,
    }}>
      <div style={{ width: '100%', maxWidth: 680 }}>
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: 48 }}>
          <div style={{ fontSize: '2.5rem', marginBottom: 12 }}>🛡️</div>
          <h1 style={{ marginBottom: 8, fontSize: '2rem' }}>CalibrAI</h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '1rem' }}>
            Governance-aligned LLM safety calibration
          </p>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: 8 }}>
            Answer three questions to get a recommended safety threshold before running calibration.
          </p>
        </div>

        {/* Input form */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          <div>
            <h2 style={{ marginBottom: 4 }}>Policy Configuration</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
              These inputs determine your regulatory risk profile and cost model.
            </p>
          </div>

          {/* Question 1: Industry */}
          <div>
            <label style={{ display: 'block', marginBottom: 10 }}>
              <div style={{ fontWeight: 600, marginBottom: 4 }}>1. What industry are you in?</div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 10 }}>
                Determines your regulatory baseline and default risk tolerance.
              </div>
            </label>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 8 }}>
              {INDUSTRIES.map(ind => (
                <button
                  key={ind}
                  onClick={() => update({ industry: ind })}
                  style={{
                    padding: '10px 8px',
                    borderRadius: 'var(--radius)',
                    border: `1px solid ${inputs.industry === ind ? 'var(--accent)' : 'var(--border)'}`,
                    background: inputs.industry === ind ? 'rgba(59,130,246,.12)' : 'var(--bg-input)',
                    color: inputs.industry === ind ? 'var(--accent)' : 'var(--text-secondary)',
                    fontFamily: 'inherit',
                    fontSize: '0.8rem',
                    fontWeight: inputs.industry === ind ? 600 : 400,
                    cursor: 'pointer',
                    transition: 'all 0.15s',
                    textAlign: 'center',
                  }}
                >
                  {ind}
                </button>
              ))}
            </div>
          </div>

          {/* Question 2: Cost per violation */}
          <div>
            <label style={{ display: 'block', marginBottom: 10 }}>
              <div style={{ fontWeight: 600, marginBottom: 4 }}>2. Estimated cost per compliance violation ($)</div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 10 }}>
                Include regulatory fines, legal fees, remediation cost, and reputational damage.
              </div>
            </label>
            <div className="input-prefix-group">
              <span className="prefix">$</span>
              <input
                type="number"
                className="input"
                min={100}
                step={500}
                value={inputs.costPerViolation}
                onChange={e => update({ costPerViolation: Number(e.target.value) })}
                placeholder="5000"
              />
            </div>
            <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
              {[500, 2500, 10000, 50000, 250000].map(v => (
                <button
                  key={v}
                  className="btn btn-ghost btn-sm"
                  style={{
                    fontSize: '0.72rem',
                    borderColor: inputs.costPerViolation === v ? 'var(--accent)' : undefined,
                    color: inputs.costPerViolation === v ? 'var(--accent)' : undefined,
                  }}
                  onClick={() => update({ costPerViolation: v })}
                >
                  ${v >= 1000 ? `${v / 1000}K` : v}
                </button>
              ))}
            </div>
          </div>

          {/* Question 3: Weekly volume */}
          <div>
            <label style={{ display: 'block', marginBottom: 10 }}>
              <div style={{ fontWeight: 600, marginBottom: 4 }}>3. Weekly query volume</div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 10 }}>
                Higher volume means false positives accumulate faster and dominate cost.
              </div>
            </label>
            <input
              type="number"
              className="input"
              min={100}
              step={1000}
              value={inputs.weeklyVolume}
              onChange={e => update({ weeklyVolume: Number(e.target.value) })}
              placeholder="10000"
            />
            <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
              {[1000, 10000, 50000, 100000, 500000].map(v => (
                <button
                  key={v}
                  className="btn btn-ghost btn-sm"
                  style={{
                    fontSize: '0.72rem',
                    borderColor: inputs.weeklyVolume === v ? 'var(--accent)' : undefined,
                    color: inputs.weeklyVolume === v ? 'var(--accent)' : undefined,
                  }}
                  onClick={() => update({ weeklyVolume: v })}
                >
                  {v >= 1000 ? `${v / 1000}K` : v}
                </button>
              ))}
            </div>
          </div>

          {error && (
            <div style={{
              padding: '10px 14px',
              background: 'var(--red-dim)',
              border: '1px solid var(--red)',
              borderRadius: 'var(--radius)',
              color: 'var(--red)',
              fontSize: '0.82rem',
            }}>
              {error}
            </div>
          )}

          <button
            className="btn btn-primary btn-lg"
            style={{ alignSelf: 'stretch', justifyContent: 'center' }}
            onClick={calculate}
            disabled={loading}
          >
            {loading ? 'Calculating…' : 'Calculate Recommended Threshold →'}
          </button>
        </div>

        {/* Recommendation result */}
        {recommendation && (
          <div
            className="card"
            style={{
              marginTop: 20,
              borderColor: LEVEL_COLORS[recommendation.recommended_level] ?? 'var(--accent)',
              borderLeftWidth: 4,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 20 }}>
              {/* Level badge */}
              <div style={{
                minWidth: 72,
                height: 72,
                background: `${LEVEL_COLORS[recommendation.recommended_level]}22`,
                border: `2px solid ${LEVEL_COLORS[recommendation.recommended_level]}`,
                borderRadius: 'var(--radius)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
              }}>
                <div style={{
                  fontSize: '1.5rem',
                  fontWeight: 800,
                  color: LEVEL_COLORS[recommendation.recommended_level],
                  lineHeight: 1,
                }}>
                  L{recommendation.recommended_level}
                </div>
                <div style={{
                  fontSize: '0.58rem',
                  color: LEVEL_COLORS[recommendation.recommended_level],
                  textTransform: 'uppercase',
                  letterSpacing: '0.06em',
                  textAlign: 'center',
                  marginTop: 2,
                }}>
                  {LEVEL_NAMES[recommendation.recommended_level]}
                </div>
              </div>

              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 700, fontSize: '1rem', marginBottom: 6 }}>
                  Recommended: Level {recommendation.recommended_level} — {recommendation.level_name}
                </div>
                <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 12 }}>
                  {recommendation.explanation}
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 14 }}>
                  <strong style={{ color: 'var(--text-secondary)' }}>Risk profile:</strong> {recommendation.industry_risk_profile}
                </div>

                {/* Cost estimates */}
                <div style={{ display: 'flex', gap: 20, marginBottom: 18 }}>
                  <CostCard
                    label="Est. Weekly FP Cost"
                    value={`$${recommendation.weekly_fp_cost_estimate.toLocaleString()}`}
                    color="var(--red)"
                  />
                  <CostCard
                    label="Est. Weekly Attack Exposure"
                    value={`$${recommendation.weekly_attack_cost_estimate.toLocaleString()}`}
                    color="var(--yellow)"
                  />
                </div>

                <button
                  className="btn btn-primary"
                  style={{ padding: '10px 28px' }}
                  onClick={handleContinue}
                >
                  Continue to Dashboard →
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Skip link */}
        {!recommendation && (
          <div style={{ textAlign: 'center', marginTop: 16 }}>
            <button
              style={{
                background: 'none',
                border: 'none',
                color: 'var(--text-muted)',
                fontSize: '0.8rem',
                cursor: 'pointer',
                textDecoration: 'underline',
              }}
              onClick={() => navigate('/dashboard')}
            >
              Skip to dashboard
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

function CostCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{
      background: `${color}12`,
      border: `1px solid ${color}44`,
      borderRadius: 'var(--radius)',
      padding: '8px 14px',
    }}>
      <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 3 }}>
        {label}
      </div>
      <div style={{ fontWeight: 700, color, fontSize: '1rem' }}>{value}</div>
    </div>
  )
}
