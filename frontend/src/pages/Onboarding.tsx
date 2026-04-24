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
      position: 'relative',
      overflow: 'hidden',
    }} className="grid-bg">

      {/* Glow orbs */}
      <div className="glow-orb" style={{
        width: 600, height: 600,
        background: 'rgba(79,142,247,0.08)',
        top: -200, left: -200,
      }} />
      <div className="glow-orb" style={{
        width: 400, height: 400,
        background: 'rgba(161,117,247,0.06)',
        bottom: -100, right: -100,
      }} />

      <div style={{ width: '100%', maxWidth: 680, position: 'relative', zIndex: 1 }}>

        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: 48 }}>
          <h1 className="gradient-text" style={{ fontSize: '2.4rem', marginBottom: 10, letterSpacing: '-0.03em' }}>
            CalibrAI
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '1rem', marginBottom: 8 }}>
            Governance-aligned LLM safety calibration
          </p>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            Answer three questions to get your recommended safety threshold.
          </p>
        </div>

        {/* Input form */}
        <div className="card card-glow" style={{ display: 'flex', flexDirection: 'column', gap: 28, borderColor: 'var(--border-bright)' }}>

          <div>
            <h2 style={{ marginBottom: 4 }}>Policy Configuration</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
              These inputs determine your regulatory risk profile and cost model.
            </p>
          </div>

          {/* Question 1 */}
          <div>
            <div style={{ fontWeight: 600, marginBottom: 4, fontSize: '0.92rem' }}>
              <span style={{ color: 'var(--accent)', marginRight: 8, fontFamily: 'var(--font-mono)', fontSize: '0.78rem' }}>01</span>
              What industry are you in?
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 12 }}>
              Determines your regulatory baseline and default risk tolerance.
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 8 }}>
              {INDUSTRIES.map(ind => {
                const selected = inputs.industry === ind
                return (
                  <button
                    key={ind}
                    onClick={() => update({ industry: ind })}
                    style={{
                      padding: '10px 8px',
                      borderRadius: 'var(--radius)',
                      border: `1px solid ${selected ? 'rgba(79,142,247,0.5)' : 'var(--border)'}`,
                      background: selected
                        ? 'linear-gradient(135deg, rgba(79,142,247,0.15), rgba(161,117,247,0.1))'
                        : 'var(--bg-input)',
                      color: selected ? 'var(--text-primary)' : 'var(--text-secondary)',
                      fontFamily: 'inherit',
                      fontSize: '0.8rem',
                      fontWeight: selected ? 600 : 400,
                      cursor: 'pointer',
                      transition: 'all 0.15s',
                      textAlign: 'center',
                      boxShadow: selected ? '0 0 12px rgba(79,142,247,0.15)' : 'none',
                    }}
                  >
                    {ind}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Question 2 */}
          <div>
            <div style={{ fontWeight: 600, marginBottom: 4, fontSize: '0.92rem' }}>
              <span style={{ color: 'var(--accent)', marginRight: 8, fontFamily: 'var(--font-mono)', fontSize: '0.78rem' }}>02</span>
              Estimated cost per compliance violation ($)
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 12 }}>
              Include regulatory fines, legal fees, remediation cost, and reputational damage.
            </div>
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
                    borderColor: inputs.costPerViolation === v ? 'rgba(79,142,247,0.5)' : undefined,
                    color: inputs.costPerViolation === v ? 'var(--accent)' : undefined,
                    background: inputs.costPerViolation === v ? 'rgba(79,142,247,0.08)' : undefined,
                  }}
                  onClick={() => update({ costPerViolation: v })}
                >
                  ${v >= 1000 ? `${v / 1000}K` : v}
                </button>
              ))}
            </div>
          </div>

          {/* Question 3 */}
          <div>
            <div style={{ fontWeight: 600, marginBottom: 4, fontSize: '0.92rem' }}>
              <span style={{ color: 'var(--accent)', marginRight: 8, fontFamily: 'var(--font-mono)', fontSize: '0.78rem' }}>03</span>
              Weekly query volume
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 12 }}>
              Higher volume means false positives accumulate faster and dominate cost.
            </div>
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
                    borderColor: inputs.weeklyVolume === v ? 'rgba(79,142,247,0.5)' : undefined,
                    color: inputs.weeklyVolume === v ? 'var(--accent)' : undefined,
                    background: inputs.weeklyVolume === v ? 'rgba(79,142,247,0.08)' : undefined,
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
              border: '1px solid rgba(240,84,100,0.3)',
              borderRadius: 'var(--radius)',
              color: 'var(--red)',
              fontSize: '0.82rem',
            }}>
              {error}
            </div>
          )}

          <button
            className="btn btn-primary btn-lg"
            style={{ alignSelf: 'stretch', justifyContent: 'center', fontSize: '0.95rem' }}
            onClick={calculate}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="dot dot-yellow" style={{ width: 6, height: 6 }} />
                Calculating…
              </>
            ) : (
              'Calculate Recommended Threshold →'
            )}
          </button>
        </div>

        {/* Recommendation result */}
        {recommendation && (
          <div
            className="card"
            style={{
              marginTop: 16,
              border: `1px solid ${LEVEL_COLORS[recommendation.recommended_level]}44`,
              borderLeft: `3px solid ${LEVEL_COLORS[recommendation.recommended_level]}`,
              boxShadow: `0 4px 24px rgba(0,0,0,0.4), 0 0 40px ${LEVEL_COLORS[recommendation.recommended_level]}18`,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 20 }}>
              {/* Level badge */}
              <div style={{
                minWidth: 76,
                height: 76,
                background: `${LEVEL_COLORS[recommendation.recommended_level]}18`,
                border: `1.5px solid ${LEVEL_COLORS[recommendation.recommended_level]}55`,
                borderRadius: 'var(--radius-lg)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
                boxShadow: `0 0 20px ${LEVEL_COLORS[recommendation.recommended_level]}20`,
              }}>
                <div style={{
                  fontSize: '1.6rem',
                  fontWeight: 800,
                  color: LEVEL_COLORS[recommendation.recommended_level],
                  lineHeight: 1,
                  fontFamily: 'var(--font-mono)',
                }}>
                  L{recommendation.recommended_level}
                </div>
                <div style={{
                  fontSize: '0.58rem',
                  color: LEVEL_COLORS[recommendation.recommended_level],
                  textTransform: 'uppercase',
                  letterSpacing: '0.06em',
                  textAlign: 'center',
                  marginTop: 4,
                  opacity: 0.8,
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
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 16 }}>
                  <strong style={{ color: 'var(--text-secondary)' }}>Risk profile:</strong> {recommendation.industry_risk_profile}
                </div>

                <div style={{ display: 'flex', gap: 12, marginBottom: 20 }}>
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
              }}
              onClick={() => navigate('/dashboard')}
            >
              Skip to dashboard →
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
      background: `${color}10`,
      border: `1px solid ${color}30`,
      borderRadius: 'var(--radius)',
      padding: '10px 16px',
      flex: 1,
    }}>
      <div style={{ fontSize: '0.67rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ fontWeight: 700, color, fontSize: '1.05rem', fontFamily: 'var(--font-mono)' }}>{value}</div>
    </div>
  )
}
