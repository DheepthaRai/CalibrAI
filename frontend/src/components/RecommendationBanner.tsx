import type { CalibrationState } from '../types'
import { LEVEL_NAMES } from '../types'

interface Props {
  calibration: CalibrationState
  costPerViolation: number
  weeklyVolume: number
  industry: string
}

const LEVEL_COLORS: Record<number, string> = {
  1: 'var(--red)',
  2: 'var(--yellow)',
  3: 'var(--green)',
  4: 'var(--accent)',
  5: 'var(--purple)',
}

export default function RecommendationBanner({
  calibration,
  costPerViolation,
  weeklyVolume,
  industry,
}: Props) {
  if (calibration.status !== 'complete' || !calibration.recommendedLevel) return null

  const lvl = calibration.recommendedLevel
  const stats = calibration.levelStats[lvl]
  const color = LEVEL_COLORS[lvl] ?? 'var(--accent)'

  return (
    <div style={{
      background: 'var(--bg-card)',
      border: `1px solid ${color}`,
      borderLeft: `4px solid ${color}`,
      borderRadius: 'var(--radius-lg)',
      padding: '18px 24px',
      display: 'flex',
      alignItems: 'flex-start',
      gap: 20,
    }}>
      {/* Level badge */}
      <div style={{
        minWidth: 64,
        height: 64,
        background: `${color}22`,
        border: `2px solid ${color}`,
        borderRadius: 'var(--radius)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <div style={{ fontSize: '1.4rem', fontWeight: 800, color, lineHeight: 1 }}>L{lvl}</div>
        <div style={{ fontSize: '0.6rem', color, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          {LEVEL_NAMES[lvl]}
        </div>
      </div>

      {/* Main text */}
      <div style={{ flex: 1 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
          <h2 style={{ color }}>Recommended: Level {lvl} — {LEVEL_NAMES[lvl]}</h2>
        </div>
        <p style={{ fontSize: '0.86rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 12 }}>
          {calibration.recommendationText}
        </p>

        {/* Stats row */}
        {stats && (
          <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
            <Stat
              label="Attack Block Rate"
              value={`${(stats.attack_block_rate * 100).toFixed(1)}%`}
              color="var(--green)"
            />
            <Stat
              label="False Positive Rate"
              value={`${(stats.fp_rate * 100).toFixed(1)}%`}
              color="var(--red)"
            />
            <Stat
              label="Safety Score"
              value={stats.score.toFixed(3)}
              color={color}
            />
            <Stat label="Industry" value={industry} />
            <Stat label="Cost / Violation" value={`$${costPerViolation.toLocaleString()}`} />
            <Stat label="Weekly Volume" value={weeklyVolume.toLocaleString()} />
          </div>
        )}
      </div>
    </div>
  )
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 2 }}>
        {label}
      </div>
      <div style={{ fontWeight: 700, fontSize: '1rem', color: color ?? 'var(--text-primary)' }}>
        {value}
      </div>
    </div>
  )
}
