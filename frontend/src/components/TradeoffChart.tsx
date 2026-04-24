import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  TooltipProps,
} from 'recharts'
import type { CalibrationSummary } from '../types'
import { LEVEL_NAMES } from '../types'

interface Props {
  levelStats: CalibrationSummary
  recommendedLevel: number | null
  isRunning: boolean
}

interface ChartPoint {
  level: number
  label: string
  attackBlockRate: number
  falsePositiveRate: number
}

function buildData(stats: CalibrationSummary): ChartPoint[] {
  return [1, 2, 3, 4, 5].map(lvl => ({
    level: lvl,
    label: LEVEL_NAMES[lvl],
    attackBlockRate: stats[lvl] ? Math.round(stats[lvl].attack_block_rate * 100) : 0,
    falsePositiveRate: stats[lvl] ? Math.round(stats[lvl].fp_rate * 100) : 0,
  }))
}

/**
 * Find the level where attack_block_rate and fp_rate cross.
 * Returns the level number at the crossover or null if none.
 */
function findCrossover(data: ChartPoint[]): number | null {
  for (let i = 0; i < data.length - 1; i++) {
    const a = data[i], b = data[i + 1]
    const crossesOver =
      (a.attackBlockRate >= a.falsePositiveRate && b.attackBlockRate <= b.falsePositiveRate) ||
      (a.attackBlockRate <= a.falsePositiveRate && b.attackBlockRate >= b.falsePositiveRate)
    if (crossesOver) {
      // Return the level that is closer to the crossover
      const aDiff = Math.abs(a.attackBlockRate - a.falsePositiveRate)
      const bDiff = Math.abs(b.attackBlockRate - b.falsePositiveRate)
      return aDiff < bDiff ? a.level : b.level
    }
  }
  return null
}

const CustomTooltip = ({ active, payload, label }: TooltipProps<number, string>) => {
  if (!active || !payload?.length) return null
  const lvl = payload[0]?.payload as ChartPoint
  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border-bright)',
      borderRadius: 8,
      padding: '10px 14px',
      fontSize: '0.82rem',
    }}>
      <div style={{ fontWeight: 600, marginBottom: 6, color: 'var(--text-primary)' }}>
        Level {lvl.level} — {lvl.label}
      </div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ display: 'flex', justifyContent: 'space-between', gap: 20, color: p.color }}>
          <span>{p.name}</span>
          <span style={{ fontWeight: 600 }}>{p.value}%</span>
        </div>
      ))}
    </div>
  )
}

export default function TradeoffChart({ levelStats, recommendedLevel, isRunning }: Props) {
  const hasData = Object.keys(levelStats).length > 0
  const data = buildData(levelStats)
  const crossover = hasData ? findCrossover(data) : null

  return (
    <div className="card" style={{ position: 'relative' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div>
          <h2 style={{ marginBottom: 2 }}>Safety vs Utility Tradeoff</h2>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Attack block rate (green) and false positive rate (red) across all 5 safety levels
          </div>
        </div>
        <div style={{ display: 'flex', gap: 10, fontSize: '0.75rem' }}>
          {isRunning && (
            <span className="badge badge-yellow">
              <span className="dot dot-yellow" style={{ width: 6, height: 6 }} />
              Live
            </span>
          )}
          {crossover && (
            <span className="badge badge-yellow">Crossover at L{crossover}</span>
          )}
          {recommendedLevel && (
            <span className="badge badge-blue">Optimal: L{recommendedLevel}</span>
          )}
        </div>
      </div>

      {!hasData ? (
        <div style={{
          height: 300,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-muted)',
          gap: 10,
          border: '1px dashed var(--border)',
          borderRadius: 'var(--radius)',
        }}>
          <span style={{ fontSize: '1.8rem' }}>📊</span>
          <span style={{ fontSize: '0.85rem' }}>Run a calibration wave to see the tradeoff chart</span>
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 30 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis
              dataKey="level"
              ticks={[1, 2, 3, 4, 5]}
              tickFormatter={v => `L${v} · ${LEVEL_NAMES[v as number]}`}
              tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
              axisLine={{ stroke: 'var(--border)' }}
              tickLine={false}
              angle={-15}
              textAnchor="end"
              height={50}
            />
            <YAxis
              domain={[0, 100]}
              tickFormatter={v => `${v}%`}
              tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
              axisLine={{ stroke: 'var(--border)' }}
              tickLine={false}
              width={40}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: '0.8rem', paddingTop: 10, color: 'var(--text-secondary)' }}
            />

            {/* Crossover reference line */}
            {crossover && (
              <ReferenceLine
                x={crossover}
                stroke="var(--yellow)"
                strokeDasharray="6 4"
                strokeWidth={1.5}
                label={{
                  value: 'Crossover',
                  position: 'top',
                  fill: 'var(--yellow)',
                  fontSize: 11,
                }}
              />
            )}

            {/* Optimal level reference line */}
            {recommendedLevel && (
              <ReferenceLine
                x={recommendedLevel}
                stroke="var(--accent)"
                strokeDasharray="4 4"
                strokeWidth={2}
                label={{
                  value: `Optimal: L${recommendedLevel}`,
                  position: 'insideTopRight',
                  fill: 'var(--accent)',
                  fontSize: 11,
                }}
              />
            )}

            <Line
              type="monotone"
              dataKey="attackBlockRate"
              name="Attack Block Rate"
              stroke="var(--green)"
              strokeWidth={2.5}
              dot={{ r: 5, fill: 'var(--green)', strokeWidth: 0 }}
              activeDot={{ r: 7 }}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="falsePositiveRate"
              name="False Positive Rate"
              stroke="var(--red)"
              strokeWidth={2.5}
              dot={{ r: 5, fill: 'var(--red)', strokeWidth: 0 }}
              activeDot={{ r: 7 }}
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
