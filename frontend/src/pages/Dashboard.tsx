import { useState } from 'react'
import { useLocation } from 'react-router-dom'
import Sidebar from '../components/Sidebar'
import TradeoffChart from '../components/TradeoffChart'
import RecommendationBanner from '../components/RecommendationBanner'
import FailureAnalysis from '../components/FailureAnalysis'
import LiveInspector from '../components/LiveInspector'
import AuditLog from '../components/AuditLog'
import { useCalibration } from '../hooks/useCalibration'
import type { PolicyRecommendation } from '../types'

export default function Dashboard() {
  const location = useLocation()
  const locationPolicy = location.state?.policy as PolicyRecommendation | undefined
  const locationInputs = location.state?.inputs

  // Config state
  const [industry, setIndustry] = useState<string>(locationInputs?.industry ?? 'Banking')
  const [waveSize, setWaveSize] = useState(10)
  const [baseUrl, setBaseUrl] = useState('http://localhost:11434')
  const [llamaModel, setLlamaModel] = useState('llama3.1:latest')
  const [deepseekModel, setDeepseekModel] = useState('deepseek-r1:7b')
  const [costPerViolation] = useState<number>(locationInputs?.costPerViolation ?? 5000)
  const [weeklyVolume] = useState<number>(locationInputs?.weeklyVolume ?? 10000)

  const { state: calibration, start, reset } = useCalibration()

  // Which level to show in the failure panel — default to recommended or 3
  const displayLevel = calibration.recommendedLevel ?? 3

  const handleStart = () => {
    start({
      industry,
      wave_size: waveSize,
      cost_per_violation: costPerViolation,
      weekly_volume: weeklyVolume,
      base_url: baseUrl,
      llama_model: llamaModel,
      deepseek_model: deepseekModel,
    })
  }

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      <Sidebar
        industry={industry}
        setIndustry={setIndustry}
        waveSize={waveSize}
        setWaveSize={setWaveSize}
        baseUrl={baseUrl}
        setBaseUrl={setBaseUrl}
        llamaModel={llamaModel}
        setLlamaModel={setLlamaModel}
        deepseekModel={deepseekModel}
        setDeepseekModel={setDeepseekModel}
        calibration={calibration}
        onStart={handleStart}
        onReset={reset}
      />

      {/* Main content */}
      <main style={{
        flex: 1,
        overflowY: 'auto',
        padding: 24,
        display: 'flex',
        flexDirection: 'column',
        gap: 20,
      }}>
        {/* Top bar */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <h1 style={{ fontSize: '1.3rem', marginBottom: 2 }}>Calibration Dashboard</h1>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              {industry} · {waveSize} queries × 5 levels
              {calibration.runId && (
                <span style={{ marginLeft: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                  run {calibration.runId.slice(0, 8)}
                </span>
              )}
            </div>
          </div>

          {calibration.status === 'complete' && (
            <span className="badge badge-green" style={{ fontSize: '0.8rem', padding: '4px 12px' }}>
              ✓ Calibration Complete
            </span>
          )}
          {calibration.status === 'running' && (
            <span className="badge badge-yellow" style={{ fontSize: '0.8rem', padding: '4px 12px' }}>
              <span className="dot dot-yellow" style={{ width: 6, height: 6 }} />
              Running…
            </span>
          )}
          {calibration.status === 'error' && (
            <span className="badge badge-red" style={{ fontSize: '0.8rem', padding: '4px 12px' }}>
              ✗ Error
            </span>
          )}
        </div>

        {/* Policy recommendation banner (from onboarding, before calibration) */}
        {locationPolicy && calibration.status === 'idle' && (
          <div style={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderLeft: '4px solid var(--accent)',
            borderRadius: 'var(--radius-lg)',
            padding: '14px 20px',
            fontSize: '0.85rem',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span className="badge badge-blue">Policy Recommendation</span>
              <span style={{ color: 'var(--text-secondary)' }}>
                Based on your policy inputs: Level {locationPolicy.recommended_level} ({locationPolicy.level_name}) — {locationPolicy.level_description}
              </span>
            </div>
          </div>
        )}

        {/* Calibration recommendation banner (after calibration) */}
        <RecommendationBanner
          calibration={calibration}
          costPerViolation={costPerViolation}
          weeklyVolume={weeklyVolume}
          industry={industry}
        />

        {/* ── TRADEOFF CHART — visual centerpiece ─────────────────── */}
        <TradeoffChart
          levelStats={calibration.levelStats}
          recommendedLevel={calibration.recommendedLevel}
          isRunning={calibration.status === 'running'}
        />

        {/* Idle prompt */}
        {calibration.status === 'idle' && (
          <div style={{
            textAlign: 'center',
            padding: '30px',
            color: 'var(--text-muted)',
            fontSize: '0.88rem',
            background: 'var(--bg-surface)',
            borderRadius: 'var(--radius-lg)',
            border: '1px dashed var(--border)',
          }}>
            Click <strong style={{ color: 'var(--text-secondary)' }}>▶ Run Calibration</strong> in the sidebar to start.
            The chart will update in real-time as queries are tested.
          </div>
        )}

        {/* Failure analysis + inspector row */}
        {(calibration.status === 'running' || calibration.status === 'complete') && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
            <FailureAnalysis
              calibration={calibration}
              costPerViolation={costPerViolation}
              selectedLevel={displayLevel}
            />
            <LiveInspector calibration={calibration} />
          </div>
        )}

        {/* Audit log */}
        {calibration.status === 'complete' && (
          <AuditLog runId={calibration.runId} />
        )}
      </main>
    </div>
  )
}
