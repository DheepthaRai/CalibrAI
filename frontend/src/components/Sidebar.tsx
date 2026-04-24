import { useState } from 'react'
import ModelStatus from './ModelStatus'
import type { CalibrationState } from '../types'
import { INDUSTRIES } from '../types'

interface Props {
  industry: string
  setIndustry: (v: string) => void
  waveSize: number
  setWaveSize: (v: number) => void
  baseUrl: string
  setBaseUrl: (v: string) => void
  llamaModel: string
  setLlamaModel: (v: string) => void
  deepseekModel: string
  setDeepseekModel: (v: string) => void
  calibration: CalibrationState
  onStart: () => void
  onReset: () => void
}

export default function Sidebar({
  industry, setIndustry,
  waveSize, setWaveSize,
  baseUrl, setBaseUrl,
  llamaModel, setLlamaModel,
  deepseekModel, setDeepseekModel,
  calibration,
  onStart,
  onReset,
}: Props) {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const running = calibration.status === 'running'

  return (
    <aside style={{
      width: 260,
      flexShrink: 0,
      background: 'var(--bg-surface)',
      borderRight: '1px solid var(--border)',
      display: 'flex',
      flexDirection: 'column',
      padding: '20px 16px',
      gap: 20,
      overflowY: 'auto',
    }}>
      {/* Logo */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <div>
          <div style={{ fontWeight: 700, fontSize: '0.95rem', letterSpacing: '-0.02em' }}>CalibrAI</div>
          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Safety Calibration</div>
        </div>
      </div>

      <div className="divider" />

      {/* Industry */}
      <div>
        <div className="section-label">Target Industry</div>
        <select
          className="select"
          value={industry}
          onChange={e => setIndustry(e.target.value)}
          disabled={running}
        >
          {INDUSTRIES.map(i => <option key={i}>{i}</option>)}
        </select>
      </div>

      {/* Wave size */}
      <div>
        <div className="section-label">Queries per Wave</div>
        <input
          type="number"
          className="input"
          min={10}
          max={200}
          step={10}
          value={waveSize}
          onChange={e => setWaveSize(Number(e.target.value))}
          disabled={running}
        />
        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 4 }}>
          ×5 levels = {waveSize * 5} total LLM calls
        </div>
      </div>

      {/* Advanced */}
      <div>
        <button
          className="btn btn-ghost btn-sm"
          style={{ width: '100%', justifyContent: 'space-between' }}
          onClick={() => setShowAdvanced(v => !v)}
        >
          <span>Advanced Settings</span>
          <span>{showAdvanced ? '▲' : '▼'}</span>
        </button>

        {showAdvanced && (
          <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div>
              <div className="section-label">Ollama URL</div>
              <input
                type="text"
                className="input"
                value={baseUrl}
                onChange={e => setBaseUrl(e.target.value)}
                placeholder="http://localhost:11434"
              />
            </div>
            <div>
              <div className="section-label">LLaMA Model</div>
              <input
                type="text"
                className="input"
                value={llamaModel}
                onChange={e => setLlamaModel(e.target.value)}
                placeholder="llama3.1:latest"
              />
            </div>
            <div>
              <div className="section-label">DeepSeek Model</div>
              <input
                type="text"
                className="input"
                value={deepseekModel}
                onChange={e => setDeepseekModel(e.target.value)}
                placeholder="deepseek-r1:32b"
              />
            </div>
          </div>
        )}
      </div>

      <div className="divider" />

      {/* Progress */}
      {running && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            <span>Running calibration…</span>
            <span>{calibration.progress.toFixed(0)}%</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${calibration.progress}%` }} />
          </div>
          <div style={{ marginTop: 6, fontSize: '0.72rem', color: 'var(--text-muted)' }}>
            {calibration.completedQueries}/{calibration.totalQueries} queries tested
          </div>
        </div>
      )}

      {/* Run button */}
      <button
        className="btn btn-primary btn-lg"
        style={{ width: '100%', justifyContent: 'center' }}
        onClick={onStart}
        disabled={running}
      >
        {running ? (
          <>
            <span className="dot dot-yellow" />
            Running…
          </>
        ) : (
          '▶  Run Calibration'
        )}
      </button>

      {calibration.status !== 'idle' && (
        <button
          className="btn btn-ghost btn-sm"
          style={{ width: '100%', justifyContent: 'center' }}
          onClick={onReset}
          disabled={running}
        >
          ↺  Reset
        </button>
      )}

      {calibration.error && (
        <div style={{
          padding: '8px 10px',
          background: 'var(--red-dim)',
          border: '1px solid var(--red)',
          borderRadius: 'var(--radius)',
          fontSize: '0.78rem',
          color: 'var(--red)',
          lineHeight: 1.5,
        }}>
          {calibration.error}
        </div>
      )}

      <div className="divider" />

      {/* Model status */}
      <ModelStatus baseUrl={baseUrl} llamaModel={llamaModel} deepseekModel={deepseekModel} />
    </aside>
  )
}
