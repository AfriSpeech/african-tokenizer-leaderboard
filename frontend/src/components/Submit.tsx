import { useState, useEffect } from 'react'
import type { LanguageTab } from '../types'
import { api } from '../api'

interface Props {
  languageTabs: LanguageTab[]
}

export default function Submit({ languageTabs }: Props) {
  const [modelName, setModelName] = useState('')
  const [revision, setRevision] = useState('main')
  const [subfolder, setSubfolder] = useState('')
  const [targetMode, setTargetMode] = useState(false)
  const [selectedLangs, setSelectedLangs] = useState<string[]>(
    languageTabs.map((l) => l.code),
  )
  const [submitting, setSubmitting] = useState(false)
  const [statusMsg, setStatusMsg] = useState<string | null>(null)
  const [statusType, setStatusType] = useState<'success' | 'error'>('success')
  const [workerData, setWorkerData] = useState<Record<string, unknown>[]>([])

  useEffect(() => {
    api.workerStatus().then((r) => setWorkerData(r.worker_status_df)).catch(() => {})
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!modelName.trim()) {
      setStatusMsg('Model name cannot be empty.')
      setStatusType('error')
      return
    }
    setSubmitting(true)
    setStatusMsg(null)
    try {
      const res = await api.submitModel({
        model_name: modelName.trim(),
        revision: revision || 'main',
        subfolder: subfolder || undefined,
        target_mode: targetMode,
        selected_langs: selectedLangs.length > 0 ? selectedLangs : undefined,
      })
      setStatusMsg(res.message)
      setStatusType(res.message.toLowerCase().includes('error') ? 'error' : 'success')
      setWorkerData(res.worker_status_df)
    } catch (e: unknown) {
      setStatusMsg(e instanceof Error ? e.message : 'Submission failed')
      setStatusType('error')
    } finally {
      setSubmitting(false)
    }
  }

  const refreshWorker = async () => {
    try {
      const r = await api.workerStatus()
      setWorkerData(r.worker_status_df)
    } catch {}
  }

  const handleLangToggle = (code: string) => {
    setSelectedLangs((prev) =>
      prev.includes(code) ? prev.filter((c) => c !== code) : [...prev, code],
    )
  }

  return (
    <div className="submit">
      <h2 className="submit__title">Submit a Tokenizer for Evaluation</h2>
      <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>
        Submit any publicly hosted Hugging Face tokenizer to be evaluated and ranked on the leaderboard.
      </p>

      <form className="submit__form" onSubmit={handleSubmit}>
        <div className="field">
          <label className="field__label">Model Name</label>
          <input
            className="input"
            type="text"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder="e.g., google-bert/bert-base-multilingual-cased"
          />
        </div>

        <div className="field">
          <label className="field__label">Revision</label>
          <input
            className="input"
            type="text"
            value={revision}
            onChange={(e) => setRevision(e.target.value)}
            placeholder="main"
          />
          <span className="field__hint">Branch, tag, or commit hash</span>
        </div>

        <div className="field">
          <label className="field__label">Subfolder (optional)</label>
          <input
            className="input"
            type="text"
            value={subfolder}
            onChange={(e) => setSubfolder(e.target.value)}
            placeholder="e.g., tokenizer"
          />
        </div>

        <div className="toggle-row">
          <div
            className={`toggle${targetMode ? ' toggle--active' : ''}`}
            onClick={() => setTargetMode(!targetMode)}
          >
            <div className="toggle__knob" />
          </div>
          <span className="toggle-label" onClick={() => setTargetMode(!targetMode)}>
            Target Mode (for models like MarianTokenizer with separate source/target vocabularies)
          </span>
        </div>

        <div className="filter-section">
          <div className="filter-section__header">
            <span className="filter-section__title">Languages to Evaluate</span>
            <div className="filter-section__actions">
              <button
                type="button"
                className="btn btn--ghost btn--sm"
                onClick={() => setSelectedLangs(languageTabs.map((l) => l.code))}
              >
                Select All
              </button>
              <button
                type="button"
                className="btn btn--ghost btn--sm"
                onClick={() => setSelectedLangs([])}
              >
                Clear
              </button>
            </div>
          </div>
          <div className="chip-grid">
            {languageTabs.map((lang) => {
              const active = selectedLangs.includes(lang.code)
              return (
                <button
                  key={lang.code}
                  type="button"
                  className={`chip${active ? ' chip--active' : ''}`}
                  onClick={() => handleLangToggle(lang.code)}
                >
                  <span className="chip__check" />
                  {lang.name}
                </button>
              )
            })}
          </div>
          <span className="field__hint">Leave empty to evaluate all languages</span>
        </div>

        <button
          type="submit"
          className="btn btn--primary"
          disabled={submitting}
          style={{ alignSelf: 'flex-start' }}
        >
          {submitting ? 'Submitting...' : '📤 Submit for Evaluation'}
        </button>
      </form>

      {statusMsg && (
        <div className={`submit__status submit__status--${statusType}`}>
          {statusMsg}
        </div>
      )}

      {/* Worker Status */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
          <h3 style={{ fontSize: '16px', fontWeight: 600 }}>Evaluation Queue</h3>
          <button className="btn btn--sm" onClick={refreshWorker}>
            🔄 Refresh
          </button>
        </div>
        {workerData.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table className="submit__worker-table">
              <thead>
                <tr>
                  {Object.keys(workerData[0]).map((h) => (
                    <th key={h}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {workerData.map((row, i) => (
                  <tr key={i}>
                    {Object.values(row).map((v, j) => (
                      <td key={j}>{v == null ? '' : String(v)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p style={{ color: 'var(--text-muted)', fontSize: '13px' }}>
            No models currently in the evaluation queue.
          </p>
        )}
      </div>
    </div>
  )
}
