import { useState } from 'react'
import { api } from '../api'

export default function Submit() {
  const [model, setModel] = useState('')
  const [revision, setRevision] = useState('main')
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!model.trim()) return
    setBusy(true)
    setMessage(null)
    setError(null)
    api.submit(model.trim(), revision.trim())
      .then((r) => setMessage(r.message))
      .catch((err: Error) => setError(err.message))
      .finally(() => setBusy(false))
  }

  return (
    <>
      <section className="page-intro">
        <h1>Submit a tokenizer</h1>
        <p>
          Any tokenizer on the Hugging Face Hub that loads with <code>AutoTokenizer</code>.
          It is evaluated on every language in the benchmark and added to the leaderboard.
        </p>
      </section>
      <form onSubmit={submit} style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'flex-end', maxWidth: 760 }}>
        <div className="field" style={{ flex: '1 1 320px' }}>
          <label htmlFor="model" className="field-label">Model or tokenizer repo</label>
          <input id="model" className="input" placeholder="e.g. Davlan/afro-xlmr-large" value={model} onChange={(e) => setModel(e.target.value)} required />
        </div>
        <div className="field" style={{ flex: '0 1 160px' }}>
          <label htmlFor="rev" className="field-label">Revision</label>
          <input id="rev" className="input" value={revision} onChange={(e) => setRevision(e.target.value)} />
        </div>
        <button className="primary" type="submit" disabled={busy}>{busy ? 'Submitting…' : 'Submit'}</button>
      </form>
      <div style={{ marginTop: 16, maxWidth: 760 }}>
        {message && <div className="notice info" role="status">{message}</div>}
        {error && <div className="notice error" role="alert">{error}</div>}
      </div>
    </>
  )
}
