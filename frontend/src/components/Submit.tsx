import { useState } from 'react'
import { SPACE_URL } from '../api'

export default function Submit() {
  const [model, setModel] = useState('')

  const title = `Evaluate tokenizer: ${model.trim()}`
  const href = `${SPACE_URL}/discussions/new?title=${encodeURIComponent(title)}`

  return (
    <>
      <section className="page-intro">
        <h1>Request a tokenizer</h1>
        <p>
          Any tokenizer on the Hugging Face Hub that loads with <code>AutoTokenizer</code>. Requests
          are posted in this Space's Community tab; we evaluate them on every language in the
          benchmark and add them to the leaderboard.
        </p>
      </section>
      <form
        onSubmit={(e) => { e.preventDefault(); if (model.trim()) window.open(href, '_blank', 'noopener') }}
        style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'flex-end', maxWidth: 760 }}
      >
        <div className="field" style={{ flex: '1 1 360px' }}>
          <label htmlFor="model" className="field-label">Model or tokenizer repo</label>
          <input id="model" className="input" placeholder="e.g. Davlan/afro-xlmr-large" value={model} onChange={(e) => setModel(e.target.value)} required />
        </div>
        <button className="primary" type="submit">Post request</button>
      </form>
      <p className="muted" style={{ fontSize: 13, marginTop: 16, maxWidth: 760 }}>
        Opens a new discussion with the title filled in; you need to be signed in to Hugging Face.
        Repositories that need <code>trust_remote_code</code> are not run. Gated repositories can be
        evaluated if their owners grant access.
      </p>
    </>
  )
}
