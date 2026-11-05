import { useEffect, useState } from 'react'
import type { RankingRow, TokenizedText } from '../types'
import { api } from '../api'

const SAMPLES: { label: string; text: string }[] = [
  { label: 'Twi', text: 'Mesrɛ wo sɛ, ɛnnɛ yɛbɛkɔ fie no akyiri.' },
  { label: 'Yoruba', text: 'Ẹ káàárọ̀, báwo ni ẹbí yín ṣe wà lónìí?' },
  { label: 'Amharic', text: 'እንኳን ደህና መጣችሁ፣ ዛሬ አየሩ በጣም ጥሩ ነው።' },
  { label: 'Swahili', text: 'Habari za asubuhi, karibu sana nyumbani kwetu.' },
  { label: 'Hausa', text: 'Ina kwana, yaya aiki da iyali?' },
]

const CHIPS = ['var(--chip-1)', 'var(--chip-2)', 'var(--chip-3)', 'var(--chip-4)']

export default function Compare() {
  const [models, setModels] = useState<RankingRow[]>([])
  const [text, setText] = useState(SAMPLES[0].text)
  const [a, setA] = useState('')
  const [b, setB] = useState('')
  const [result, setResult] = useState<[TokenizedText, TokenizedText] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api.ranking('all', true).then((r) => {
      setModels(r.rows)
      setA(r.rows.find((x) => x.model.includes('en-twi'))?.model ?? r.rows[0]?.model ?? '')
      setB(r.rows.find((x) => x.model.includes('gpt-4o'))?.model ?? r.rows[1]?.model ?? '')
    }).catch((e: Error) => setError(e.message))
  }, [])

  const run = () => {
    if (!text.trim() || !a || !b) return
    setBusy(true)
    setError(null)
    api.compare(text, a, b)
      .then((r) => {
        if (r.error) setError(r.error)
        else setResult([r.model1, r.model2])
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setBusy(false))
  }

  const words = text.trim() ? text.trim().split(/\s+/).length : 0
  const sides: { label: string; value: string; set: (v: string) => void; res: TokenizedText | null }[] = [
    { label: 'Tokenizer A', value: a, set: setA, res: result?.[0] ?? null },
    { label: 'Tokenizer B', value: b, set: setB, res: result?.[1] ?? null },
  ]

  return (
    <>
      <section className="page-intro">
        <h1>See how two tokenizers split your text</h1>
        <p>Type or paste a sentence in any language. Each coloured block is one token.</p>
      </section>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
        <div className="field">
          <label htmlFor="cmp-text" className="field-label">Your text</label>
          <textarea id="cmp-text" className="input" rows={3} value={text} onChange={(e) => setText(e.target.value)} />
          <div className="chips">
            <span className="muted">Try:</span>
            {SAMPLES.map((s) => (
              <button key={s.label} className="chip" onClick={() => setText(s.text)}>{s.label}</button>
            ))}
          </div>
        </div>
        <div className="compare-grid">
          {sides.map((s) => (
            <section key={s.label} className="compare-card" aria-label={s.label}>
              <div className="field">
                <label className="field-label" htmlFor={`sel-${s.label}`}>{s.label}</label>
                <select id={`sel-${s.label}`} className="select" value={s.value} onChange={(e) => s.set(e.target.value)}>
                  {models.map((m) => <option key={m.model} value={m.model}>{m.name} — {m.org}</option>)}
                </select>
              </div>
              {s.res && (
                <>
                  <div className="stats">
                    <div><div className="stat-value">{s.res.count}</div><div className="stat-label">tokens</div></div>
                    <div><div className="stat-value">{words ? (s.res.count / words).toFixed(2) : '—'}</div><div className="stat-label">tokens / word</div></div>
                  </div>
                  <div className="tokens">
                    {s.res.highlight_data.map(([tok], i) => (
                      <span key={i} className="token" style={{ background: CHIPS[i % CHIPS.length] }}>{tok}</span>
                    ))}
                  </div>
                </>
              )}
            </section>
          ))}
        </div>
        <div><button className="primary" onClick={run} disabled={busy || !a || !b}>{busy ? 'Tokenizing…' : 'Compare'}</button></div>
        {error && <div className="notice error" role="alert">{error}</div>}
      </div>
    </>
  )
}
