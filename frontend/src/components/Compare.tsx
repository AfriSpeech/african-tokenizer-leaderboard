import { useEffect, useState } from 'react'
import type { TokenizedText } from '../types'
import { api } from '../api'
import { loadData, type TokenizerData } from '../data'

const SAMPLES: { label: string; text: string }[] = [
  { label: 'Twi', text: 'Mesrɛ wo sɛ, ɛnnɛ yɛbɛkɔ fie no akyiri.' },
  { label: 'Yoruba', text: 'Ẹ káàárọ̀, báwo ni ẹbí yín ṣe wà lónìí?' },
  { label: 'Amharic', text: 'እንኳን ደህና መጣችሁ፣ ዛሬ አየሩ በጣም ጥሩ ነው።' },
  { label: 'Swahili', text: 'Habari za asubuhi, karibu sana nyumbani kwetu.' },
  { label: 'Hausa', text: 'Ina kwana, yaya aiki da iyali?' },
]

const CHIPS = ['var(--chip-1)', 'var(--chip-2)', 'var(--chip-3)', 'var(--chip-4)']

interface Side {
  model: string
  result: TokenizedText | null
  error: string | null
  busy: boolean
}

export default function Compare() {
  const [models, setModels] = useState<TokenizerData[]>([])
  const [text, setText] = useState(SAMPLES[0].text)
  const [sides, setSides] = useState<Side[]>([
    { model: '', result: null, error: null, busy: false },
    { model: '', result: null, error: null, busy: false },
  ])

  useEffect(() => {
    loadData().then((d) => {
      const usable = d.tokenizers.filter((t) => t.browser).sort((a, b) => a.name.localeCompare(b.name))
      setModels(usable)
      const pick = (s: string, i: number) => usable.find((t) => t.model.includes(s))?.model ?? usable[i]?.model ?? ''
      setSides((ss) => [{ ...ss[0], model: pick('en-twi', 0) }, { ...ss[1], model: pick('gpt-4o', 1) }])
    })
  }, [])

  const words = text.trim() ? text.trim().split(/\s+/).length : 0
  const set = (i: number, patch: Partial<Side>) =>
    setSides((ss) => ss.map((s, j) => (j === i ? { ...s, ...patch } : s)))

  const run = () => {
    if (!text.trim()) return
    sides.forEach((s, i) => {
      const t = models.find((m) => m.model === s.model)
      if (!t?.browser) return
      set(i, { busy: true, error: null })
      api.tokenize(t.browser, text)
        .then((result) => set(i, { result, busy: false }))
        .catch((e: Error) => set(i, { error: `This tokenizer could not run in the browser (${e.message}).`, busy: false, result: null }))
    })
  }

  const busy = sides.some((s) => s.busy)

  return (
    <>
      <section className="page-intro">
        <h1>See how two tokenizers split your text</h1>
        <p>
          Type or paste a sentence in any language. Each coloured block is one token. Tokenizers
          run in your browser; the first run downloads each one's vocabulary.
        </p>
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
          {sides.map((s, i) => (
            <section key={i} className="compare-card" aria-label={`Tokenizer ${'AB'[i]}`}>
              <div className="field">
                <label className="field-label" htmlFor={`sel-${i}`}>Tokenizer {'AB'[i]}</label>
                <select id={`sel-${i}`} className="select" value={s.model} onChange={(e) => set(i, { model: e.target.value, result: null, error: null })}>
                  {models.map((m) => <option key={m.model} value={m.model}>{m.name} — {m.org}</option>)}
                </select>
              </div>
              {s.busy && <div className="muted" role="status">Loading tokenizer…</div>}
              {s.error && <div className="notice error" role="alert">{s.error}</div>}
              {s.result && (
                <>
                  <div className="stats">
                    <div><div className="stat-value">{s.result.count}</div><div className="stat-label">tokens</div></div>
                    <div><div className="stat-value">{words ? (s.result.count / words).toFixed(2) : '—'}</div><div className="stat-label">tokens / word</div></div>
                  </div>
                  <div className="tokens">
                    {s.result.highlight_data.map(([tok], j) => (
                      <span key={j} className="token" style={{ background: CHIPS[j % CHIPS.length] }}>{tok}</span>
                    ))}
                  </div>
                </>
              )}
            </section>
          ))}
        </div>
        <div><button className="primary" onClick={run} disabled={busy || !models.length}>{busy ? 'Tokenizing…' : 'Compare'}</button></div>
        <p className="muted" style={{ fontSize: 13, margin: 0 }}>
          Gated tokenizers (Llama, Gemma, Aya Expanse) need a Hugging Face login to download, so
          they are ranked on the leaderboard but can't be compared here.
        </p>
      </div>
    </>
  )
}
