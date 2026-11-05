import { useState, useMemo } from 'react'
import type { BattleResponse, ModelResult } from '../types'
import { api } from '../api'

interface Props {
  modelChoices: { display: string; value: string }[]
}

const DEFAULT_TEXT = `Hello world! This is an English sentence.
Olá mundo! Esta é uma frase em português.
مرحبا بالعالم! هذه جملة باللغة العربية.
こんにちは世界！これは日本語の文章です。

\`\`\`python
def greet(lang):
  greetings = {
    "en": "Hello",
    "pt": "Olá",
    "ar": "مرحبا",
    "ja": "こんにちは"
  }
  print(f"{greetings.get(lang, 'Hello')}!")
greet("en")
\`\`\``

function TokenHighlight({ result }: { result: ModelResult }) {
  if (!result.highlight_data?.length) {
    return <div className="battle__tokens" style={{ color: 'var(--text-muted)' }}>No tokens</div>
  }
  return (
    <div className="battle__tokens">
      {result.highlight_data.map(([text, label], i) => {
        const color = label != null ? result.color_map[label] : undefined
        return (
          <span
            key={i}
            className="battle__token"
            style={{
              background: color ? `${color}33` : 'transparent',
              borderBottom: color ? `2px solid ${color}` : 'none',
            }}
          >
            {text}
          </span>
        )
      })}
    </div>
  )
}

function TokenTable({ result }: { result: ModelResult }) {
  if (!result.table?.length) return null
  const cols = Object.keys(result.table[0])
  return (
    <div style={{ maxHeight: '300px', overflow: 'auto' }}>
      <table className="battle__token-table">
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {result.table.map((row, i) => (
            <tr key={i}>
              {cols.map((c) => (
                <td key={c}>{String(row[c] ?? '')}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function Battle({ modelChoices }: Props) {
  const [text, setText] = useState(DEFAULT_TEXT)
  const [model1, setModel1] = useState(modelChoices[0]?.value || '')
  const [model2, setModel2] = useState(modelChoices[1]?.value || modelChoices[0]?.value || '')
  const [betterTokenization, setBetterTokenization] = useState(false)
  const [result, setResult] = useState<BattleResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const sortedChoices = useMemo(
    () => [...modelChoices].sort((a, b) => a.display.localeCompare(b.display)),
    [modelChoices],
  )

  const handleTokenize = async () => {
    if (!text.trim()) {
      setError('Please enter some text to tokenize.')
      return
    }
    if (!model1 || !model2) {
      setError('Please select both tokenizers.')
      return
    }
    setError(null)
    setLoading(true)
    try {
      const res = await api.battleTokenize({
        text,
        model_selection_1: model1,
        model_selection_2: model2,
        better_tokenization: betterTokenization,
      })
      if (res.error) {
        setError(res.error)
      } else {
        setResult(res)
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Tokenization failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="battle">
      <div className="battle__header">
        <h2 className="battle__title">Tokenizer Battle</h2>
        <p className="battle__subtitle">
          Compare two tokenizers side-by-side on custom text
        </p>
      </div>

      <textarea
        className="battle__textarea"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text to tokenize..."
        rows={8}
      />

      <div className="battle__selectors">
        <div className="battle__select-group">
          <label className="battle__select-label">Tokenizer 1</label>
          <select
            className="select"
            value={model1}
            onChange={(e) => setModel1(e.target.value)}
          >
            {sortedChoices.map((c) => (
              <option key={c.value} value={c.value}>
                {c.display}
              </option>
            ))}
          </select>
        </div>
        <div className="battle__select-group">
          <label className="battle__select-label">Tokenizer 2</label>
          <select
            className="select"
            value={model2}
            onChange={(e) => setModel2(e.target.value)}
          >
            {sortedChoices.map((c) => (
              <option key={c.value} value={c.value}>
                {c.display}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="battle__options">
        <div className="toggle-row">
          <div
            className={`toggle${betterTokenization ? ' toggle--active' : ''}`}
            onClick={() => setBetterTokenization(!betterTokenization)}
          >
            <div className="toggle__knob" />
          </div>
          <span
            className="toggle-label"
            onClick={() => setBetterTokenization(!betterTokenization)}
          >
            Improved visual highlighting for non-Latin alphabets (slower)
          </span>
        </div>
      </div>

      <button
        className="btn btn--primary"
        onClick={handleTokenize}
        disabled={loading}
        style={{ alignSelf: 'flex-start' }}
      >
        {loading ? 'Tokenizing...' : '⚡ Tokenize'}
      </button>

      {error && <div className="error-banner">{error}</div>}

      {result && (
        <div className="battle__results">
          <div className="battle__column">
            <div className="battle__column-title">Tokenizer 1</div>
            <div className="battle__stats">
              <div className="battle__stat">
                <span className="battle__stat-label">Tokens</span>
                <span className="battle__stat-value">
                  {result.model1.count}
                </span>
              </div>
              <div className="battle__stat">
                <span className="battle__stat-label">Vocab Size</span>
                <span className="battle__stat-value">
                  {result.model1.vocab_size?.toLocaleString() ?? 'N/A'}
                </span>
              </div>
            </div>
            <TokenHighlight result={result.model1} />
            <TokenTable result={result.model1} />
          </div>

          <div className="battle__vs">VS</div>

          <div className="battle__column">
            <div className="battle__column-title">Tokenizer 2</div>
            <div className="battle__stats">
              <div className="battle__stat">
                <span className="battle__stat-label">Tokens</span>
                <span className="battle__stat-value">
                  {result.model2.count}
                </span>
              </div>
              <div className="battle__stat">
                <span className="battle__stat-label">Vocab Size</span>
                <span className="battle__stat-value">
                  {result.model2.vocab_size?.toLocaleString() ?? 'N/A'}
                </span>
              </div>
            </div>
            <TokenHighlight result={result.model2} />
            <TokenTable result={result.model2} />
          </div>
        </div>
      )}
    </div>
  )
}
