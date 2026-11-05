import { Fragment, useEffect, useMemo, useState } from 'react'
import type { LanguagesResponse, ModelType, RankBy, RankingResponse, RankingRow, TokenizerDetail } from '../types'
import { MODEL_TYPES } from '../types'
import { api } from '../api'
import LanguagePicker, { ALL } from './LanguagePicker'

const RANK_BY: { id: RankBy; label: string; lowerIsBetter: boolean }[] = [
  { id: 'fertility', label: 'Tokens per word (lower is better)', lowerIsBetter: true },
  { id: 'parity', label: 'Parity vs English (lower is better)', lowerIsBetter: true },
  { id: 'bytes_per_token', label: 'Bytes per token (higher is better)', lowerIsBetter: false },
]

const PAGE_SIZE = 25
const fmt = (v: number | null, d = 2) => (v == null ? '—' : v.toFixed(d))
const fmtInt = (v: number | null) => (v == null ? '—' : v.toLocaleString('en-US'))

const MAX_LANG_COLUMNS = 5

function readParams() {
  const p = new URLSearchParams(window.location.hash.split('?')[1] ?? '')
  const lang = p.get('lang') ?? ALL
  return {
    langs: lang.split(',').filter(Boolean),
    by: (p.get('by') ?? 'fertility') as RankBy,
    hr: p.get('hr') !== '0',
    types: (p.get('types') ?? '').split(',').filter((x): x is ModelType => MODEL_TYPES.some((m) => m.id === x)),
  }
}

export default function Leaderboard({ langs: info }: { langs: LanguagesResponse | null }) {
  const initial = readParams()
  const [langs, setLangs] = useState<string[]>(initial.langs)
  const [rankBy, setRankBy] = useState<RankBy>(initial.by)
  const [hr, setHr] = useState(initial.hr)
  const [types, setTypes] = useState<ModelType[]>(initial.types)
  const [result, setResult] = useState<RankingResponse | null>(null)
  const rows: RankingRow[] | null = result?.rows ?? null
  const [error, setError] = useState<string | null>(null)
  const [showAll, setShowAll] = useState(false)
  const [open, setOpen] = useState<string | null>(null)

  const isAll = langs.length === 1 && langs[0] === ALL
  const lang = langs.join(',')
  const byCode = (c: string) => info?.languages.find((l) => l.code === c)
  // Languages actually scored (picked + companions), as the server resolved them.
  const scored = (result?.langs ?? []).map(byCode).filter((l) => l != null)
  const langColumns = scored.length > 1 && scored.length <= MAX_LANG_COLUMNS ? scored : []
  const companions = Object.entries(result?.companions ?? {})

  useEffect(() => {
    const params = new URLSearchParams({ lang, by: rankBy, ...(hr ? {} : { hr: '0' }), ...(types.length ? { types: types.join(',') } : {}) })
    history.replaceState(null, '', `#/leaderboard?${params}`)
  }, [lang, rankBy, hr, types])

  useEffect(() => {
    const ctrl = new AbortController()
    setResult(null)
    setError(null)
    setOpen(null)
    if (!lang) return
    api.ranking(lang, hr, types)
      .then((r) => { if (!ctrl.signal.aborted) setResult(r) })
      .catch((e: Error) => { if (e.name !== 'AbortError') setError(e.message) })
    return () => ctrl.abort()
  }, [lang, hr, types])

  const metric = RANK_BY.find((m) => m.id === rankBy)!
  const sorted = useMemo(() => {
    if (!rows) return null
    const val = (r: RankingRow) => r[rankBy]
    return [...rows].sort((a, b) => {
      const va = val(a), vb = val(b)
      if (va == null) return 1
      if (vb == null) return -1
      return metric.lowerIsBetter ? va - vb : vb - va
    })
  }, [rows, rankBy, metric.lowerIsBetter])

  const maxFert = Math.max(...(sorted ?? []).map((r) => r.fertility ?? 0), 1)
  const visible = sorted ? (showAll ? sorted : sorted.slice(0, PAGE_SIZE)) : null
  const nLangs = result?.n_langs
  const title = !langs.length
    ? 'No languages selected'
    : isAll
      ? `${hr ? 'All languages' : 'All African languages'}${nLangs ? ` (${nLangs})` : ''}`
      : scored.length && scored.length <= MAX_LANG_COLUMNS
        ? scored.map((l) => l.name).join(' + ')
        : `${nLangs ?? langs.length} languages`

  return (
    <>
      <section className="page-intro">
        <h1>Which tokenizer handles your language best?</h1>
        <p>
          Fewer tokens per word means cheaper, faster models that see more of the text.
          Pick a language to rank every tokenizer on it.
        </p>
      </section>

      <section className="filters" aria-label="Filters">
        {info ? (
          <LanguagePicker
            languages={info.languages}
            includeHighResource={hr}
            value={langs}
            onChange={setLangs}
          />
        ) : (
          <div className="field picker"><span className="field-label">Languages</span><div className="skeleton" style={{ height: 44 }} /></div>
        )}
        <div className="field">
          <label htmlFor="rank-by" className="field-label">Rank by</label>
          <select id="rank-by" className="select" value={rankBy} onChange={(e) => setRankBy(e.target.value as RankBy)}>
            {RANK_BY.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
        </div>
        <div className="field">
          <span className="field-label" id="type-label">Model type</span>
          <div className="segmented" role="group" aria-labelledby="type-label">
            <button aria-pressed={types.length === 0} onClick={() => setTypes([])}>All</button>
            {MODEL_TYPES.map((m) => (
              <button
                key={m.id}
                title={m.hint}
                aria-pressed={types.includes(m.id)}
                onClick={() => setTypes(types.includes(m.id) ? types.filter((x) => x !== m.id) : [...types, m.id])}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>
        <label className="check">
          <input
            type="checkbox"
            checked={hr}
            onChange={(e) => {
              setHr(e.target.checked)
              if (!e.target.checked) {
                const african = langs.filter((c) => c === ALL || byCode(c)?.group === 'african')
                if (african.length !== langs.length) setLangs(african.length ? african : [ALL])
              }
            }}
          />
          <span>
            Include high-resource languages
            <span className="field-hint">English, French, Arabic, Portuguese, Spanish</span>
          </span>
        </label>
      </section>

      <div className="table-head">
        <h2>{title}{lang && <> · {sorted ? `${sorted.length} tokenizers` : 'loading…'}</>}</h2>
      </div>

      {companions.length > 0 && (
        <p className="companion-note">
          Ranked with the high-resource language each one is mainly used alongside:{' '}
          {companions.map(([c, comp], i) => (
            <span key={c}>{i > 0 && ', '}{byCode(c)?.name ?? c} + {byCode(comp)?.name ?? comp}</span>
          ))}. Untick “Include high-resource languages” to score African languages alone.
        </p>
      )}
      {error && <div className="notice error" role="alert">{error}</div>}
      {!lang && (
        <div className="notice info" role="status">
          Pick one or more languages above, or{' '}
          <button className="link-button" onClick={() => setLangs([ALL])}>rank across all languages</button>.
        </div>
      )}
      {sorted && sorted.length === 0 && <div className="notice info">No results yet for this selection.</div>}

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col" style={{ width: 48 }}>#</th>
              <th scope="col">Tokenizer</th>
              <th scope="col" style={{ width: 260 }}>{scored.length === 1 ? 'Tokens / word ↓' : 'Avg tokens / word ↓'}</th>
              {langColumns.map((l) => <th key={l.code} scope="col" className="lang-col">{l.name}</th>)}
              <th scope="col" className="num">Parity vs EN ↓</th>
              <th scope="col" className="num">Bytes / token ↑</th>
              <th scope="col" className="num">Vocab</th>
            </tr>
          </thead>
          <tbody>
            {!visible && lang && Array.from({ length: 8 }, (_, i) => (
              <tr key={i}>{Array.from({ length: 6 + langColumns.length }, (__, j) => <td key={j}><div className="skeleton" /></td>)}</tr>
            ))}
            {visible?.map((r, i) => (
              <Fragment key={r.model}>
                <tr
                  className={`row${i === 0 ? ' best' : ''}`}
                  aria-expanded={open === r.model}
                  onClick={() => setOpen(open === r.model ? null : r.model)}
                >
                  <td className="mono muted">{i + 1}</td>
                  <td>
                    <button className="row-button" aria-expanded={open === r.model} aria-label={`${r.name}: show details`}>
                      <div className="row-name">{r.name}</div>
                      <div className="row-sub">{[r.org, MODEL_TYPES.find((m) => m.id === r.type)?.label, r.algorithm].filter(Boolean).join(' · ')}</div>
                    </button>
                  </td>
                  <td>
                    <div className="bar-cell">
                      <span className="bar-value">{fmt(r.fertility)}</span>
                      <div className="bar-track" aria-hidden="true">
                        <div className="bar-fill" style={{ width: `${Math.round(((r.fertility ?? 0) / maxFert) * 100)}%` }} />
                      </div>
                    </div>
                  </td>
                  {langColumns.map((l) => <td key={l.code} className="lang-col mono">{fmt(r.per_lang?.[l.code] ?? null)}</td>)}
                  <td className="num mono">{fmt(r.parity)}</td>
                  <td className="num mono">{fmt(r.bytes_per_token, 1)}</td>
                  <td className="num mono muted">{fmtInt(r.vocab_size)}</td>
                </tr>
                {open === r.model && (
                  <tr><td colSpan={6 + langColumns.length} style={{ padding: 0 }}><Detail model={r.model} lang={lang} hr={hr} /></td></tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
      <div className="table-foot">
        <span>Click a row for its summary and its best and hardest African languages.</span>
        {sorted && sorted.length > PAGE_SIZE && (
          <button className="link-button" onClick={() => setShowAll(!showAll)}>
            {showAll ? 'Show top 25' : `Show all ${sorted.length} tokenizers`}
          </button>
        )}
      </div>

      <section className="guide" aria-label="Metric guide">
        <div className="card">
          <div className="card-title">Tokens per word</div>
          <div className="card-text">How many pieces an average word is split into. 1.0 means whole words.</div>
        </div>
        <div className="card">
          <div className="card-title">Parity vs English</div>
          <div className="card-text">Tokens for the same Bible verses, divided by the English count. 1.0 means as cheap as English.</div>
        </div>
        <div className="card">
          <div className="card-title">Bytes per token</div>
          <div className="card-text">How much text each token carries. Higher means more compact.</div>
        </div>
      </section>
    </>
  )
}

function Detail({ model, lang, hr }: { model: string; lang: string; hr: boolean }) {
  const [d, setD] = useState<TokenizerDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    api.tokenizer(model, lang, hr).then(setD).catch((e: Error) => setError(e.message))
  }, [model, lang, hr])
  if (error) return <div className="detail"><div className="notice error">{error}</div></div>
  if (!d) return <div className="detail"><div className="skeleton" style={{ width: '40%' }} /></div>
  const href = d.source.startsWith('http') ? d.source : `https://huggingface.co/${d.source}`
  return (
    <div className="detail">
      <div className="detail-grid">
        <div>
          <h3>{d.n_langs === 1 ? 'On this language' : `Across ${d.n_langs} languages`}</h3>
          <ul>
            <li><span>Tokens per word</span><span className="mono">{fmt(d.fertility)}</span></li>
            <li><span>Parity vs English</span><span className="mono">{fmt(d.parity)}</span></li>
            <li><span>Bytes per token</span><span className="mono">{fmt(d.bytes_per_token, 1)}</span></li>
          </ul>
          <p className="muted" style={{ fontSize: 13, margin: '12px 0 0' }}>
            <a href={href} target="_blank" rel="noreferrer">{d.source.replace('https://github.com/', '')}</a>
            {d.vocab_size != null && <> · {d.vocab_size.toLocaleString('en-US')} tokens</>}
          </p>
        </div>
        <div>
          <h3>Best African languages</h3>
          <ol>
            {d.best_languages.map((x) => <li key={x.code}><span>{x.name}</span><span className="mono">{fmt(x.fertility)}</span></li>)}
          </ol>
        </div>
        <div>
          <h3>Hardest African languages</h3>
          <ol>
            {d.worst_languages.map((x) => <li key={x.code}><span>{x.name}</span><span className="mono">{fmt(x.fertility)}</span></li>)}
          </ol>
        </div>
      </div>
    </div>
  )
}
