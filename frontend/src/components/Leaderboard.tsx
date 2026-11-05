import { Fragment, useEffect, useMemo, useState } from 'react'
import type { LanguagesResponse, RankBy, RankingRow, TokenizerDetail } from '../types'
import { api } from '../api'
import LanguagePicker, { ALL } from './LanguagePicker'

const DOMAINS: { id: string; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'bible', label: 'Bible' },
  { id: 'watchtower', label: 'Watchtower' },
  { id: 'web', label: 'Web' },
]

const RANK_BY: { id: RankBy; label: string; lowerIsBetter: boolean }[] = [
  { id: 'fertility', label: 'Tokens per word (lower is better)', lowerIsBetter: true },
  { id: 'parity', label: 'Parity vs English (lower is better)', lowerIsBetter: true },
  { id: 'bytes_per_token', label: 'Bytes per token (higher is better)', lowerIsBetter: false },
]

const PAGE_SIZE = 25
const fmt = (v: number | null, d = 2) => (v == null ? '—' : v.toFixed(d))
const fmtInt = (v: number | null) => (v == null ? '—' : v.toLocaleString('en-US'))

function readParams() {
  const p = new URLSearchParams(window.location.hash.split('?')[1] ?? '')
  return { lang: p.get('lang') ?? ALL, domain: p.get('domain') ?? 'all', by: (p.get('by') ?? 'fertility') as RankBy }
}

export default function Leaderboard({ langs }: { langs: LanguagesResponse | null }) {
  const initial = readParams()
  const [lang, setLang] = useState(initial.lang)
  const [domain, setDomain] = useState(initial.domain)
  const [rankBy, setRankBy] = useState<RankBy>(initial.by)
  const [rows, setRows] = useState<RankingRow[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [showAll, setShowAll] = useState(false)
  const [open, setOpen] = useState<string | null>(null)

  const language = langs?.languages.find((l) => l.code === lang)
  const availableDomains = lang === ALL ? null : new Set(language?.domains ?? [])

  useEffect(() => {
    const params = new URLSearchParams({ lang, domain, by: rankBy })
    history.replaceState(null, '', `#/leaderboard?${params}`)
  }, [lang, domain, rankBy])

  useEffect(() => {
    const ctrl = new AbortController()
    setRows(null)
    setError(null)
    setOpen(null)
    api.ranking(lang, domain, ctrl.signal)
      .then((r) => setRows(r.rows))
      .catch((e: Error) => { if (e.name !== 'AbortError') setError(e.message) })
    return () => ctrl.abort()
  }, [lang, domain])

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
  const title = lang === ALL ? `All African languages${langs ? ` (${langs.n_african})` : ''}` : language?.name ?? lang
  const domainLabel = DOMAINS.find((d) => d.id === domain)?.label.toLowerCase() ?? domain

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
        {langs ? (
          <LanguagePicker
            languages={langs.languages}
            nAfrican={langs.n_african}
            value={lang}
            onChange={(c) => {
              setLang(c)
              const l = langs.languages.find((x) => x.code === c)
              if (c !== ALL && domain !== 'all' && !l?.domains.includes(domain)) setDomain('all')
            }}
          />
        ) : (
          <div className="field picker"><span className="field-label">Language</span><div className="skeleton" style={{ height: 44 }} /></div>
        )}
        <div className="field">
          <span className="field-label" id="domain-label">Text domain</span>
          <div className="segmented" role="group" aria-labelledby="domain-label">
            {DOMAINS.map((d) => {
              const disabled = d.id !== 'all' && availableDomains !== null && !availableDomains.has(d.id)
              return (
                <button
                  key={d.id}
                  aria-pressed={domain === d.id}
                  disabled={disabled}
                  title={disabled ? `No ${d.label} text for this language` : undefined}
                  onClick={() => setDomain(d.id)}
                >
                  {d.label}
                </button>
              )
            })}
          </div>
        </div>
        <div className="field">
          <label htmlFor="rank-by" className="field-label">Rank by</label>
          <select id="rank-by" className="select" value={rankBy} onChange={(e) => setRankBy(e.target.value as RankBy)}>
            {RANK_BY.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
        </div>
      </section>

      <div className="table-head">
        <h2>{title} · {domainLabel === 'all' ? 'all domains' : domainLabel} · {sorted ? `${sorted.length} tokenizers` : 'loading…'}</h2>
      </div>

      {error && <div className="notice error" role="alert">{error}</div>}
      {sorted && sorted.length === 0 && <div className="notice info">No results yet for this selection.</div>}

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col" style={{ width: 48 }}>#</th>
              <th scope="col">Tokenizer</th>
              <th scope="col" style={{ width: 260 }}>Tokens / word ↓</th>
              <th scope="col" className="num">Parity vs EN ↓</th>
              <th scope="col" className="num">Bytes / token ↑</th>
              <th scope="col" className="num">Vocab</th>
            </tr>
          </thead>
          <tbody>
            {!visible && Array.from({ length: 8 }, (_, i) => (
              <tr key={i}>{Array.from({ length: 6 }, (__, j) => <td key={j}><div className="skeleton" /></td>)}</tr>
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
                      <div className="row-sub">{[r.org, r.algorithm].filter(Boolean).join(' · ')}</div>
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
                  <td className="num mono">{fmt(r.parity)}</td>
                  <td className="num mono">{fmt(r.bytes_per_token, 1)}</td>
                  <td className="num mono muted">{fmtInt(r.vocab_size)}</td>
                </tr>
                {open === r.model && (
                  <tr><td colSpan={6} style={{ padding: 0 }}><Detail model={r.model} lang={lang} /></td></tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
      <div className="table-foot">
        <span>Click a row for per-domain scores and its best and worst languages.</span>
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

function Detail({ model, lang }: { model: string; lang: string }) {
  const [d, setD] = useState<TokenizerDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    api.tokenizer(model, lang).then(setD).catch((e: Error) => setError(e.message))
  }, [model, lang])
  if (error) return <div className="detail"><div className="notice error">{error}</div></div>
  if (!d) return <div className="detail"><div className="skeleton" style={{ width: '40%' }} /></div>
  const isLocal = model.startsWith('tokenizers/')
  return (
    <div className="detail">
      <div className="detail-grid">
        <div>
          <h3>Tokens per word by domain</h3>
          <ul>
            {d.domains.map((x) => (
              <li key={x.domain}><span style={{ textTransform: 'capitalize' }}>{x.domain}</span><span className="mono">{fmt(x.fertility)}</span></li>
            ))}
          </ul>
          <p className="muted" style={{ fontSize: 13, margin: '12px 0 0' }}>
            {isLocal ? model.replace('tokenizers/', '') : <a href={`https://huggingface.co/${model}`} target="_blank" rel="noreferrer">{model}</a>}
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
