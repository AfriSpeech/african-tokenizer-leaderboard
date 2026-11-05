// Leaderboard data and rankings, computed in the browser from data/leaderboard.json
// (written by benchmark/export_static.py from the evaluation results).
//
// Every score averages per-language values with each language counting equally. With
// high-resource languages included (the default), "all" covers English, French, Arabic,
// Portuguese and Spanish too, and each picked African language brings its companion: the
// high-resource language it is mainly used alongside (Twi -> Twi + English).

import type { Language, LangScore, ModelType, RankingResponse, RankingRow, TokenizerDetail } from './types'

export const ALL = 'all'

export interface TokenizerData {
  model: string
  name: string
  org: string
  vocab_size: number | null
  algorithm: string | null
  type: ModelType
  browser: string | null
  source: string
  fertility: (number | null)[]
  bytes_per_token: (number | null)[]
  parity: (number | null)[]
}

export interface LeaderboardData {
  languages: Language[]
  tokenizers: TokenizerData[]
}

let cache: Promise<LeaderboardData> | null = null

export function loadData(): Promise<LeaderboardData> {
  cache ??= fetch('./data/leaderboard.json').then((r) => {
    if (!r.ok) throw new Error(`Could not load leaderboard data (${r.status})`)
    return r.json()
  })
  return cache
}

const mean = (xs: (number | null)[]) => {
  const v = xs.filter((x): x is number => x != null)
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null
}
const round = (x: number | null, d: number) => (x == null ? null : Math.round(x * 10 ** d) / 10 ** d)

export interface Pool {
  picked: string[] | null // null = all languages
  scored: string[] // languages averaged
  companions: Record<string, string>
}

export function resolvePool(data: LeaderboardData, lang: string, hr: boolean): Pool {
  const byCode = new Map(data.languages.map((l) => [l.code, l]))
  const african = data.languages.filter((l) => l.group === 'african').map((l) => l.code)
  if (lang === ALL) {
    return { picked: null, scored: hr ? data.languages.map((l) => l.code) : african, companions: {} }
  }
  const picked = [...new Set(lang.split(',').filter((c) => byCode.has(c)))]
  if (!hr) {
    const p = picked.filter((c) => byCode.get(c)!.group === 'african')
    return { picked: p, scored: p, companions: {} }
  }
  const companions: Record<string, string> = {}
  for (const c of picked) {
    const comp = byCode.get(c)!.companion
    if (comp && byCode.has(comp)) companions[c] = comp
  }
  return { picked, scored: [...new Set([...picked, ...Object.values(companions)])], companions }
}

/** types: model architectures to include; empty = all. Ranks count only the shown tokenizers. */
export function ranking(data: LeaderboardData, lang: string, hr: boolean, types: ModelType[] = []): RankingResponse {
  const pool = resolvePool(data, lang, hr)
  const idx = new Map(data.languages.map((l, i) => [l.code, i]))
  const at = pool.scored.map((c) => idx.get(c)!)
  const showPerLang = pool.picked != null && pool.scored.length > 1
  const shown = types.length ? data.tokenizers.filter((t) => types.includes(t.type)) : data.tokenizers
  const rows: RankingRow[] = shown.map((t) => ({
    rank: 0,
    model: t.model,
    name: t.name,
    org: t.org,
    algorithm: t.algorithm,
    type: t.type,
    vocab_size: t.vocab_size,
    fertility: round(mean(at.map((i) => t.fertility[i])), 3),
    parity: round(mean(at.map((i) => t.parity[i])), 3),
    bytes_per_token: round(mean(at.map((i) => t.bytes_per_token[i])), 2),
    unk_ratio: null,
    n_langs: at.filter((i) => t.fertility[i] != null).length,
    per_lang: showPerLang
      ? Object.fromEntries(pool.scored.map((c) => [c, round(t.fertility[idx.get(c)!], 3)]))
      : undefined,
  }))
  rows.sort((a, b) => (a.fertility ?? Infinity) - (b.fertility ?? Infinity))
  rows.forEach((r, i) => (r.rank = i + 1))
  return {
    rows,
    lang,
    hr,
    langs: pool.picked == null ? null : pool.scored,
    companions: pool.companions,
    n_langs: pool.scored.length,
  }
}

export function tokenizerDetail(data: LeaderboardData, model: string, lang: string, hr: boolean): TokenizerDetail {
  const t = data.tokenizers.find((x) => x.model === model)
  if (!t) throw new Error(`Unknown tokenizer ${model}`)
  const pool = resolvePool(data, lang, hr)
  const idx = new Map(data.languages.map((l, i) => [l.code, i]))
  const at = pool.scored.map((c) => idx.get(c)!)
  const african: LangScore[] = data.languages
    .map((l, i) => ({ code: l.code, name: l.name, group: l.group, fertility: t.fertility[i], parity: t.parity[i] }))
    .filter((x) => x.group === 'african' && x.fertility != null)
    .sort((a, b) => a.fertility! - b.fertility!)
    .map(({ code, name, fertility, parity }) => ({ code, name, fertility: round(fertility, 3), parity: round(parity, 3) }))
  return {
    model: t.model,
    name: t.name,
    org: t.org,
    vocab_size: t.vocab_size,
    algorithm: t.algorithm,
    fertility: round(mean(at.map((i) => t.fertility[i])), 3),
    parity: round(mean(at.map((i) => t.parity[i])), 3),
    bytes_per_token: round(mean(at.map((i) => t.bytes_per_token[i])), 2),
    n_langs: at.length,
    best_languages: african.slice(0, 5),
    worst_languages: african.slice(-5).reverse(),
    source: t.source,
  }
}
