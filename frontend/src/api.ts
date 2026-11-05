// The static Space has no server: rankings come from data/leaderboard.json (data.ts) and
// "Compare" tokenizes in the browser with tokenizers.js (no model runtime needed).

import type { LanguagesResponse, ModelType, RankingResponse, TokenizedText, TokenizerDetail } from './types'
import { BYTES, fromJson, pieces, type LoadedTokenizer } from './tokenize'
import { loadData, ranking, tokenizerDetail } from './data'

export const SPACE_URL = 'https://huggingface.co/spaces/AfriSpeech/african-tokenizer-leaderboard'

const tokenizers = new Map<string, Promise<LoadedTokenizer>>()

async function fetchJson(url: string, optional = false): Promise<object> {
  const r = await fetch(url)
  if (!r.ok) {
    if (optional) return {}
    throw new Error(r.status === 404 ? 'no tokenizer.json in this repo' : `HTTP ${r.status}`)
  }
  return r.json()
}

/** "local:<folder>" = a folder shipped with the Space; "builtin:bytes" = UTF-8 bytes (ByT5);
 * otherwise a Hub repo id. */
async function loadTokenizer(browser: string): Promise<LoadedTokenizer> {
  if (browser === 'builtin:bytes') return BYTES
  const base = browser.startsWith('local:')
    ? new URL(`./tokenizers/${browser.slice('local:'.length)}/`, window.location.href).toString()
    : `https://huggingface.co/${browser}/resolve/main/`
  const [json, config] = await Promise.all([
    fetchJson(`${base}tokenizer.json`),
    fetchJson(`${base}tokenizer_config.json`, true),
  ])
  return fromJson(json, config)
}

function tokenizerFor(browser: string): Promise<LoadedTokenizer> {
  let t = tokenizers.get(browser)
  if (!t) {
    t = loadTokenizer(browser)
    t.catch(() => tokenizers.delete(browser))
    tokenizers.set(browser, t)
  }
  return t
}

export const api = {
  languages: async (): Promise<LanguagesResponse> => {
    const d = await loadData()
    return {
      languages: d.languages,
      n_african: d.languages.filter((l) => l.group === 'african').length,
      data_version: null,
    }
  },

  ranking: async (lang: string, hr: boolean, types: ModelType[] = []): Promise<RankingResponse> =>
    ranking(await loadData(), lang, hr, types),

  tokenizer: async (name: string, lang: string, hr: boolean): Promise<TokenizerDetail> =>
    tokenizerDetail(await loadData(), name, lang, hr),

  tokenize: async (browser: string, text: string): Promise<TokenizedText> => {
    const tok = await tokenizerFor(browser)
    const p = pieces(tok, text)
    return { highlight_data: p.map((x) => [x, null]), color_map: {}, count: p.length, vocab_size: null }
  },
}
