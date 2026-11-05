export type Page = 'leaderboard' | 'compare' | 'submit' | 'about'

export interface Language {
  code: string
  name: string
  group: 'african' | 'high_resource'
  domains: string[]
  has_parity: boolean
}

export interface LanguagesResponse {
  languages: Language[]
  n_african: number
  domains: string[]
  data_version: string | null
}

export interface RankingRow {
  rank: number
  model: string
  name: string
  org: string
  algorithm: string | null
  vocab_size: number | null
  fertility: number | null
  parity: number | null
  bytes_per_token: number | null
  unk_ratio: number | null
  n_langs: number
  per_lang?: Record<string, number | null>
}

export interface RankingResponse {
  rows: RankingRow[]
  lang: string
  langs?: string[] | null
  domain: string
  n_african?: number
}

export interface LangScore {
  code: string
  name: string
  fertility: number | null
  parity: number | null
}

export interface TokenizerDetail {
  model: string
  name: string
  org: string
  vocab_size: number | null
  algorithm: string | null
  domains: { domain: string; fertility: number | null }[]
  best_languages: LangScore[]
  worst_languages: LangScore[]
}

export interface TokenizedText {
  highlight_data: [string, string | null][]
  color_map: Record<string, string>
  count: number
  vocab_size: number | null
}

export interface CompareResponse {
  error: string | null
  model1: TokenizedText
  model2: TokenizedText
}

export type RankBy = 'fertility' | 'parity' | 'bytes_per_token'
