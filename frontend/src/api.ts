import type {
  Metadata,
  FilterState,
  LeaderboardResponse,
  LangResponse,
  MultilingualResponse,
  BattleResponse,
  SubmitResponse,
  WorkerStatus,
} from './types'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(path, options)
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`API ${res.status}: ${text || res.statusText}`)
  }
  return res.json()
}

function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
}

export const api = {
  health: () =>
    request<{ status: string; initialized: boolean }>('/api/health'),

  metadata: () => request<Metadata>('/api/metadata'),

  leaderboardInit: () => request<LeaderboardResponse>('/api/leaderboard/init'),

  leaderboard: (filters: FilterState, signal?: AbortSignal) =>
    post<LeaderboardResponse>('/api/leaderboard', {
      selected_langs: filters.selected_langs,
      selected_domains: filters.selected_domains,
      selected_multilingual_domains: filters.selected_multilingual_domains,
    }, signal),

  refreshLeaderboard: () =>
    post<LeaderboardResponse>('/api/leaderboard/refresh', {}),

  leaderboardLang: (langCode: string, selectedDomains: string[], signal?: AbortSignal) =>
    post<LangResponse>(`/api/leaderboard/lang/${langCode}`, {
      selected_domains: selectedDomains,
    }, signal),

  leaderboardMultilingual: (selectedLangs: string[], selectedMultilingualDomains: string[], signal?: AbortSignal) =>
    post<MultilingualResponse>('/api/leaderboard/multilingual', {
      selected_langs: selectedLangs,
      selected_multilingual_domains: selectedMultilingualDomains,
    }, signal),

  submitModel: (data: {
    model_name: string
    revision: string
    subfolder?: string
    target_mode: boolean
    selected_langs?: string[]
  }) => post<SubmitResponse>('/api/submit', data),

  workerStatus: () => request<WorkerStatus>('/api/worker/status'),

  battleTokenize: (data: {
    text: string
    model_selection_1: string
    model_selection_2: string
    better_tokenization: boolean
  }) => post<BattleResponse>('/api/battle/tokenize', data),

  validateSearch: (query: string) =>
    post<{ is_valid: boolean; error_message: string }>(
      '/api/search/validate',
      { query },
    ),

  domainsForLanguages: (codes: string[]) =>
    post<{ domains: [string, string][] }>('/api/domains/for-languages', {
      selected_lang_codes: codes,
    }),

  content: (page: string) =>
    request<{ content: string }>(`/api/content/${page}`),
}
