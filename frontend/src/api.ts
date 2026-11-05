import type {
  CompareResponse,
  LanguagesResponse,
  RankingResponse,
  TokenizerDetail,
} from './types'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(path, options)
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(text || `${res.status} ${res.statusText}`)
  }
  return res.json()
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

const q = (params: Record<string, string>) => new URLSearchParams(params).toString()

export const api = {
  languages: () => request<LanguagesResponse>('/api/v2/languages'),

  ranking: (lang: string, domain: string, signal?: AbortSignal) =>
    request<RankingResponse>(`/api/v2/ranking?${q({ lang, domain })}`, { signal }),

  tokenizer: (name: string, lang: string) =>
    request<TokenizerDetail>(`/api/v2/tokenizer?${q({ name, lang })}`),

  compare: (text: string, a: string, b: string) =>
    post<CompareResponse>('/api/battle/tokenize', {
      text,
      model_selection_1: a,
      model_selection_2: b,
      better_tokenization: false,
    }),

  submit: (modelName: string, revision: string) =>
    post<{ message: string }>('/api/submit', {
      model_name: modelName,
      revision: revision || 'main',
      target_mode: false,
    }),

  content: (page: string) => request<{ content: string }>(`/api/content/${page}`),
}
