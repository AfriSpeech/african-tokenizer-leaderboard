export interface LanguageTab {
  code: string
  name: string
  domains: [string, string][]
}

export interface Metadata {
  data_version: string
  language_tabs: LanguageTab[]
  all_domains: [string, string][]
  display_columns_map: Record<string, string>
  default_visible_columns: string[]
  avg_multilingual_columns: string[]
  default_search_query: string
  filter_help_text: string
  dataframe_datatypes: string[]
  limit_models_show: number
  column_descriptions: Record<string, string>
  filter_fields: { field: string; description: string }[]
}

export interface FilterState {
  selected_langs: string[]
  selected_domains: Record<string, string[]>
  selected_multilingual_domains: string[]
}

export interface LocalFilterState {
  search_query: string
  hide_duplicates: boolean
}

export interface CountInfo {
  filtered: number
  total: number
}

// Counts returned by the server — reflects domain-level filtering only
export interface ServerCounts {
  total: number          // before any domain filtering
  domain_filtered: number  // after domain selection, before search/dedup
}

export interface LeaderboardResponse {
  multilingual_df: Record<string, unknown>[]
  multilingual_counts: ServerCounts
  lang_dfs: Record<string, Record<string, unknown>[]>
  lang_count_infos: Record<string, ServerCounts>
  model_choices: { display: string; value: string }[]
  filter_state: FilterState
}

export interface LangResponse {
  lang_df: Record<string, unknown>[]
  lang_counts: ServerCounts
}

export interface MultilingualResponse {
  multilingual_df: Record<string, unknown>[]
  multilingual_counts: ServerCounts
}

export interface ModelResult {
  highlight_data: [string, string | null][]
  color_map: Record<string, string>
  count: number
  vocab_size: number | null
  table: Record<string, unknown>[]
}

export interface BattleResponse {
  error: string | null
  model1: ModelResult
  model2: ModelResult
}

export interface SubmitResponse {
  message: string
  worker_status_df: Record<string, unknown>[]
}

export interface WorkerStatus {
  worker_status_df: Record<string, unknown>[]
}

export type ActivePage = 'leaderboard' | 'battle' | 'submit' | 'guide' | 'about'

export interface ColumnHeader {
  key: string
  emoji: string
  label: string
  unit: string
  fullName: string
  datatype: string
}

export function parseColumnHeaders(
  displayMap: Record<string, string>,
  datatypes: string[],
): ColumnHeader[] {
  const keys = Object.keys(displayMap)
  return keys.map((key, i) => {
    const fullName = displayMap[key]
    const lines = fullName.split('\n').map((l) => l.trim())
    const firstLine = lines[0]
    const spaceIdx = firstLine.indexOf(' ')
    const emoji = spaceIdx > 0 ? firstLine.slice(0, spaceIdx) : ''
    const label =
      spaceIdx > 0 ? firstLine.slice(spaceIdx + 1) : firstLine
    const unit = lines[1] || ''
    return {
      key,
      emoji,
      label,
      unit,
      fullName,
      datatype: datatypes[i] || 'str',
    }
  })
}

