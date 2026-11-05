// Port of Python SearchFilter to TypeScript — applies the same text + @field filter logic client-side.

type Row = Record<string, unknown>

interface FieldFilter {
  field: string
  operator: string
  value: number
}

interface ParsedQuery {
  textSearch: string
  fieldFilters: FieldFilter[]
}

const NUMERIC_FIELDS = new Set([
  'fertility', 'compression_chars', 'compression_bytes', 'reversible_ratio',
  'unk_ratio', 'total_tokens', 'total_words', 'total_chars', 'total_bytes',
  'byte_edit_distance_total', 'char_edit_distance_total',
  'total_docs', 'reversible_docs_count', 'unk_tokens_count', 'vocab_size',
  'vocab_near_duplicates', 'parity_en',
  // Experimental fidelity metrics
  'byte_fidelity', 'effective_bytes_per_token', 'ebpb', 'ebpb_full',
  'char_fidelity', 'effective_chars_per_token', 'ebpc',
  // Experimental frequency / Zipf metrics
  'freq_cardinality', 'freq_cardinality_ratio', 'freq_auc', 'freq_slope', 'freq_power_law',
  'freq_entropy', 'freq_renyi_entropy',
  'freq_shannon_efficiency', 'freq_renyi_efficiency',
  'freq_shannon_efficiency_full', 'freq_renyi_efficiency_full',
  'freq_percentile_freq',
  'freq_hapax_count', 'freq_hapax_ratio',
  'freq_half_mass_count', 'freq_90pct_mass_count', 'freq_90pct_mass_ratio', 'freq_gini',
])

const PERCENTAGE_FIELDS = new Set([
  'reversible_ratio', 'unk_ratio', 'vocab_near_duplicates',
  'byte_fidelity', 'char_fidelity',
  'freq_hapax_ratio', 'freq_gini',
  'freq_shannon_efficiency', 'freq_renyi_efficiency',
  'freq_shannon_efficiency_full', 'freq_renyi_efficiency_full',
  'freq_percentile_freq',
  'freq_cardinality_ratio',
  'freq_90pct_mass_ratio',
])

const TEXT_FIELDS = [
  'model', 'revision', 'subfolder', 'target_mode',
  'tokenizer_algorithm', 'tokenizer_implementation',
  'vocab_hash', 'tokenizer_config_hash',
]

const FIELD_FILTER_RE = /^([a-zA-Z_][a-zA-Z0-9_]*)(>=|<=|>|<|==|!=|=)(.+)$/

type OpFn = (a: number, b: number) => boolean
const OPS: Record<string, OpFn> = {
  '>=': (a, b) => a >= b,
  '<=': (a, b) => a <= b,
  '>':  (a, b) => a > b,
  '<':  (a, b) => a < b,
  '=':  (a, b) => Math.abs(a - b) < 1e-10,
  '==': (a, b) => Math.abs(a - b) < 1e-10,
  '!=': (a, b) => Math.abs(a - b) >= 1e-10,
}

function parseSearchQuery(query: string): ParsedQuery {
  if (!query || !query.trim()) return { textSearch: '', fieldFilters: [] }

  const parts = query.trim().split(/\s+/)
  const textParts: string[] = []
  const fieldFilters: FieldFilter[] = []

  for (const part of parts) {
    if (part.startsWith('@')) {
      const filterStr = part.slice(1)
      const m = filterStr.match(FIELD_FILTER_RE)
      if (m) {
        const [, field, operator, valueStr] = m
        if (NUMERIC_FIELDS.has(field)) {
          let value = parseFloat(valueStr)
          if (!isNaN(value)) {
            // Percentage fields: if user types >1, treat as percentage
            if (PERCENTAGE_FIELDS.has(field) && value > 1) value = value / 100
            fieldFilters.push({ field, operator, value })
          }
        }
      }
    } else {
      textParts.push(part)
    }
  }

  return { textSearch: textParts.join(' '), fieldFilters }
}

function matchesTextSearch(row: Row, textSearch: string): boolean {
  const lower = textSearch.toLowerCase()
  for (const field of TEXT_FIELDS) {
    const raw = row[field]
    if (raw == null) continue
    let str = String(raw)
    // Mirror Python's special display formatting for text matching
    if (field === 'revision' && str !== 'main') str = `(rev: ${str})`
    else if (field === 'revision') str = ''
    else if (field === 'subfolder' && str) str = `(sub: ${str})`
    else if (field === 'target_mode') str = raw ? '(target)' : ''
    if (str.toLowerCase().includes(lower)) return true
  }
  return false
}

function matchesFieldFilter(row: Row, filter: FieldFilter): boolean {
  const raw = row[filter.field]
  if (raw == null) return false
  const num = parseFloat(String(raw))
  if (isNaN(num)) return false
  const op = OPS[filter.operator]
  return op ? op(num, filter.value) : false
}

export function applySearchFilter(rows: Row[], query: string): Row[] {
  if (!query || !query.trim()) return rows
  const { textSearch, fieldFilters } = parseSearchQuery(query)

  return rows.filter((row) => {
    if (textSearch && !matchesTextSearch(row, textSearch)) return false
    for (const f of fieldFilters) {
      if (!matchesFieldFilter(row, f)) return false
    }
    return true
  })
}

export function validateSearchQuery(query: string): { isValid: boolean; errorMessage: string } {
  if (!query || !query.trim()) return { isValid: true, errorMessage: '' }
  const parts = query.trim().split(/\s+/)
  for (const part of parts) {
    if (!part.startsWith('@')) continue
    const m = part.slice(1).match(FIELD_FILTER_RE)
    if (!m) continue
    const [, field, operator] = m
    if (!NUMERIC_FIELDS.has(field)) {
      return { isValid: false, errorMessage: `Unknown field: ${field}` }
    }
    if (!OPS[operator]) {
      return { isValid: false, errorMessage: `Unknown operator: ${operator}` }
    }
  }
  return { isValid: true, errorMessage: '' }
}
