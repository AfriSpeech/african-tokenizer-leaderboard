import { useState, useCallback, useMemo, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type {
  Metadata,
  FilterState,
  LocalFilterState,
  LeaderboardResponse,
  ColumnHeader,
  CountInfo,
} from '../types'
import { parseColumnHeaders } from '../types'
import { applySearchFilter, validateSearchQuery } from '../utils/searchFilter'
import { filterDuplicates } from '../utils/filterDuplicates'
import LeaderboardTable from './LeaderboardTable'

interface Props {
  metadata: Metadata
  data: LeaderboardResponse | null
  filters: FilterState
  localFilters: LocalFilterState
  loading: boolean
  onFiltersChange: (f: FilterState) => void
  onLocalFiltersChange: (f: LocalFilterState) => void
  onRefresh: () => void
}

const EXAMPLES = [
  '@char_fidelity>0.9',
  '@vocab_size<100000 @unk_ratio<0.05 @ch>0.25',
  'google-bert/bert-base-multilingual-cased',
  'meta @fertility<2',
  '@compression_bytes>4 @parity_en<1.4 @tokenizer_algorithm=BPE',
  '@ebpb<2.5 @byte_fidelity>0.99',
  '@freq_renyi_efficiency>0.55 @freq_gini<0.8',
]

const FILTER_PATTERN = /(@[a-zA-Z_][a-zA-Z0-9_]*(?:>=|<=|>|<|==|!=|=)[^\s]+)/g

function parseFilterChips(input: string): string[] {
  FILTER_PATTERN.lastIndex = 0
  const chips: string[] = []
  let m: RegExpExecArray | null
  while ((m = FILTER_PATTERN.exec(input)) !== null) {
    chips.push(m[0])
  }
  return chips
}

function removeChipFromInput(input: string, chip: string): string {
  return input.replace(chip, '').replace(/\s{2,}/g, ' ').trim()
}

export default function Leaderboard({
  metadata,
  data,
  filters,
  localFilters,
  loading,
  onFiltersChange,
  onLocalFiltersChange,
  onRefresh,
}: Props) {
  const [activeLanguage, setActiveLanguage] = useState('multilingual')
  const [showExamples, setShowExamples] = useState(false)
  const [searchInput, setSearchInput] = useState(localFilters.search_query)
  const [searchError, setSearchError] = useState<string | null>(null)
  const [showFilterHelp, setShowFilterHelp] = useState(false)
  const [suggestions, setSuggestions] = useState<{ field: string; description: string }[]>([])
  const [activeSuggestion, setActiveSuggestion] = useState(0)
  const searchRef = useRef<HTMLInputElement>(null)
  const suggestionsRef = useRef<HTMLDivElement>(null)

  // Column visibility — initialized from server defaults; "model" is always visible
  const [visibleColumns, setVisibleColumns] = useState<Set<string>>(
    () => new Set(metadata.default_visible_columns ?? Object.keys(metadata.display_columns_map)),
  )

  const allColumns: ColumnHeader[] = useMemo(
    () =>
      parseColumnHeaders(
        metadata.display_columns_map,
        metadata.dataframe_datatypes,
      ),
    [metadata],
  )

  // The subset of columns actually shown in the table (model is always pinned)
  const columns: ColumnHeader[] = useMemo(
    () => allColumns.filter((c) => c.key === 'model' || visibleColumns.has(c.key)),
    [allColumns, visibleColumns],
  )

  const defaultVisibleSet = useMemo(
    () => new Set(metadata.default_visible_columns ?? Object.keys(metadata.display_columns_map)),
    [metadata],
  )

  const handleToggleColumn = useCallback((key: string) => {
    if (key === 'model') return
    setVisibleColumns((prev) => {
      const next = new Set(prev)
      if (next.has(key)) {
        next.delete(key)
      } else {
        next.add(key)
      }
      return next
    })
  }, [])

  const handleShowDefault = useCallback(() => {
    setVisibleColumns(new Set(defaultVisibleSet))
  }, [defaultVisibleSet])

  const handleShowAll = useCallback(() => {
    setVisibleColumns(new Set(allColumns.map((c) => c.key)))
  }, [allColumns])

  const isMultilingual = activeLanguage === 'multilingual'

  // Raw data from server for the active tab only — deps are granular so patching
  // a different tab does not trigger a recompute here.
  const activeSlice = isMultilingual
    ? data?.multilingual_df
    : data?.lang_dfs[activeLanguage]

  // True when the active tab's data has not yet arrived from the server
  const isTabLoading = !loading && activeSlice === undefined

  const rawData = useMemo(() => {
    return activeSlice ?? []
  }, [activeSlice])

  // Client-side filtering pipeline: search → dedup
  const searchedData = useMemo(
    () => applySearchFilter(rawData, localFilters.search_query),
    [rawData, localFilters.search_query],
  )

  const dedupedData = useMemo(
    () => filterDuplicates(searchedData, localFilters.hide_duplicates),
    [searchedData, localFilters.hide_duplicates],
  )

  // CountInfo computed entirely client-side
  const currentCounts: CountInfo = useMemo(() => {
    return { total: rawData.length, filtered: dedupedData.length }
  }, [rawData, dedupedData])

  const filterChips = useMemo(() => parseFilterChips(searchInput), [searchInput])

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value
    setSearchInput(val)

    const cursor = e.target.selectionStart ?? val.length
    const before = val.slice(0, cursor)
    const atMatch = before.match(/@([a-zA-Z_]*)$/)
    if (atMatch) {
      const prefix = atMatch[1].toLowerCase()
      const filtered = (metadata.filter_fields || []).filter((f) =>
        f.field.startsWith(prefix),
      )
      setSuggestions(filtered)
      setActiveSuggestion(0)
    } else {
      setSuggestions([])
    }
  }

  const insertSuggestion = (field: string) => {
    const el = searchRef.current
    if (!el) return
    const cursor = el.selectionStart ?? searchInput.length
    const before = searchInput.slice(0, cursor)
    const after = searchInput.slice(cursor)
    const atMatch = before.match(/@([a-zA-Z_]*)$/)
    if (atMatch) {
      const newBefore = before.slice(0, before.length - atMatch[0].length) + `@${field}`
      setSearchInput(newBefore + after)
      setTimeout(() => {
        if (el) {
          el.selectionStart = el.selectionEnd = newBefore.length
          el.focus()
        }
      }, 0)
    }
    setSuggestions([])
  }

  const handleSearchKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (suggestions.length === 0) return
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActiveSuggestion((i) => Math.min(i + 1, suggestions.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveSuggestion((i) => Math.max(i - 1, 0))
    } else if (e.key === 'Enter' || e.key === 'Tab') {
      e.preventDefault()
      insertSuggestion(suggestions[activeSuggestion].field)
    } else if (e.key === 'Escape') {
      setSuggestions([])
    }
  }

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (
        suggestionsRef.current &&
        !suggestionsRef.current.contains(e.target as Node) &&
        searchRef.current &&
        !searchRef.current.contains(e.target as Node)
      ) {
        setSuggestions([])
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const doSearch = useCallback(
    (query: string) => {
      const { isValid, errorMessage } = validateSearchQuery(query)
      if (!isValid) {
        setSearchError(errorMessage)
        return
      }
      setSearchError(null)
      onLocalFiltersChange({ ...localFilters, search_query: query })
    },
    [localFilters, onLocalFiltersChange],
  )

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setSuggestions([])
    doSearch(searchInput)
  }

  const handleExampleClick = (q: string) => {
    setSearchInput(q)
    setSuggestions([])
    doSearch(q)
  }

  const handleClearSearch = () => {
    setSearchInput('')
    setSearchError(null)
    setSuggestions([])
    onLocalFiltersChange({ ...localFilters, search_query: '' })
  }

  const handleRemoveChip = (chip: string) => {
    const newInput = removeChipFromInput(searchInput, chip)
    setSearchInput(newInput)
    doSearch(newInput)
  }

  const handleLanguageToggle = (code: string) => {
    const selected = filters.selected_langs.includes(code)
      ? filters.selected_langs.filter((c) => c !== code)
      : [...filters.selected_langs, code]
    onFiltersChange({ ...filters, selected_langs: selected })
  }

  const handleDomainToggle = (domain: string) => {
    if (isMultilingual) {
      const selected = filters.selected_multilingual_domains.includes(domain)
        ? filters.selected_multilingual_domains.filter((d) => d !== domain)
        : [...filters.selected_multilingual_domains, domain]
      onFiltersChange({
        ...filters,
        selected_multilingual_domains: selected,
      })
    } else {
      const langDomains = filters.selected_domains[activeLanguage] || []
      const selected = langDomains.includes(domain)
        ? langDomains.filter((d) => d !== domain)
        : [...langDomains, domain]
      onFiltersChange({
        ...filters,
        selected_domains: {
          ...filters.selected_domains,
          [activeLanguage]: selected,
        },
      })
    }
  }

  const handleHideDuplicates = () => {
    onLocalFiltersChange({ ...localFilters, hide_duplicates: !localFilters.hide_duplicates })
  }

  const currentDomains = useMemo(() => {
    if (isMultilingual) {
      return metadata.all_domains
    }
    const tab = metadata.language_tabs.find(
      (l) => l.code === activeLanguage,
    )
    return tab?.domains || []
  }, [activeLanguage, isMultilingual, metadata])

  const selectedDomains = useMemo(() => {
    if (isMultilingual) {
      return filters.selected_multilingual_domains
    }
    return filters.selected_domains[activeLanguage] || []
  }, [activeLanguage, isMultilingual, filters])

  const handleSelectAllDomains = () => {
    const allNames = currentDomains.map((d) => d[1])
    if (isMultilingual) {
      onFiltersChange({
        ...filters,
        selected_multilingual_domains: allNames,
      })
    } else {
      onFiltersChange({
        ...filters,
        selected_domains: {
          ...filters.selected_domains,
          [activeLanguage]: allNames,
        },
      })
    }
  }

  const handleClearAllDomains = () => {
    if (isMultilingual) {
      onFiltersChange({ ...filters, selected_multilingual_domains: [] })
    } else {
      onFiltersChange({
        ...filters,
        selected_domains: {
          ...filters.selected_domains,
          [activeLanguage]: [],
        },
      })
    }
  }

  const handleSelectAllLangs = () => {
    onFiltersChange({
      ...filters,
      selected_langs: metadata.language_tabs.map((l) => l.code),
    })
  }

  const handleClearAllLangs = () => {
    onFiltersChange({ ...filters, selected_langs: [] })
  }

  const handleDownloadCSV = () => {
    if (!dedupedData.length) return

    // Collect all keys present in the data (all display columns first, then extras)
    const displayKeys = allColumns.map((c) => c.key)
    const extraKeys = Object.keys(dedupedData[0]).filter((k) => !displayKeys.includes(k))
    const allKeys = [...displayKeys, ...extraKeys]

    // Use friendly header names for display columns, raw key names for extras
    const displayLabelMap = Object.fromEntries(allColumns.map((c) => [c.key, c.label]))
    const headers = allKeys.map((k) => displayLabelMap[k] ?? k)

    const escape = (v: unknown) => {
      const s = v == null ? '' : String(v)
      return s.includes(',') || s.includes('"') || s.includes('\n')
        ? `"${s.replace(/"/g, '""')}"`
        : s
    }

    const csvRows = [headers.join(',')]
    for (const row of dedupedData) {
      csvRows.push(allKeys.map((key) => escape(row[key])).join(','))
    }

    const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `leaderboard_${activeLanguage}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const langTabs = [
    { code: 'multilingual', name: 'Multilingual' },
    ...metadata.language_tabs.map((l) => ({
      code: l.code,
      name: l.name,
    })),
  ]

  return (
    <div className="leaderboard">
      {/* Search row */}
      <form className="search-row" onSubmit={handleSearchSubmit}>
        <div className="search-input-wrap">
          <button type="submit" className="search-submit-icon" title="Search">🔍</button>
          <input
            ref={searchRef}
            className="search-input"
            type="text"
            value={searchInput}
            onChange={handleSearchChange}
            onKeyDown={handleSearchKeyDown}
            placeholder="Search models or type @ to filter: @fertility<2 @unk_ratio<0.01 ..."
            autoComplete="off"
          />
          {suggestions.length > 0 && (
            <div className="search-suggestions" ref={suggestionsRef}>
              {suggestions.map((s, i) => (
                <div
                  key={s.field}
                  className={`search-suggestion${i === activeSuggestion ? ' search-suggestion--active' : ''}`}
                  onMouseDown={(e) => { e.preventDefault(); insertSuggestion(s.field) }}
                  onMouseEnter={() => setActiveSuggestion(i)}
                >
                  <span className="search-suggestion__field">@{s.field}</span>
                  <span className="search-suggestion__desc">{s.description}</span>
                </div>
              ))}
            </div>
          )}
        </div>
        <button type="button" className="btn btn--icon" onClick={handleClearSearch} title="Clear search">
          ✕
        </button>
        <button
          type="button"
          className={`btn btn--icon${showFilterHelp ? ' btn--primary' : ''}`}
          onClick={() => setShowFilterHelp(!showFilterHelp)}
          title="Filter syntax help"
        >
          ?
        </button>
      </form>

      {/* Filter help overlay */}
      {showFilterHelp && (
        <>
          <div className="filter-help-backdrop" onClick={() => setShowFilterHelp(false)} />
          <div className="filter-help-overlay">
            <div className="filter-help-panel__header">
              <strong>Filter Syntax Help</strong>
              <button className="btn btn--ghost btn--sm" onClick={() => setShowFilterHelp(false)}>✕ Close</button>
            </div>
            <div className="filter-help-panel__body markdown">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {metadata.filter_help_text}
              </ReactMarkdown>
            </div>
          </div>
        </>
      )}

      {searchError && <div className="search-validation">❌ {searchError}</div>}

      {/* Active filter chips */}
      {filterChips.length > 0 && (
        <div className="filter-chips">
          {filterChips.map((chip) => (
            <span key={chip} className="search-tag">
              <span className="search-tag__text">{chip}</span>
              <button
                type="button"
                className="search-tag__remove"
                onClick={() => handleRemoveChip(chip)}
                title={`Remove filter ${chip}`}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}

      {/* Examples — collapsed by default */}
      <div className="examples-section">
        <button
          type="button"
          className="examples-toggle"
          onClick={() => setShowExamples((s) => !s)}
        >
          Examples {showExamples ? '▴' : '▾'}
        </button>
        {showExamples && (
          <div className="example-filters">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                className="example-pill"
                onClick={() => handleExampleClick(ex)}
              >
                {ex}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Language tabs — full width, all tabs visible */}
      <div className="lang-tabs-row">
        <div className="lang-tabs">
          {langTabs.map((tab) => {
            const isLangLoading = tab.code !== 'multilingual' && data != null && !(tab.code in (data.lang_dfs ?? {}))
            return (
              <button
                key={tab.code}
                className={`lang-tab${activeLanguage === tab.code ? ' lang-tab--active' : ''}${isLangLoading ? ' lang-tab--loading' : ''}`}
                onClick={() => setActiveLanguage(tab.code)}
                title={isLangLoading ? 'Loading...' : undefined}
              >
                {tab.name}
                {isLangLoading && <span className="lang-tab__spinner" />}
              </button>
            )
          })}
        </div>
      </div>

      {/* Hide duplicates + count — below the tabs */}
      <div className="tabs-meta-row">
        <div className="toggle-inline" onClick={handleHideDuplicates}>
          <div className={`toggle${localFilters.hide_duplicates ? ' toggle--active' : ''}`}>
            <div className="toggle__knob" />
          </div>
          <span>Hide duplicates by Tokenizer Hash</span>
        </div>
        <div className="count-badge">
          {currentCounts.filtered} filtered of {currentCounts.total}
        </div>
      </div>

      {/* Multilingual note */}
      {isMultilingual && (
        <div className="multilingual-note">
          ¹ Multilingual metrics are averaged across all selected languages, balanced regardless of dataset size.
        </div>
      )}

      {/* Always-visible filter bar */}
      <div className="filter-bar">
        {isMultilingual && (
          <div className="filter-section">
            <div className="filter-section__header">
              <span className="filter-section__title">Languages</span>
              <div className="filter-section__actions">
                <button className="btn btn--ghost btn--sm" onClick={handleSelectAllLangs}>
                  Select All
                </button>
                <button className="btn btn--ghost btn--sm" onClick={handleClearAllLangs}>
                  Clear
                </button>
              </div>
            </div>
            <div className="chip-grid">
              {metadata.language_tabs.map((lang) => {
                const active = filters.selected_langs.includes(lang.code)
                return (
                  <button
                    key={lang.code}
                    className={`chip${active ? ' chip--active' : ''}`}
                    onClick={() => handleLanguageToggle(lang.code)}
                  >
                    <span className="chip__check" />
                    {lang.name}
                  </button>
                )
              })}
            </div>
          </div>
        )}

        <div className="filter-section">
          <div className="filter-section__header">
            <span className="filter-section__title">
              {isMultilingual
                ? 'Domains'
                : `${langTabs.find((l) => l.code === activeLanguage)?.name || ''} Domains`}
            </span>
            <div className="filter-section__actions">
              <button className="btn btn--ghost btn--sm" onClick={handleSelectAllDomains}>
                Select All
              </button>
              <button className="btn btn--ghost btn--sm" onClick={handleClearAllDomains}>
                Clear
              </button>
            </div>
          </div>
          <div className="chip-grid">
            {currentDomains.map(([pretty, name]) => {
              const active = selectedDomains.includes(name)
              return (
                <button
                  key={name}
                  className={`chip${active ? ' chip--active' : ''}`}
                  onClick={() => handleDomainToggle(name)}
                >
                  <span className="chip__check" />
                  {pretty}
                </button>
              )
            })}
          </div>
        </div>
      </div>

      {/* Data Table */}
      <details className="col-visibility-accordion">
        <summary className="col-visibility-summary">
          <span className="col-visibility-summary__label">Column Visibility</span>
          <span className="col-visibility-summary__count">
            {columns.length - 1} of {allColumns.length - 1} shown
          </span>
        </summary>
        <div className="col-visibility-body">
          <div className="col-visibility-actions">
            <button className="btn btn--ghost btn--sm" onClick={handleShowDefault}>
              Restore defaults
            </button>
            <button className="btn btn--ghost btn--sm" onClick={handleShowAll}>
              Show all
            </button>
          </div>
          {/* Main columns */}
          <div className="col-visibility-section">
            <div className="col-visibility-section__title">Main</div>
            <div className="col-visibility-grid">
              {allColumns.filter((c) => c.key !== 'model' && defaultVisibleSet.has(c.key)).map((col) => {
                const checked = visibleColumns.has(col.key)
                const desc = (metadata.column_descriptions || {})[col.key]
                return (
                  <label key={col.key} className="col-visibility-item" title={desc}>
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => handleToggleColumn(col.key)}
                    />
                    <span className="col-visibility-item__emoji">{col.emoji}</span>
                    <span className="col-visibility-item__label">{col.label}</span>
                  </label>
                )
              })}
            </div>
          </div>

          {/* Experimental columns */}
          <div className="col-visibility-section">
            <div className="col-visibility-section__title">Experimental</div>
            <div className="col-visibility-grid">
              {allColumns.filter((c) => c.key !== 'model' && !defaultVisibleSet.has(c.key)).map((col) => {
                const checked = visibleColumns.has(col.key)
                const desc = (metadata.column_descriptions || {})[col.key]
                return (
                  <label key={col.key} className="col-visibility-item col-visibility-item--experimental" title={desc}>
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => handleToggleColumn(col.key)}
                    />
                    <span className="col-visibility-item__emoji">{col.emoji}</span>
                    <span className="col-visibility-item__label">{col.label}</span>
                  </label>
                )
              })}
            </div>
          </div>
        </div>
      </details>

      {/* Data Table */}
      <LeaderboardTable
        data={dedupedData}
        columns={columns}
        loading={loading || isTabLoading}
        isMultilingual={isMultilingual}
        avgColumns={metadata.avg_multilingual_columns || []}
        columnDescriptions={metadata.column_descriptions || {}}
      />

      {/* Footer */}
      <div className="table-footer">
        <div className="table-footer__note">
          {currentCounts.total > 0 && (
            <>Showing {currentCounts.filtered} of {currentCounts.total} tokenizers</>
          )}
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button className="btn" onClick={handleDownloadCSV} disabled={!dedupedData.length}>
            📥 Download CSV
          </button>
          <button className="btn" onClick={onRefresh} disabled={loading}>
            🔄 Refresh & Reset
          </button>
        </div>
      </div>
    </div>
  )
}
