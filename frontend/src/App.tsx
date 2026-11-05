import { useState, useEffect, useCallback, useRef } from 'react'
import type {
  Metadata,
  FilterState,
  LocalFilterState,
  LeaderboardResponse,
  ActivePage,
} from './types'
import { api } from './api'
import Leaderboard from './components/Leaderboard'
import Battle from './components/Battle'
import Submit from './components/Submit'
import Guide from './components/Guide'
import About from './components/About'

const NAV_ITEMS: { id: ActivePage; label: string; icon: string }[] = [
  { id: 'leaderboard', label: 'Leaderboard', icon: '🏆' },
  { id: 'battle', label: 'Battle', icon: '⚔️' },
  { id: 'submit', label: 'Submit', icon: '📤' },
  { id: 'guide', label: 'Guide', icon: '🛠️' },
  { id: 'about', label: 'About', icon: '📖' },
]

type Theme = 'dark' | 'light'

function initTheme(): Theme {
  const saved = localStorage.getItem('theme') as Theme | null
  if (saved === 'dark' || saved === 'light') return saved
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

function sameSet(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false
  const setA = new Set(a)
  return b.every((x) => setA.has(x))
}

function findChangedLangDomain(
  prevDomains: Record<string, string[]>,
  nextDomains: Record<string, string[]>,
): string | null {
  const keys = new Set([...Object.keys(prevDomains), ...Object.keys(nextDomains)])
  let changed: string | null = null
  for (const k of keys) {
    if (!sameSet(prevDomains[k] ?? [], nextDomains[k] ?? [])) {
      if (changed !== null) return null // more than one changed
      changed = k
    }
  }
  return changed
}

export default function App() {
  const [activePage, setActivePage] = useState<ActivePage>('leaderboard')
  const [metadata, setMetadata] = useState<Metadata | null>(null)
  const [leaderboardData, setLeaderboardData] =
    useState<LeaderboardResponse | null>(null)
  const [filters, setFilters] = useState<FilterState | null>(null)
  const [localFilters, setLocalFilters] = useState<LocalFilterState | null>(null)
  const [loading, setLoading] = useState(true)
  const [tableLoading, setTableLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [theme, setTheme] = useState<Theme>(initTheme)

  const filtersRef = useRef<FilterState | null>(null)
  const metadataRef = useRef<Metadata | null>(null)
  const leaderboardDataRef = useRef<LeaderboardResponse | null>(null)

  const dataVersionRef = useRef<string>('')
  const abortControllerRef = useRef<AbortController | null>(null)
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('theme', theme)
  }, [theme])

  const toggleTheme = () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))

  useEffect(() => {
    let cancelled = false
    async function init() {
      try {
        const meta = await api.metadata()
        if (cancelled) return
        setMetadata(meta)
        metadataRef.current = meta
        dataVersionRef.current = meta.data_version ?? ''

        const defaultFilters: FilterState = {
          selected_langs: meta.language_tabs.map((l) => l.code),
          selected_domains: Object.fromEntries(
            meta.language_tabs.map((l) => [
              l.code,
              l.domains.map((d) => d[1]),
            ]),
          ),
          selected_multilingual_domains: meta.all_domains.map((d) => d[1]),
        }
        const defaultLocalFilters: LocalFilterState = {
          search_query: meta.default_search_query,
          hide_duplicates: true,
        }
        setFilters(defaultFilters)
        filtersRef.current = defaultFilters
        setLocalFilters(defaultLocalFilters)

        // Phase 1: fetch only multilingual table + model_choices (~5MB).
        // The user can see and interact with the main page as soon as this resolves.
        const initData = await api.leaderboardInit()
        if (cancelled) return
        setLeaderboardData(initData)
        leaderboardDataRef.current = initData
      } catch (e: unknown) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      } finally {
        if (!cancelled) setLoading(false)
      }

      // Phase 2: load priority language tabs (code, en) in parallel while the
      // user is already browsing the multilingual view.
      const meta = metadataRef.current
      if (!meta || cancelled) return
      const PRIORITY_LANGS = ['code', 'en']
      const patchLang = (code: string, lang_df: Record<string, unknown>[], lang_counts: { total: number; domain_filtered: number }) => {
        if (cancelled) return
        setLeaderboardData((prev) => {
          if (!prev) return prev
          const next = {
            ...prev,
            lang_dfs: { ...prev.lang_dfs, [code]: lang_df },
            lang_count_infos: { ...prev.lang_count_infos, [code]: lang_counts },
          }
          leaderboardDataRef.current = next
          return next
        })
      }

      const phase2 = PRIORITY_LANGS
        .filter(code => meta.language_tabs.some(l => l.code === code))
        .map((code) => {
          const langTab = meta.language_tabs.find((l) => l.code === code)
          const domains = langTab ? langTab.domains.map((d) => d[1]) : []
          return api.leaderboardLang(code, domains)
            .then((r) => patchLang(code, r.lang_df, r.lang_counts))
            .catch((e) => console.error(`Failed to load lang tab ${code}:`, e))
        })
      await Promise.all(phase2)

      // Phase 3: load remaining language tabs in parallel.
      if (cancelled) return
      const remainingLangs = meta.language_tabs
        .map((l) => l.code)
        .filter((c) => !PRIORITY_LANGS.includes(c))
      const phase3 = remainingLangs.map((code) => {
        const langTab = meta.language_tabs.find((l) => l.code === code)
        const domains = langTab ? langTab.domains.map((d) => d[1]) : []
        return api.leaderboardLang(code, domains)
          .then((r) => patchLang(code, r.lang_df, r.lang_counts))
          .catch((e) => console.error(`Failed to load lang tab ${code}:`, e))
      })
      await Promise.all(phase3)
    }
    init()
    return () => {
      cancelled = true
    }
  }, [])

  // Poll /api/metadata every 60s to detect data_version changes (new submissions).
  // When a version change is detected, re-run the progressive load sequence.
  useEffect(() => {
    const POLL_INTERVAL = 60_000
    const PRIORITY_LANGS = ['code', 'en']

    const intervalId = setInterval(async () => {
      try {
        const meta = await api.metadata()
        const newVersion = meta.data_version ?? ''
        if (!dataVersionRef.current || newVersion === dataVersionRef.current) return

        console.log(`data_version changed: ${dataVersionRef.current} -> ${newVersion}`)
        dataVersionRef.current = newVersion
        setMetadata(meta)
        metadataRef.current = meta

        const patchLang = (code: string, lang_df: Record<string, unknown>[], lang_counts: { total: number; domain_filtered: number }) => {
          setLeaderboardData((prev) => {
            if (!prev) return prev
            const next = {
              ...prev,
              lang_dfs: { ...prev.lang_dfs, [code]: lang_df },
              lang_count_infos: { ...prev.lang_count_infos, [code]: lang_counts },
            }
            leaderboardDataRef.current = next
            return next
          })
        }

        const initData = await api.leaderboardInit()
        setLeaderboardData(initData)
        leaderboardDataRef.current = initData

        const priorityFetches = PRIORITY_LANGS
          .filter(code => meta.language_tabs.some(l => l.code === code))
          .map((code) => {
            const langTab = meta.language_tabs.find((l) => l.code === code)
            const domains = langTab ? langTab.domains.map((d) => d[1]) : []
            return api.leaderboardLang(code, domains)
              .then((r) => patchLang(code, r.lang_df, r.lang_counts))
              .catch((e) => console.error(`Refresh: failed to load lang tab ${code}:`, e))
          })
        await Promise.all(priorityFetches)

        const remainingLangs = meta.language_tabs
          .map((l) => l.code)
          .filter((c) => !PRIORITY_LANGS.includes(c))
        const restFetches = remainingLangs.map((code) => {
          const langTab = meta.language_tabs.find((l) => l.code === code)
          const domains = langTab ? langTab.domains.map((d) => d[1]) : []
          return api.leaderboardLang(code, domains)
            .then((r) => patchLang(code, r.lang_df, r.lang_counts))
            .catch((e) => console.error(`Refresh: failed to load lang tab ${code}:`, e))
        })
        await Promise.all(restFetches)
      } catch (e) {
        console.error('Version poll error:', e)
      }
    }, POLL_INTERVAL)

    return () => clearInterval(intervalId)
  }, [])

  const fetchLeaderboard = useCallback(
    async (newFilters: FilterState) => {
      abortControllerRef.current?.abort()
      const controller = new AbortController()
      abortControllerRef.current = controller

      setTableLoading(true)
      try {
        const data = await api.leaderboard(newFilters, controller.signal)
        setLeaderboardData(data)
        leaderboardDataRef.current = data
      } catch (e: unknown) {
        if ((e as DOMException)?.name === 'AbortError') return
        console.error('Leaderboard fetch error:', e)
      } finally {
        if (!controller.signal.aborted) {
          setTableLoading(false)
        }
      }
    },
    [],
  )

  // Updates filter UI immediately but debounces the actual API call by 400ms
  // so rapid successive clicks (e.g. selecting multiple language chips) only
  // fire one request after the user pauses.
  const handleFiltersChange = useCallback(
    (newFilters: FilterState) => {
      const prev = filtersRef.current  // capture before update
      setFilters(newFilters)
      filtersRef.current = newFilters
      if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current)
      debounceTimerRef.current = setTimeout(async () => {
        if (!prev) {
          fetchLeaderboard(newFilters)
          return
        }

        const meta = metadataRef.current
        const currentData = leaderboardDataRef.current

        abortControllerRef.current?.abort()
        const controller = new AbortController()
        abortControllerRef.current = controller

        const langsChanged = !sameSet(prev.selected_langs, newFilters.selected_langs)
        const multiDomainsChanged = !sameSet(
          prev.selected_multilingual_domains,
          newFilters.selected_multilingual_domains,
        )
        const changedLangDomain = findChangedLangDomain(
          prev.selected_domains,
          newFilters.selected_domains,
        )

        setTableLoading(true)
        try {
          if (!currentData || !meta) {
            const data = await api.leaderboard(newFilters, controller.signal)
            setLeaderboardData(data)
            leaderboardDataRef.current = data
          } else if (langsChanged) {
            const result = await api.leaderboardMultilingual(
              newFilters.selected_langs,
              newFilters.selected_multilingual_domains,
              controller.signal,
            )
            setLeaderboardData((prev) => {
              if (!prev) return prev
              const next = {
                ...prev,
                multilingual_df: result.multilingual_df,
                multilingual_counts: result.multilingual_counts,
              }
              leaderboardDataRef.current = next
              return next
            })
          } else if (multiDomainsChanged) {
            const result = await api.leaderboardMultilingual(
              newFilters.selected_langs,
              newFilters.selected_multilingual_domains,
              controller.signal,
            )
            setLeaderboardData((prev) => {
              if (!prev) return prev
              const next = {
                ...prev,
                multilingual_df: result.multilingual_df,
                multilingual_counts: result.multilingual_counts,
              }
              leaderboardDataRef.current = next
              return next
            })
          } else if (changedLangDomain) {
            const domains = newFilters.selected_domains[changedLangDomain] ?? []
            const result = await api.leaderboardLang(
              changedLangDomain,
              domains,
              controller.signal,
            )
            setLeaderboardData((prev) => {
              if (!prev) return prev
              const next = {
                ...prev,
                lang_dfs: { ...prev.lang_dfs, [changedLangDomain]: result.lang_df },
                lang_count_infos: {
                  ...prev.lang_count_infos,
                  [changedLangDomain]: result.lang_counts,
                },
              }
              leaderboardDataRef.current = next
              return next
            })
          }
        } catch (e: unknown) {
          if ((e as DOMException)?.name === 'AbortError') return
          console.error('Leaderboard fetch error:', e)
        } finally {
          if (!controller.signal.aborted) {
            setTableLoading(false)
          }
        }
      }, 400)
    },
    [fetchLeaderboard],
  )

  // Local filter changes (search, hide_duplicates) are instant — no API call
  const handleLocalFiltersChange = useCallback((newLocalFilters: LocalFilterState) => {
    setLocalFilters(newLocalFilters)
  }, [])

  const handleRefresh = useCallback(async () => {
    setTableLoading(true)
    try {
      const data = await api.refreshLeaderboard()
      setLeaderboardData(data)
      leaderboardDataRef.current = data
      if (data.filter_state) {
        setFilters(data.filter_state)
        filtersRef.current = data.filter_state
      }
    } catch (e: unknown) {
      console.error('Refresh error:', e)
    } finally {
      setTableLoading(false)
    }
  }, [])

  if (loading) {
    return (
      <div className="app">
        <div className="loading-screen">
          <div className="loading-spinner" />
          <div className="loading-text">
            Loading tokenizer data...
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="app">
        <div className="loading-screen">
          <div className="error-banner">Failed to load: {error}</div>
          <button className="btn btn--primary" onClick={() => window.location.reload()}>
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header__inner">
          <div
            className="header__logo"
            onClick={() => setActivePage('leaderboard')}
          >
            <span className="header__trophy">🏆</span>
            <span className="header__title">
              Multilingual Tokenizer Leaderboard
            </span>
          </div>
          <nav className="header__nav">
            {NAV_ITEMS.map((item) => (
              <button
                key={item.id}
                className={`header__nav-item${activePage === item.id ? ' header__nav-item--active' : ''}`}
                onClick={() => setActivePage(item.id)}
              >
                {item.icon} {item.label}
              </button>
            ))}
            <button
              className="btn btn--icon header__theme-toggle"
              onClick={toggleTheme}
              title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              aria-label="Toggle theme"
            >
              {theme === 'dark' ? '☀️' : '🌙'}
            </button>
          </nav>
        </div>
      </header>

      <main className="main">
        {activePage === 'leaderboard' && metadata && filters && localFilters && (
          <Leaderboard
            metadata={metadata}
            data={leaderboardData}
            filters={filters}
            localFilters={localFilters}
            loading={tableLoading}
            onFiltersChange={handleFiltersChange}
            onLocalFiltersChange={handleLocalFiltersChange}
            onRefresh={handleRefresh}
          />
        )}
        {activePage === 'battle' && leaderboardData && (
          <Battle modelChoices={leaderboardData.model_choices} />
        )}
        {activePage === 'submit' && metadata && (
          <Submit languageTabs={metadata.language_tabs} />
        )}
        {activePage === 'guide' && <Guide />}
        {activePage === 'about' && <About />}
      </main>
    </div>
  )
}
