import { useEffect, useState } from 'react'
import type { LanguagesResponse, Page } from './types'
import { api } from './api'
import Leaderboard from './components/Leaderboard'
import Compare from './components/Compare'
import Submit from './components/Submit'
import About from './components/About'

const PAGES: { id: Page; label: string }[] = [
  { id: 'leaderboard', label: 'Leaderboard' },
  { id: 'compare', label: 'Compare' },
  { id: 'submit', label: 'Submit a tokenizer' },
  { id: 'about', label: 'About' },
]

function pageFromHash(): Page {
  const id = window.location.hash.replace(/^#\/?/, '').split('?')[0]
  return (PAGES.find((p) => p.id === id)?.id ?? 'leaderboard') as Page
}

export default function App() {
  const [page, setPage] = useState<Page>(pageFromHash)
  const [langs, setLangs] = useState<LanguagesResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const onHash = () => setPage(pageFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    api.languages().then(setLangs).catch((e: Error) => setError(e.message))
  }, [])

  return (
    <>
      <header className="header">
        <div className="header-inner container">
          <div className="brand">
            <div className="brand-mark" aria-hidden="true">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="4" width="5" height="16" rx="1.5" />
                <rect x="10" y="9" width="5" height="11" rx="1.5" />
                <rect x="17" y="13" width="4" height="7" rx="1.5" />
              </svg>
            </div>
            <div>
              <div className="brand-title">African Tokenizer Leaderboard</div>
              <div className="brand-sub">by AfriSpeech</div>
            </div>
          </div>
          <nav className="nav" aria-label="Main">
            {PAGES.map((p) => (
              <a key={p.id} href={`#/${p.id}`} aria-current={page === p.id ? 'page' : undefined}>
                {p.label}
              </a>
            ))}
          </nav>
        </div>
      </header>
      <main className="container">
        {error && <div className="notice error" role="alert">Could not load the leaderboard: {error}</div>}
        {page === 'leaderboard' && <Leaderboard langs={langs} />}
        {page === 'compare' && <Compare />}
        {page === 'submit' && <Submit />}
        {page === 'about' && <About />}
      </main>
    </>
  )
}
