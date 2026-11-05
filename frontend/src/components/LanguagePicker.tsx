import { useId, useMemo, useRef, useState } from 'react'
import type { Language } from '../types'

export const ALL = 'all'
export const AFRICAN = 'african'

interface Props {
  languages: Language[]
  nAfrican: number
  /** ['all'] = every language (default), ['african'] = African preset, [] = none yet, else codes. */
  value: string[]
  onChange: (codes: string[]) => void
}

interface Option {
  code: string
  name: string
  group: string
}

const GROUP_LABEL: Record<string, string> = {
  high_resource: 'High-resource languages of Africa',
  african: 'African languages',
}

export default function LanguagePicker({ languages, nAfrican, value, onChange }: Props) {
  const id = useId()
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const byCode = useMemo(() => new Map(languages.map((l) => [l.code, l])), [languages])
  const allLabel = `All languages (${languages.length})`
  const africanLabel = `All African languages (${nAfrican})`
  const isPreset = value.length === 1 && (value[0] === ALL || value[0] === AFRICAN)
  const chipName = (c: string) =>
    c === ALL ? allLabel : c === AFRICAN ? africanLabel : byCode.get(c)?.name ?? c

  const options = useMemo(() => {
    const qq = query.trim().toLowerCase()
    const match = (l: Language) =>
      !qq || l.name.toLowerCase().includes(qq) || l.code.toLowerCase().startsWith(qq)
    const out: Option[] = []
    if (!qq || allLabel.toLowerCase().includes(qq)) out.push({ code: ALL, name: allLabel, group: '' })
    if (!qq || africanLabel.toLowerCase().includes(qq)) out.push({ code: AFRICAN, name: africanLabel, group: '' })
    for (const group of ['high_resource', 'african'] as const) {
      for (const l of languages.filter((x) => x.group === group && match(x))) {
        out.push({ code: l.code, name: l.name, group })
      }
    }
    return out
  }, [query, languages, allLabel, africanLabel])

  const toggle = (o: Option) => {
    if (o.code === ALL || o.code === AFRICAN) {
      onChange([o.code])
      setOpen(false)
      inputRef.current?.blur()
    } else {
      const base = isPreset ? [] : value
      onChange(base.includes(o.code) ? base.filter((c) => c !== o.code) : [...base, o.code])
      inputRef.current?.focus()
    }
    setQuery('')
    setActive(0)
  }

  const onKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setOpen(true)
      setActive((a) => Math.min(a + 1, options.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActive((a) => Math.max(a - 1, 0))
    } else if (e.key === 'Enter' && open && options[active]) {
      e.preventDefault()
      toggle(options[active])
    } else if (e.key === 'Backspace' && !query && value.length) {
      onChange(value.slice(0, -1))
    } else if (e.key === 'Escape') {
      setOpen(false)
      setQuery('')
    }
  }

  let lastGroup: string | null = null

  return (
    <div className="field picker">
      <label htmlFor={`${id}-input`} className="field-label">Languages</label>
      <div className="picker-box" onClick={() => inputRef.current?.focus()}>
        <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" style={{ color: 'var(--text-muted)', flex: 'none' }}>
          <circle cx="11" cy="11" r="7" />
          <path d="M20 20l-3.5-3.5" />
        </svg>
        {value.map((c) => (
          <span key={c} className="lang-chip">
            {chipName(c)}
            <button
              type="button"
              aria-label={`Remove ${chipName(c)}`}
              onClick={(e) => { e.stopPropagation(); onChange(value.filter((x) => x !== c)) }}
            >
              <svg aria-hidden="true" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M6 6l12 12M18 6L6 18" /></svg>
            </button>
          </span>
        ))}
        <input
          id={`${id}-input`}
          ref={inputRef}
          role="combobox"
          aria-expanded={open}
          aria-controls={`${id}-list`}
          aria-autocomplete="list"
          aria-activedescendant={open && options[active] ? `${id}-opt-${options[active].code}` : undefined}
          placeholder={value.length ? 'Add a language…' : 'Pick languages, or “All languages”'}
          value={query}
          onFocus={() => { setOpen(true); setActive(0) }}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          onChange={(e) => { setQuery(e.target.value); setOpen(true); setActive(0) }}
          onKeyDown={onKey}
        />
      </div>
      {open && (
        <ul id={`${id}-list`} role="listbox" aria-multiselectable="true" className="picker-list" aria-label="Languages">
          {options.length === 0 && <li className="picker-empty">No language matches “{query}”</li>}
          {options.map((o, i) => {
            const header = o.group && o.group !== lastGroup ? GROUP_LABEL[o.group] : null
            lastGroup = o.group || lastGroup
            const checked = value.includes(o.code)
            return (
              <li key={o.code} role="presentation">
                {header && <div className="picker-group" aria-hidden="true">{header}</div>}
                <div
                  id={`${id}-opt-${o.code}`}
                  role="option"
                  aria-selected={checked}
                  className={`picker-option${i === active ? ' active' : ''}`}
                  onMouseDown={(e) => { e.preventDefault(); toggle(o) }}
                  onMouseEnter={() => setActive(i)}
                >
                  <span className="picker-check" aria-hidden="true">{checked ? '✓' : ''}</span>
                  <span style={{ flex: 1 }}>{o.name}</span>
                  {o.code !== ALL && o.code !== AFRICAN && <span className="code">{o.code}</span>}
                </div>
              </li>
            )
          })}
        </ul>
      )}
      <span className="field-hint">Remove “All languages” to pick your own, e.g. “Twi” and “English”; several are averaged, each counting equally.</span>
    </div>
  )
}
