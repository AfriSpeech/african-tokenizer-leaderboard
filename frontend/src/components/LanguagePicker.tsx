import { useId, useMemo, useRef, useState } from 'react'
import type { Language } from '../types'

export const ALL = 'all'

interface Props {
  languages: Language[]
  nAfrican: number
  value: string
  onChange: (code: string) => void
}

interface Option {
  code: string
  name: string
  group: string
}

export default function LanguagePicker({ languages, nAfrican, value, onChange }: Props) {
  const id = useId()
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)

  const allOption: Option = { code: ALL, name: `All African languages (${nAfrican})`, group: '' }
  const selected =
    value === ALL ? allOption : languages.find((l) => l.code === value) ?? allOption

  const options = useMemo(() => {
    const qq = query.trim().toLowerCase()
    const match = (l: Language) =>
      !qq || l.name.toLowerCase().includes(qq) || l.code.toLowerCase().startsWith(qq)
    const out: Option[] = []
    if (!qq || allOption.name.toLowerCase().includes(qq)) out.push(allOption)
    for (const group of ['high_resource', 'african'] as const) {
      for (const l of languages.filter((x) => x.group === group && match(x))) {
        out.push({ code: l.code, name: l.name, group })
      }
    }
    return out
  }, [query, languages, nAfrican]) // eslint-disable-line react-hooks/exhaustive-deps

  const choose = (o: Option) => {
    onChange(o.code)
    setQuery('')
    setOpen(false)
    inputRef.current?.blur()
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
      choose(options[active])
    } else if (e.key === 'Escape') {
      setOpen(false)
      setQuery('')
    }
  }

  let lastGroup: string | null = null
  const groupLabel: Record<string, string> = {
    high_resource: 'High-resource languages of Africa',
    african: 'African languages',
  }

  return (
    <div className="field picker">
      <label htmlFor={`${id}-input`} className="field-label">Language</label>
      <div className="picker-box">
        <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" style={{ color: 'var(--text-muted)' }}>
          <circle cx="11" cy="11" r="7" />
          <path d="M20 20l-3.5-3.5" />
        </svg>
        <input
          id={`${id}-input`}
          ref={inputRef}
          role="combobox"
          aria-expanded={open}
          aria-controls={`${id}-list`}
          aria-autocomplete="list"
          aria-activedescendant={open && options[active] ? `${id}-opt-${options[active].code}` : undefined}
          placeholder={selected.name}
          value={open ? query : selected.name}
          onFocus={() => { setOpen(true); setActive(0) }}
          onBlur={() => setTimeout(() => setOpen(false), 120)}
          onChange={(e) => { setQuery(e.target.value); setOpen(true); setActive(0) }}
          onKeyDown={onKey}
        />
        {value !== ALL && <span className="mono muted" style={{ fontSize: 12 }}>{value}</span>}
      </div>
      {open && (
        <ul id={`${id}-list`} role="listbox" className="picker-list" aria-label="Languages">
          {options.length === 0 && <li className="picker-empty">No language matches “{query}”</li>}
          {options.map((o, i) => {
            const header = o.group && o.group !== lastGroup ? groupLabel[o.group] : null
            lastGroup = o.group || lastGroup
            return (
              <li key={o.code} role="presentation">
                {header && <div className="picker-group" aria-hidden="true">{header}</div>}
                <div
                  id={`${id}-opt-${o.code}`}
                  role="option"
                  aria-selected={i === active}
                  className="picker-option"
                  onMouseDown={(e) => { e.preventDefault(); choose(o) }}
                  onMouseEnter={() => setActive(i)}
                >
                  <span>{o.name}</span>
                  {o.code !== ALL && <span className="code">{o.code}</span>}
                </div>
              </li>
            )
          })}
        </ul>
      )}
      <span className="field-hint">Type a name or code, e.g. “Twi”, “yor”, “Amharic”.</span>
    </div>
  )
}
