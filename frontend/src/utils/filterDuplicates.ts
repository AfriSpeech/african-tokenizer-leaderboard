// Port of Python filter_duplicates to TypeScript — same sort-priority and hash-based dedup.

type Row = Record<string, unknown>

function bool(v: unknown): boolean {
  return !!v && v !== 'false' && v !== '0'
}

function dateMs(v: unknown): number {
  if (v == null) return Infinity
  const d = new Date(String(v))
  return isNaN(d.getTime()) ? Infinity : d.getTime()
}

function num(v: unknown): number {
  if (v == null) return 0
  const n = parseFloat(String(v))
  return isNaN(n) ? 0 : n
}

function str(v: unknown): string {
  return v == null ? '' : String(v)
}

export function filterDuplicates(
  rows: Row[],
  hide: boolean,
  sortCol = 'fertility',
  sortAsc = true,
): Row[] {
  if (rows.length === 0) return rows

  // Sort using frontend duplicate-ranking priority:
  // [sortCol asc, author_is_verified desc, author_followers desc, likes desc,
  //  created_at asc, is_main desc, is_target asc, is_subfolder asc,
  //  model_key asc, tokenizer_config_hash asc]
  const sorted = [...rows].sort((a, b) => {
    // 1. Primary sort column
    const av = num(a[sortCol])
    const bv = num(b[sortCol])
    const nullA = a[sortCol] == null || a[sortCol] === ''
    const nullB = b[sortCol] == null || b[sortCol] === ''
    if (nullA && !nullB) return 1
    if (!nullA && nullB) return -1
    if (av !== bv) return sortAsc ? av - bv : bv - av

    // 2. author_is_verified (desc — verified preferred)
    const va = bool(a.author_is_verified)
    const vb = bool(b.author_is_verified)
    if (va !== vb) return va ? -1 : 1

    // 3. author_followers (desc)
    const fa = num(a.author_followers)
    const fb = num(b.author_followers)
    if (fa !== fb) return fb - fa

    // 4. Likes (desc)
    const la = num(a.likes)
    const lb = num(b.likes)
    if (la !== lb) return lb - la

    // 5. created_at (asc)
    const ca = dateMs(a.created_at)
    const cb = dateMs(b.created_at)
    if (ca !== cb) return ca - cb

    // 6. is_main: revision === 'main' (desc — main preferred, so false < true → -1 when b is main)
    const ma = a.revision == null || a.revision === 'main'
    const mb = b.revision == null || b.revision === 'main'
    if (ma !== mb) return ma ? -1 : 1

    // 7. is_target (asc — non-target preferred)
    const ta = bool(a.target_mode)
    const tb = bool(b.target_mode)
    if (ta !== tb) return ta ? 1 : -1

    // 8. is_subfolder (asc — no subfolder preferred)
    const sa = !!(a.subfolder && a.subfolder !== '')
    const sb = !!(b.subfolder && b.subfolder !== '')
    if (sa !== sb) return sa ? 1 : -1

    // 9. model_key (asc)
    const mka = str(a.model_key)
    const mkb = str(b.model_key)
    if (mka !== mkb) return mka < mkb ? -1 : 1

    // 10. Hash (asc)
    const ha = str(a.tokenizer_config_hash)
    const hb = str(b.tokenizer_config_hash)
    return ha < hb ? -1 : ha > hb ? 1 : 0
  })

  if (!hide) return sorted

  // Drop duplicates by tokenizer_config_hash, keeping first (= best-ranked)
  const seen = new Set<string>()
  return sorted.filter((row) => {
    const hash = str(row.tokenizer_config_hash)
    if (!hash) return true // no hash → keep always
    if (seen.has(hash)) return false
    seen.add(hash)
    return true
  })
}
