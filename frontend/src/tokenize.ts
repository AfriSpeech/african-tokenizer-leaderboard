// Tokenization for the Compare page, matching the Python `tokenizers` library the leaderboard
// was scored with:
// - Unigram models with byte_fallback (SentencePiece-style, e.g. Pocket TTS) spell characters
//   missing from the vocabulary as UTF-8 byte tokens (<0xE1> ...). tokenizers.js instead
//   returns the whole span around them as a single unknown token, so those spans are
//   re-segmented here the way the Python library does: best-scoring path over vocabulary
//   pieces, each uncovered character an "unknown" step, then unknowns spelled as bytes.
// - "builtin:bytes" (ByT5) is one token per UTF-8 byte.

import { Tokenizer } from '@huggingface/tokenizers'

interface Unigram {
  scores: Map<string, number>
  maxLen: number
  unkId: number
  unkScore: number
}

export interface LoadedTokenizer {
  tok: Tokenizer | null
  unigram: Unigram | null // set for Unigram models with byte_fallback
}

const encoder = new TextEncoder()
const hex = (b: number) => `<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`
const BYTE_PIECE = /^<0x[0-9A-F]{2}>$/

export function fromJson(json: any, config: object): LoadedTokenizer {
  const tok = new Tokenizer(json, config)
  const model = json?.model ?? {}
  if (model.type !== 'Unigram' || !model.byte_fallback || model.unk_id == null) {
    return { tok, unigram: null }
  }
  const vocab: [string, number][] = model.vocab ?? []
  const scores = new Map<string, number>()
  let maxLen = 1
  let minScore = 0
  vocab.forEach(([piece, score], i) => {
    if (i === model.unk_id || BYTE_PIECE.test(piece)) return
    scores.set(piece, score)
    maxLen = Math.max(maxLen, [...piece].length)
    minScore = Math.min(minScore, score)
  })
  // The Python library scores an unknown character as (lowest piece score - 10).
  return { tok, unigram: { scores, maxLen, unkId: model.unk_id, unkScore: minScore - 10 } }
}

export const BYTES: LoadedTokenizer = { tok: null, unigram: null }

/** Best path over vocabulary pieces; characters no piece covers become byte tokens. */
function segment(u: Unigram, span: string): string[] {
  const chars = [...span]
  const n = chars.length
  const best = new Array<number>(n + 1).fill(-Infinity)
  const back = new Array<[number, string | null]>(n + 1)
  best[0] = 0
  for (let i = 0; i < n; i++) {
    if (best[i] === -Infinity) continue
    let piece = ''
    let matched = false
    for (let j = i; j < Math.min(n, i + u.maxLen); j++) {
      piece += chars[j]
      const s = u.scores.get(piece)
      if (s !== undefined) {
        if (j === i) matched = true
        if (best[i] + s > best[j + 1]) { best[j + 1] = best[i] + s; back[j + 1] = [i, piece] }
      }
    }
    if (!matched && best[i] + u.unkScore > best[i + 1]) {
      best[i + 1] = best[i] + u.unkScore
      back[i + 1] = [i, null] // unknown character
    }
  }
  const out: string[] = []
  for (let k = n; k > 0; ) {
    const [i, piece] = back[k]
    out.push(...(piece ?? [...encoder.encode(chars[i])].map(hex).reverse()))
    k = i
  }
  return out.reverse()
}

/** Display pieces, one per token. */
export function pieces(t: LoadedTokenizer, text: string): string[] {
  if (!t.tok) return Array.from(encoder.encode(text), hex)
  const { ids, tokens } = t.tok.encode(text, { add_special_tokens: false })
  if (!t.unigram) {
    return ids.map((id, i) => {
      const s = t.tok!.decode([id], { skip_special_tokens: false })
      return s.includes('�') ? (tokens[i] ?? s) : s.replace(/^ /, '·')
    })
  }
  const out: string[] = []
  ids.forEach((id, i) => {
    const pcs = id === t.unigram!.unkId ? segment(t.unigram!, tokens[i]) : [tokens[i]]
    for (const p of pcs) out.push(p.replace(/^▁/, '·'))
  })
  return out
}
