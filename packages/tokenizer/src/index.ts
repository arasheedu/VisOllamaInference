/**
 * Byte-level BPE (GPT-2 / Llama-3 / Qwen style), compatible with the
 * `tokenizer.ggml.tokens` / `tokenizer.ggml.merges` arrays that Ollama returns
 * from `/api/show` (verbose). Also exposes the merge sequence so the UI can
 * animate it, and a tiny trainer used by the mock to build a synthetic vocab.
 */

// ---- byte <-> unicode mapping (GPT-2) -------------------------------------

function buildByteMaps(): { b2u: string[]; u2b: Map<string, number> } {
  const bs: number[] = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n++);
    }
  }
  const b2u: string[] = new Array(256);
  const u2b = new Map<string, number>();
  bs.forEach((b, i) => {
    const ch = String.fromCodePoint(cs[i]);
    b2u[b] = ch;
    u2b.set(ch, b);
  });
  return { b2u, u2b };
}

const { b2u: BYTE_TO_UNICODE, u2b: UNICODE_TO_BYTE } = buildByteMaps();
const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8', { fatal: false });

export const bytesToSymbols = (bytes: Uint8Array): string[] =>
  Array.from(bytes, (b) => BYTE_TO_UNICODE[b]);

export function symbolsToBytes(s: string): Uint8Array {
  const out: number[] = [];
  for (const ch of s) {
    const b = UNICODE_TO_BYTE.get(ch);
    if (b === undefined) out.push(...encoder.encode(ch));
    else out.push(b);
  }
  return Uint8Array.from(out);
}

// ---- pre-tokenizers -------------------------------------------------------

const CONTRACTIONS = "'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]";
export const PRETOKENIZERS: Record<string, RegExp> = {
  gpt2: new RegExp(
    `${CONTRACTIONS}| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+`,
    'gu',
  ),
  llama3: new RegExp(
    `${CONTRACTIONS}|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+`,
    'gu',
  ),
  qwen2: new RegExp(
    `${CONTRACTIONS}|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+`,
    'gu',
  ),
};

/** Map a GGUF `tokenizer.ggml.pre` value to a pre-tokenizer. */
export function pretokenizerFor(pre: string | undefined): RegExp {
  switch (pre) {
    case 'llama3':
    case 'llama-bpe':
    case 'smaug-bpe':
    case 'falcon3':
    case 'mock':
      return PRETOKENIZERS.llama3;
    case 'qwen2':
    case 'deepseek-r1-qwen':
      return PRETOKENIZERS.qwen2;
    default:
      return PRETOKENIZERS.gpt2;
  }
}

export function pretokenize(text: string, re: RegExp): string[] {
  return Array.from(text.matchAll(new RegExp(re.source, re.flags)), (m) => m[0]);
}

// ---- tokenizer ------------------------------------------------------------

export interface MergeStep {
  /** index of the pre-token word within the encoded text */
  word: number;
  /** merge round within that word (0-based) */
  round: number;
  /** symbol position (within the word, after previous merges) of the left piece */
  pos: number;
  left: string;
  right: string;
  /** decoded text of merged piece */
  text: string;
  rank: number;
  resultId: number;
}

export interface EncodedWord {
  /** decoded text of the word, or the special token string */
  text: string;
  /** char offset of the word in the input */
  start: number;
  special: boolean;
  /** initial symbols (byte-unicode chars) before merges */
  initial: string[];
  ids: number[];
}

export interface EncodeTrace {
  ids: number[];
  pieces: string[];
  words: EncodedWord[];
  merges: MergeStep[];
  /** ids that had to fall back to raw bytes (symbol not in vocab) */
  byteFallback: number;
}

export interface TokenizerSpec {
  tokens: string[];
  merges: string[];
  /** GGUF token types: 1 normal, 2 unknown, 3 control, 4 user-defined, 6 byte */
  tokenTypes?: number[];
  pre?: string;
}

export class BpeTokenizer {
  readonly tokens: string[];
  readonly tokenTypes: number[];
  private readonly vocab = new Map<string, number>();
  private readonly ranks = new Map<string, number>();
  private readonly re: RegExp;
  private readonly specials: string[];
  private readonly specialRe: RegExp | null;

  constructor(spec: TokenizerSpec) {
    this.tokens = spec.tokens;
    this.tokenTypes = spec.tokenTypes ?? [];
    spec.tokens.forEach((t, i) => {
      if (!this.vocab.has(t)) this.vocab.set(t, i);
    });
    spec.merges.forEach((m, i) => this.ranks.set(m, i));
    this.re = pretokenizerFor(spec.pre);
    this.specials = spec.tokens
      .filter((_, i) => this.tokenTypes[i] === 3 || this.tokenTypes[i] === 4)
      .filter((t) => t.length > 1)
      .sort((a, b) => b.length - a.length);
    this.specialRe = this.specials.length
      ? new RegExp(this.specials.map(escapeRe).join('|'), 'g')
      : null;
  }

  get size(): number {
    return this.tokens.length;
  }

  idOf(token: string): number | undefined {
    return this.vocab.get(token);
  }

  isSpecial(id: number): boolean {
    const t = this.tokenTypes[id];
    return t === 3 || t === 4;
  }

  /** Raw bytes for a token id (special tokens yield their literal text). */
  tokenBytes(id: number): Uint8Array {
    const t = this.tokens[id] ?? '';
    if (this.isSpecial(id)) return encoder.encode(t);
    return symbolsToBytes(t);
  }

  /** Decoded text for a token id (may contain U+FFFD for partial UTF-8). */
  piece(id: number): string {
    return decoder.decode(this.tokenBytes(id));
  }

  decode(ids: number[]): string {
    const parts: number[] = [];
    for (const id of ids) parts.push(...this.tokenBytes(id));
    return decoder.decode(Uint8Array.from(parts));
  }

  encode(text: string): number[] {
    return this.encodeWithTrace(text).ids;
  }

  encodeWithTrace(text: string, parseSpecial = true): EncodeTrace {
    const out: EncodeTrace = { ids: [], pieces: [], words: [], merges: [], byteFallback: 0 };
    const segments = parseSpecial ? this.splitSpecial(text) : [{ text, special: false, start: 0 }];
    for (const seg of segments) {
      if (seg.special) {
        const id = this.vocab.get(seg.text)!;
        out.words.push({
          text: seg.text,
          start: seg.start,
          special: true,
          initial: [seg.text],
          ids: [id],
        });
        out.ids.push(id);
        out.pieces.push(seg.text);
        continue;
      }
      let offset = seg.start;
      for (const w of pretokenize(seg.text, this.re)) {
        const wordIdx = out.words.length;
        const initial = bytesToSymbols(encoder.encode(w));
        const ids = this.bpeWord(initial, wordIdx, out);
        out.words.push({ text: w, start: offset, special: false, initial, ids });
        for (const id of ids) {
          out.ids.push(id);
          out.pieces.push(this.piece(id));
        }
        offset += w.length;
      }
    }
    return out;
  }

  private bpeWord(initial: string[], wordIdx: number, out: EncodeTrace): number[] {
    const syms = initial.slice();
    let round = 0;
    while (syms.length > 1) {
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i < syms.length - 1; i++) {
        const r = this.ranks.get(syms[i] + ' ' + syms[i + 1]);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          best = i;
        }
      }
      if (best < 0) break;
      const left = syms[best];
      const right = syms[best + 1];
      // merge every occurrence of this pair, left to right
      for (let i = 0; i < syms.length - 1; i++) {
        if (syms[i] === left && syms[i + 1] === right) {
          const merged = left + right;
          syms.splice(i, 2, merged);
          out.merges.push({
            word: wordIdx,
            round,
            pos: i,
            left,
            right,
            text: decoder.decode(symbolsToBytes(merged)),
            rank: bestRank,
            resultId: this.vocab.get(merged) ?? -1,
          });
        }
      }
      round++;
    }
    const ids: number[] = [];
    for (const s of syms) {
      const id = this.vocab.get(s);
      if (id !== undefined) {
        ids.push(id);
        continue;
      }
      // byte fallback: emit each byte symbol individually
      for (const ch of s) {
        const bid = this.vocab.get(ch);
        if (bid !== undefined) {
          ids.push(bid);
          out.byteFallback++;
        }
      }
    }
    return ids;
  }

  private splitSpecial(text: string): Array<{ text: string; special: boolean; start: number }> {
    if (!this.specialRe) return [{ text, special: false, start: 0 }];
    const segs: Array<{ text: string; special: boolean; start: number }> = [];
    let last = 0;
    for (const m of text.matchAll(new RegExp(this.specialRe.source, 'g'))) {
      const i = m.index ?? 0;
      if (i > last) segs.push({ text: text.slice(last, i), special: false, start: last });
      segs.push({ text: m[0], special: true, start: i });
      last = i + m[0].length;
    }
    if (last < text.length) segs.push({ text: text.slice(last), special: false, start: last });
    return segs;
  }
}

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Compare our encoding against reference ids (e.g. from `tokenize.done`). */
export function validateIds(
  ours: number[],
  reference: number[],
): { ok: boolean; firstMismatch: number } {
  const n = Math.max(ours.length, reference.length);
  for (let i = 0; i < n; i++) if (ours[i] !== reference[i]) return { ok: false, firstMismatch: i };
  return { ok: true, firstMismatch: -1 };
}

// ---- trainer (for the mock vocab) -----------------------------------------

/**
 * Train a small byte-level BPE on `corpus`. Deterministic. Base vocab is the
 * 256 byte symbols; `specials` are appended as control tokens (type 3).
 */
export function trainBpe(
  corpus: string,
  numMerges: number,
  specials: string[] = [],
  pre = 'mock',
): Required<TokenizerSpec> {
  const re = pretokenizerFor(pre);
  const freq = new Map<string, number>();
  for (const w of pretokenize(corpus, re)) freq.set(w, (freq.get(w) ?? 0) + 1);
  const words = Array.from(freq.entries()).map(([w, f]) => ({
    syms: bytesToSymbols(encoder.encode(w)),
    f,
  }));
  const tokens: string[] = [];
  for (let b = 0; b < 256; b++) tokens.push(BYTE_TO_UNICODE[b]);
  const merges: string[] = [];
  const have = new Set(tokens);
  for (let m = 0; m < numMerges; m++) {
    const counts = new Map<string, number>();
    for (const { syms, f } of words) {
      for (let i = 0; i < syms.length - 1; i++) {
        const k = syms[i] + ' ' + syms[i + 1];
        counts.set(k, (counts.get(k) ?? 0) + f);
      }
    }
    let best = '';
    let bestN = 1; // require a pair to occur at least twice
    for (const [k, n] of counts) {
      if (n > bestN || (n === bestN && best !== '' && k < best)) {
        best = k;
        bestN = n;
      }
    }
    if (!best) break;
    const [l, r] = best.split(' ');
    const merged = l + r;
    merges.push(best);
    if (!have.has(merged)) {
      tokens.push(merged);
      have.add(merged);
    }
    for (const w of words) {
      const s = w.syms;
      for (let i = 0; i < s.length - 1; i++) {
        if (s[i] === l && s[i + 1] === r) s.splice(i, 2, merged);
      }
    }
  }
  const tokenTypes: number[] = tokens.map((t) => (t.length === 1 ? 6 : 1));
  for (const s of specials) {
    tokens.push(s);
    tokenTypes.push(3);
  }
  return { tokens, merges, tokenTypes, pre };
}
