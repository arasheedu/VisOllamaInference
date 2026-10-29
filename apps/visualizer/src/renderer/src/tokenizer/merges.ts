import { BpeTokenizer, validateIds, type EncodeTrace, type TokenizerSpec } from 'tokenizer';

export interface MergePlan {
  ok: boolean;
  reason?: string;
  firstMismatch: number;
  trace: EncodeTrace | null;
  /** number of merge rounds (max over words); words animate in parallel */
  rounds: number;
}

const cache = new Map<string, Promise<BpeTokenizer | null>>();
const ready = new Map<string, BpeTokenizer | null>();

/** Load (once) and cache a tokenizer for `model`. */
export function loadTokenizer(model: string, load: () => Promise<TokenizerSpec | null>) {
  let p = cache.get(model);
  if (!p) {
    p = load()
      .then((spec) => (spec ? new BpeTokenizer(spec) : null))
      .catch(() => null)
      .then((t) => {
        ready.set(model, t);
        return t;
      });
    cache.set(model, p);
  }
  return p;
}

export const tokenizerFor = (model: string): BpeTokenizer | null | undefined => ready.get(model);

/**
 * Re-run BPE on the detokenized prompt and check it reproduces the runner's
 * ids. Only then is the merge animation trustworthy.
 */
export function planMerges(
  tok: BpeTokenizer | null | undefined,
  tokens: Array<{ id: number; text: string }>,
): MergePlan {
  if (tok === undefined)
    return {
      ok: false,
      reason: 'tokenizer still loading',
      firstMismatch: -1,
      trace: null,
      rounds: 0,
    };
  if (tok === null)
    return {
      ok: false,
      reason: 'tokenizer unavailable for this model',
      firstMismatch: -1,
      trace: null,
      rounds: 0,
    };
  const text = tokens.map((t) => t.text).join('');
  const trace = tok.encodeWithTrace(text);
  const v = validateIds(
    trace.ids,
    tokens.map((t) => t.id),
  );
  if (!v.ok)
    return {
      ok: false,
      reason: `JS tokenizer differs at token ${v.firstMismatch}`,
      firstMismatch: v.firstMismatch,
      trace,
      rounds: 0,
    };
  let rounds = 0;
  for (const m of trace.merges) rounds = Math.max(rounds, m.round + 1);
  return { ok: true, firstMismatch: -1, trace, rounds };
}

/** Symbols of a word after applying merges of rounds < r. */
export function wordAtRound(
  trace: EncodeTrace,
  wordIdx: number,
  r: number,
  decode: (s: string) => string,
): string[] {
  const w = trace.words[wordIdx];
  if (w.special) return [w.text];
  const syms = w.initial.slice();
  for (const m of trace.merges) {
    if (m.word !== wordIdx || m.round >= r) continue;
    // positions shift as we merge; find the pair from the recorded pos
    const i = m.pos;
    if (syms[i] === m.left && syms[i + 1] === m.right) syms.splice(i, 2, m.left + m.right);
  }
  return syms.map(decode);
}
