import { BpeTokenizer, trainBpe, type TokenizerSpec } from 'tokenizer';

/** Deterministic PRNG (mulberry32). */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(r: () => number): number {
  const u = Math.max(r(), 1e-12);
  const v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Response sentences the mock "model" writes. */
export const RESPONSE_CORPUS = [
  'A language model reads your prompt as a sequence of tokens.',
  'First the text is split into pieces, and each piece becomes an integer id.',
  'During prefill the model processes every prompt token in parallel and fills the key-value cache.',
  'Then decoding begins: one new token per step, each step attending to everything before it.',
  'Every layer refines the hidden state with attention and a feed-forward block.',
  'At the end of the stack, the logits are turned into probabilities.',
  'The sampler applies temperature, top-k and top-p before drawing the next token.',
  'Low-entropy steps are confident; high-entropy steps could have gone many ways.',
  'The chosen token is appended to the context and the loop repeats.',
  'Streaming sends each piece to you as soon as its bytes form valid UTF-8 → like this ✓.',
  'Generation stops when the model emits an end-of-sequence token or reaches the limit.',
  'Caching keys and values means each new step only computes one position.',
  'Larger models have more layers, so each step takes a little longer.',
  'Quantized weights trade a little accuracy for a lot of speed and memory.',
];

export const DEFAULT_PROMPT =
  'Explain, step by step, how a large language model turns my prompt into an answer.';

const EXTRA_CORPUS =
  ' Explain step by step how large language model turns my prompt into an answer. What is attention? Why is the sky blue? Write a haiku about tokens. Summarize this paragraph. hello world the and of to in is it that for you with on as are this ';

export const SPECIALS = {
  bos: '<|begin|>',
  user: '<|user|>',
  assistant: '<|assistant|>',
  end: '<|end|>',
  eos: '<|eot|>',
} as const;

let cached: { spec: Required<TokenizerSpec>; tok: BpeTokenizer } | null = null;

/** The mock model's tokenizer (trained once, deterministic). */
export function mockTokenizer(): { spec: Required<TokenizerSpec>; tok: BpeTokenizer } {
  if (!cached) {
    const corpus = (RESPONSE_CORPUS.join(' ') + EXTRA_CORPUS).repeat(3);
    const spec = trainBpe(corpus, 700, Object.values(SPECIALS), 'mock');
    cached = { spec, tok: new BpeTokenizer(spec) };
  }
  return cached;
}

/** Chat template applied by the mock before tokenizing. */
export function renderPrompt(userPrompt: string): string {
  return `${SPECIALS.bos}${SPECIALS.user}\n${userPrompt}${SPECIALS.end}\n${SPECIALS.assistant}\n`;
}

/** Pick response text whose tokenization has at least `n` tokens. */
export function responseText(nTokens: number, r: () => number): string {
  const { tok } = mockTokenizer();
  const start = Math.floor(r() * RESPONSE_CORPUS.length);
  let text = '';
  let i = start;
  while (tok.encode(text).length < nTokens) {
    text += (text ? ' ' : '') + RESPONSE_CORPUS[i % RESPONSE_CORPUS.length];
    i++;
  }
  return text;
}

/** Filler prompt text targeting about `n` tokens. */
export function promptText(nTokens: number): string {
  const { tok } = mockTokenizer();
  let text = DEFAULT_PROMPT;
  let i = 0;
  while (tok.encode(renderPrompt(text)).length < nTokens) {
    text += ' ' + RESPONSE_CORPUS[i++ % RESPONSE_CORPUS.length];
  }
  return text;
}

export const MOCK_MODELS: Record<
  string,
  { layers: number; params: string; tokPerSec: number; size: number }
> = {
  'mock-llama:1b': { layers: 16, params: '1.2B', tokPerSec: 120, size: 800e6 },
  'mock-llama:8b': { layers: 32, params: '8.0B', tokPerSec: 60, size: 4.9e9 },
  'mock-llama:70b': { layers: 80, params: '70.6B', tokPerSec: 12, size: 42e9 },
};
