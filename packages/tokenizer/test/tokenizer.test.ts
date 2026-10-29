import { describe, expect, it } from 'vitest';
import { BpeTokenizer, trainBpe, validateIds, pretokenize, PRETOKENIZERS } from '../src';

const corpus =
  'The quick brown fox jumps over the lazy dog. The dog sleeps. Tokenizers merge bytes into tokens; tokens become ids. Café → naïve ✓ 12345 '.repeat(
    20,
  );

describe('bpe', () => {
  const spec = trainBpe(corpus, 200, ['<|user|>', '<|end|>']);
  const tok = new BpeTokenizer(spec);

  it('round-trips text through encode/decode', () => {
    const text = 'The lazy fox merges tokens. Café ✓ 42!';
    expect(tok.decode(tok.encode(text))).toBe(text);
  });

  it('handles special tokens atomically', () => {
    const ids = tok.encode('<|user|>hi<|end|>');
    expect(ids[0]).toBe(tok.idOf('<|user|>'));
    expect(ids[ids.length - 1]).toBe(tok.idOf('<|end|>'));
  });

  it('learned merges produce multi-byte tokens', () => {
    const ids = tok.encode(' the');
    expect(ids.length).toBe(1);
  });

  it('records merge steps that rebuild the final pieces', () => {
    const tr = tok.encodeWithTrace(' tokens dog');
    expect(tr.merges.length).toBeGreaterThan(0);
    expect(tr.merges.every((m) => m.resultId >= 0)).toBe(true);
    expect(tr.pieces.join('')).toBe(' tokens dog');
  });

  it('validateIds reports first mismatch', () => {
    expect(validateIds([1, 2, 3], [1, 2, 3]).ok).toBe(true);
    expect(validateIds([1, 2, 3], [1, 9, 3]).firstMismatch).toBe(1);
  });

  it('llama3 pretokenizer splits like reference', () => {
    expect(pretokenize("Hello world's 12345!!\n\n", PRETOKENIZERS.llama3)).toEqual([
      'Hello',
      ' world',
      "'s",
      ' ',
      '123',
      '45',
      '!!\n\n',
    ]);
  });
});
