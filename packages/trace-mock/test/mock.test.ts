import { describe, expect, it } from 'vitest';
import { parseEvent, serializeEvent, validateEventObject, type TraceEvent } from 'trace-schema';
import { BpeTokenizer } from 'tokenizer';
import { generateTrace, playItems, Utf8Assembler, mockTokenizer } from '../src';

describe('generateTrace', () => {
  for (const detail of ['basic', 'layers', 'full'] as const) {
    it(`${detail}: every event validates and text is consistent`, () => {
      const tr = generateTrace({
        detail,
        layers: 8,
        responseTokens: 40,
        promptTokens: 30,
        seed: 3,
      });
      for (const e of tr.events) {
        const r = parseEvent(serializeEvent(e));
        if (!r.ok) throw new Error(r.error.message);
      }
      const tokens = tr.events.filter((e) => e.type === 'decode.token') as Extract<
        TraceEvent,
        { type: 'decode.token' }
      >[];
      expect(tokens.length).toBe(40);
      expect(tokens.at(-1)!.is_eos).toBe(true);
      expect(tokens.map((t) => t.text).join('')).toBe(tr.responseText);
      expect(tr.chunks.map((c) => c.text).join('')).toBe(tr.responseText);
      const types = new Set(tr.events.map((e) => e.type));
      expect(types.has('decode.layer')).toBe(detail !== 'basic');
      expect(types.has('decode.sample.stage')).toBe(detail === 'full');
    });
  }

  it('sampling stages draw the token that was emitted', () => {
    const tr = generateTrace({ detail: 'full', layers: 4, responseTokens: 30, seed: 5 });
    const samples = tr.events.filter((e) => e.type === 'decode.sample') as any[];
    for (const s of samples) {
      const draw = tr.events.find(
        (e: any) => e.type === 'decode.sample.stage' && e.step === s.step && e.stage === 'draw',
      ) as any;
      expect(draw.candidates[draw.cutoff].id).toBe(s.chosen_id);
      let cum = 0;
      for (let i = 0; i < draw.cutoff; i++) cum += draw.candidates[i].v;
      expect(draw.draw).toBeGreaterThanOrEqual(cum - 1e-4);
    }
  });

  it('prompt tokens match the mock tokenizer (in-app BPE path)', () => {
    const tr = generateTrace({
      prompt: 'Why is the sky blue?',
      responseTokens: 3,
      detail: 'basic',
    });
    const { spec } = mockTokenizer();
    const tok = new BpeTokenizer(spec);
    const text = tr.promptTokens.map((t) => t.text).join('');
    expect(tok.encode(text)).toEqual(tr.promptTokens.map((t) => t.id));
  });
});

describe('Utf8Assembler', () => {
  it('holds partial sequences', () => {
    const a = new Utf8Assembler();
    const b = new TextEncoder().encode('✓');
    expect(a.push(b.slice(0, 1))).toBe('');
    expect(a.pending).toBe(1);
    expect(a.push(b.slice(1))).toBe('✓');
    expect(a.pending).toBe(0);
  });
});

describe('replay timing', () => {
  it('error < 5 ms at speed 1.0', async () => {
    const tr = generateTrace({
      detail: 'basic',
      layers: 4,
      responseTokens: 30,
      tokPerSec: 50,
      loadMs: 50,
      seed: 1,
    });
    const t0 = tr.events[0].t_ns;
    const errs: number[] = [];
    let start = 0;
    await playItems(tr.items, null, {
      speed: 1,
      onEvent: (e) => {
        const now = performance.now();
        if (!start) start = now;
        errs.push(Math.abs(now - start - (e.t_ns - t0) / 1e6));
      },
    });
    expect(Math.max(...errs)).toBeLessThan(5);
  }, 10_000);
});

describe('deep tier', () => {
  it('emits valid deep events and leaves lower tiers unchanged', () => {
    const deep = generateTrace({ layers: 6, detail: 'deep', responseTokens: 14, seed: 3 });
    const full = generateTrace({ layers: 6, detail: 'full', responseTokens: 14, seed: 3 });
    for (const e of deep.events) expect(validateEventObject(e).ok).toBe(true);
    const types = new Set(deep.events.map((e) => e.type));
    for (const t of ['decode.node', 'decode.lens', 'decode.attn.heads'])
      expect(types.has(t as never)).toBe(true);
    // deep events only on steps < 3 and every 12th step
    const deepSteps = new Set(
      deep.events.filter((e) => e.type === 'decode.lens').map((e) => (e as { step: number }).step),
    );
    expect([...deepSteps].sort((a, b) => a - b)).toEqual([0, 1, 2, 12]);
    const strip = (es: typeof deep.events) =>
      es
        .filter(
          (e) =>
            !['decode.node', 'decode.lens', 'decode.attn.heads', 'request.received'].includes(
              e.type,
            ),
        )
        .map(({ seq: _s, t_ns: _t, req: _r, ...x }) => JSON.stringify(x));
    expect(strip(deep.events)).toEqual(strip(full.events));
    expect(deep.responseText).toBe(full.responseText);
  });
});
