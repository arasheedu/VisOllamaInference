import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { parseNdjson, type TraceEvent } from 'trace-schema';
import { generateTrace } from 'trace-mock';
import { Player } from './player';
import { DEFAULT_CLOCK_CONFIG, DEFAULT_MINIMUMS } from './clock';
import type { Phase, ViewModel } from './types';

const FIX = resolve(__dirname, '../../../../../../traces/fixtures');
const fixture = (name: string) => parseNdjson(readFileSync(resolve(FIX, name), 'utf8')).events;

const FRAME = 1000 / 60;

interface SimResult {
  phases: Phase[];
  vm: ViewModel;
  /** wall time each event was shown */
  shown: Array<{ e: TraceEvent; wall: number }>;
  sawCatchup: boolean;
  maxSpeed: number;
  sawDegraded: boolean;
  frames: number;
}

/** Simulate a live run: events arrive at their trace times; rAF at 60 fps. */
function simulate(
  events: TraceEvent[],
  o: { detail?: 'basic' | 'layers' | 'full'; maxWallMs?: number; noCatchup?: boolean } = {},
): SimResult {
  const p = new Player(events[0].req, {
    detail: o.detail ?? 'full',
    clock: o.noCatchup
      ? { ...DEFAULT_CLOCK_CONFIG, lagThresholdMs: Infinity, fullDetailLagThresholdMs: Infinity }
      : undefined,
  });
  const shown: SimResult['shown'] = [];
  const consume = p.consume.bind(p);
  p.consume = (e, wall, skip) => {
    consume(e, wall, skip);
    if (!skip) shown.push({ e, wall });
  };
  const t0 = events[0].t_ns;
  let i = 0;
  let now = 0;
  let vm = p.snapshot();
  const phases: Phase[] = [vm.phase];
  let sawCatchup = false;
  let sawDegraded = false;
  let maxSpeed = 1;
  let frames = 0;
  const limit = o.maxWallMs ?? 600_000;
  while (now < limit) {
    while (i < events.length && (events[i].t_ns - t0) / 1e6 <= now) p.push(events[i++], now);
    vm = p.tick(now);
    frames++;
    if (phases[phases.length - 1] !== vm.phase) phases.push(vm.phase);
    sawCatchup ||= vm.clock.catchingUp;
    sawDegraded ||= vm.clock.degraded;
    maxSpeed = Math.max(maxSpeed, vm.clock.speed);
    if (i >= events.length && p.atEnd && (vm.phase === 'done' || vm.phase === 'error')) break;
    now += FRAME;
  }
  return { phases, vm, shown, sawCatchup, maxSpeed, sawDegraded, frames };
}

const BOUNDARY_TYPES = [
  'model.load',
  'tokenize.done',
  'prefill.batch',
  'prefill.done',
  'decode.step.start',
  'request.done',
];

function minFor(e: TraceEvent): number {
  if (e.type === 'model.load') return e.phase === 'start' ? DEFAULT_MINIMUMS['model.load'] : 0;
  if (e.type === 'tokenize.done') return DEFAULT_MINIMUMS.tokenize;
  if (e.type === 'prefill.batch') return DEFAULT_MINIMUMS['prefill.batch'];
  if (e.type === 'decode.step.start') return DEFAULT_MINIMUMS['decode.step'];
  return 0;
}

describe('engine: short-answer fixture', () => {
  const events = fixture('short-answer.ndjson');
  const expectedText = readFileSync(resolve(FIX, 'short-answer.response.txt'), 'utf8');

  it('walks the phase sequence and reproduces the response', () => {
    const r = simulate(events, { detail: 'layers' });
    expect(r.phases).toEqual(['idle', 'loading', 'tokenizing', 'prefill', 'decoding', 'done']);
    expect(r.vm.responseText).toBe(expectedText);
    expect(r.vm.promptTokens.length).toBeGreaterThan(0);
    expect(r.vm.stats?.eval_count).toBe(20);
    expect(r.vm.derived.ttftMs).toBeGreaterThan(800);
  });

  it('keeps every stage on screen for at least its minimum (catch-up off)', () => {
    const r = simulate(events, { detail: 'layers', noCatchup: true });
    const boundary = r.shown.filter(({ e }) => BOUNDARY_TYPES.includes(e.type));
    for (let k = 1; k < boundary.length; k++) {
      const prev = boundary[k - 1];
      expect(boundary[k].wall - prev.wall).toBeGreaterThanOrEqual(minFor(prev.e) - 1e-6);
    }
    const layers = r.shown.filter(({ e }) => e.type === 'decode.layer');
    for (let k = 1; k < layers.length; k++)
      expect(layers[k].wall - layers[k - 1].wall).toBeGreaterThanOrEqual(
        DEFAULT_MINIMUMS['decode.layer'] - 1e-6,
      );
  });

  it('dropping all decode.layer events changes neither phases nor tokens', () => {
    const a = simulate(events, { detail: 'layers' });
    const b = simulate(
      events.filter((e) => e.type !== 'decode.layer'),
      { detail: 'layers' },
    );
    expect(b.phases).toEqual(a.phases);
    expect(b.vm.responseText).toBe(a.vm.responseText);
    expect(b.vm.responseTokens.map((t) => t.step)).toEqual(a.vm.responseTokens.map((t) => t.step));
  });
});

describe('engine: catch-up', () => {
  it('engages on a burst of 2000 layer events and settles back to 1x', () => {
    const tr = generateTrace({
      detail: 'layers',
      layers: 80,
      responseTokens: 25,
      promptTokens: 20,
      tokPerSec: 60,
      loadMs: 0,
      seed: 9,
    });
    expect(tr.events.filter((e) => e.type === 'decode.layer').length).toBe(2000);
    const r = simulate(tr.events, { detail: 'layers' });
    expect(r.sawCatchup).toBe(true);
    expect(r.maxSpeed).toBeGreaterThan(2);
    expect(r.vm.responseText).toBe(tr.responseText);
    expect(r.vm.clock.catchingUp).toBe(false);
  });
});

describe('engine: auto-degrade (Full detail)', () => {
  it('skips full-detail backlog without changing phases or tokens, then recovers', () => {
    const tr = generateTrace({
      detail: 'full',
      layers: 32,
      responseTokens: 30,
      promptTokens: 20,
      tokPerSec: 60,
      loadMs: 0,
      seed: 4,
    });
    const r = simulate(tr.events, { detail: 'full' });
    expect(r.sawDegraded).toBe(true);
    expect(r.phases).toEqual(['idle', 'loading', 'tokenizing', 'prefill', 'decoding', 'done']);
    expect(r.vm.responseText).toBe(tr.responseText);
    expect(r.vm.clock.degraded).toBe(false);
  });

  it('basic detail ignores higher-tier events at ingestion', () => {
    const tr = generateTrace({ detail: 'full', layers: 8, responseTokens: 5, seed: 2, loadMs: 0 });
    const p = new Player(tr.req, { detail: 'basic' });
    for (const e of tr.events) p.push(e, 0);
    expect(
      p.events.every((e) => !['decode.op', 'decode.layer', 'decode.attn'].includes(e.type)),
    ).toBe(true);
    expect(p.filtered).toBeGreaterThan(0);
  });
});

describe('engine: buffer', () => {
  it('drops oldest droppable events first, never critical', () => {
    const tr = generateTrace({
      detail: 'layers',
      layers: 32,
      responseTokens: 60,
      seed: 3,
      loadMs: 0,
    });
    const p = new Player(tr.req, { detail: 'layers', capacity: 500 });
    for (const e of tr.events) p.push(e, 0);
    const crit = tr.events.filter((e) => e.type !== 'decode.layer').length;
    expect(p.events.filter((e) => e.type !== 'decode.layer').length).toBe(crit);
    expect(p.dropped).toBeGreaterThan(0);
  });

  it('sorts out-of-order arrivals by seq', () => {
    const tr = generateTrace({ detail: 'basic', layers: 4, responseTokens: 10, seed: 1 });
    const ev = tr.events.slice();
    [ev[5], ev[6]] = [ev[6], ev[5]];
    const p = new Player(tr.req, { detail: 'basic' });
    for (const e of ev) p.push(e, 0);
    expect(p.events.map((e) => e.seq)).toEqual(tr.events.map((e) => e.seq));
  });
});

describe('engine: seek and step', () => {
  const events = fixture('full-detail.ndjson');
  const loaded = () => {
    const p = new Player(events[0].req, { detail: 'full', live: false });
    for (const e of events) p.push(e, 0);
    return p;
  };
  const runUntilPaused = (p: Player, start: number) => {
    let now = start;
    let vm = p.tick(now);
    for (let k = 0; k < 10_000 && !vm.clock.paused; k++) vm = p.tick((now += FRAME));
    return { vm, now };
  };

  it('seek rebuilds state at a time instantly', () => {
    const p = loaded();
    const mid = events.find((e) => e.type === 'decode.token' && e.step === 2)!;
    p.seek(mid.t_ns);
    const vm = p.snapshot();
    expect(vm.responseTokens.length).toBe(3);
    expect(vm.clock.paused).toBe(true);
    p.seek(events[events.length - 1].t_ns);
    expect(p.snapshot().phase).toBe('done');
  });

  it('stepUntil(token) shows exactly one more token then pauses', () => {
    const p = loaded();
    const firstStep = events.find((e) => e.type === 'decode.step.start')!;
    p.seek(firstStep.t_ns - 1);
    p.stepUntil('token');
    const { vm } = runUntilPaused(p, 0);
    expect(vm.responseTokens.length).toBe(1);
    expect(vm.steps[0].samplingStages.length).toBeGreaterThanOrEqual(5);
    expect(vm.steps[0].ops.length).toBe(8 * 8);
  });

  it('stepUntil(sample) advances one sampling stage at a time', () => {
    const p = loaded();
    const firstStep = events.find((e) => e.type === 'decode.step.start')!;
    p.seek(firstStep.t_ns - 1);
    p.stepUntil('sample');
    let r = runUntilPaused(p, 0);
    expect(r.vm.steps[0].samplingStages.length).toBe(1);
    p.stepUntil('sample');
    r = runUntilPaused(p, r.now + FRAME);
    expect(r.vm.steps[0].samplingStages.length).toBe(2);
  });
});
