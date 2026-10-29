import { readFileSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  EVENT_TYPES,
  PRIORITY,
  EVENT_DETAIL,
  parseEvent,
  parseNdjson,
  includedAtDetail,
} from '../src';

const dir = resolve(__dirname, '../../../traces/fixtures');
const fixtures = readdirSync(dir).filter((f) => f.endsWith('.ndjson'));

describe('fixtures', () => {
  it('exist', () => expect(fixtures.length).toBeGreaterThan(0));
  for (const f of fixtures) {
    it(`${f}: every line parses`, () => {
      const { events, errors } = parseNdjson(readFileSync(resolve(dir, f), 'utf8'));
      expect(errors).toEqual([]);
      expect(events.length).toBeGreaterThan(10);
    });
    it(`${f}: seq strictly increasing per req, t_ns non-decreasing`, () => {
      const { events } = parseNdjson(readFileSync(resolve(dir, f), 'utf8'));
      const last = new Map<string, { seq: number; t: number }>();
      for (const e of events) {
        const p = last.get(e.req);
        if (p) {
          expect(e.seq).toBeGreaterThan(p.seq);
          expect(e.t_ns).toBeGreaterThanOrEqual(p.t);
        }
        last.set(e.req, { seq: e.seq, t: e.t_ns });
      }
    });
  }
});

describe('parseEvent errors', () => {
  const ok = { v: 1, req: 'r', seq: 0, t_ns: 0, type: 'request.error', message: 'x' };
  it('accepts a valid event', () => expect(parseEvent(JSON.stringify(ok)).ok).toBe(true));
  it('json', () => {
    const r = parseEvent('{nope');
    expect(r.ok === false && r.error.kind).toBe('json');
  });
  it('version', () => {
    const r = parseEvent(JSON.stringify({ ...ok, v: 2 }));
    expect(r.ok === false && r.error.kind).toBe('version');
  });
  it('unknown_type', () => {
    const r = parseEvent(JSON.stringify({ ...ok, type: 'decode.magic' }));
    expect(r.ok === false && r.error.kind).toBe('unknown_type');
  });
  it('schema', () => {
    const r = parseEvent(JSON.stringify({ ...ok, message: 5 }));
    expect(r.ok === false && r.error.kind).toBe('schema');
    expect(r.ok === false && r.error.message).toContain('message');
  });
  it('non-object', () => {
    const r = parseEvent('[1,2]');
    expect(r.ok === false && r.error.kind).toBe('schema');
  });
});

describe('metadata maps', () => {
  it('cover every event type', () => {
    for (const t of EVENT_TYPES) {
      expect(PRIORITY[t]).toBeDefined();
      expect(EVENT_DETAIL[t]).toBeDefined();
    }
  });
  it('decode.layer is droppable, tokens critical', () => {
    expect(PRIORITY['decode.layer']).toBe('droppable');
    expect(PRIORITY['decode.token']).toBe('critical');
  });
  it('detail filter', () => {
    expect(includedAtDetail('decode.layer', 'basic')).toBe(false);
    expect(includedAtDetail('decode.layer', 'layers')).toBe(true);
    expect(includedAtDetail('decode.op', 'layers')).toBe(false);
    expect(includedAtDetail('decode.op', 'full')).toBe(true);
  });
});
