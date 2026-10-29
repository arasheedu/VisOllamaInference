import { EventEmitter } from 'node:events';
import WebSocket from 'ws';
import { AckMessage, DEFAULT_TRACE_PORT, HUB_PATHS, type TraceEvent } from 'trace-schema';
import type { Granularity, TraceItem } from './generate';

export const DEFAULT_HUB_URL = `ws://127.0.0.1:${DEFAULT_TRACE_PORT}${HUB_PATHS.ingest}`;

/**
 * Producer-side connection to the app's trace hub. Reconnects with backoff,
 * buffers while disconnected, and surfaces `ack` control messages.
 */
export class HubClient extends EventEmitter {
  private ws: WebSocket | null = null;
  private queue: string[] = [];
  private backoff = 250;
  private warned = false;
  private closed = false;
  private acks: AckMessage[] = [];
  connected = false;

  constructor(
    readonly url: string = DEFAULT_HUB_URL,
    private readonly log: (msg: string) => void = (m) => console.error(m),
  ) {
    super();
    this.connect();
  }

  private connect() {
    if (this.closed) return;
    const ws = new WebSocket(this.url);
    this.ws = ws;
    ws.on('open', () => {
      this.connected = true;
      this.backoff = 250;
      if (this.warned) this.log(`[trace-mock] connected to ${this.url}`);
      this.warned = false;
      for (const line of this.queue) ws.send(line);
      this.queue = [];
      this.emit('open');
    });
    ws.on('message', (data) => {
      for (const line of data.toString().split('\n')) {
        if (!line.trim()) continue;
        try {
          const r = AckMessage.safeParse(JSON.parse(line));
          if (r.success) {
            this.acks.push(r.data);
            this.emit('ack', r.data);
          }
        } catch {
          /* ignore non-JSON */
        }
      }
    });
    ws.on('close', () => {
      this.connected = false;
      this.ws = null;
      if (this.closed) return;
      if (!this.warned) {
        this.log(`[trace-mock] app not listening at ${this.url}; retrying quietly…`);
        this.warned = true;
      }
      setTimeout(() => this.connect(), this.backoff);
      this.backoff = Math.min(this.backoff * 2, 5000);
    });
    ws.on('error', () => {
      /* close follows */
    });
  }

  send(e: TraceEvent | string) {
    const line = typeof e === 'string' ? e : JSON.stringify(e);
    if (this.connected && this.ws) this.ws.send(line);
    else if (this.queue.length < 200_000) this.queue.push(line);
  }

  /** Resolve on the next ack for `req` (or any ack without a req match), or on timeout. */
  waitAck(req: string, timeoutMs: number): Promise<boolean> {
    if (this.acks.some((a) => a.req === req && a.continue)) return Promise.resolve(true);
    const i = this.acks.findIndex((a) => a.req === req);
    if (i >= 0) {
      this.acks.splice(i, 1);
      return Promise.resolve(true);
    }
    return new Promise((resolve) => {
      const onAck = (a: AckMessage) => {
        if (a.req !== req) return;
        const j = this.acks.indexOf(a);
        if (j >= 0 && !a.continue) this.acks.splice(j, 1);
        clearTimeout(timer);
        this.off('ack', onAck);
        resolve(true);
      };
      const timer = setTimeout(() => {
        this.off('ack', onAck);
        resolve(false);
      }, timeoutMs);
      this.on('ack', onAck);
    });
  }

  async flush(): Promise<void> {
    const ws = this.ws;
    if (!ws || !this.connected) return;
    while (ws.bufferedAmount > 0) await sleep(5);
  }

  close() {
    this.closed = true;
    this.ws?.close();
  }
}

export const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/**
 * Wait until performance.now() >= target with ~1 ms precision. Timers on
 * Windows have ~15 ms granularity, so sleep coarsely and yield-spin the rest.
 */
async function waitUntil(target: number) {
  for (;;) {
    const d = target - performance.now();
    if (d <= 0) return;
    if (d > 20) await sleep(d - 17);
    else await new Promise<void>((r) => setImmediate(r));
  }
}

export interface PlayOptions {
  speed?: number;
  /** honour gates of this granularity (step mode) */
  step?: Granularity | null;
  stepTimeoutMs?: number;
  /** shift t_ns by time spent paused at gates (like a real runner would) */
  shiftForPauses?: boolean;
  signal?: AbortSignal;
  onEvent: (e: TraceEvent) => void;
}

/**
 * Emit items in real time, honouring t_ns deltas scaled by `speed`. In step
 * mode, blocks at matching gates until an ack arrives (or times out, emitting
 * `step.timeout`).
 */
export async function playItems(
  items: TraceItem[],
  hub: HubClient | null,
  o: PlayOptions,
): Promise<void> {
  const speed = o.speed ?? 1;
  const first = items.find((i) => i.kind === 'event');
  if (!first || first.kind !== 'event') return;
  const t0 = first.event.t_ns;
  const req = first.event.req;
  const start = performance.now();
  let pausedMs = 0;
  // Renumber seq per request so injected step.timeout events stay strictly increasing.
  const seqs = new Map<string, number>();
  const nextSeq = (r: string) => {
    const n = seqs.get(r) ?? 0;
    seqs.set(r, n + 1);
    return n;
  };
  for (const it of items) {
    if (o.signal?.aborted) return;
    if (it.kind === 'gate') {
      if (!o.step || it.granularity !== o.step || !hub) continue;
      const before = performance.now();
      const ok = await hub.waitAck(req, o.stepTimeoutMs ?? 60_000);
      pausedMs += performance.now() - before;
      if (!ok) {
        const t_ns = Math.round(t0 + (performance.now() - start) * 1e6);
        o.onEvent({
          v: 1,
          req,
          seq: nextSeq(req),
          t_ns,
          type: 'step.timeout',
          step: it.step,
          granularity: it.granularity,
        });
      }
      continue;
    }
    const e = it.event;
    const target = start + (e.t_ns - t0) / 1e6 / speed + pausedMs;
    await waitUntil(target);
    const t_ns = o.shiftForPauses ? Math.round(e.t_ns + pausedMs * 1e6) : e.t_ns;
    o.onEvent({ ...e, seq: nextSeq(e.req), t_ns });
  }
}
