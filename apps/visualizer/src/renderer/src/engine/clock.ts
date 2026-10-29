import type { TraceEvent } from 'trace-schema';

/** Minimum on-screen durations (wall ms). */
export interface StageMinimums {
  'model.load': number;
  tokenize: number;
  'prefill.batch': number;
  'decode.step': number;
  'decode.layer': number;
  // Full-detail sub-steps
  'decode.op': number;
  'decode.attn': number;
  'decode.sample.stage': number;
  'tokenize.merge': number;
  'stream.chunk': number;
}

export const DEFAULT_MINIMUMS: StageMinimums = {
  'model.load': 600,
  tokenize: 400,
  'prefill.batch': 150,
  'decode.step': 80,
  'decode.layer': 8,
  'decode.op': 25,
  'decode.attn': 60,
  'decode.sample.stage': 250,
  'tokenize.merge': 60,
  'stream.chunk': 40,
};

export interface ClockConfig {
  minimums: StageMinimums;
  /** catch-up engages when backlog exceeds this (ms of trace time) */
  lagThresholdMs: number;
  /** catch-up disengages below this */
  lagReleaseMs: number;
  maxSpeed: number;
  /** Full detail degrades to Layers when backlog exceeds this */
  fullDetailLagThresholdMs: number;
}

export const DEFAULT_CLOCK_CONFIG: ClockConfig = {
  minimums: DEFAULT_MINIMUMS,
  lagThresholdMs: 2000,
  lagReleaseMs: 250,
  maxSpeed: 8,
  fullDetailLagThresholdMs: 4000,
};

/** Events that end one stage and begin the next. */
const BOUNDARY = new Set<TraceEvent['type']>([
  'model.load',
  'tokenize.done',
  'prefill.batch',
  'prefill.done',
  'decode.step.start',
  'request.done',
  'request.error',
]);

export type HoldKind = 'stage' | 'sub';

/**
 * Hold applied after an event is shown. `stage` holds delay only the next
 * stage boundary; `sub` holds delay the next event of any kind (sequential
 * sub-steps like layers, ops, sampler stages).
 */
export function defaultHold(
  e: TraceEvent,
  m: StageMinimums,
): { kind: HoldKind; ms: number } | null {
  switch (e.type) {
    case 'model.load':
      return e.phase === 'start' ? { kind: 'stage', ms: m['model.load'] } : null;
    case 'tokenize.done':
      return { kind: 'stage', ms: m.tokenize };
    case 'prefill.batch':
      return { kind: 'stage', ms: m['prefill.batch'] };
    case 'decode.step.start':
      return { kind: 'stage', ms: m['decode.step'] };
    case 'decode.layer':
    case 'prefill.layer':
      return { kind: 'sub', ms: m['decode.layer'] };
    case 'decode.op':
      return { kind: 'sub', ms: m['decode.op'] };
    case 'decode.attn':
      return { kind: 'sub', ms: m['decode.attn'] };
    case 'decode.sample.stage':
      return { kind: 'sub', ms: m['decode.sample.stage'] };
    case 'tokenize.merge':
      return { kind: 'sub', ms: m['tokenize.merge'] };
    case 'stream.chunk':
      return { kind: 'sub', ms: m['stream.chunk'] };
    default:
      return null;
  }
}

export interface ClockSource {
  peek(): TraceEvent | undefined;
  /** Show the event. `skip` = consume without applying (auto-degrade). */
  consume(e: TraceEvent, wall: number, skip: boolean): void;
  /** Latest t_ns received so far (live head). */
  headT(): number;
  /** Wall time the head event arrived. */
  headWall(): number;
  /** Whether the producer has finished (request.done/error received). */
  finished(): boolean;
}

/**
 * Playback clock. Advances playback time `tp` (trace ns) by `dt * speed`,
 * holding it back so no stage finishes before its minimum on-screen duration,
 * and speeding up smoothly when the live backlog grows.
 */
export class PlaybackClock {
  tp = 0;
  t0 = 0;
  userSpeed = 1;
  catchup = 1;
  catchingUp = false;
  degraded = false;
  paused = false;
  live = true;
  dilation = 1;
  lagMs = 0;
  /** extra hold for specific events (e.g. tokenizer merge animation length) */
  extraHold: ((e: TraceEvent) => number) | null = null;
  /** when set, playback pauses right after an event matching this is shown */
  stopWhen: ((e: TraceEvent) => boolean) | null = null;
  stoppingLabel: string | null = null;
  /** true when the detail level is Full (enables auto-degrade) */
  fullDetail = false;
  isFullTier: (e: TraceEvent) => boolean = () => false;

  private lastWall: number | null = null;
  private stageHoldUntil = -Infinity;
  private subHoldUntil = -Infinity;
  private started = false;

  constructor(public config: ClockConfig = DEFAULT_CLOCK_CONFIG) {}

  get speed(): number {
    return this.userSpeed * this.catchup;
  }

  start(t0: number) {
    this.t0 = t0;
    this.tp = t0;
    this.started = true;
  }

  get isStarted() {
    return this.started;
  }

  pause() {
    this.paused = true;
  }
  resume() {
    this.paused = false;
    this.lastWall = null;
  }
  setSpeed(x: number) {
    this.userSpeed = Math.max(0.05, x);
  }
  /** Jump to trace time t; caller is responsible for rebuilding state. */
  seek(t: number) {
    this.tp = t;
    this.stageHoldUntil = this.subHoldUntil = -Infinity;
    this.catchup = 1;
    this.catchingUp = false;
    this.degraded = false;
  }

  /**
   * Advance to wall time `now` (ms), consuming due events from `src`.
   * Returns the number of events consumed.
   */
  tick(now: number, src: ClockSource): number {
    const prevWall = this.lastWall ?? now;
    const dt = Math.min(100, Math.max(0, now - prevWall));
    this.lastWall = now;
    if (!this.started) {
      const first = src.peek();
      if (!first) return 0;
      this.start(first.t_ns);
    }
    this.updateLag(src, dt, now);
    if (this.paused) {
      this.dilation += (0 - this.dilation) * 0.2;
      return 0;
    }

    const tpBefore = this.tp;
    const frameStart = now - dt;
    let target = this.tp + dt * 1e6 * this.speed;
    let consumed = 0;
    for (;;) {
      const e = src.peek();
      if (!e) break;
      const skip = this.degraded && this.isFullTier(e);
      if (skip) {
        // Backlog of full-detail events: skipped, not played faster.
        if (e.t_ns <= Math.max(target, src.headT())) {
          src.consume(e, now, true);
          continue;
        }
        break;
      }
      if (e.t_ns > target) break;
      const until = Math.max(
        this.subHoldUntil,
        BOUNDARY.has(e.type) ? this.stageHoldUntil : -Infinity,
      );
      if (now < until) {
        target = Math.max(this.tp, Math.min(target, e.t_ns));
        break;
      }
      // Moment the event became showable: inside this frame if a hold just expired.
      const wall = until > frameStart ? until : now;
      src.consume(e, wall, false);
      consumed++;
      if (e.t_ns > this.tp) this.tp = e.t_ns;
      const h = this.holdFor(e);
      if (h) {
        const ms = h.ms / this.catchup / Math.max(this.userSpeed, 1e-3);
        if (h.kind === 'stage') this.stageHoldUntil = wall + ms;
        else this.subHoldUntil = wall + ms;
      }
      if (this.stopWhen && this.stopWhen(e)) {
        this.stopWhen = null;
        this.stoppingLabel = null;
        this.paused = true;
        target = this.tp;
        break;
      }
    }
    // Never run (much) ahead of what the producer has sent.
    const head = src.finished()
      ? src.headT()
      : src.headT() + Math.min(250, Math.max(0, now - src.headWall())) * 1e6;
    this.tp = Math.max(this.tp, Math.min(target, head));
    if (dt > 0) {
      const inst = (this.tp - tpBefore) / (dt * 1e6);
      this.dilation += (inst - this.dilation) * Math.min(1, dt / 400);
    }
    // refresh lag flags after consuming (dt=0: no extra smoothing step)
    this.updateLag(src, 0, now);
    return consumed;
  }

  private holdFor(e: TraceEvent) {
    const h = defaultHold(e, this.config.minimums);
    const extra = this.extraHold?.(e) ?? 0;
    if (!h && !extra) return null;
    return { kind: h?.kind ?? 'stage', ms: Math.max(h?.ms ?? 0, extra) };
  }

  private updateLag(src: ClockSource, dt: number, now: number) {
    // Lag = how far the display trails the producer's real-time clock. With no
    // backlog left there is nothing to catch up on.
    const nowNs = src.headT() + Math.max(0, now - src.headWall()) * 1e6;
    this.lagMs = src.peek() ? Math.max(0, (nowNs - this.tp) / 1e6) : 0;
    const cfg = this.config;
    if (!this.live || this.paused || this.stopWhen) {
      this.catchingUp = false;
      this.catchup = 1;
      this.degraded = false;
      return;
    }
    if (this.lagMs > cfg.lagThresholdMs) this.catchingUp = true;
    else if (this.lagMs < cfg.lagReleaseMs) this.catchingUp = false;
    const goal = this.catchingUp ? cfg.maxSpeed : 1;
    // smooth approach, ~400 ms time constant
    this.catchup += (goal - this.catchup) * Math.min(1, dt / 400);
    if (Math.abs(this.catchup - goal) < 0.01) this.catchup = goal;
    if (this.fullDetail) {
      if (this.lagMs > cfg.fullDetailLagThresholdMs) this.degraded = true;
      else if (this.lagMs < cfg.lagReleaseMs) this.degraded = false;
    } else this.degraded = false;
  }
}
