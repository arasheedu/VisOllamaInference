import {
  EVENT_DETAIL,
  PRIORITY,
  includedAtDetail,
  type DetailLevel,
  type TraceEvent,
} from 'trace-schema';
import { DEFAULT_CLOCK_CONFIG, PlaybackClock, type ClockConfig, type ClockSource } from './clock';
import { InferenceStateMachine } from './stateMachine';
import type { ClockState, ViewModel } from './types';

export type StepKind = 'token' | 'layer' | 'sample';

export const STEP_PREDICATES: Record<StepKind, (e: TraceEvent) => boolean> = {
  token: (e) => e.type === 'decode.token' || e.type === 'request.done',
  layer: (e) => e.type === 'decode.layer' || e.type === 'request.done',
  sample: (e) => e.type === 'decode.sample.stage' || e.type === 'request.done',
};

export interface PlayerOptions {
  capacity?: number;
  detail?: DetailLevel;
  clock?: ClockConfig;
  live?: boolean;
}

/**
 * One request's playback: a seq-ordered ring buffer, the playback clock and
 * the state machine. `tick(now)` is called once per animation frame.
 */
export class Player implements ClockSource {
  readonly events: TraceEvent[] = [];
  readonly clock: PlaybackClock;
  readonly machine = new InferenceStateMachine();
  private cursor = 0;
  private capacity: number;
  private lastT = 0;
  private lastWall = 0;
  private done = false;
  dropped = 0;
  filtered = 0;
  late = 0;
  detail: DetailLevel;

  constructor(
    readonly req: string,
    o: PlayerOptions = {},
  ) {
    this.capacity = o.capacity ?? 50_000;
    this.detail = o.detail ?? 'full';
    this.clock = new PlaybackClock(o.clock ?? DEFAULT_CLOCK_CONFIG);
    this.clock.live = o.live ?? true;
    this.clock.fullDetail = this.detail === 'full';
    this.clock.isFullTier = (e) => EVENT_DETAIL[e.type] === 'full';
  }

  setDetail(d: DetailLevel) {
    this.detail = d;
    this.clock.fullDetail = d === 'full';
  }

  /** Accepts whether the event type is wanted at the current detail level. */
  wants(type: TraceEvent['type']): boolean {
    return includedAtDetail(type, this.detail);
  }

  push(e: TraceEvent, wall = performance.now()) {
    if (!this.wants(e.type)) {
      this.filtered++;
      return;
    }
    const ev = this.events;
    if (e.type === 'request.done' || e.type === 'request.error') this.done = true;
    if (ev.length === 0 || e.seq > ev[ev.length - 1].seq) {
      ev.push(e);
    } else {
      // out-of-order: binary search insert by seq, dedupe
      let lo = 0;
      let hi = ev.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (ev[mid].seq < e.seq) lo = mid + 1;
        else hi = mid;
      }
      if (ev[lo]?.seq === e.seq) return;
      ev.splice(lo, 0, e);
      if (lo < this.cursor) {
        // arrived after we already played past it
        this.cursor++;
        if (PRIORITY[e.type] === 'critical') this.machine.apply(e, wall);
        else this.late++;
      }
    }
    if (e.t_ns > this.lastT) {
      this.lastT = e.t_ns;
      this.lastWall = wall;
    }
    if (ev.length > this.capacity) this.compact();
  }

  /** Drop oldest droppable events until under ~90% capacity. Never critical. */
  private compact() {
    const target = Math.floor(this.capacity * 0.9);
    let excess = this.events.length - target;
    let removedBeforeCursor = 0;
    const kept: TraceEvent[] = [];
    for (let i = 0; i < this.events.length; i++) {
      const e = this.events[i];
      if (excess > 0 && PRIORITY[e.type] === 'droppable') {
        excess--;
        this.dropped++;
        if (i < this.cursor) removedBeforeCursor++;
        continue;
      }
      kept.push(e);
    }
    this.events.length = 0;
    this.events.push(...kept);
    this.cursor -= removedBeforeCursor;
  }

  // ---- ClockSource ----
  peek() {
    return this.events[this.cursor];
  }
  consume(e: TraceEvent, wall: number, skip: boolean) {
    this.cursor++;
    if (!skip) this.machine.apply(e, wall);
  }
  headT() {
    return this.lastT;
  }
  headWall() {
    return this.lastWall;
  }
  finished() {
    return this.done;
  }

  get position() {
    return this.cursor;
  }
  get atEnd() {
    return this.cursor >= this.events.length;
  }
  get t0() {
    return this.events[0]?.t_ns ?? 0;
  }
  get tEnd() {
    return this.events[this.events.length - 1]?.t_ns ?? 0;
  }

  tick(now: number): ViewModel {
    this.clock.tick(now, this);
    return this.snapshot();
  }

  snapshot(): ViewModel {
    const c = this.clock;
    const clock: ClockState = {
      tp: c.tp,
      t0: this.t0,
      tEnd: this.tEnd,
      speed: c.speed,
      dilation: c.dilation,
      lagMs: c.lagMs,
      catchingUp: c.catchingUp,
      degraded: c.degraded,
      dropped: this.dropped,
      paused: c.paused,
      live: c.live,
      stepping: c.stoppingLabel,
    };
    return this.machine.snapshot(clock);
  }

  /** Rebuild state at trace time t instantly (no holds). Pauses and detaches. */
  seek(t: number) {
    this.machine.reset();
    this.cursor = 0;
    while (this.cursor < this.events.length && this.events[this.cursor].t_ns <= t) {
      this.machine.apply(this.events[this.cursor], -Infinity);
      this.cursor++;
    }
    this.clock.seek(Math.max(this.t0, t));
    this.clock.pause();
    this.clock.live = false;
  }

  /** Re-attach to the live head. */
  goLive() {
    this.seek(this.lastT);
    this.clock.live = true;
    this.clock.resume();
  }

  /** Play until the next event of `kind` has been shown, then pause. */
  stepUntil(kind: StepKind) {
    this.clock.stopWhen = STEP_PREDICATES[kind];
    this.clock.stoppingLabel = kind;
    this.clock.resume();
  }
}
