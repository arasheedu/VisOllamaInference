import type { TraceEvent } from 'trace-schema';
import type { ClockState, DerivedStats, Phase, StepDetail, ViewModel } from './types';

const RECENT = 60;
const KEEP_STEPS = 64;

function emptyClock(): ClockState {
  return {
    tp: 0,
    t0: 0,
    tEnd: 0,
    speed: 1,
    dilation: 1,
    lagMs: 0,
    catchingUp: false,
    degraded: false,
    dropped: 0,
    paused: false,
    live: true,
    stepping: null,
  };
}

export function emptyViewModel(): ViewModel {
  return {
    req: null,
    phase: 'idle',
    model: null,
    detail: null,
    layersCount: 0,
    ctx: 0,
    vramMb: 0,
    options: {},
    capabilities: [],
    promptTokens: [],
    tokenizeWall: -Infinity,
    responseTokens: [],
    responseText: '',
    currentStep: -1,
    prefill: { batch: -1, of: 0, progress: 0, from: 0, to: 0, wall: -Infinity },
    kv: { used: 0, total: 0 },
    layers: [],
    sampling: {
      step: -1,
      topk: [],
      temp: 0,
      top_p: 0,
      top_k: 0,
      entropy: 0,
      method: '',
      wall: -Infinity,
    },
    stats: null,
    derived: { ttftMs: null, promptTokPerSec: null, genTokPerSec: null, totalMs: null },
    timeline: [],
    phaseDurations: {},
    error: null,
    stepDetail: null,
    steps: {},
    streamChunks: [],
    deepSteps: [],
    lastEvent: null,
    recent: [],
    eventsApplied: 0,
    clock: emptyClock(),
    version: 0,
  };
}

function emptyStep(step: number): StepDetail {
  return {
    step,
    ops: [],
    residual: [],
    attn: [],
    kvWrites: [],
    samplingStages: [],
    layerMs: [],
    moe: [],
    nodes: [],
    lens: [],
    heads: [],
  };
}

/**
 * Pure-ish reducer from trace events to a view model. Every mutation replaces
 * the touched array/object (copy-on-write) so React can compare by reference.
 * Tolerant of missing droppable events: nothing here depends on them.
 */
export class InferenceStateMachine {
  private vm: ViewModel = emptyViewModel();
  private t0 = 0;
  private firstTokenT: number | null = null;
  private decodeStartT: number | null = null;
  private dirty = true;

  reset() {
    this.vm = emptyViewModel();
    this.t0 = 0;
    this.firstTokenT = null;
    this.decodeStartT = null;
    this.dirty = true;
    this.last = null;
  }

  get phase(): Phase {
    return this.vm.phase;
  }

  get current(): ViewModel {
    return this.vm;
  }

  /** Displayed phase: changes when the event that reveals it is shown. */
  private setPhase(p: Phase, _t: number) {
    this.vm.phase = p;
  }

  /**
   * Timeline bands use the true stage boundaries in trace time: a stage runs
   * from the end of the previous one to the event that reports its completion
   * (load → model.load done, tokenize → tokenize.done, prefill → prefill.done,
   * decode → request.done).
   */
  private mark(p: Phase | null, t: number) {
    const vm = this.vm;
    const tl = vm.timeline.slice();
    const last = tl[tl.length - 1];
    if (last && last.phase === p) return;
    if (last && last.to === null) {
      tl[tl.length - 1] = { ...last, to: t };
      vm.phaseDurations = { ...vm.phaseDurations, [last.phase]: (t - last.from) / 1e6 };
    }
    if (p) tl.push({ phase: p, from: t, to: null });
    vm.timeline = tl;
  }

  private ensureLayers(n: number) {
    if (this.vm.layers.length >= n) return;
    const layers = this.vm.layers.slice();
    for (let i = layers.length; i < n; i++)
      layers.push({ idx: i, active: false, lastMs: 0, norm: 0, wall: -Infinity, step: -1 });
    this.vm.layers = layers;
  }

  private step(step: number): StepDetail {
    const vm = this.vm;
    let sd = vm.steps[step];
    if (!sd) {
      sd = emptyStep(step);
      const steps = { ...vm.steps, [step]: sd };
      const old = step - KEEP_STEPS;
      if (old in steps) delete steps[old];
      vm.steps = steps;
    }
    return sd;
  }

  private updateStep(step: number, f: (sd: StepDetail) => StepDetail) {
    const next = f(this.step(step));
    this.vm.steps = { ...this.vm.steps, [step]: next };
    if (step === this.vm.currentStep || !this.vm.stepDetail || this.vm.stepDetail.step <= step)
      this.vm.stepDetail = next;
  }

  apply(e: TraceEvent, wall: number) {
    const vm = this.vm;
    this.dirty = true;
    vm.eventsApplied++;
    vm.lastEvent = e;
    const recent = vm.recent.length >= RECENT ? vm.recent.slice(1) : vm.recent.slice();
    recent.push(e);
    vm.recent = recent;
    if (vm.req === null) {
      vm.req = e.req;
      this.t0 = e.t_ns;
    }

    switch (e.type) {
      case 'request.received':
        this.t0 = e.t_ns;
        vm.model = e.model;
        vm.options = e.options;
        vm.detail = e.detail ?? null;
        vm.capabilities = e.capabilities ?? [];
        this.setPhase('loading', e.t_ns);
        this.mark('loading', e.t_ns);
        break;
      case 'model.load':
        vm.layersCount = e.layers;
        vm.ctx = e.ctx;
        vm.vramMb = e.vram_mb;
        vm.kv = { used: vm.kv.used, total: e.ctx };
        this.ensureLayers(e.layers);
        if (e.phase === 'start') this.setPhase('loading', e.t_ns);
        else if (vm.phase === 'idle') this.setPhase('loading', e.t_ns);
        if (vm.timeline.length === 0) this.mark('loading', e.t_ns - (e.ms ?? 0) * 1e6);
        if (e.phase === 'done') this.mark('tokenizing', e.t_ns);
        break;
      case 'tokenize.done':
        vm.promptTokens = e.tokens;
        vm.tokenizeWall = wall;
        this.setPhase('tokenizing', e.t_ns);
        this.mark('prefill', e.t_ns);
        break;
      case 'prefill.batch':
        this.setPhase('prefill', e.t_ns);
        vm.prefill = {
          batch: e.batch_i,
          of: e.of,
          from: e.from,
          to: e.to,
          progress: (e.batch_i + 1) / e.of,
          wall,
        };
        vm.kv = { ...vm.kv, used: e.to };
        break;
      case 'prefill.layer':
        this.ensureLayers(e.layer + 1);
        this.flashLayer(e.layer, e.ms, undefined, wall, -1);
        break;
      case 'prefill.done':
        this.mark('decoding', e.t_ns);
        vm.prefill = { ...vm.prefill, progress: 1 };
        vm.kv = { ...vm.kv, used: Math.max(vm.kv.used, e.n_tokens) };
        break;
      case 'decode.step.start':
        if (this.decodeStartT === null) this.decodeStartT = e.t_ns;
        this.setPhase('decoding', e.t_ns);
        this.mark('decoding', e.t_ns);
        vm.currentStep = e.step;
        vm.kv = { used: e.kv_used, total: e.kv_total || vm.kv.total };
        this.step(e.step);
        vm.stepDetail = vm.steps[e.step];
        break;
      case 'decode.layer':
        this.ensureLayers(e.layer + 1);
        this.flashLayer(e.layer, e.ms, e.norm, wall, e.step, e.device);
        this.updateStep(e.step, (sd) => ({
          ...sd,
          layerMs: [...sd.layerMs, { layer: e.layer, ms: e.ms }],
        }));
        break;
      case 'decode.logits':
        vm.sampling = {
          ...vm.sampling,
          step: e.step,
          topk: e.topk,
          entropy: e.entropy,
          chosen: undefined,
          wall,
        };
        break;
      case 'decode.sample':
        vm.sampling = {
          ...vm.sampling,
          step: e.step,
          chosen: e.chosen_id,
          temp: e.temp,
          top_p: e.top_p,
          top_k: e.top_k,
          method: e.method,
        };
        break;
      case 'decode.token': {
        if (this.firstTokenT === null) this.firstTokenT = e.t_ns;
        const s = vm.sampling.step === e.step ? vm.sampling : null;
        vm.responseTokens = [
          ...vm.responseTokens,
          {
            step: e.step,
            text: e.text,
            isEos: e.is_eos,
            id: s?.chosen,
            topk: s?.topk,
            entropy: s?.entropy,
            wall,
          },
        ];
        vm.responseText = vm.responseText + e.text;
        if (vm.phase !== 'decoding') this.setPhase('decoding', e.t_ns);
        this.updateDerived(e.t_ns);
        break;
      }
      case 'decode.op':
        this.updateStep(e.step, (sd) => {
          const slot = sd.ops.filter((o) => o.layer === e.layer && o.op === e.op).length;
          return { ...sd, ops: [...sd.ops, { layer: e.layer, op: e.op, slot, ms: e.ms, wall }] };
        });
        this.ensureLayers(e.layer + 1);
        break;
      case 'decode.residual':
        this.updateStep(e.step, (sd) => ({
          ...sd,
          residual: [...sd.residual, { layer: e.layer, norm: e.norm, delta: e.delta_norm, wall }],
        }));
        break;
      case 'decode.attn':
        this.updateStep(e.step, (sd) => ({
          ...sd,
          attn: [...sd.attn, { layer: e.layer, head: e.head, top: e.top, wall }],
        }));
        break;
      case 'decode.kv.write':
        this.updateStep(e.step, (sd) => ({
          ...sd,
          kvWrites: [...sd.kvWrites, { layer: e.layer, pos: e.pos, wall }],
        }));
        break;
      case 'decode.sample.stage':
        this.updateStep(e.step, (sd) => ({
          ...sd,
          samplingStages: [...sd.samplingStages, { ...e, wall }],
        }));
        break;
      case 'stream.chunk':
        vm.streamChunks = [
          ...(vm.streamChunks.length > 400 ? vm.streamChunks.slice(-300) : vm.streamChunks),
          {
            step: e.step,
            bytes: e.bytes,
            complete: e.utf8_complete,
            text: e.text,
            t_ns: e.t_ns,
            wall,
          },
        ];
        break;
      case 'tokenize.merge':
        // Merges are synthesised in-app (see tokenizer/merges.ts); recorded only.
        break;
      case 'request.done':
        vm.stats = e.stats;
        this.setPhase('done', e.t_ns);
        this.mark(null, e.t_ns);
        this.updateDerived(e.t_ns);
        break;
      case 'request.error':
        vm.error = e.message;
        this.setPhase('error', e.t_ns);
        this.mark(null, e.t_ns);
        break;
      case 'decode.moe':
        this.updateStep(e.step, (sd) => ({
          ...sd,
          moe: [...sd.moe, { layer: e.layer, experts: e.experts, wall }],
        }));
        break;
      case 'decode.node': {
        const { step, layer, i, name, op, shape, ms, src, stats } = e;
        this.updateStep(step, (sd) => ({
          ...sd,
          nodes: [...sd.nodes, { layer, i, name, op, shape, ms, src, stats, wall }],
        }));
        break;
      }
      case 'decode.lens':
        if (!vm.deepSteps.includes(e.step))
          vm.deepSteps = [...vm.deepSteps, e.step].sort((a, b) => a - b);
        this.updateStep(e.step, (sd) => ({
          ...sd,
          lens: [...sd.lens, { layer: e.layer, top: e.top, entropy: e.entropy, wall }],
        }));
        break;
      case 'decode.attn.heads':
        this.updateStep(e.step, (sd) => ({
          ...sd,
          heads: [
            ...sd.heads,
            { layer: e.layer, from: e.from, nKvHead: e.n_kv_head, rows: e.rows, wall },
          ],
        }));
        break;
      case 'step.timeout':
        break;
    }
  }

  private flashLayer(
    layer: number,
    ms: number,
    norm: number | undefined,
    wall: number,
    step: number,
    device?: string,
  ) {
    const layers = this.vm.layers.slice();
    for (const l of layers) if (l.active) layers[l.idx] = { ...l, active: false };
    const prev = layers[layer];
    layers[layer] = {
      idx: layer,
      active: true,
      lastMs: ms,
      norm: norm ?? prev.norm,
      wall,
      step,
      device: device ?? prev.device,
    };
    this.vm.layers = layers;
  }

  private updateDerived(t: number) {
    const vm = this.vm;
    const d: DerivedStats = { ...vm.derived };
    if (this.firstTokenT !== null) d.ttftMs = (this.firstTokenT - this.t0) / 1e6;
    if (vm.stats) {
      const s = vm.stats;
      d.promptTokPerSec =
        s.prompt_eval_ms > 0 ? s.prompt_eval_count / (s.prompt_eval_ms / 1000) : null;
      d.genTokPerSec = s.eval_ms > 0 ? s.eval_count / (s.eval_ms / 1000) : null;
      d.totalMs = s.total_ms;
    } else if (this.decodeStartT !== null && t > this.decodeStartT) {
      d.genTokPerSec = vm.responseTokens.length / ((t - this.decodeStartT) / 1e9);
      d.totalMs = (t - this.t0) / 1e6;
    }
    vm.derived = d;
  }

  /** Forget deep steps whose events were dropped from the buffer. */
  pruneDeepSteps(keep: Set<number>) {
    const next = this.vm.deepSteps.filter((s) => keep.has(s));
    if (next.length !== this.vm.deepSteps.length) {
      this.vm.deepSteps = next;
      this.dirty = true;
    }
  }

  /** Immutable snapshot; returns the same object if nothing changed. */
  snapshot(clock: ClockState): ViewModel {
    const prev = this.last;
    const c = prev?.clock ?? emptyClock();
    const clockChanged =
      c.tp !== clock.tp ||
      c.speed !== clock.speed ||
      c.paused !== clock.paused ||
      c.live !== clock.live ||
      c.catchingUp !== clock.catchingUp ||
      c.degraded !== clock.degraded ||
      c.dropped !== clock.dropped ||
      c.stepping !== clock.stepping ||
      c.tEnd !== clock.tEnd ||
      Math.abs(c.dilation - clock.dilation) > 1e-3 ||
      Math.abs(c.lagMs - clock.lagMs) > 1;
    if (prev && !this.dirty && !clockChanged) return prev;
    this.dirty = false;
    // Internal state keeps mutating top-level scalars; nested arrays/objects are
    // copy-on-write, so a shallow copy is a stable snapshot.
    this.last = { ...this.vm, clock: { ...clock }, version: (prev?.version ?? 0) + 1 };
    return this.last;
  }

  private last: ViewModel | null = null;
}
