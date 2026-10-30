import {
  atLeast,
  includedAtDetail,
  type DetailLevel,
  type EventType,
  type Op,
  type SampleStage,
  type TraceEvent,
} from 'trace-schema';
import {
  gaussian,
  mockTokenizer,
  promptText,
  renderPrompt,
  responseText,
  rng,
  SPECIALS,
} from './model';
import { headRows, layerNodes, lensTop, modelDims, nodeStats } from './deep';

export type Granularity = 'token' | 'layer' | 'sample';
export type TraceItem =
  { kind: 'event'; event: TraceEvent } | { kind: 'gate'; granularity: Granularity; step: number };

export interface GenerateOptions {
  req?: string;
  model?: string;
  layers?: number;
  ctx?: number;
  prompt?: string;
  promptTokens?: number;
  responseTokens?: number;
  tokPerSec?: number;
  prefillTokPerSec?: number;
  batchSize?: number;
  loadMs?: number;
  detail?: DetailLevel;
  /** emit decode.layer for every Nth step after step 8 (Ollama-style downsampling) */
  layerStride?: number;
  seed?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  minP?: number;
  repeatPenalty?: number;
  /** emit per-head decode.attn for this many heads (0 = heads averaged) */
  heads?: number;
  /** layers [0, gpuLayers) run on CUDA0, the rest on CPU (undefined = no device info) */
  gpuLayers?: number;
  /** mixture-of-experts: n experts, k routed per token */
  experts?: { n: number; k: number };
  /** deep tier: emit deep events on steps < 3 and every deepStride-th step */
  deepStride?: number;
  /** deep tier: attention window (positions per head row) */
  attnWindow?: number;
  /** start of the trace on the emitter's monotonic clock */
  t0Ns?: number;
}

export interface GeneratedTrace {
  req: string;
  items: TraceItem[];
  events: TraceEvent[];
  promptTokens: Array<{ id: number; text: string }>;
  /** exact concatenation of decode.token texts */
  responseText: string;
  /** what /api/chat streams: one content chunk per non-empty decode.token */
  chunks: Array<{ t_ns: number; text: string }>;
}

const OP_SEQUENCE: Array<{ op: Op; w: number }> = [
  { op: 'norm', w: 0.04 },
  { op: 'attn_qkv', w: 0.18 },
  { op: 'attn_scores', w: 0.14 },
  { op: 'attn_out', w: 0.1 },
  { op: 'norm', w: 0.04 },
  { op: 'mlp_up', w: 0.22 },
  { op: 'mlp_act', w: 0.06 },
  { op: 'mlp_down', w: 0.22 },
];

const N_CANDIDATES = 64;

export function generateTrace(o: GenerateOptions = {}): GeneratedTrace {
  const opt = {
    model: 'mock-llama:8b',
    layers: 32,
    ctx: 4096,
    responseTokens: 200,
    tokPerSec: 60,
    prefillTokPerSec: 400,
    batchSize: 512,
    loadMs: 800,
    detail: 'full' as DetailLevel,
    layerStride: 1,
    seed: 42,
    temperature: 0.8,
    topK: 40,
    topP: 0.9,
    minP: 0,
    repeatPenalty: 1.1,
    heads: 0,
    deepStride: 12,
    attnWindow: 256,
    t0Ns: 1_000_000_000,
    ...o,
  };
  const req = o.req ?? globalThis.crypto.randomUUID();
  const r = rng(opt.seed);
  const { tok } = mockTokenizer();
  const eosId = tok.idOf(SPECIALS.eos)!;

  const userPrompt = opt.prompt ?? promptText(o.promptTokens ?? 120);
  const rendered = renderPrompt(userPrompt);
  const promptIds = tok.encode(rendered);
  const promptTokens = promptIds.map((id) => ({ id, text: tok.piece(id) }));
  const nPrompt = promptIds.length;

  // like a runner sizing num_ctx: grow to the next power of two that fits
  while (opt.ctx < nPrompt + opt.responseTokens) opt.ctx *= 2;

  const respIds = tok
    .encode(responseText(opt.responseTokens, r))
    .slice(0, Math.max(0, opt.responseTokens - 1));
  respIds.push(eosId);

  const items: TraceItem[] = [];
  const events: TraceEvent[] = [];
  const chunks: GeneratedTrace['chunks'] = [];
  let seq = 0;
  let t = opt.t0Ns;
  const ms = (x: number) => x * 1e6;
  const jitter = (x: number, f = 0.15) => Math.max(x * 0.05, x * (1 + gaussian(r) * f));

  const emit = (type: EventType, payload: Record<string, unknown>) => {
    if (!includedAtDetail(type, opt.detail)) return;
    const event = { v: 1, req, seq: seq++, t_ns: Math.round(t), type, ...payload } as TraceEvent;
    events.push(event);
    items.push({ kind: 'event', event });
  };
  const gate = (granularity: Granularity, step: number) =>
    items.push({ kind: 'gate', granularity, step });

  // ---- request + load ----
  const tStart = t;
  emit('request.received', {
    model: opt.model,
    prompt_chars: userPrompt.length,
    options: {
      temperature: opt.temperature,
      top_k: opt.topK,
      top_p: opt.topP,
      min_p: opt.minP,
      repeat_penalty: opt.repeatPenalty,
      num_predict: opt.responseTokens,
    },
    detail: opt.detail,
  });
  t += ms(0.3);
  const vram = Math.round(opt.layers * 140 + opt.ctx * 0.125);
  if (opt.loadMs > 0) {
    emit('model.load', { phase: 'start', layers: opt.layers, ctx: opt.ctx, vram_mb: vram });
    t += ms(opt.loadMs);
  }
  emit('model.load', {
    phase: 'done',
    layers: opt.layers,
    ctx: opt.ctx,
    vram_mb: vram,
    ms: opt.loadMs,
  });
  const loadMs = (t - tStart) / 1e6;

  // ---- tokenize ----
  t += ms(2);
  emit('tokenize.done', { n: nPrompt, tokens: promptTokens });

  // ---- prefill ----
  const tPrefill = t;
  const nBatches = Math.max(1, Math.ceil(nPrompt / opt.batchSize));
  for (let b = 0; b < nBatches; b++) {
    const from = b * opt.batchSize;
    const to = Math.min(nPrompt, from + opt.batchSize);
    const batchMs = jitter(((to - from) / opt.prefillTokPerSec) * 1000, 0.05);
    for (let l = 0; l < opt.layers; l++) {
      t += ms(batchMs / opt.layers);
      emit('prefill.layer', { batch_i: b, layer: l, ms: round3(batchMs / opt.layers) });
    }
    if (!includedAtDetail('prefill.layer', opt.detail)) t += ms(batchMs);
    emit('prefill.batch', { batch_i: b, of: nBatches, from, to, ms: round3(batchMs) });
  }
  const prefillMs = (t - tPrefill) / 1e6;
  emit('prefill.done', { n_tokens: nPrompt, ms: round3(prefillMs) });

  // ---- decode ----
  const tDecode = t;
  const anchors = [1, Math.floor(nPrompt * 0.3), Math.floor(nPrompt * 0.6)].filter(
    (p) => p > 0 && p < nPrompt,
  );
  const history: number[] = [];
  const utf8 = new Utf8Assembler();
  let responseTextOut = '';
  const stepMs = 1000 / opt.tokPerSec;
  const candidatePool = Array.from({ length: tok.size }, (_, i) => i).filter(
    (i) => !tok.isSpecial(i),
  );

  // deep tier uses its own RNG so lower tiers stay identical for a given seed
  const rd = rng(opt.seed ^ 0x5eed);
  const dims = modelDims(opt.layers);
  const piece = (id: number) => tok.piece(id);

  for (let step = 0; step < respIds.length; step++) {
    const target = respIds[step];
    const deep =
      atLeast(opt.detail, 'deep') && (step < 3 || step % Math.max(1, opt.deepStride) === 0);
    const nodes = deep ? layerNodes(dims, pos0(nPrompt, step) + 1) : [];
    const finalP = 0.55 + 0.4 * rd();
    const pos = nPrompt + step;
    const thisStepMs = jitter(stepMs, 0.1);
    const layerMs = (thisStepMs * 0.85) / opt.layers;
    gate('token', step);
    emit('decode.step.start', { step, kv_used: pos, kv_total: opt.ctx });
    t += ms(0.05);
    const emitLayers = step < 8 || step % opt.layerStride === 0;
    let norm = 4 + r();
    for (let l = 0; l < opt.layers; l++) {
      gate('layer', step);
      const cpuLayer = o.gpuLayers !== undefined && l >= o.gpuLayers;
      const thisLayerMs =
        jitter(layerMs, 0.2) * (l === 0 || l === opt.layers - 1 ? 1.3 : 1) * (cpuLayer ? 4 : 1);
      const residRms = ((norm + 1) / Math.sqrt(dims.embd)) * 3;
      let topAttn = 0.5;
      if (atLeast(opt.detail, 'full')) {
        for (let g = 0; g < OP_SEQUENCE.length; g++) {
          const { op, w } = OP_SEQUENCE[g];
          const opMs = thisLayerMs * w;
          const inGroup = nodes.map((n, i) => ({ n, i })).filter((x) => x.n.group === g);
          if (inGroup.length) {
            const wTot = inGroup.reduce((a, x) => a + x.n.w, 0);
            for (const { n, i } of inGroup) {
              const nodeMs = (opMs * n.w) / wTot;
              t += ms(nodeMs);
              emit('decode.node', {
                step,
                layer: l,
                i,
                name: n.name,
                op: n.op,
                shape: n.shape,
                ms: round4(nodeMs),
                ...(n.src.length ? { src: n.src } : {}),
                stats: nodeStats(n, residRms, topAttn, pos + 1, rd),
              });
            }
          } else t += ms(opMs);
          emit('decode.op', { step, layer: l, op, ms: round3(opMs) });
          if (op === 'attn_qkv') emit('decode.kv.write', { step, layer: l, pos });
          if (op === 'attn_scores') {
            if (opt.heads > 0) {
              for (let h = 0; h < opt.heads; h++)
                emit('decode.attn', {
                  step,
                  layer: l,
                  head: h,
                  top: attention(pos, l, anchors, r),
                });
            } else {
              const top = attention(pos, l, anchors, r);
              topAttn = top[0]?.w ?? topAttn;
              emit('decode.attn', { step, layer: l, top });
            }
            if (deep) {
              const hr = headRows(pos, l, opt.layers, dims.heads, anchors, opt.attnWindow, rd);
              emit('decode.attn.heads', {
                step,
                layer: l,
                from: hr.from,
                n_kv_head: dims.kvHeads,
                rows: hr.rows,
              });
            }
          }
        }
      } else {
        t += ms(thisLayerMs);
      }
      const delta = Math.abs(1.1 + gaussian(r) * 0.35) * (l > opt.layers * 0.85 ? 1.8 : 1);
      norm += delta;
      emit('decode.residual', { step, layer: l, norm: round3(norm), delta_norm: round3(delta) });
      if (deep) {
        const lens = lensTop(l, opt.layers, target, finalP, candidatePool, piece, rd);
        emit('decode.lens', { step, layer: l, top: lens.top, entropy: lens.entropy });
      }
      if (o.experts) {
        const picked = new Set<number>();
        while (picked.size < o.experts.k) picked.add(Math.floor(r() * o.experts.n));
        const ws = [...picked].map(() => r() + 0.2);
        const tot = ws.reduce((a, b) => a + b, 0);
        emit('decode.moe', {
          step,
          layer: l,
          experts: [...picked]
            .map((id, i) => ({ id, w: round3(ws[i] / tot) }))
            .sort((a, b) => b.w - a.w),
        });
      }
      if (emitLayers)
        emit('decode.layer', {
          step,
          layer: l,
          ms: round3(thisLayerMs),
          norm: round3(norm),
          ...(o.gpuLayers !== undefined ? { device: l < o.gpuLayers ? 'CUDA0' : 'CPU' } : {}),
        });
    }

    // ---- sampling ----
    const lowConfidence = r() < 0.12;
    const sampled = sample(target, candidatePool, history, lowConfidence, opt, r);
    const sampleMs = thisStepMs * 0.1;
    const top10 = sampled.post
      .slice(0, 10)
      .map((c) => ({ id: c.id, text: tok.piece(c.id), p: round5(c.p) }));
    emit('decode.logits', { step, topk: top10, entropy: round3(sampled.entropy) });
    for (const st of sampled.stages) {
      gate('sample', step);
      t += ms(sampleMs / sampled.stages.length);
      emit('decode.sample.stage', { ...st, step });
    }
    emit('decode.sample', {
      step,
      chosen_id: target,
      temp: opt.temperature,
      top_p: opt.topP,
      top_k: opt.topK,
      method: 'dist',
    });
    t += ms(Math.max(0.05, thisStepMs * 0.05 - 0.05));

    const isEos = target === eosId;
    const bytes = isEos ? new Uint8Array() : tok.tokenBytes(target);
    const text = isEos ? '' : utf8.push(bytes);
    responseTextOut += text;
    emit('decode.token', { step, text, is_eos: isEos });
    if (!isEos) {
      emit('stream.chunk', { step, bytes: bytes.length, utf8_complete: utf8.pending === 0, text });
      if (text) chunks.push({ t_ns: Math.round(t), text });
    }
    history.push(target);
  }
  const evalMs = (t - tDecode) / 1e6;
  t += ms(0.5);
  emit('request.done', {
    stats: {
      prompt_eval_count: nPrompt,
      prompt_eval_ms: round3(prefillMs),
      eval_count: respIds.length,
      eval_ms: round3(evalMs),
      load_ms: round3(loadMs),
      total_ms: round3((t - tStart) / 1e6),
    },
  });

  return { req, items, events, promptTokens, responseText: responseTextOut, chunks };
}

// ---- helpers --------------------------------------------------------------

const round3 = (x: number) => Math.round(x * 1000) / 1000;
const round4 = (x: number) => Math.round(x * 1e4) / 1e4;
const pos0 = (nPrompt: number, step: number) => nPrompt + step;
const round5 = (x: number) => Math.round(x * 1e5) / 1e5;

/** Releases decoded text only once complete UTF-8 sequences are available. */
export class Utf8Assembler {
  private buf: number[] = [];
  private dec = new TextDecoder('utf-8');
  get pending(): number {
    return this.buf.length;
  }
  push(bytes: Uint8Array): string {
    this.buf.push(...bytes);
    // find longest prefix that ends on a complete sequence
    let end = this.buf.length;
    let i = end - 1;
    let back = 0;
    while (i >= 0 && back < 4 && (this.buf[i] & 0xc0) === 0x80) {
      i--;
      back++;
    }
    if (i >= 0) {
      const lead = this.buf[i];
      const need = lead >= 0xf0 ? 4 : lead >= 0xe0 ? 3 : lead >= 0xc0 ? 2 : 1;
      if (end - i < need) end = i;
    }
    const out = this.dec.decode(Uint8Array.from(this.buf.slice(0, end)));
    this.buf = this.buf.slice(end);
    return out;
  }
}

function attention(pos: number, layer: number, anchors: number[], r: () => number) {
  const w = new Map<number, number>();
  const add = (p: number, x: number) => {
    if (p >= 0 && p <= pos) w.set(p, (w.get(p) ?? 0) + x);
  };
  add(0, 0.25 + 0.2 * r()); // attention sink
  add(pos, 0.1 + 0.1 * r());
  for (let k = 1; k <= 4; k++) add(pos - k, (0.18 / k) * (0.5 + r()));
  for (const a of anchors) add(a, (layer % 3 === 0 ? 0.12 : 0.04) * (0.5 + r()));
  add(Math.floor(r() * pos), 0.05 * r());
  const top = [...w.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8);
  const total = top.reduce((s, [, x]) => s + x, 0);
  const scale = (0.75 + 0.2 * r()) / total;
  return top.map(([p, x]) => ({ pos: p, w: round5(x * scale) }));
}

interface Cand {
  id: number;
  v: number;
}

function softmax(c: Cand[]): Array<{ id: number; p: number }> {
  const m = Math.max(...c.map((x) => x.v));
  const e = c.map((x) => Math.exp(x.v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return c.map((x, i) => ({ id: x.id, p: e[i] / s }));
}

function sample(
  target: number,
  pool: number[],
  history: number[],
  lowConfidence: boolean,
  opt: { temperature: number; topK: number; topP: number; minP: number; repeatPenalty: number },
  r: () => number,
) {
  // synthetic logit vector: target + distractors (some from recent history)
  const ids = new Set<number>([target]);
  for (const h of history.slice(-6)) if (ids.size < 6) ids.add(h);
  while (ids.size < N_CANDIDATES) ids.add(pool[Math.floor(r() * pool.length)]);
  let raw: Cand[] = [...ids].map((id) => ({ id, v: 8 + gaussian(r) * 2 }));
  const tgt = raw.find((c) => c.id === target)!;
  const others = raw.filter((c) => c.id !== target).sort((a, b) => b.v - a.v);
  if (lowConfidence) {
    tgt.v = others[0].v + gaussian(r) * 0.4;
    others[1].v = tgt.v - 0.2;
  } else {
    tgt.v = others[0].v + 3 + r() * 4;
  }

  for (let attempt = 0; attempt < 4; attempt++) {
    const stages: Array<{
      step: number;
      stage: SampleStage;
      candidates: Cand[];
      cutoff?: number;
      draw?: number;
    }> = [];
    const sorted = (c: Cand[]) => [...c].sort((a, b) => b.v - a.v);
    const r5 = (c: Cand[]) => c.map((x) => ({ id: x.id, v: round5(x.v) }));
    let cur = sorted(raw);
    stages.push({ step: 0, stage: 'raw', candidates: r5(cur) });
    if (opt.repeatPenalty !== 1) {
      const recent = new Set(history.slice(-64));
      cur = sorted(
        cur.map((c) =>
          recent.has(c.id)
            ? { id: c.id, v: c.v > 0 ? c.v / opt.repeatPenalty : c.v * opt.repeatPenalty }
            : c,
        ),
      );
      stages.push({ step: 0, stage: 'rep_penalty', candidates: r5(cur) });
    }
    cur = cur.map((c) => ({ id: c.id, v: c.v / Math.max(opt.temperature, 1e-3) }));
    stages.push({ step: 0, stage: 'temperature', candidates: r5(cur) });
    const post = softmax(cur).sort((a, b) => b.p - a.p);
    const entropy = -post.reduce((s, x) => s + (x.p > 0 ? x.p * Math.log(x.p) : 0), 0);

    let kept = cur.length;
    if (opt.topK > 0) {
      kept = Math.min(kept, opt.topK);
      stages.push({ step: 0, stage: 'top_k', candidates: r5(cur), cutoff: kept });
    }
    {
      const pr = softmax(cur.slice(0, kept));
      let cum = 0;
      let n = 0;
      for (const x of pr) {
        cum += x.p;
        n++;
        if (cum >= opt.topP) break;
      }
      kept = Math.max(1, n);
      stages.push({ step: 0, stage: 'top_p', candidates: r5(cur), cutoff: kept });
    }
    if (opt.minP > 0) {
      const pr = softmax(cur.slice(0, kept));
      const thr = pr[0].p * opt.minP;
      kept = Math.max(1, pr.filter((x) => x.p >= thr).length);
      stages.push({ step: 0, stage: 'min_p', candidates: r5(cur), cutoff: kept });
    }
    const final = softmax(cur.slice(0, kept));
    const ti = final.findIndex((x) => x.id === target);
    if (ti < 0) {
      // target was pruned: make it the favourite and retry
      raw = raw.map((c) =>
        c.id === target ? { id: c.id, v: Math.max(...raw.map((x) => x.v)) + 0.5 } : c,
      );
      continue;
    }
    const probs = final.map((x) => ({ id: x.id, v: round5(x.p) }));
    stages.push({ step: 0, stage: 'softmax', candidates: probs, cutoff: kept });
    let a = 0;
    for (let i = 0; i < ti; i++) a += final[i].p;
    const drawU = a + final[ti].p * (0.1 + 0.8 * r());
    stages.push({ step: 0, stage: 'draw', candidates: probs, cutoff: ti, draw: round5(drawU) });
    return { stages, post, entropy };
  }
  throw new Error('sampler synthesis failed');
}
