// Synthetic deep-tier data (docs/deep-tier.md): the graph nodes of a llama-style
// decoder layer, a logit lens and per-head attention rows.
import { gaussian } from './model';

export interface ModelDims {
  embd: number;
  heads: number;
  kvHeads: number;
  headDim: number;
  ff: number;
}

/** Llama-family shapes for the mock models, picked by layer count. */
export function modelDims(layers: number): ModelDims {
  if (layers >= 64) return { embd: 8192, heads: 64, kvHeads: 8, headDim: 128, ff: 28672 };
  if (layers >= 40) return { embd: 2048, heads: 32, kvHeads: 4, headDim: 128, ff: 6144 };
  if (layers >= 24) return { embd: 4096, heads: 32, kvHeads: 8, headDim: 128, ff: 14336 };
  return { embd: 2048, heads: 32, kvHeads: 8, headDim: 64, ff: 8192 };
}

export interface NodeTemplate {
  name: string;
  op: string;
  shape: number[];
  src: number[];
  /** index into the decode.op sequence this node's time is charged to */
  group: number;
  /** relative cost within the group */
  w: number;
}

/**
 * Nodes of one decode layer in execution order (non-flash-attention graph).
 * Groups follow the decode.op sequence: 0 norm, 1 attn_qkv, 2 attn_scores,
 * 3 attn_out, 4 norm, 5 mlp_up, 6 mlp_act, 7 mlp_down.
 */
export function layerNodes(d: ModelDims, nKv: number): NodeTemplate[] {
  const { embd: E, heads: H, kvHeads: KV, headDim: hd, ff: F } = d;
  return [
    { name: 'norm', op: 'RMS_NORM', shape: [E], src: [], group: 0, w: 1 },
    { name: 'attn_norm', op: 'MUL', shape: [E], src: [0], group: 0, w: 0.6 },
    { name: 'Qcur', op: 'MUL_MAT', shape: [H * hd], src: [1], group: 1, w: H },
    { name: 'Kcur', op: 'MUL_MAT', shape: [KV * hd], src: [1], group: 1, w: KV },
    { name: 'Vcur', op: 'MUL_MAT', shape: [KV * hd], src: [1], group: 1, w: KV },
    { name: 'Qcur', op: 'ROPE', shape: [hd, H], src: [2], group: 1, w: 1 },
    { name: 'Kcur', op: 'ROPE', shape: [hd, KV], src: [3], group: 1, w: 0.5 },
    { name: 'k_cache', op: 'SET_ROWS', shape: [hd * KV], src: [6], group: 2, w: 0.4 },
    { name: 'v_cache', op: 'SET_ROWS', shape: [hd * KV], src: [4], group: 2, w: 0.4 },
    { name: 'kq', op: 'MUL_MAT', shape: [nKv, H], src: [5, 7], group: 2, w: 3 },
    { name: 'kq_soft_max', op: 'SOFT_MAX', shape: [nKv, H], src: [9], group: 2, w: 1 },
    { name: 'kqv', op: 'MUL_MAT', shape: [hd, H], src: [10, 8], group: 3, w: 2 },
    { name: 'kqv_out', op: 'CONT', shape: [H * hd], src: [11], group: 3, w: 0.4 },
    { name: 'attn_out', op: 'MUL_MAT', shape: [E], src: [12], group: 3, w: 4 },
    { name: 'ffn_inp', op: 'ADD', shape: [E], src: [13], group: 3, w: 0.3 },
    { name: 'norm', op: 'RMS_NORM', shape: [E], src: [14], group: 4, w: 1 },
    { name: 'ffn_norm', op: 'MUL', shape: [E], src: [15], group: 4, w: 0.6 },
    { name: 'ffn_gate', op: 'MUL_MAT', shape: [F], src: [16], group: 5, w: 1 },
    { name: 'ffn_up', op: 'MUL_MAT', shape: [F], src: [16], group: 5, w: 1 },
    { name: 'ffn_swiglu', op: 'GLU', shape: [F], src: [17, 18], group: 6, w: 1 },
    { name: 'ffn_out', op: 'MUL_MAT', shape: [E], src: [19], group: 7, w: 5 },
    { name: 'l_out', op: 'ADD', shape: [E], src: [20, 14], group: 7, w: 0.3 },
  ];
}

const r4 = (x: number) => Math.round(x * 1e4) / 1e4;

/** Plausible value statistics for a node's result. */
export function nodeStats(
  n: NodeTemplate,
  residRms: number,
  topAttn: number,
  nKv: number,
  r: () => number,
) {
  const j = () => 0.85 + 0.3 * r();
  let rms: number;
  let spread = 4 + 4 * r();
  switch (n.name) {
    case 'norm':
      rms = 1;
      break;
    case 'attn_norm':
    case 'ffn_norm':
      rms = 0.9 * j();
      spread = 6 + 6 * r();
      break;
    case 'Qcur':
    case 'Kcur':
      rms = 2.2 * j();
      break;
    case 'Vcur':
    case 'v_cache':
      rms = 0.45 * j();
      break;
    case 'k_cache':
      rms = 2.2 * j();
      break;
    case 'kq':
      rms = 3 * j();
      break;
    case 'kq_soft_max': {
      const mean = 1 / Math.max(1, nKv);
      return { min: 0, max: r4(topAttn), mean: r4(mean), rms: r4(Math.sqrt(mean * topAttn)) };
    }
    case 'kqv':
    case 'kqv_out':
      rms = 0.3 * j();
      break;
    case 'attn_out':
      rms = 0.25 * j();
      break;
    case 'ffn_gate':
      rms = 1.1 * j();
      break;
    case 'ffn_up':
      rms = 0.6 * j();
      break;
    case 'ffn_swiglu':
      rms = 0.18 * j();
      spread = 10 + 10 * r();
      break;
    case 'ffn_out':
      rms = 0.35 * j();
      break;
    default:
      // residual stream (ffn_inp, l_out): grows with depth, heavy outliers
      rms = residRms;
      spread = 15 + 25 * r();
  }
  const max = rms * spread * (0.7 + 0.3 * r());
  const min = -rms * spread * (0.7 + 0.3 * r());
  return { min: r4(min), max: r4(max), mean: r4(rms * 0.02 * gaussian(r)), rms: r4(rms) };
}

/**
 * Logit lens for one layer: early layers spread their mass over unrelated
 * tokens, the eventual token takes over in the last third of the stack.
 */
export function lensTop(
  layer: number,
  layers: number,
  target: number,
  finalP: number,
  pool: number[],
  piece: (id: number) => string,
  r: () => number,
  k = 5,
) {
  const x = layers > 1 ? layer / (layers - 1) : 1;
  const pT = finalP / (1 + Math.exp(-(x - 0.62) * 12));
  const junkMass = Math.max(0.02, (1 - pT) * (0.08 + 0.5 * x * x));
  const ids = new Set<number>([target]);
  while (ids.size < k + 2) ids.add(pool[Math.floor(r() * pool.length)]);
  const rest = [...ids].filter((id) => id !== target);
  const ws = rest.map(() => Math.pow(r(), 2));
  const tot = ws.reduce((a, b) => a + b, 0) || 1;
  const cands = rest.map((id, i) => ({ id, p: (ws[i] / tot) * junkMass }));
  cands.push({ id: target, p: pT });
  cands.sort((a, b) => b.p - a.p);
  const top = cands.slice(0, k).map((c) => ({ id: c.id, p: r4(c.p), text: piece(c.id) }));
  const entropy = 0.3 + (1 - pT) * (9 - 6 * x) + r() * 0.3;
  return { top, entropy: Math.round(entropy * 1000) / 1000 };
}

/**
 * Softmaxed attention rows for every query head over positions [from, pos].
 * Heads get stable roles (sink, previous token, local, content, broad).
 */
export function headRows(
  pos: number,
  layer: number,
  layers: number,
  heads: number,
  anchors: number[],
  window: number,
  r: () => number,
) {
  const from = Math.max(0, pos + 1 - window);
  const n = pos + 1 - from;
  const depth = layers > 1 ? layer / (layers - 1) : 0;
  const rows: number[][] = [];
  for (let h = 0; h < heads; h++) {
    const role = (h * 7 + layer * 3) % 6;
    const w = new Float64Array(n);
    for (let j = 0; j < n; j++) w[j] = 0.02 * r();
    const at = (p: number, x: number) => {
      if (p >= from && p <= pos) w[p - from] += x;
    };
    const sink = 0.2 + 2.5 * depth;
    switch (role) {
      case 0: // attention sink
        at(0, 3 + 3 * depth);
        at(pos, 0.3);
        break;
      case 1: // previous token
        at(pos - 1, 3);
        at(0, sink * 0.5);
        break;
      case 2: // local window
        for (let k = 0; k < 8; k++) at(pos - k, 1.2 / (k + 1));
        at(0, sink * 0.6);
        break;
      case 3: // content: prompt anchors
        for (const a of anchors) at(a, 0.8 + r());
        at(0, sink);
        break;
      case 4: // broad
        for (let j = 0; j < n; j++) w[j] += 0.05;
        at(0, sink * 0.4);
        break;
      default:
        at(0, sink);
        at(pos, 0.6);
        at(Math.floor(r() * (pos + 1)), 0.8);
    }
    let s = 0;
    for (let j = 0; j < n; j++) s += w[j];
    rows.push(Array.from(w, (x) => r4(x / s)));
  }
  return { from, rows };
}
