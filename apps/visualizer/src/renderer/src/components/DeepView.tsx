import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { session } from '../session';
import { setUi, useUi } from '../store';
import { OP_COLOR, showToken, useReducedMotion } from '../theme';
import type { NodeState, StepDetail, ViewModel } from '../engine/types';

// ---- data helpers -----------------------------------------------------------

const hasDeep = (d: StepDetail | null | undefined) =>
  !!d && (d.nodes.length > 0 || d.lens.length > 0 || d.heads.length > 0);

/** Deep data of one step straight from the player buffer (older steps leave the view model). */
function deepFromBuffer(step: number): StepDetail {
  const sd: StepDetail = {
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
  for (const e of session.active?.events ?? []) {
    if (!('step' in e) || e.step !== step) continue;
    if (e.type === 'decode.node') {
      const { layer, i, name, op, shape, ms, src, stats } = e;
      sd.nodes.push({ layer, i, name, op, shape, ms, src, stats, wall: -Infinity });
    } else if (e.type === 'decode.lens')
      sd.lens.push({ layer: e.layer, top: e.top, entropy: e.entropy, wall: -Infinity });
    else if (e.type === 'decode.attn.heads')
      sd.heads.push({
        layer: e.layer,
        from: e.from,
        nKvHead: e.n_kv_head,
        rows: e.rows,
        wall: -Infinity,
      });
  }
  return sd;
}

/** Response token picked in the response panel (via the inspector), if any. */
const useSelectedStep = () =>
  useUi((s) => (s.inspector?.kind === 'token' ? s.inspector.step : null));

/**
 * The step to show: a token clicked in the response panel, else the current
 * step if it has deep data, else the latest step that does.
 */
function useDeepStep(): StepDetail | null {
  const sd = useUi((s) => s.vm.stepDetail);
  const steps = useUi((s) => s.vm.steps);
  const selected = useSelectedStep();
  const applied = useUi((s) => s.vm.eventsApplied);
  return useMemo(() => {
    if (selected !== null) {
      const own = steps[selected];
      return hasDeep(own) ? own : deepFromBuffer(selected);
    }
    const has = hasDeep;
    if (has(sd)) return sd;
    let best: StepDetail | null = null;
    for (const d of Object.values(steps)) if (has(d) && (!best || d.step > best.step)) best = d;
    return best;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sd, steps, selected, selected !== null ? applied : 0]);
}

/** Text of the token at KV position `pos` (prompt first, then the response). */
function tokenAt(vm: ViewModel, pos: number): string {
  const n = vm.promptTokens.length;
  const t = pos < n ? vm.promptTokens[pos]?.text : vm.responseTokens[pos - n]?.text;
  return t === undefined ? `#${pos}` : showToken(t);
}

type Group =
  | 'norm'
  | 'attn_qkv'
  | 'attn_scores'
  | 'attn_out'
  | 'kv'
  | 'residual'
  | 'mlp_up'
  | 'mlp_act'
  | 'mlp_down'
  | 'other';

export function groupOf(name: string, op: string): Group {
  if (op === 'RMS_NORM' || op === 'NORM' || name.includes('norm')) return 'norm';
  if (op === 'SET_ROWS' || op === 'CPY' || name.includes('cache')) return 'kv';
  if (/^[QKV]cur/.test(name) || op === 'ROPE') return 'attn_qkv';
  if (name.startsWith('kq_') || name === 'kq' || op === 'SOFT_MAX' || op === 'FLASH_ATTN_EXT')
    return 'attn_scores';
  if (name.startsWith('kqv') || name.startsWith('attn')) return 'attn_out';
  if (name === 'ffn_inp' || name === 'l_out' || op === 'ADD') return 'residual';
  if (name === 'ffn_gate' || name === 'ffn_up' || name.includes('up')) return 'mlp_up';
  if (op === 'GLU' || op === 'UNARY' || /silu|gelu|swiglu|geglu|relu|act/.test(name))
    return 'mlp_act';
  if (name.startsWith('ffn')) return 'mlp_down';
  return 'other';
}

const GROUP_COLOR: Record<Group, string> = {
  norm: hex(OP_COLOR.norm),
  attn_qkv: hex(OP_COLOR.attn_qkv),
  attn_scores: hex(OP_COLOR.attn_scores),
  attn_out: hex(OP_COLOR.attn_out),
  kv: '#c084fc',
  residual: hex(OP_COLOR.residual),
  mlp_up: hex(OP_COLOR.mlp_up),
  mlp_act: hex(OP_COLOR.mlp_act),
  mlp_down: hex(OP_COLOR.mlp_down),
  other: '#64748b',
};

function hex(n: number) {
  return `#${n.toString(16).padStart(6, '0')}`;
}

const OP_CAPTION: Record<string, string> = {
  MUL_MAT: 'matmul',
  MUL_MAT_ID: 'expert matmul',
  ROPE: 'rotary position',
  RMS_NORM: 'RMS norm',
  NORM: 'layer norm',
  MUL: '× weight',
  ADD: '+ residual',
  SOFT_MAX: 'softmax',
  FLASH_ATTN_EXT: 'flash attention',
  GLU: 'gated activation',
  SWIGLU: 'SwiGLU (gate · up)',
  GEGLU: 'GeGLU (gate · up)',
  SILU: 'SiLU',
  GELU: 'GELU',
  UNARY: 'activation',
  SET_ROWS: 'write KV cache',
  CPY: 'copy to KV cache',
  CONT: 'merge heads',
  SCALE: 'scale',
  GET_ROWS: 'gather',
};

/** ggml ne order is innermost first; show outermost first (heads × dim). */
const fmtShape = (s: number[]) => (s.length ? s.slice().reverse().join(' × ') : 'scalar');

// ---- layout of one layer's graph -----------------------------------------------

/** Structure of a layer's graph (same for every layer of a model). */
type Skel = Pick<NodeState, 'i' | 'name' | 'op' | 'shape' | 'src'>;

interface Box {
  n: Skel;
  x: number;
  y: number;
  lane: number;
  /** a second input comes from outside the layer (the residual stream) */
  residualIn: boolean;
}

const BW = 128;
const BH = 54;
const GX = 26;
const GY = 12;
const IN_W = 64;
const LANE_GAP = 30;
/** MAX_COLS: depth columns per lane before a band wraps onto the next lane */
function layoutLayer(nodes: Skel[], MAX_COLS: number) {
  const byI = new Map<number, Skel>();
  for (const n of nodes) byI.set(n.i, n);
  const list = [...byI.values()].sort((a, b) => a.i - b.i);
  // split into attention / MLP bands at the residual add before the MLP
  const split = list.find((n) => n.name === 'ffn_inp');
  const bandOf = (n: Skel) => (split && n.i > split.i ? 1 : 0);
  const depth = new Map<number, number>();
  for (const n of list) {
    let d = 0;
    for (const s of n.src ?? []) {
      const sn = byI.get(s);
      if (sn && bandOf(sn) === bandOf(n)) d = Math.max(d, (depth.get(s) ?? 0) + 1);
    }
    depth.set(n.i, d);
  }
  // lanes: (band, wrap line); each lane is a row of up to MAX_COLS depth columns
  const bandDepth = [0, 0];
  for (const n of list) bandDepth[bandOf(n)] = Math.max(bandDepth[bandOf(n)], depth.get(n.i)! + 1);
  const linesOf = (b: number) => Math.max(1, Math.ceil(bandDepth[b] / MAX_COLS));
  const laneOf = (n: Skel) =>
    (bandOf(n) === 1 ? linesOf(0) : 0) + Math.floor(depth.get(n.i)! / MAX_COLS);
  const nLanes = linesOf(0) + (split ? linesOf(1) : 0);
  const laneRows = new Array<number>(nLanes).fill(1);
  const slot = new Map<string, number>();
  const placed: Array<{ n: Skel; lane: number; col: number; row: number }> = [];
  for (const n of list) {
    const lane = laneOf(n);
    const col = depth.get(n.i)! % MAX_COLS;
    const key = `${lane}:${col}`;
    const row = slot.get(key) ?? 0;
    slot.set(key, row + 1);
    laneRows[lane] = Math.max(laneRows[lane], row + 1);
    placed.push({ n, lane, col, row });
  }
  const laneY: number[] = [];
  let y = 22;
  for (let l = 0; l < nLanes; l++) {
    // a gap with a band title before the MLP band
    if (split && l === linesOf(0)) y += 16;
    laneY.push(y);
    y += laneRows[l] * (BH + GY) + LANE_GAP;
  }
  const boxes: Box[] = placed.map(({ n, lane, col, row }) => {
    const inLayer = (n.src ?? []).filter((s) => byI.has(s)).length;
    return {
      n,
      x: IN_W + GX + col * (BW + GX),
      y: laneY[lane] + row * (BH + GY),
      lane,
      residualIn: (n.op === 'ADD' && inLayer < 2) || inLayer === 0,
    };
  });
  const cols = Math.min(MAX_COLS, Math.max(bandDepth[0], bandDepth[1]));
  const width = IN_W + GX + cols * (BW + GX) + 10;
  const height = y - LANE_GAP + 6;
  const mlpY = split ? laneY[linesOf(0)] : null;
  return { boxes, byI, width, height, top: laneY[0], mlpY };
}

/** Last complete layer structure seen per model, so a layer can be drawn before it runs. */
const skeletons = new Map<string, { base: Skel[]; byLayer: Map<number, Skel[]> }>();

const toSkel = (m: Map<number, NodeState>): Skel[] =>
  [...m.values()]
    .sort((x, y) => x.i - y.i)
    .map(({ i, name, op, shape, src }) => ({ i, name, op, shape, src }));

/**
 * The full graph of a layer, so ops that haven't run yet can be drawn dimmed.
 * Remembered per layer (the last layer can differ, e.g. extra GET_ROWS nodes)
 * with layer 0's graph as the default for layers not seen complete yet.
 */
function useSkeleton(sd: StepDetail, layer: number, layerNodes: NodeState[]): Skel[] {
  const model = useUi((s) => s.vm.model) ?? '';
  return useMemo(() => {
    let cache = skeletons.get(model);
    if (!cache) skeletons.set(model, (cache = { base: [], byLayer: new Map() }));
    const perLayer = new Map<number, Map<number, NodeState>>();
    for (const n of sd.nodes) {
      let m = perLayer.get(n.layer);
      if (!m) perLayer.set(n.layer, (m = new Map()));
      m.set(n.i, n);
    }
    for (const [l, m] of perLayer) {
      const prev = cache.byLayer.get(l);
      if (!prev || m.size >= prev.length) cache.byLayer.set(l, toSkel(m));
    }
    if (!cache.base.length || (perLayer.get(0)?.size ?? 0) >= cache.base.length) {
      const l0 = perLayer.get(0);
      if (l0) cache.base = toSkel(l0);
    }
    const own = cache.byLayer.get(layer) ?? [];
    const cand = own.length >= cache.base.length ? own : cache.base;
    // use what actually ran when it diverges from the remembered graph
    const fits = layerNodes.every((n) => {
      const k = cand.find((c) => c.i === n.i);
      return k && k.name === n.name && k.op === n.op;
    });
    return fits && cand.length >= layerNodes.length ? cand : layerNodes;
  }, [sd.nodes, layer, layerNodes, model]);
}

/** Readable label: llama.cpp leaves the KV-cache stores unnamed. */
function labelOf(n: Skel, byI: Map<number, Skel>): string {
  if (n.op === 'SET_ROWS' || n.name === 'set_rows' || n.name === 'cpy') {
    const src = byI.get(n.src?.[0] ?? -1)?.name ?? '';
    if (src.startsWith('K')) return 'K → cache';
    if (src.startsWith('V')) return 'V → cache';
  }
  return n.name;
}

// ---- block diagram -----------------------------------------------------------

function BlockDiagram({ sd, layer }: { sd: StepDetail; layer: number }) {
  const nodes = useMemo(() => sd.nodes.filter((n) => n.layer === layer), [sd.nodes, layer]);
  const skel = useSkeleton(sd, layer, nodes);
  const hasSkel = skel.length > 0;
  const wrap = useRef<HTMLDivElement>(null);
  const [area, setArea] = useState({ w: 800, h: 300 });
  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setArea({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, [hasSkel]);
  // wrap width that makes the boxes largest in the space available
  const L = useMemo(() => {
    let best = layoutLayer(skel, 7);
    let bestScale = 0;
    for (let c = 3; c <= 12; c++) {
      const l = layoutLayer(skel, c);
      const sc = Math.min(area.w / l.width, area.h / l.height);
      if (sc > bestScale + 1e-3) {
        bestScale = sc;
        best = l;
      }
    }
    return best;
  }, [skel, area]);
  const ran = useMemo(() => new Map(nodes.map((n) => [n.i, n])), [nodes]);
  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(null);
  const lastWall = nodes.reduce((m, n) => Math.max(m, n.wall), -Infinity);
  const reduced = useReducedMotion();
  const maxMs = nodes.reduce((m, n) => Math.max(m, n.ms), 0) || 1;
  const totalMs = nodes.reduce((a, n) => a + n.ms, 0);
  const boxOf = new Map(L.boxes.map((b) => [b.n.i, b]));

  if (!skel.length)
    return (
      <div className="deep-empty">
        No graph-node data for layer {layer} at step {sd.step} yet.
      </div>
    );

  // the op that ran last: its incoming arrows carry the data flow animation
  let hotI = -1;
  if (lastWall > -Infinity)
    for (const n of nodes) if (n.wall === lastWall) hotI = Math.max(hotI, n.i);
  const edge = (key: string, d: string, cls: string, hot: boolean, color: string) => {
    if (!hot)
      return <path key={key} d={d} className={`deep-edge ${cls}`} markerEnd="url(#deep-arrow)" />;
    return (
      <g key={`${key}-${lastWall}`} className="deep-flow">
        <path
          d={d}
          className="deep-edge hot"
          style={{ stroke: color }}
          markerEnd="url(#deep-arrow-hot)"
        />
        {!reduced &&
          [0, 0.33, 0.66].map((t) => (
            <circle key={t} r={3.2} fill={color} className="deep-particle">
              <animateMotion dur="1.1s" begin={`${t * 1.1}s`} repeatCount="indefinite" path={d} />
            </circle>
          ))}
      </g>
    );
  };
  const edges: ReactNode[] = [];
  const hotEdges: ReactNode[] = [];
  for (const b of L.boxes) {
    const live = ran.has(b.n.i);
    const hot = b.n.i === hotI;
    for (const s of b.n.src ?? []) {
      const a = boxOf.get(s);
      if (!a) continue;
      let d: string;
      if (a.lane === b.lane && b.x > a.x) {
        const x1 = a.x + BW;
        const y1 = a.y + BH / 2;
        const x2 = b.x;
        const y2 = b.y + BH / 2;
        const mx = (x1 + x2) / 2;
        d = `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
      } else {
        const x1 = a.x + BW / 2;
        const y1 = a.y + BH;
        const x2 = b.x + BW / 2;
        const y2 = b.y;
        const k = Math.max(30, (y2 - y1) / 2);
        d = `M${x1},${y1} C${x1},${y1 + k} ${x2},${y2 - k} ${x2},${y2}`;
      }
      const g = groupOf(a.n.name, a.n.op);
      (hot ? hotEdges : edges).push(
        edge(
          `${s}-${b.n.i}`,
          d,
          `${g === 'residual' ? 'res' : ''} ${live ? '' : 'pending'}`,
          hot,
          GROUP_COLOR[g],
        ),
      );
    }
    if (b.residualIn) {
      // residual stream / layer input enters from the left rail
      const y2 = b.y + BH / 2;
      (hot ? hotEdges : edges).push(
        edge(
          `in-${b.n.i}`,
          `M${IN_W - 6},${y2} L${b.x},${y2}`,
          `res ${live ? '' : 'pending'}`,
          hot,
          GROUP_COLOR.residual,
        ),
      );
    }
  }

  const hn = hover ? (ran.get(hover.i) ?? null) : null;
  const hs = hover ? L.byI.get(hover.i) : undefined;
  return (
    <div className="deep-diagram" data-testid="block-diagram">
      <div className="deep-caption mono">
        layer {layer} · step {sd.step} · {nodes.length}/{skel.length} ops ran · {totalMs.toFixed(3)}{' '}
        ms
      </div>
      <div ref={wrap} className="deep-svg-wrap">
        <svg
          viewBox={`0 0 ${L.width} ${L.height}`}
          preserveAspectRatio="xMinYMin meet"
          style={{ width: '100%', height: '100%' }}
          onPointerLeave={() => setHover(null)}
        >
          <defs>
            <marker
              id="deep-arrow"
              viewBox="0 0 8 8"
              refX="7"
              refY="4"
              markerWidth="6"
              markerHeight="6"
              orient="auto"
            >
              <path d="M0,0 L8,4 L0,8 z" fill="#4b5470" />
            </marker>
            <marker
              id="deep-arrow-hot"
              viewBox="0 0 8 8"
              refX="7"
              refY="4"
              markerWidth="6"
              markerHeight="6"
              orient="auto"
            >
              <path d="M0,0 L8,4 L0,8 z" fill="#e6e9f2" />
            </marker>
          </defs>
          {/* residual stream rail */}
          <rect
            x={4}
            y={L.top}
            width={IN_W - 12}
            height={L.height - L.top - 6}
            rx={6}
            className="deep-rail"
          />
          <text x={IN_W / 2 - 2} y={L.top + 16} className="deep-rail-t" textAnchor="middle">
            residual
          </text>
          <text x={IN_W / 2 - 2} y={L.top + 28} className="deep-rail-t" textAnchor="middle">
            stream
          </text>
          <text x={IN_W + GX} y={L.top - 8} className="deep-band-t">
            ATTENTION
          </text>
          {L.mlpY !== null && (
            <text x={IN_W + GX} y={L.mlpY - 8} className="deep-band-t">
              MLP
            </text>
          )}
          {edges}
          {hotEdges}
          {L.boxes.map((b) => {
            const n = ran.get(b.n.i);
            const g = groupOf(b.n.name, b.n.op);
            const c = GROUP_COLOR[g];
            const latest = !!n && n.wall === lastWall && lastWall > -Infinity;
            return (
              <g
                key={`${b.n.i}-${n?.wall ?? 'pending'}`}
                transform={`translate(${b.x},${b.y})`}
                className={`deep-node ${n ? 'ran' : 'pending'} ${latest ? 'latest' : ''}`}
                onPointerEnter={(e) => setHover({ i: b.n.i, x: e.clientX, y: e.clientY })}
                onPointerMove={(e) => setHover({ i: b.n.i, x: e.clientX, y: e.clientY })}
              >
                <rect
                  width={BW}
                  height={BH}
                  rx={7}
                  fill={n ? `${c}2e` : '#ffffff05'}
                  stroke={c}
                  strokeWidth={latest ? 2.4 : 1}
                  strokeDasharray={n ? undefined : '3 3'}
                />
                {n && (
                  <rect
                    x={0}
                    y={BH - 4}
                    width={Math.max(2, (BW * n.ms) / maxMs)}
                    height={4}
                    rx={2}
                    fill={c}
                    opacity={0.85}
                  />
                )}
                <text x={8} y={17} className="deep-node-name">
                  {labelOf(b.n, L.byI)}
                </text>
                <text x={8} y={31} className="deep-node-op">
                  {OP_CAPTION[b.n.op] ?? b.n.op.toLowerCase()}
                </text>
                <text x={8} y={44} className="deep-node-shape">
                  {fmtShape(b.n.shape)}
                  {n ? ` · ${n.ms < 0.01 ? n.ms.toFixed(4) : n.ms.toFixed(3)} ms` : ''}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      {hover && hs && (
        <div className="tip" style={{ left: hover.x + 14, top: hover.y + 10 }}>
          <div className="mono">
            {labelOf(hs, L.byI)} · {hs.op}
          </div>
          <div className="mono">shape {fmtShape(hs.shape)}</div>
          <div className="mono">{hn ? `${hn.ms.toFixed(4)} ms` : 'not run yet in this layer'}</div>
          {hn?.stats && (
            <div className="mono" style={{ color: 'var(--muted)' }}>
              min {hn.stats.min.toFixed(3)} · max {hn.stats.max.toFixed(3)}
              <br />
              mean {hn.stats.mean.toFixed(4)} · rms {hn.stats.rms.toFixed(4)}
            </div>
          )}
          {!!hs.src?.length && (
            <div className="mono" style={{ color: 'var(--muted)' }}>
              from{' '}
              {hs.src
                .map((s) => (L.byI.get(s) ? labelOf(L.byI.get(s)!, L.byI) : `#${s}`))
                .join(', ')}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---- logit lens ---------------------------------------------------------------

/** A word without deep data: say why, and offer the nearest words that have it. */
function NoDeepData({ step }: { step: number }) {
  const deep = useUi((s) => s.vm.deepSteps);
  const tokens = useUi((s) => s.vm.responseTokens);
  const word = (st: number) => showToken(tokens.find((t) => t.step === st)?.text ?? `#${st}`);
  let prev: number | undefined;
  let next: number | undefined;
  for (const s of deep) {
    if (s < step) prev = s;
    else if (s > step && next === undefined) next = s;
  }
  const go = (s: number) => setUi({ inspector: { kind: 'token', step: s } });
  return (
    <div className="deep-empty" data-testid="no-deep-data">
      No layer data for “{word(step)}”. While a Deep run plays live, Ollama traces the first few
      words and then every 4th word, to keep generation fast; words generated with Step on are
      always traced. Words that have layer data are underlined in blue in the Response panel.
      {(prev !== undefined || next !== undefined) && (
        <div className="deep-nearest">
          Nearest traced:{' '}
          {prev !== undefined && (
            <button onClick={() => go(prev)}>
              ← “{word(prev)}” (step {prev})
            </button>
          )}
          {next !== undefined && (
            <button onClick={() => go(next)}>
              “{word(next)}” (step {next}) →
            </button>
          )}
        </div>
      )}
      {deep.length === 0 && (
        <div className="deep-nearest">
          This run has no layer data in memory (very long runs drop the oldest).
        </div>
      )}
    </div>
  );
}

function LogitLens({
  sd,
  focus,
  pinned,
}: {
  sd: StepDetail | null;
  focus: number;
  pinned: boolean;
}) {
  const n = useUi((s) => s.vm.layersCount);
  const chosen = useUi((s) => {
    const t = s.vm.responseTokens.find((x) => x.step === sd?.step);
    return t?.id;
  });
  const byLayer = useMemo(() => {
    const m = new Map<number, StepDetail['lens'][number]>();
    for (const l of sd?.lens ?? []) m.set(l.layer, l);
    return m;
  }, [sd]);
  const rows = Math.max(n, byLayer.size);
  const noData = !!sd && sd.lens.length === 0;
  return (
    <div className="deep-lens" data-testid="logit-lens">
      <div className="deep-sub">
        Logit lens <span className="muted">· layer’s best guess</span>
      </div>
      {noData && (
        <div className="deep-lens-none" data-testid="lens-none">
          No lens for this word (step {sd.step}).
        </div>
      )}
      <div className="deep-lens-rows" style={noData ? { opacity: 0.35 } : undefined}>
        {Array.from({ length: rows }, (_, layer) => {
          const l = byLayer.get(layer);
          const top = l?.top[0];
          const hit = top && chosen !== undefined && top.id === chosen;
          return (
            <div
              key={layer}
              className={`deep-lens-row ${layer === focus ? 'focus' : ''}`}
              title={
                l
                  ? `layer ${layer}${l.entropy !== undefined ? ` · entropy ${l.entropy.toFixed(2)}` : ''}\n` +
                    l.top
                      .map((t) => `${showToken(t.text ?? `#${t.id}`)}  ${(t.p * 100).toFixed(1)}%`)
                      .join('\n')
                  : `layer ${layer}`
              }
              onClick={() => setUi({ deepLayer: pinned && layer === focus ? null : layer })}
            >
              <span className="mono deep-lens-l">L{layer}</span>
              <span className="deep-lens-bar">
                {top && (
                  <>
                    <span
                      className="deep-lens-fill"
                      style={{ width: `${Math.max(2, top.p * 100)}%`, opacity: hit ? 0.55 : 0.3 }}
                    />
                    <span className={`mono deep-lens-tok ${hit ? 'hit' : ''}`}>
                      {showToken(top.text ?? `#${top.id}`).slice(0, 14)}
                    </span>
                    <span className="mono deep-lens-p">{(top.p * 100).toFixed(0)}%</span>
                  </>
                )}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---- attention heads heatmap -------------------------------------------------------

function HeadsHeatmap({ sd, layer }: { sd: StepDetail; layer: number }) {
  const entry = useMemo(() => {
    let e: StepDetail['heads'][number] | undefined;
    for (const h of sd.heads) if (h.layer === layer) e = h;
    return e;
  }, [sd.heads, layer]);
  const canvas = useRef<HTMLCanvasElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 400, h: 120 });
  const [hover, setHover] = useState<{ h: number; j: number; x: number; y: number } | null>(null);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const c = canvas.current;
    if (!c || !entry) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.max(1, Math.floor(size.w * dpr));
    c.height = Math.max(1, Math.floor(size.h * dpr));
    const ctx = c.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#0d1017';
    ctx.fillRect(0, 0, size.w, size.h);
    const H = entry.rows.length;
    const N = entry.rows[0]?.length ?? 0;
    if (!H || !N) return;
    const cw = size.w / N;
    const rh = size.h / H;
    for (let h = 0; h < H; h++) {
      const row = entry.rows[h];
      for (let j = 0; j < N; j++) {
        const w = row[j];
        if (w < 0.002) continue;
        const v = Math.sqrt(Math.min(1, w));
        ctx.fillStyle = heat(v);
        ctx.fillRect(j * cw, h * rh, Math.max(cw, 0.75), Math.max(rh - (rh > 4 ? 0.5 : 0), 0.75));
      }
    }
    // KV-head groups (GQA): thin separators
    if (entry.nKvHead && entry.nKvHead > 1 && H % entry.nKvHead === 0) {
      const g = H / entry.nKvHead;
      ctx.fillStyle = '#ffffff22';
      for (let k = 1; k < entry.nKvHead; k++) ctx.fillRect(0, k * g * rh - 0.5, size.w, 1);
    }
  }, [entry, size]);

  const vm = useUi.getState().vm;
  const H = entry?.rows.length ?? 0;
  const N = entry?.rows[0]?.length ?? 0;
  return (
    <div className="deep-heads" data-testid="attn-heads">
      <div className="deep-sub">
        Attention by head{' '}
        <span className="muted">
          · layer {layer}
          {entry ? ` · ${H} heads × ${N} positions` : ''}
          {entry?.nKvHead ? ` · ${entry.nKvHead} KV heads` : ''}
        </span>
      </div>
      <div
        ref={wrap}
        className="deep-heads-canvas"
        onPointerMove={(e) => {
          if (!entry || !wrap.current) return;
          const r = wrap.current.getBoundingClientRect();
          const j = Math.floor(((e.clientX - r.left) / r.width) * N);
          const h = Math.floor(((e.clientY - r.top) / r.height) * H);
          if (h >= 0 && h < H && j >= 0 && j < N) setHover({ h, j, x: e.clientX, y: e.clientY });
        }}
        onPointerLeave={() => setHover(null)}
      >
        {entry ? (
          <canvas ref={canvas} style={{ width: '100%', height: '100%', display: 'block' }} />
        ) : (
          <div className="deep-empty">
            No per-head attention for layer {layer} at step {sd.step}.
          </div>
        )}
      </div>
      {entry && (
        <div className="deep-axis mono">
          <span>{tokenAt(vm, entry.from).slice(0, 10)} ←</span>
          <span>
            positions {entry.from}–{entry.from + N - 1}
          </span>
          <span>→ {tokenAt(vm, entry.from + N - 1).slice(0, 10)}</span>
        </div>
      )}
      {hover && entry && (
        <div className="tip" style={{ left: hover.x + 14, top: hover.y + 10 }}>
          <div className="mono">
            head {hover.h}
            {entry.nKvHead ? ` (KV ${Math.floor(hover.h / (H / entry.nKvHead))})` : ''} → pos{' '}
            {entry.from + hover.j}
          </div>
          <div className="mono">“{tokenAt(vm, entry.from + hover.j)}”</div>
          <div className="mono">{(entry.rows[hover.h][hover.j] * 100).toFixed(2)}%</div>
        </div>
      )}
    </div>
  );
}

/** Dark → indigo → amber → white ramp for v in [0,1]. */
function heat(v: number): string {
  const stops = [
    [13, 16, 23],
    [67, 56, 202],
    [236, 72, 153],
    [251, 191, 36],
    [255, 250, 230],
  ];
  const x = v * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = stops[i];
  const b = stops[i + 1];
  return `rgb(${a.map((c, k) => Math.round(c + (b[k] - c) * f)).join(',')})`;
}

// ---- container ----------------------------------------------------------------------

/**
 * Deep tier: logit lens down the left (click a layer to focus it), the focused
 * layer's real operation graph, and its per-head attention.
 */
export function DeepView() {
  const sd = useDeepStep();
  const pinned = useUi((s) => s.deepLayer);
  // follow the layer that ran most recently unless one is pinned
  const following = useMemo(() => {
    if (!sd) return 0;
    let best = -1;
    let wall = -Infinity;
    for (const n of sd.nodes) {
      if (n.wall >= wall) {
        wall = n.wall;
        best = n.layer;
      }
    }
    if (best < 0) for (const l of sd.lens) best = Math.max(best, l.layer);
    return Math.max(0, best);
  }, [sd]);
  const focus = pinned ?? following;
  const selected = useSelectedStep();
  const selText = useUi((s) => {
    const t = selected === null ? null : s.vm.responseTokens.find((x) => x.step === selected);
    return t ? showToken(t.text) : null;
  });
  const empty = !!sd && !hasDeep(sd);

  return (
    <div className="deep" data-testid="deep-view">
      <LogitLens sd={sd} focus={focus} pinned={pinned !== null} />
      <div className="deep-right">
        {selected !== null && (
          <div className="deep-selected mono" data-testid="deep-selected">
            showing “{selText ?? '…'}” (step {selected})
            <button onClick={() => setUi({ inspector: null })}>back to live</button>
          </div>
        )}
        {empty && sd ? (
          <NoDeepData step={sd.step} />
        ) : sd ? (
          <>
            <BlockDiagram sd={sd} layer={focus} />
            <HeadsHeatmap sd={sd} layer={focus} />
          </>
        ) : (
          <div className="deep-empty">
            Deep data shows up for traced steps. Turn on Step mode, run a prompt, then use “Step op”
            or “Step layer” to walk through one layer’s operations.
          </div>
        )}
        <div className="deep-pin mono">
          {pinned !== null ? (
            <button onClick={() => setUi({ deepLayer: null })}>
              📌 layer {pinned} · follow live
            </button>
          ) : (
            <span className="muted">following the running layer · click a lens row to pin</span>
          )}
        </div>
      </div>
    </div>
  );
}
