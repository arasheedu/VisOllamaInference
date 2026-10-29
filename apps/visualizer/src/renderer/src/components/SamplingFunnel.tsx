import { useMemo } from 'react';
import { motion } from 'motion/react';
import { SAMPLE_STAGES, type SampleStage } from 'trace-schema';
import { session } from '../session';
import { setUi, useUi } from '../store';
import { tokenizerFor } from '../tokenizer/merges';
import { SPRING, showToken, useReducedMotion } from '../theme';
import type { SampleStageEvent, StepDetail } from '../engine/types';

const N = 64;
const W = 340;
const H_FULL = 92;
const H_COMPACT = 22;

const STAGE_LABEL: Record<SampleStage, string> = {
  raw: 'raw logits',
  rep_penalty: 'repetition penalty',
  temperature: 'temperature',
  top_k: 'top-k',
  top_p: 'top-p (nucleus)',
  min_p: 'min-p',
  softmax: 'softmax',
  draw: 'draw',
};

type Cand = { id: number; v: number };

function softmax(vals: number[]): number[] {
  if (!vals.length) return [];
  const m = Math.max(...vals);
  const e = vals.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

const entropy = (p: number[]) => -p.reduce((s, x) => s + (x > 0 ? x * Math.log(x) : 0), 0);

function useTextFor() {
  const model = useUi((s) => s.vm.model);
  const topk = useUi((s) => s.vm.sampling.topk);
  const resp = useUi((s) => s.vm.responseTokens);
  return useMemo(() => {
    const known = new Map<number, string>();
    for (const t of resp) for (const c of t.topk ?? []) known.set(c.id, c.text);
    for (const c of topk) known.set(c.id, c.text);
    const tok = model
      ? tokenizerFor((model.startsWith('mock-') ? 'mock:' : 'ollama:') + model)
      : undefined;
    return (id: number) => {
      const k = known.get(id);
      if (k !== undefined) return k;
      if (tok) return tok.piece(id);
      return `#${id}`;
    };
  }, [model, topk, resp]);
}

/** Pick the step detail with sampling stages (current, else latest with any). */
function useStageSource(): StepDetail | null {
  const sd = useUi((s) => s.vm.stepDetail);
  const steps = useUi((s) => s.vm.steps);
  return useMemo(() => {
    if (sd && sd.samplingStages.length) return sd;
    let best: StepDetail | null = null;
    for (const k of Object.keys(steps)) {
      const d = steps[Number(k)];
      if (d.samplingStages.length && (!best || d.step > best.step)) best = d;
    }
    return best ?? sd;
  }, [sd, steps]);
}

export function SamplingFunnel() {
  const detail = useStageSource();
  const rm = useReducedMotion();
  const textFor = useTextFor();
  const resp = useUi((s) => s.vm.responseTokens);
  const temp = useUi((s) => s.vm.sampling.temp);
  const topP = useUi((s) => s.vm.sampling.top_p);
  const options = useUi((s) => s.vm.options);
  const T = temp || Number(options.temperature) || 0;
  const P = topP || Number(options.top_p) || 0;

  const stages = detail?.samplingStages ?? [];
  const byStage = new Map<SampleStage, SampleStageEvent>();
  for (const s of stages) byStage.set(s.stage, s);
  const current = stages.length ? stages[stages.length - 1].stage : null;
  const step = detail?.step ?? -1;

  const outputCounts = useMemo(() => {
    const m = new Map<number, number>();
    for (const t of resp)
      if (t.id !== undefined && t.step < step) m.set(t.id, (m.get(t.id) ?? 0) + 1);
    return m;
  }, [resp, step]);

  if (!detail || !stages.length) {
    return (
      <div className="funnel" style={{ color: 'var(--muted)', padding: 12 }}>
        Waiting for sampler stages… (Full detail)
      </div>
    );
  }
  const trans = rm ? { duration: 0 } : SPRING;
  const raw = byStage.get('raw');

  return (
    <div className="funnel">
      <div
        style={{
          fontFamily: 'var(--mono)',
          fontSize: 11,
          color: 'var(--muted)',
          padding: '2px 2px 4px',
        }}
      >
        step {step}
      </div>
      {SAMPLE_STAGES.map((name, idx) => {
        const ev = byStage.get(name);
        // a stage not (yet) seen: absent if a later stage exists, otherwise pending
        const laterSeen = SAMPLE_STAGES.slice(idx + 1).some((s) => byStage.has(s));
        if (!ev) {
          return (
            <div
              key={name}
              className={`stage${laterSeen ? ' absent' : ''}`}
              style={laterSeen ? undefined : { opacity: 0.5 }}
            >
              <header>
                <span>{STAGE_LABEL[name]}</span>
                <span>{laterSeen ? 'not configured' : '…'}</span>
              </header>
            </div>
          );
        }
        const isCur = name === current;
        const prevIdx = SAMPLE_STAGES.slice(0, idx)
          .reverse()
          .find((s) => byStage.has(s));
        const prev = prevIdx ? byStage.get(prevIdx) : undefined;
        return (
          <div key={name} className={`stage${isCur ? ' current' : ''}`}>
            <header
              onClick={() => {
                session.pause();
                setUi({ inspector: { kind: 'stage', step, stage: name } });
              }}
            >
              <span>{STAGE_LABEL[name]}</span>
              <span>{summary(ev, T, P)}</span>
            </header>
            <StageBody
              ev={ev}
              prev={prev}
              raw={raw}
              expanded={isCur}
              trans={trans}
              textFor={textFor}
              outputCounts={outputCounts}
              T={T}
              P={P}
            />
          </div>
        );
      })}
    </div>
  );
}

function summary(ev: SampleStageEvent, T: number, P: number): string {
  switch (ev.stage) {
    case 'temperature':
      return `T = ${T || '?'}`;
    case 'top_k':
      return `keep ${ev.cutoff ?? '?'}`;
    case 'top_p':
      return `p = ${P || '?'} → keep ${ev.cutoff ?? '?'}`;
    case 'min_p':
      return `keep ${ev.cutoff ?? '?'}`;
    case 'softmax':
      return `${ev.candidates.length} candidates`;
    case 'draw':
      return `u = ${ev.draw?.toFixed(3) ?? '?'}`;
    default:
      return `${ev.candidates.length} candidates`;
  }
}

interface BodyProps {
  ev: SampleStageEvent;
  prev?: SampleStageEvent;
  raw?: SampleStageEvent;
  expanded: boolean;
  trans: object;
  textFor: (id: number) => string;
  outputCounts: Map<number, number>;
  T: number;
  P: number;
}

function StageBody(p: BodyProps) {
  const { ev, expanded, trans } = p;
  const H = expanded ? H_FULL : H_COMPACT;
  const cands = ev.candidates.slice(0, N);
  const isProb = ev.stage === 'softmax' || ev.stage === 'draw';
  const vals = cands.map((c) => c.v);
  const lo = isProb ? 0 : Math.min(...vals);
  const hi = Math.max(...vals, lo + 1e-9);
  const norm = (v: number) => (isProb ? v / hi : 0.06 + (0.94 * (v - lo)) / (hi - lo || 1));
  const bw = W / Math.max(cands.length, 1);
  const cutoff = ev.cutoff;
  const pruneStage = ev.stage === 'top_k' || ev.stage === 'top_p' || ev.stage === 'min_p';

  if (ev.stage === 'draw') return <DrawBody {...p} H={H} />;

  // previous-stage outline (to show stretch / shrink)
  const prevMap = new Map<number, number>();
  if (p.prev && (ev.stage === 'temperature' || ev.stage === 'rep_penalty')) {
    const pv = p.prev.candidates.slice(0, N).map((c) => c.v);
    const plo = Math.min(...pv);
    const phi = Math.max(...pv, plo + 1e-9);
    for (const c of p.prev.candidates.slice(0, N))
      prevMap.set(c.id, 0.06 + (0.94 * (c.v - plo)) / (phi - plo || 1));
  }

  const curve = ev.stage === 'top_p' && expanded ? cumulativeCurve(ev, p.prev) : null;

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} height={H} preserveAspectRatio="none">
        {cands.map((c, i) => {
          const h = norm(c.v) * (H - 4);
          const excluded = pruneStage && cutoff !== undefined && i >= cutoff;
          const inOut = ev.stage === 'rep_penalty' && p.outputCounts.has(c.id);
          const fill = inOut
            ? '#f59e0b'
            : excluded
              ? '#2a3045'
              : i === 0 && isProb
                ? '#34d399'
                : '#5b6fb0';
          const ph = prevMap.get(c.id);
          return (
            <g key={c.id}>
              {ph !== undefined && expanded && (
                <rect
                  x={i * bw + 0.5}
                  width={Math.max(bw - 1, 0.5)}
                  y={H - ph * (H - 4)}
                  height={ph * (H - 4)}
                  fill="none"
                  stroke="#8a93a8"
                  strokeOpacity={0.35}
                  strokeWidth={0.6}
                />
              )}
              <motion.rect
                x={i * bw + 0.5}
                width={Math.max(bw - 1, 0.5)}
                initial={false}
                animate={{ y: H - h, height: h, opacity: excluded ? 0.25 : 1 }}
                transition={trans}
                fill={fill}
              >
                <title>{`${showToken(p.textFor(c.id))}  ${c.v.toFixed(4)}`}</title>
              </motion.rect>
            </g>
          );
        })}
        {pruneStage && cutoff !== undefined && (
          <motion.line
            y1={0}
            y2={H}
            stroke="#f87171"
            strokeWidth={1.5}
            strokeDasharray="3 2"
            initial={{ x1: W, x2: W }}
            animate={{ x1: cutoff * bw, x2: cutoff * bw }}
            transition={p.trans}
          />
        )}
        {curve && (
          <>
            <line
              x1={0}
              x2={W}
              y1={H - p.P * (H - 4)}
              y2={H - p.P * (H - 4)}
              stroke="#fbbf24"
              strokeOpacity={0.5}
              strokeDasharray="2 3"
            />
            <motion.polyline
              points={curve.map((c, i) => `${(i + 0.5) * bw},${H - c * (H - 4)}`).join(' ')}
              fill="none"
              stroke="#fbbf24"
              strokeWidth={1.5}
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={
                (p.trans as { duration?: number }).duration === 0
                  ? { duration: 0 }
                  : { duration: 0.6 }
              }
            />
          </>
        )}
        {ev.stage === 'raw' && expanded && <Sparkline cands={ev.candidates} H={H} />}
      </svg>
      {expanded && <StageFooter {...p} cands={cands} />}
    </div>
  );
}

function cumulativeCurve(ev: SampleStageEvent, prev?: SampleStageEvent): number[] {
  // softmax over the candidates kept by top-k (prev stage cutoff), cumulative
  const keep = prev?.stage === 'top_k' && prev.cutoff ? prev.cutoff : ev.candidates.length;
  const probs = softmax(ev.candidates.slice(0, Math.min(keep, N)).map((c) => c.v));
  let cum = 0;
  return probs.map((x) => (cum += x));
}

function Sparkline({ cands, H }: { cands: Cand[]; H: number }) {
  // full distribution shape (all candidates, softmaxed), drawn faintly on top
  const p = softmax(cands.map((c) => c.v));
  const max = Math.max(...p, 1e-9);
  const pts = p
    .map((x, i) => `${(i / Math.max(p.length - 1, 1)) * W},${H - 2 - (x / max) * (H * 0.45)}`)
    .join(' ');
  return <polyline points={pts} fill="none" stroke="#22d3ee" strokeOpacity={0.7} strokeWidth={1} />;
}

function StageFooter(p: BodyProps & { cands: Cand[] }) {
  const { ev, cands } = p;
  const style = {
    fontFamily: 'var(--mono)',
    fontSize: 11,
    color: 'var(--muted)',
    padding: '2px 8px 6px',
  } as const;
  const top = cands
    .slice(0, 3)
    .map((c) => showToken(p.textFor(c.id)))
    .join('  ');
  switch (ev.stage) {
    case 'raw':
      return (
        <div style={style}>
          top: {top} · range {fmt(cands.at(-1)?.v)}…{fmt(cands[0]?.v)}
        </div>
      );
    case 'rep_penalty': {
      const hits = cands.filter((c) => p.outputCounts.has(c.id));
      return (
        <div style={style}>
          {hits.length === 0
            ? 'no candidates already in output'
            : hits.slice(0, 4).map((c) => {
                const before = p.raw?.candidates.find((r) => r.id === c.id)?.v;
                return (
                  <span key={c.id} style={{ marginRight: 10, color: '#f59e0b' }}>
                    {showToken(p.textFor(c.id))} ×{p.outputCounts.get(c.id)} in output
                    {before !== undefined ? ` (${fmt(before)}→${fmt(c.v)})` : ''}
                  </span>
                );
              })}
        </div>
      );
    }
    case 'temperature':
      return (
        <div style={style}>
          p ∝ exp(zᵢ / T), T = <b style={{ color: 'var(--text)' }}>{p.T || '?'}</b>{' '}
          {p.T && p.T < 1 ? '(sharper)' : p.T > 1 ? '(flatter)' : ''}
        </div>
      );
    case 'top_k':
    case 'min_p':
      return (
        <div style={style}>
          kept {ev.cutoff ?? '?'} of {ev.candidates.length}
        </div>
      );
    case 'top_p':
      return (
        <div style={style}>
          smallest set with Σp ≥ {p.P || '?'} → kept {ev.cutoff ?? '?'}
        </div>
      );
    case 'softmax': {
      const probs = cands.map((c) => c.v);
      const H = entropy(probs);
      const maxH = Math.log(Math.max(probs.length, 2));
      return (
        <div style={{ ...style, display: 'flex', alignItems: 'center', gap: 8 }}>
          <span>entropy {H.toFixed(2)} nats</span>
          <div
            style={{
              flex: 1,
              height: 6,
              background: '#1a1f2c',
              borderRadius: 3,
              overflow: 'hidden',
            }}
          >
            <motion.div
              initial={false}
              animate={{ width: `${Math.min(100, (H / maxH) * 100)}%` }}
              transition={p.trans}
              style={{ height: '100%', background: H / maxH > 0.5 ? '#f59e0b' : '#34d399' }}
            />
          </div>
          <span>
            top {showToken(p.textFor(cands[0]?.id ?? -1))} {((cands[0]?.v ?? 0) * 100).toFixed(1)}%
          </span>
        </div>
      );
    }
    default:
      return null;
  }
}

function DrawBody(p: BodyProps & { H: number }) {
  const { ev, expanded, trans } = p;
  const cands = ev.candidates;
  const chosenIdx = ev.cutoff ?? 0;
  const chosen = cands[chosenIdx];
  const H = expanded ? 34 : H_COMPACT;
  let x = 0;
  const total = cands.reduce((s, c) => s + c.v, 0) || 1;
  const segs = cands.map((c, i) => {
    const w = (c.v / total) * W;
    const s = { x, w, i, c };
    x += w;
    return s;
  });
  const u = ev.draw ?? 0;
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} height={H} preserveAspectRatio="none">
        {segs.map((s) => (
          <rect
            key={s.c.id}
            x={s.x}
            width={Math.max(s.w - 0.5, 0.3)}
            y={6}
            height={H - 10}
            fill={s.i === chosenIdx ? '#34d399' : s.i % 2 ? '#3b4a7a' : '#4b5c94'}
          >
            <title>{`${showToken(p.textFor(s.c.id))}  p=${s.c.v.toFixed(4)}`}</title>
          </rect>
        ))}
        <motion.g
          initial={{ x: 0, opacity: 0 }}
          animate={{ x: (u / total) * W, opacity: 1 }}
          transition={trans}
        >
          <polygon points="-5,0 5,0 0,7" fill="#fff" />
          <line x1={0} x2={0} y1={6} y2={H - 4} stroke="#fff" strokeWidth={1.5} />
        </motion.g>
      </svg>
      {expanded && chosen && (
        <div
          style={{
            fontFamily: 'var(--mono)',
            fontSize: 12,
            padding: '2px 8px 6px',
            color: 'var(--muted)',
          }}
        >
          u = {u.toFixed(4)} → <b style={{ color: '#34d399' }}>{showToken(p.textFor(chosen.id))}</b>{' '}
          (p = {(chosen.v * 100).toFixed(1)}%, rank {chosenIdx + 1})
        </div>
      )}
    </div>
  );
}

const fmt = (v: number | undefined) => (v === undefined ? '?' : v.toFixed(2));
