import { motion } from 'motion/react';
import type { Phase } from '../engine/types';
import { useUi } from '../store';
import { fmtMs, PHASE_COLOR, PHASE_LABEL, useReducedMotion } from '../theme';

const NODES: Phase[] = ['loading', 'tokenizing', 'prefill', 'decoding', 'done'];
const ORDER: Record<Phase, number> = {
  idle: -1,
  loading: 0,
  tokenizing: 1,
  prefill: 2,
  decoding: 3,
  done: 4,
  error: 5,
};

export function PhasePipeline() {
  const phase = useUi((s) => s.vm.phase);
  const durations = useUi((s) => s.vm.phaseDurations);
  const model = useUi((s) => s.vm.model);
  const vram = useUi((s) => s.vm.vramMb);
  const layers = useUi((s) => s.vm.layersCount);
  const nPrompt = useUi((s) => s.vm.promptTokens.length);
  const prefill = useUi((s) => s.vm.prefill);
  const step = useUi((s) => s.vm.currentStep);
  const kv = useUi((s) => s.vm.kv);
  const nResp = useUi((s) => s.vm.responseTokens.length);
  const error = useUi((s) => s.vm.error);
  const totalMs = useUi((s) => s.vm.stats?.total_ms ?? null);
  const reduced = useReducedMotion();

  const cur = ORDER[phase];
  const H = 96;
  const top = 34;

  const sub = (p: Phase): string => {
    switch (p) {
      case 'loading':
        return model
          ? `${model}${layers ? ` · ${layers} layers` : ''}${vram ? ` · ${(vram / 1024).toFixed(1)} GB` : ''}`
          : '';
      case 'tokenizing':
        return nPrompt ? `${nPrompt} tokens` : '';
      case 'prefill':
        return prefill.of ? `batch ${prefill.batch + 1}/${prefill.of} · ${prefill.to} tok` : '';
      case 'decoding':
        return step >= 0 ? `step ${step} · kv ${kv.used}/${kv.total || '?'}` : '';
      case 'done':
        return phase === 'done' ? `${nResp} tokens · ${fmtMs(totalMs)}` : '';
      default:
        return '';
    }
  };

  return (
    <div className="pipeline panel-body" data-testid="pipeline" data-phase={phase}>
      <svg viewBox={`0 0 220 ${top + H * NODES.length}`} preserveAspectRatio="xMidYMin meet">
        {NODES.slice(0, -1).map((p, i) => {
          const y1 = top + i * H + 22;
          const y2 = top + (i + 1) * H - 22;
          const done = cur > i;
          const flowing = cur === i + 1 || (cur === i && phase !== 'done');
          return (
            <g key={`edge-${p}`}>
              <line
                x1={30}
                x2={30}
                y1={y1}
                y2={y2}
                stroke={done ? PHASE_COLOR[NODES[i]] : '#2a3044'}
                strokeWidth={2}
              />
              {flowing && !reduced && (
                <motion.circle
                  cx={30}
                  r={3}
                  fill={PHASE_COLOR[p]}
                  initial={{ cy: y1, opacity: 0 }}
                  animate={{ cy: [y1, y2], opacity: [0, 1, 0] }}
                  transition={{ duration: 0.9, repeat: Infinity, ease: 'easeInOut' }}
                />
              )}
            </g>
          );
        })}
        {NODES.map((p, i) => {
          const y = top + i * H;
          const isCur = cur === i && phase !== 'done';
          const isDone = cur > i || (p === 'done' && phase === 'done');
          const color =
            phase === 'error' && cur === 5 && i === NODES.length - 1
              ? PHASE_COLOR.error
              : PHASE_COLOR[p];
          const label = phase === 'error' && i === NODES.length - 1 ? 'Error' : PHASE_LABEL[p];
          const d = durations[p];
          return (
            <g
              key={p}
              data-testid={`node-${p}`}
              data-state={isCur ? 'active' : isDone ? 'done' : 'pending'}
            >
              {isCur && !reduced && (
                <motion.circle
                  cx={30}
                  cy={y}
                  r={14}
                  fill="none"
                  stroke={color}
                  strokeWidth={2}
                  initial={{ scale: 1, opacity: 0.7 }}
                  animate={{ scale: [1, 1.7], opacity: [0.7, 0] }}
                  transition={{ duration: 1.2, repeat: Infinity, ease: 'easeOut' }}
                  style={{ transformOrigin: `30px ${y}px` }}
                />
              )}
              <motion.circle
                cx={30}
                cy={y}
                r={12}
                initial={false}
                animate={{
                  fill: isDone || isCur ? color : '#141824',
                  stroke: isDone || isCur ? color : '#2f364b',
                }}
                transition={{ duration: reduced ? 0 : 0.3 }}
                strokeWidth={2}
              />
              {isDone && (
                <path
                  d={`M ${24} ${y} l 4 4 l 8 -8`}
                  stroke="#0b0d12"
                  strokeWidth={2.5}
                  fill="none"
                  strokeLinecap="round"
                />
              )}
              <text
                x={54}
                y={y - 2}
                className="node-label"
                style={{ fill: isCur ? color : isDone ? '#e6e9f2' : '#6b7390' }}
              >
                {label}
                {d != null && p !== 'done' && (
                  <tspan className="node-sub" dx={8}>
                    {fmtMs(d)}
                  </tspan>
                )}
              </text>
              <text x={54} y={y + 15} className="node-sub">
                {truncate(sub(p), 26)}
              </text>
              {p === 'done' && error && (
                <text x={54} y={y + 30} className="node-sub" style={{ fill: PHASE_COLOR.error }}>
                  {truncate(error, 28)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

const truncate = (s: string, n: number) => (s.length > n ? s.slice(0, n - 1) + '…' : s);
