import type { Phase } from './engine/types';
import { useUi } from './store';

export const PHASE_COLOR: Record<Phase, string> = {
  idle: '#4a5168',
  loading: '#a78bfa',
  tokenizing: '#22d3ee',
  prefill: '#f59e0b',
  decoding: '#34d399',
  done: '#94a3b8',
  error: '#f87171',
};
export const PHASE_HEX: Record<Phase, number> = Object.fromEntries(
  Object.entries(PHASE_COLOR).map(([k, v]) => [k, parseInt(v.slice(1), 16)]),
) as Record<Phase, number>;

export const PHASE_LABEL: Record<Phase, string> = {
  idle: 'Idle',
  loading: 'Load',
  tokenizing: 'Tokenize',
  prefill: 'Prefill',
  decoding: 'Decode',
  done: 'Done',
  error: 'Error',
};

export const OP_COLOR: Record<string, number> = {
  norm: 0x64748b,
  attn_qkv: 0x60a5fa,
  attn_scores: 0x818cf8,
  attn_out: 0xa78bfa,
  residual: 0xfbbf24,
  mlp_up: 0x34d399,
  mlp_act: 0x2dd4bf,
  mlp_down: 0x22c55e,
};

/** Shared spring for bars and layout motion. */
export const SPRING = { type: 'spring', stiffness: 380, damping: 34, mass: 0.8 } as const;
export const EASE = [0.22, 1, 0.36, 1] as const;

/** True when animations should be replaced by instant state changes. */
export function prefersReducedMotion(): boolean {
  const s = useUi.getState().settings.reducedMotion;
  if (s === 'on') return true;
  if (s === 'off') return false;
  return (
    typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches
  );
}

export function useReducedMotion(): boolean {
  const s = useUi((u) => u.settings.reducedMotion);
  if (s === 'on') return true;
  if (s === 'off') return false;
  return (
    typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches
  );
}

/** Visible-ish rendering of whitespace in token text. */
export function showToken(t: string): string {
  return t.replace(/\n/g, '↵').replace(/\t/g, '⇥').replace(/^ /, '·').replace(/ $/, '·') || '∅';
}

export const fmtMs = (ms: number | null | undefined) =>
  ms == null
    ? '–'
    : ms >= 1000
      ? `${(ms / 1000).toFixed(2)} s`
      : `${ms.toFixed(ms < 10 ? 1 : 0)} ms`;
