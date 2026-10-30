import type { DetailLevel, EventOf, Op, TraceEvent } from 'trace-schema';

export type Phase = 'idle' | 'loading' | 'tokenizing' | 'prefill' | 'decoding' | 'done' | 'error';
export const PHASE_ORDER: Phase[] = [
  'idle',
  'loading',
  'tokenizing',
  'prefill',
  'decoding',
  'done',
];

export interface Candidate {
  id: number;
  text: string;
  p: number;
}

export interface ResponseToken {
  step: number;
  text: string;
  isEos: boolean;
  id?: number;
  topk?: Candidate[];
  entropy?: number;
  /** wall time (ms) when this token appeared on screen */
  wall: number;
}

export interface LayerState {
  idx: number;
  active: boolean;
  lastMs: number;
  norm: number;
  /** wall time of the last activation (for flash/fade), -Infinity if never */
  wall: number;
  step: number;
  /** backend that ran it ("CPU", "CUDA0", …) when the runner reports it */
  device?: string;
}

export interface OpState {
  layer: number;
  op: Op;
  /** occurrence index within the layer (norm appears twice) */
  slot: number;
  ms: number;
  wall: number;
}

export interface NodeState {
  layer: number;
  i: number;
  name: string;
  op: string;
  shape: number[];
  ms: number;
  src?: number[];
  stats?: { min: number; max: number; mean: number; rms: number };
  wall: number;
}

export type SampleStageEvent = EventOf<'decode.sample.stage'>;

export interface StepDetail {
  step: number;
  ops: OpState[];
  residual: Array<{ layer: number; norm: number; delta: number; wall: number }>;
  attn: Array<{
    layer: number;
    head?: number;
    top: Array<{ pos: number; w: number }>;
    wall: number;
  }>;
  kvWrites: Array<{ layer: number; pos: number; wall: number }>;
  samplingStages: Array<SampleStageEvent & { wall: number }>;
  layerMs: Array<{ layer: number; ms: number }>;
  /** deep tier: graph nodes of each layer, in arrival order */
  nodes: NodeState[];
  /** deep tier: logit lens per layer */
  lens: Array<{
    layer: number;
    top: Array<{ id: number; p: number; text?: string }>;
    entropy?: number;
    wall: number;
  }>;
  /** deep tier: per-head attention rows per layer */
  heads: Array<{ layer: number; from: number; nKvHead?: number; rows: number[][]; wall: number }>;
  /** MoE routing per layer (MoE models only) */
  moe: Array<{ layer: number; experts: Array<{ id: number; w: number }>; wall: number }>;
  stepMs?: number;
}

export interface StreamChunkState {
  step: number;
  bytes: number;
  complete: boolean;
  text: string;
  t_ns: number;
  wall: number;
}

export interface Stats {
  prompt_eval_count: number;
  prompt_eval_ms: number;
  eval_count: number;
  eval_ms: number;
  load_ms: number;
  total_ms: number;
}

export interface DerivedStats {
  ttftMs: number | null;
  promptTokPerSec: number | null;
  genTokPerSec: number | null;
  totalMs: number | null;
}

export interface TimelineBand {
  phase: Phase;
  from: number;
  to: number | null;
}

export interface ClockState {
  tp: number;
  t0: number;
  tEnd: number;
  /** user speed × catch-up factor */
  speed: number;
  /** measured playback dilation: trace-time / wall-time */
  dilation: number;
  lagMs: number;
  catchingUp: boolean;
  degraded: boolean;
  dropped: number;
  paused: boolean;
  live: boolean;
  stepping: string | null;
}

export interface ViewModel {
  req: string | null;
  phase: Phase;
  model: string | null;
  detail: DetailLevel | null;
  layersCount: number;
  ctx: number;
  vramMb: number;
  options: Record<string, unknown>;
  capabilities: string[];
  promptTokens: Array<{ id: number; text: string }>;
  /** wall time when tokenize.done was shown (drives staggered chip reveal) */
  tokenizeWall: number;
  responseTokens: ResponseToken[];
  responseText: string;
  currentStep: number;
  prefill: { batch: number; of: number; progress: number; from: number; to: number; wall: number };
  kv: { used: number; total: number };
  layers: LayerState[];
  sampling: {
    step: number;
    topk: Candidate[];
    chosen?: number;
    temp: number;
    top_p: number;
    top_k: number;
    entropy: number;
    method: string;
    wall: number;
  };
  stats: Stats | null;
  derived: DerivedStats;
  timeline: TimelineBand[];
  phaseDurations: Partial<Record<Phase, number>>;
  error: string | null;
  stepDetail: StepDetail | null;
  /** recent per-step details for the inspector (bounded) */
  steps: Record<number, StepDetail>;
  streamChunks: StreamChunkState[];
  lastEvent: TraceEvent | null;
  recent: TraceEvent[];
  eventsApplied: number;
  clock: ClockState;
  version: number;
}
