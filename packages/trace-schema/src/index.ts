import { z } from 'zod';

/**
 * Trace event schema, version 1.
 *
 * Every event is one line of NDJSON. The Go emitter in `ollama-patch/` mirrors
 * these shapes by hand, so treat any change here as a breaking change.
 */
export const SCHEMA_VERSION = 1 as const;

const base = {
  v: z.literal(SCHEMA_VERSION),
  req: z.string().min(1),
  seq: z.number().int().nonnegative(),
  t_ns: z.number().nonnegative(),
};

const tokenRef = z.object({ id: z.number().int(), text: z.string() });
const candidate = z.object({ id: z.number().int(), text: z.string(), p: z.number() });

export const DETAIL_LEVELS = ['basic', 'layers', 'full', 'deep'] as const;
export const DetailLevel = z.enum(DETAIL_LEVELS);
export type DetailLevel = z.infer<typeof DetailLevel>;

export const OPS = [
  'norm',
  'attn_qkv',
  'attn_scores',
  'attn_out',
  'mlp_up',
  'mlp_act',
  'mlp_down',
] as const;
export type Op = (typeof OPS)[number];

export const SAMPLE_STAGES = [
  'raw',
  'rep_penalty',
  'temperature',
  'top_k',
  'top_p',
  'min_p',
  'softmax',
  'draw',
] as const;
/** Extra llama.cpp sampler names (real chains); any other string is also accepted. */
export const EXTRA_SAMPLE_STAGES = ['dry', 'typical_p', 'xtc', 'top_n_sigma'] as const;
export type SampleStage =
  (typeof SAMPLE_STAGES)[number] | (typeof EXTRA_SAMPLE_STAGES)[number] | (string & {});

const ev = <T extends string, S extends z.ZodRawShape>(type: T, shape: S) =>
  z.object({ ...base, type: z.literal(type), ...shape });

export const RequestReceived = ev('request.received', {
  model: z.string(),
  prompt_chars: z.number().int().nonnegative(),
  options: z.record(z.string(), z.unknown()).default({}),
  detail: DetailLevel.optional(),
  /** Features the emitter could not provide, e.g. ["no_residual"]. */
  capabilities: z.array(z.string()).optional(),
});
export const ModelLoad = ev('model.load', {
  phase: z.enum(['start', 'done']),
  layers: z.number().int().nonnegative(),
  ctx: z.number().int().nonnegative(),
  vram_mb: z.number().nonnegative(),
  ms: z.number().nonnegative().optional(),
});
export const TokenizeDone = ev('tokenize.done', {
  n: z.number().int().nonnegative(),
  tokens: z.array(tokenRef),
});
export const PrefillBatch = ev('prefill.batch', {
  batch_i: z.number().int().nonnegative(),
  of: z.number().int().positive(),
  from: z.number().int().nonnegative(),
  to: z.number().int().nonnegative(),
  ms: z.number().nonnegative(),
});
export const PrefillDone = ev('prefill.done', {
  n_tokens: z.number().int().nonnegative(),
  ms: z.number().nonnegative(),
});
export const DecodeStepStart = ev('decode.step.start', {
  step: z.number().int().nonnegative(),
  kv_used: z.number().int().nonnegative(),
  kv_total: z.number().int().nonnegative(),
});
export const DecodeLayer = ev('decode.layer', {
  step: z.number().int().nonnegative(),
  layer: z.number().int().nonnegative(),
  ms: z.number().nonnegative(),
  norm: z.number().optional(),
  /** backend that ran the layer, e.g. "CPU", "CUDA0" */
  device: z.string().optional(),
});
export const DecodeLogits = ev('decode.logits', {
  step: z.number().int().nonnegative(),
  topk: z.array(candidate),
  entropy: z.number(),
});
export const DecodeSample = ev('decode.sample', {
  step: z.number().int().nonnegative(),
  chosen_id: z.number().int(),
  temp: z.number(),
  top_p: z.number(),
  top_k: z.number().int(),
  method: z.string(),
});
export const DecodeToken = ev('decode.token', {
  step: z.number().int().nonnegative(),
  text: z.string(),
  is_eos: z.boolean(),
});
export const RequestDone = ev('request.done', {
  stats: z.object({
    prompt_eval_count: z.number(),
    prompt_eval_ms: z.number(),
    eval_count: z.number(),
    eval_ms: z.number(),
    load_ms: z.number(),
    total_ms: z.number(),
  }),
});
export const RequestError = ev('request.error', { message: z.string() });

// ---- Detailed tier --------------------------------------------------------

export const TokenizeMerge = ev('tokenize.merge', {
  step: z.number().int().nonnegative(),
  span: z.tuple([z.number().int(), z.number().int()]),
  pieces: z.array(z.string()),
  result_id: z.number().int(),
});
export const PrefillLayer = ev('prefill.layer', {
  batch_i: z.number().int().nonnegative(),
  layer: z.number().int().nonnegative(),
  ms: z.number().nonnegative(),
});
export const DecodeOp = ev('decode.op', {
  step: z.number().int().nonnegative(),
  layer: z.number().int().nonnegative(),
  op: z.enum(OPS),
  ms: z.number().nonnegative(),
});
export const DecodeResidual = ev('decode.residual', {
  step: z.number().int().nonnegative(),
  layer: z.number().int().nonnegative(),
  norm: z.number(),
  delta_norm: z.number(),
});
export const DecodeAttn = ev('decode.attn', {
  step: z.number().int().nonnegative(),
  layer: z.number().int().nonnegative(),
  head: z.number().int().nonnegative().optional(),
  top: z.array(z.object({ pos: z.number().int().nonnegative(), w: z.number() })),
});
export const DecodeKvWrite = ev('decode.kv.write', {
  step: z.number().int().nonnegative(),
  layer: z.number().int().nonnegative(),
  pos: z.number().int().nonnegative(),
});
export const DecodeSampleStage = ev('decode.sample.stage', {
  step: z.number().int().nonnegative(),
  /** stages arrive in the runner's real chain order; unknown names allowed */
  stage: z.string().min(1),
  candidates: z.array(z.object({ id: z.number().int(), v: z.number() })),
  cutoff: z.number().optional(),
  draw: z.number().optional(),
});
export const StreamChunk = ev('stream.chunk', {
  step: z.number().int().nonnegative(),
  bytes: z.number().int().nonnegative(),
  utf8_complete: z.boolean(),
  text: z.string(),
});
/** Mixture-of-experts routing for one layer (MoE models only). */
export const DecodeMoe = ev('decode.moe', {
  step: z.number().int().nonnegative(),
  layer: z.number().int().nonnegative(),
  experts: z.array(z.object({ id: z.number().int().nonnegative(), w: z.number() })),
});
// ---- Deep tier (step mode; see docs/deep-tier.md) -------------------------

/** One graph node of a decode layer, in execution order. */
export const DecodeNode = ev('decode.node', {
  step: z.number().int().nonnegative(),
  layer: z.number().int().nonnegative(),
  /** index within the layer, execution order */
  i: z.number().int().nonnegative(),
  /** llama.cpp tensor name without the -<layer> suffix, e.g. "Qcur" */
  name: z.string().min(1),
  /** ggml op, e.g. "MUL_MAT" */
  op: z.string().min(1),
  shape: z.array(z.number().int().nonnegative()),
  ms: z.number().nonnegative(),
  /** indices (i) of sources that are nodes of this layer */
  src: z.array(z.number().int().nonnegative()).optional(),
  stats: z
    .object({ min: z.number(), max: z.number(), mean: z.number(), rms: z.number() })
    .optional(),
});
/** Logit lens: the residual after `layer` projected through the output head. */
export const DecodeLens = ev('decode.lens', {
  step: z.number().int().nonnegative(),
  layer: z.number().int().nonnegative(),
  top: z.array(z.object({ id: z.number().int(), p: z.number(), text: z.string().optional() })),
  entropy: z.number().optional(),
});
/** Per-head attention rows: rows[h][j] = weight of head h on KV position from + j. */
export const DecodeAttnHeads = ev('decode.attn.heads', {
  step: z.number().int().nonnegative(),
  layer: z.number().int().nonnegative(),
  from: z.number().int().nonnegative(),
  n_kv_head: z.number().int().positive().optional(),
  rows: z.array(z.array(z.number())),
});

/** Emitted by the runner when a step-mode wait expired without an ack. */
export const StepTimeout = ev('step.timeout', {
  step: z.number().int().nonnegative(),
  granularity: z.enum(['token', 'layer', 'sample']),
});

export const TraceEvent = z.discriminatedUnion('type', [
  RequestReceived,
  ModelLoad,
  TokenizeDone,
  PrefillBatch,
  PrefillDone,
  DecodeStepStart,
  DecodeLayer,
  DecodeLogits,
  DecodeSample,
  DecodeToken,
  RequestDone,
  RequestError,
  TokenizeMerge,
  PrefillLayer,
  DecodeOp,
  DecodeResidual,
  DecodeAttn,
  DecodeKvWrite,
  DecodeSampleStage,
  StreamChunk,
  StepTimeout,
  DecodeMoe,
  DecodeNode,
  DecodeLens,
  DecodeAttnHeads,
]);
export type TraceEvent = z.infer<typeof TraceEvent>;
export type EventType = TraceEvent['type'];
export type EventOf<T extends EventType> = Extract<TraceEvent, { type: T }>;

export const EVENT_TYPES = TraceEvent.options.map((o) => o.shape.type.value) as EventType[];

export type Priority = 'critical' | 'droppable';

/** Ring-buffer / emitter priority. Critical events are never dropped. */
export const PRIORITY: Record<EventType, Priority> = {
  'request.received': 'critical',
  'model.load': 'critical',
  'tokenize.done': 'critical',
  'prefill.batch': 'critical',
  'prefill.done': 'critical',
  'decode.step.start': 'critical',
  'decode.layer': 'droppable',
  'decode.logits': 'critical',
  'decode.sample': 'critical',
  'decode.token': 'critical',
  'request.done': 'critical',
  'request.error': 'critical',
  'tokenize.merge': 'droppable',
  'prefill.layer': 'droppable',
  'decode.op': 'droppable',
  'decode.residual': 'droppable',
  'decode.attn': 'droppable',
  'decode.kv.write': 'droppable',
  'decode.sample.stage': 'critical',
  'stream.chunk': 'droppable',
  'step.timeout': 'critical',
  'decode.moe': 'droppable',
  'decode.node': 'droppable',
  'decode.lens': 'droppable',
  'decode.attn.heads': 'droppable',
};

/** Minimum detail level at which each event type is emitted / consumed. */
export const EVENT_DETAIL: Record<EventType, DetailLevel> = {
  'request.received': 'basic',
  'model.load': 'basic',
  'tokenize.done': 'basic',
  'prefill.batch': 'basic',
  'prefill.done': 'basic',
  'decode.step.start': 'basic',
  'decode.layer': 'layers',
  'decode.logits': 'basic',
  'decode.sample': 'basic',
  'decode.token': 'basic',
  'request.done': 'basic',
  'request.error': 'basic',
  'tokenize.merge': 'full',
  'prefill.layer': 'full',
  'decode.op': 'full',
  'decode.residual': 'full',
  'decode.attn': 'full',
  'decode.kv.write': 'full',
  'decode.sample.stage': 'full',
  'stream.chunk': 'full',
  'step.timeout': 'basic',
  'decode.moe': 'full',
  'decode.node': 'deep',
  'decode.lens': 'deep',
  'decode.attn.heads': 'deep',
};

export const detailRank = (d: DetailLevel): number => DETAIL_LEVELS.indexOf(d);
/** True when detail level d includes everything at level `min`. */
export const atLeast = (d: DetailLevel | null | undefined, min: DetailLevel): boolean =>
  d != null && detailRank(d) >= detailRank(min);
export const includedAtDetail = (type: EventType, d: DetailLevel): boolean =>
  detailRank(EVENT_DETAIL[type]) <= detailRank(d);

export type ParseErrorKind = 'json' | 'version' | 'unknown_type' | 'schema';
export interface ParseError {
  kind: ParseErrorKind;
  message: string;
  line: string;
}
export type ParseResult = { ok: true; event: TraceEvent } | { ok: false; error: ParseError };

const knownTypes = new Set<string>(EVENT_TYPES);

/** Validate one NDJSON line. Never throws. */
export function parseEvent(line: string): ParseResult {
  let raw: unknown;
  try {
    raw = JSON.parse(line);
  } catch (e) {
    return { ok: false, error: { kind: 'json', message: (e as Error).message, line } };
  }
  return validateEventObject(raw, line);
}

/** Validate an already-parsed object (lets callers pre-filter by `type`). */
export function validateEventObject(raw: unknown, line = ''): ParseResult {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    return { ok: false, error: { kind: 'schema', message: 'event must be an object', line } };
  }
  const obj = raw as Record<string, unknown>;
  if (obj.v !== SCHEMA_VERSION) {
    return {
      ok: false,
      error: { kind: 'version', message: `unsupported schema version ${String(obj.v)}`, line },
    };
  }
  if (typeof obj.type !== 'string' || !knownTypes.has(obj.type)) {
    return {
      ok: false,
      error: { kind: 'unknown_type', message: `unknown event type ${String(obj.type)}`, line },
    };
  }
  const r = TraceEvent.safeParse(raw);
  if (!r.success) {
    const msg = r.error.issues
      .map((i) => `${i.path.join('.') || '<root>'}: ${i.message}`)
      .join('; ');
    return { ok: false, error: { kind: 'schema', message: msg, line } };
  }
  return { ok: true, event: r.data };
}

/** Parse a whole NDJSON document (blank lines ignored). */
export function parseNdjson(text: string): { events: TraceEvent[]; errors: ParseError[] } {
  const events: TraceEvent[] = [];
  const errors: ParseError[] = [];
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const r = parseEvent(line);
    if (r.ok) events.push(r.event);
    else errors.push(r.error);
  }
  return { events, errors };
}

/** Serialize an event to a single NDJSON line (no trailing newline). */
export const serializeEvent = (e: TraceEvent): string => JSON.stringify(e);

// ---- Control messages (app -> producer over the same socket) -------------

export const AckMessage = z.object({
  v: z.literal(SCHEMA_VERSION),
  type: z.literal('ack'),
  req: z.string(),
  granularity: z.enum(['token', 'layer', 'sample']).optional(),
  /** leave step mode for this request (runner proceeds freely) */
  continue: z.boolean().optional(),
});
export type AckMessage = z.infer<typeof AckMessage>;

/**
 * Hub socket paths. Producers (Ollama, mock) connect to INGEST; renderer
 * windows connect to VIEW. Events flow ingest -> view; acks flow view -> ingest.
 */
export const HUB_PATHS = { ingest: '/ingest', view: '/view' } as const;

export const TRACE_HEADERS = {
  requestId: 'X-Trace-Request-Id',
  detail: 'X-Trace-Detail',
  step: 'X-Trace-Step',
} as const;

export const DEFAULT_TRACE_PORT = 4711;
