# Trace event reference (schema v1)

_Generated from `packages/trace-schema/src/index.ts` by `pnpm --filter trace-schema docs`. Do not edit by hand._

Every event is one NDJSON line carrying the common envelope:

| field | type | meaning |
|---|---|---|
| `v` | `1` | schema version |
| `req` | string | request id (the app's `X-Trace-Request-Id`) |
| `seq` | int | strictly increasing per `req` |
| `t_ns` | number | monotonic emitter clock, nanoseconds |
| `type` | string | one of the types below |

## `request.received`

priority **critical** · detail ≥ **basic**

| field | type | required |
|---|---|---|
| `model` | `string` | yes |
| `prompt_chars` | `int` | yes |
| `options` | `object` | no |
| `detail` | `"basic" \| "layers" \| "full" \| "deep"` | no |
| `capabilities` | `Array<string>` | no |

## `model.load`

priority **critical** · detail ≥ **basic**

| field | type | required |
|---|---|---|
| `phase` | `"start" \| "done"` | yes |
| `layers` | `int` | yes |
| `ctx` | `int` | yes |
| `vram_mb` | `number` | yes |
| `ms` | `number` | no |

## `tokenize.done`

priority **critical** · detail ≥ **basic**

| field | type | required |
|---|---|---|
| `n` | `int` | yes |
| `tokens` | `Array<{ id: int, text: string }>` | yes |

## `prefill.batch`

priority **critical** · detail ≥ **basic**

| field | type | required |
|---|---|---|
| `batch_i` | `int` | yes |
| `of` | `int` | yes |
| `from` | `int` | yes |
| `to` | `int` | yes |
| `ms` | `number` | yes |

## `prefill.done`

priority **critical** · detail ≥ **basic**

| field | type | required |
|---|---|---|
| `n_tokens` | `int` | yes |
| `ms` | `number` | yes |

## `decode.step.start`

priority **critical** · detail ≥ **basic**

| field | type | required |
|---|---|---|
| `step` | `int` | yes |
| `kv_used` | `int` | yes |
| `kv_total` | `int` | yes |

## `decode.layer`

priority **droppable** · detail ≥ **layers**

| field | type | required |
|---|---|---|
| `step` | `int` | yes |
| `layer` | `int` | yes |
| `ms` | `number` | yes |
| `norm` | `number` | no |
| `device` | `string` | no |

## `decode.logits`

priority **critical** · detail ≥ **basic**

| field | type | required |
|---|---|---|
| `step` | `int` | yes |
| `topk` | `Array<{ id: int, text: string, p: number }>` | yes |
| `entropy` | `number` | yes |

## `decode.sample`

priority **critical** · detail ≥ **basic**

| field | type | required |
|---|---|---|
| `step` | `int` | yes |
| `chosen_id` | `int` | yes |
| `temp` | `number` | yes |
| `top_p` | `number` | yes |
| `top_k` | `int` | yes |
| `method` | `string` | yes |

## `decode.token`

priority **critical** · detail ≥ **basic**

| field | type | required |
|---|---|---|
| `step` | `int` | yes |
| `text` | `string` | yes |
| `is_eos` | `boolean` | yes |

## `request.done`

priority **critical** · detail ≥ **basic**

| field | type | required |
|---|---|---|
| `stats` | `{ prompt_eval_count: number, prompt_eval_ms: number, eval_count: number, eval_ms: number, load_ms: number, total_ms: number }` | yes |

## `request.error`

priority **critical** · detail ≥ **basic**

| field | type | required |
|---|---|---|
| `message` | `string` | yes |

## `tokenize.merge`

priority **droppable** · detail ≥ **full**

| field | type | required |
|---|---|---|
| `step` | `int` | yes |
| `span` | `[int, int]` | yes |
| `pieces` | `Array<string>` | yes |
| `result_id` | `int` | yes |

## `prefill.layer`

priority **droppable** · detail ≥ **full**

| field | type | required |
|---|---|---|
| `batch_i` | `int` | yes |
| `layer` | `int` | yes |
| `ms` | `number` | yes |

## `decode.op`

priority **droppable** · detail ≥ **full**

| field | type | required |
|---|---|---|
| `step` | `int` | yes |
| `layer` | `int` | yes |
| `op` | `"norm" \| "attn_qkv" \| "attn_scores" \| "attn_out" \| "mlp_up" \| "mlp_act" \| "mlp_down"` | yes |
| `ms` | `number` | yes |

## `decode.residual`

priority **droppable** · detail ≥ **full**

| field | type | required |
|---|---|---|
| `step` | `int` | yes |
| `layer` | `int` | yes |
| `norm` | `number` | yes |
| `delta_norm` | `number` | yes |

## `decode.attn`

priority **droppable** · detail ≥ **full**

| field | type | required |
|---|---|---|
| `step` | `int` | yes |
| `layer` | `int` | yes |
| `head` | `int` | no |
| `top` | `Array<{ pos: int, w: number }>` | yes |

## `decode.kv.write`

priority **droppable** · detail ≥ **full**

| field | type | required |
|---|---|---|
| `step` | `int` | yes |
| `layer` | `int` | yes |
| `pos` | `int` | yes |

## `decode.sample.stage`

priority **critical** · detail ≥ **full**

| field | type | required |
|---|---|---|
| `step` | `int` | yes |
| `stage` | `string` | yes |
| `candidates` | `Array<{ id: int, v: number }>` | yes |
| `cutoff` | `number` | no |
| `draw` | `number` | no |

## `stream.chunk`

priority **droppable** · detail ≥ **full**

| field | type | required |
|---|---|---|
| `step` | `int` | yes |
| `bytes` | `int` | yes |
| `utf8_complete` | `boolean` | yes |
| `text` | `string` | yes |

## `step.timeout`

priority **critical** · detail ≥ **basic**

| field | type | required |
|---|---|---|
| `step` | `int` | yes |
| `granularity` | `"token" \| "layer" \| "sample"` | yes |

## `decode.moe`

priority **droppable** · detail ≥ **full**

| field | type | required |
|---|---|---|
| `step` | `int` | yes |
| `layer` | `int` | yes |
| `experts` | `Array<{ id: int, w: number }>` | yes |

## `decode.node`

priority **droppable** · detail ≥ **deep**

| field | type | required |
|---|---|---|
| `step` | `int` | yes |
| `layer` | `int` | yes |
| `i` | `int` | yes |
| `name` | `string` | yes |
| `op` | `string` | yes |
| `shape` | `Array<int>` | yes |
| `ms` | `number` | yes |
| `src` | `Array<int>` | no |
| `stats` | `{ min: number, max: number, mean: number, rms: number }` | no |

## `decode.lens`

priority **droppable** · detail ≥ **deep**

| field | type | required |
|---|---|---|
| `step` | `int` | yes |
| `layer` | `int` | yes |
| `top` | `Array<{ id: int, p: number, text?: string }>` | yes |
| `entropy` | `number` | no |

## `decode.attn.heads`

priority **droppable** · detail ≥ **deep**

| field | type | required |
|---|---|---|
| `step` | `int` | yes |
| `layer` | `int` | yes |
| `from` | `int` | yes |
| `n_kv_head` | `int` | no |
| `rows` | `Array<Array<number>>` | yes |

## Control: `ack` (app → producer)

`{ v: 1, type: "ack", req: string, granularity?: "token" \| "layer" \| "sample", continue?: boolean }`

Sent over the same socket in step mode; `continue: true` releases the request from step mode.
