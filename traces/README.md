# Traces

Recorded inference traces in **NDJSON**: one JSON event per line, UTF-8, `\n`-separated.

```
{"v":1,"req":"…","seq":0,"t_ns":1000000000,"type":"request.received","model":"llama3.2","prompt_chars":42,"options":{}}
{"v":1,"req":"…","seq":1,"t_ns":1000300000,"type":"model.load","phase":"done","layers":28,"ctx":4096,"vram_mb":2300,"ms":0}
…
```

## Envelope

| field  | meaning                                                                 |
| ------ | ----------------------------------------------------------------------- |
| `v`    | schema version, currently `1`. Readers reject other versions.           |
| `req`  | request id. The app sends it as `X-Trace-Request-Id`; producers echo it. |
| `seq`  | strictly increasing per `req`. Readers sort by it, so order of arrival may differ. |
| `t_ns` | emitter's monotonic clock in nanoseconds. Only deltas are meaningful.   |
| `type` | event type. Payload fields depend on it.                                |

The full per-type field reference is generated from the zod schema into [`docs/events.md`](../docs/events.md). The schema itself lives in [`packages/trace-schema`](../packages/trace-schema/src/index.ts).

## Priorities and detail tiers

- **critical** events (phase boundaries, tokens, logits, sampler stages) are never dropped.
- **droppable** events (`decode.layer`, `decode.op`, attention, residual, KV writes, stream chunks) may be dropped by the emitter or by the app's ring buffer under pressure. The app must produce the same phases and tokens without them.
- Detail tiers: `basic` has phase boundaries plus tokens and logits. `layers` adds `decode.layer`. `full` adds ops, attention, residual, KV writes, sampler stages and stream chunks.

## Layout

- `fixtures/` contains deterministic traces produced by `pnpm gen-fixtures` from `packages/trace-mock`. Each `*.response.txt` is the exact concatenation of that trace's `decode.token` texts.
  - `short-answer.ndjson`: 20 tokens, 8 layers, `layers` detail, load 800 ms, 3 prefill batches.
  - `full-detail.ndjson`: 6 tokens, 8 layers, `full` detail.
- `real/` holds traces recorded from patched Ollama (see `ollama-patch/`).

## Replaying

```sh
pnpm mock replay traces/fixtures/short-answer.ndjson --speed 1 --loop   # into a running app
```

You can also open a trace in the app with **Open…** and scrub through it offline.
