# Ollama instrumentation notes

Checkout: `ollama/` at v0.40.13 + 2 commits (`8861adaa`), `LLAMA_CPP_VERSION = b11963`, `go.mod` requires Go 1.26.10.
Branch `visualizer-trace` adds 12 commits (`00*.patch`). Build and run instructions are in `build.md`.
The llama.cpp side is described in `llama.cpp/README.md`.

## Architecture (differs from the original plan)

- There is **no `runner/`, no Go `ollamarunner`, no `ml/backend/ggml`, no `sample/`**. No Go-native GGML engine remains.
- Every GGUF model runs in an **upstream llama.cpp `llama-server` subprocess**. Ollama talks to it over HTTP on `127.0.0.1:<random port>`.
- `mlxrunner/` is a Go-native MLX engine, used only for MLX-format models. It is not relevant here.
- llama.cpp is **not vendored**. The root CMake project clones it (`GIT_TAG` from `LLAMA_CPP_VERSION`) and applies `llama/compat/*.patch` with `cmake/apply-git-patches.cmake`. Our llama.cpp change is carried there as `llama/compat/100-visualizer-trace.patch`.

## Hook points (Go side)

| Concern | Location |
|---|---|
| Route registration | `server/routes.go` (`/api/generate`, `/api/chat`) |
| Header access / middleware pattern | `server/inference_request_log.go` (`withInferenceRequestLogging`) |
| GenerateHandler / ChatHandler entry | `startTrace` right after the body is bound |
| Model scheduling (load) | `scheduleRunnerForModel`; `Scheduler.load` `server/sched.go`; spawn `llm.NewLlamaServer` |
| llama-server spawn / flags | `llm/llama_server.go` `startLlamaServer` |
| Rendered path → `POST /completion` | `llamaServerRunner.Completion`; request struct `llamaServerCompletionRequest` (now with `Trace`) |
| Native chat path → `POST /v1/chat/completions` | `llamaServerRunner.Chat` (used when the model has no Go template, a renderer/parser, or `OLLAMA_GO_TEMPLATE=false`) |
| Runner trace protocol | `llm/llama_server_runner_trace.go` (request object, probe, chunk forwarding, ack forwarding) |
| Basic-tier decode events | `llm/llama_server_trace.go` |
| Tokenize | `r.Tokenize` → llama-server `POST /tokenize` with `with_pieces: true` |
| Streaming to client | `streamResponse`; `writeChatResponse` |

## What is real and what is synthesised

With the native payload from `build.md` (patched llama-server) and a request at `layers`/`full` detail
or in `layer`/`sample` step mode:

| Event | Source | Notes |
|---|---|---|
| `request.received`, `model.load`, `request.done`, `request.error` | Go (`server/`) | unchanged |
| `tokenize.done` | Go: llama-server `/tokenize` with `with_pieces` on the exact prompt | unchanged |
| `prefill.batch`, `prefill.done` | **llama-server**, one `prefill.batch` per real ubatch | the synthesised single batch is used only when llama-server sent neither |
| `prefill.layer` | **llama-server** (`full`) | |
| `decode.step.start` | Go, one per generated token | `t_ns` = first runner event of the step; `kv_used` = prompt tokens + step |
| `decode.layer` (`device`, `norm` at full) | **llama-server** | every step < 8, then every `OLLAMA_TRACE_LAYER_STRIDE`-th (default 4) |
| `decode.op`, `decode.residual`, `decode.kv.write`, `decode.attn`, `decode.moe` | **llama-server** (`full`) | attention on every `OLLAMA_TRACE_ATTN_LAYERS`-th layer (default 4), heads averaged unless `OLLAMA_TRACE_ATTN_HEADS` |
| `decode.node`, `decode.lens`, `decode.attn.heads` | **llama-server** (`deep`) | every layer of each traced step (stride as above; step mode forces 1). See `llama.cpp/README.md` §10 |
| `decode.sample.stage` | **llama-server** (`full`, or `sample` step mode) | the real chain in llama.cpp order, e.g. `raw → top_k → top_p → temperature → softmax → draw` for Ollama's defaults (penalties, DRY, min-p 0 etc. are no-ops and not emitted) |
| `decode.sample` | **llama-server** | real `chosen_id` and `method`; `temp`/`top_p` rounded from float32 (0.8, not 0.800000011920929). Go synthesises it only for a step llama-server sent none for. |
| `decode.logits` | Go: `n_probs` / `top_logprobs` (top 10) | still the pre-sampling distribution from llama-server's probabilities |
| `decode.token`, `stream.chunk` | Go (`server/`, after Ollama's parsers) | unchanged |
| `step.timeout` | Go (token gate) or **llama-server** (layer/sample pauses) | |
| `tokenize.merge` | not produced | left to the app (`no_tokenize_merge`) |

At `basic` detail, with tracing off, or with a stock llama-server, nothing changes from the Go-only
tier (below). The llama-server request then has no `trace` field.

### How runner events are forwarded (`llm/llama_server_runner_trace.go`)

- **Detection.** `request.received` is sent before a runner is scheduled, so its capabilities come from
  `llm.RunnerTraceSupported()`. That is the cached result of the last runner probe, or, before
  the first one, a scan of `llama-server*.exe/dll` for the protocol's `__trace_caps` marker. Each
  runner is then probed once, the first time a request wants runner events: `POST /trace/ack
  {"id_slot":2147483647}`. A patched server answers 409 (no such slot in step mode; this never
  counts as an ack), and a stock one 404. On a stock runner the request runs untraced on the
  llama.cpp side and layer/sample step mode falls back to the Go token gate.
- **Request.** `trace: {detail: "layers"|"full", layer_stride, attn_layers, attn_heads[, step, step_timeout_ms][, deep: true]}`
  is added to `/completion` and `/v1/chat/completions` requests. Basic-detail requests with layer
  step mode send `detail: "layers"`.
- **Chunks.** `__trace`, `__trace_t_ns`, `__trace_caps` and `__trace_slot` are parsed from every SSE chunk,
  including `/completion` trace-only chunks (`content:""`, `tokens:[]`), which also skip the
  token-repeat check, and chat chunks with `choices:[]`. On the chat path, the chunks after the
  `finish_reason` chunk (the rest of llama-server's final array) are drained for their events
  before the last step is closed.
- **Steps.** llama-server counts steps from 1, while the tracer and the app count from 0. Every runner
  `step` is decremented. The tracer starts one step per generated token, driven by the
  runner events and `timings.predicted_n`. Earlier, steps were per stream chunk, and a chunk with
  held-back text could cover several tokens. Now the two always line up, and the extra steps get an
  empty `decode.token`, as before. This only happens when runner events are forwarded; the basic
  tier keeps one step per chunk.
- **Order and time.** Events within a chunk are sorted by `t_ns`, because `prefill.batch` is held back
  by llama-server. They are rebased with `offset = go_receive_ns − __trace_t_ns`, keeping
  the smallest offset seen in the request. The tracer assigns `seq` and `t_ns` under a lock and clamps
  `t_ns` so it never goes backwards in `seq` order and never lies in the future.
- **De-duplication.** Real `prefill.*` replaces the synthesised batch. Per step, the runner's
  `decode.sample` is held until the step's token arrives, so the order is
  `decode.step.start → forward-pass events → sampler stages → decode.logits → decode.sample → decode.token`,
  with exactly one `decode.logits` and one `decode.sample` per step. A grammar resample's second
  `raw…draw` sequence is passed through.
- **Detail filtering.** `layers` forwards `decode.layer` and the basic-tier types and drops the
  full-tier ones. `full` forwards everything the schema knows except the deep tier; `deep` adds
  `decode.node`, `decode.lens` and `decode.attn.heads`. Unknown types are dropped. In
  step mode, `layer` also lets `decode.layer` through and `sample` lets `decode.sample.stage` through,
  since those are the events the runner pauses on.
- **`__trace_caps`** (e.g. `no_attn`, `no_sample_stages`, `no_step`) is logged
  (`llama-server trace capabilities missing`); the schema has no later capabilities event.
  `no_step` (another request holds llama-server's step gate) switches the request back to the Go
  token gate.

### Step mode

| `X-Trace-Step` | Gate | Ack path |
|---|---|---|
| `token` (`1`, `true`, …) | Go, before each `decode.step.start` (unchanged). llama-server keeps generating into the HTTP stream. | hub ack → tracer |
| `layer`, `sample` (patched runner) | llama-server, after every layer's `l_out` / every sampler stage. Stride is forced to 1. | hub ack → tracer → `POST /trace/ack {"id_slot":<__trace_slot>}` (`continue:true` passed on) |

- Before each pause, llama-server sends a trace-only chunk. Its events go out right away, so the app
  sees layer *n* while llama-server waits for the ack for layer *n*. There is no Go token gate in
  this mode.
- Ollama sends `{"continue":true}` when the request ends (before closing the stream), when the client
  goes away, or when the visualizer socket drops (checked every 250 ms), so llama-server is never
  left waiting. Its own `step_timeout_ms` (`OLLAMA_TRACE_STEP_TIMEOUT`) and client-disconnect
  detection are the last line of defence. Step mode is only requested while the hub is connected.
- Only one request holds step mode (the Go owner and llama-server's gate). Acks that arrive before
  llama-server's first pause count as credits there.

### Capabilities in `request.received`

| Runner | Capabilities |
|---|---|
| patched, `layers`/`full` or layer/sample step | `["no_tokenize_merge"]` |
| patched, `basic` | `["no_tokenize_merge","prefill_synthesized"]` |
| stock | `["no_layers","no_ops","no_residual","no_attn","no_kv_writes","no_sample_stages","no_tokenize_merge","prefill_synthesized"]` (+ `"step_token_only"` for layer/sample step) |

## Verification (native CUDA build, RTX 4060, llama3.2 3B Q4_K_M)

- Rendered (`/completion`) and native (`/v1/chat/completions`, `OLLAMA_GO_TEMPLATE=false`) chat and
  `/api/generate` at `full`, `layers` and `basic`: every event passes `parseEvent`, `seq` is contiguous,
  `t_ns` monotonic, every step has `decode.step.start` before its other events and exactly one
  `decode.logits` + `decode.sample`, and concat(`decode.token`) equals the response. `basic` sends no
  `trace` field (unit test, plus live counts identical to the Go-only tier).
- Layer step mode with a scripted hub: 140 layer pauses acked (5 steps × 28), 2.6 s; with no acks,
  `step.timeout` per pause, and a client disconnect releases llama-server; hub disconnect → continue.
  Sample step mode at `layers` detail: 6 stages × 5 steps acked.
- In the Electron app (`e2e/probe-real.ts`, plus a scratch probe for replay screenshots and
  `stepGranularity: 'layer'`), capabilities are `["no_tokenize_merge"]` and there are no mismatch
  warnings. Replay shows real per-op timings, residual bars, attention arcs and the llama.cpp
  sampler chain. In layer step mode, each *Step layer* click advances exactly one layer.
- Recorded: `traces/real/llama3.2-chat-sky-full.ndjson` (8,100 events, 81 steps) and
  `traces/real/llama3.2-chat-copper-full.ndjson` (7,438 events, 73 steps), each with its
  `.response.txt`. 0 schema errors.
- Unit tests: `trace/runner_test.go` (capabilities, filtering, t_ns clamp, ack routing),
  `llm/llama_server_runner_trace_test.go` (fake llama-server + real WebSocket hub: chunk parsing,
  step shift, offset rebasing, de-dup, layers filtering, no `trace` at basic/stock/untraced, chat
  `choices:[]` and final array, ack and continue forwarding).

## Go-only tier (stock llama-server, or `basic` detail)

### Emitter (`trace/`)

- `trace.NewRequest(id, detail, step)` returns `nil` when `OLLAMA_TRACE_URL` is unset. Every
  `*RequestTracer` method starts with `if t == nil { return }`. `testing.AllocsPerRun` checks that the disabled path makes 0 allocations.
- One ordered queue (`chan []byte`, cap 10 000) and one writer goroutine that owns the socket.
  When the queue is full, **droppable** events are counted and dropped. **Critical** events (same split as
  `PRIORITY` in trace-schema) go to an overflow list instead (capped at 200 000). Callers never block.
- The writer batches up to 256 queued events into one newline-delimited text frame and resends a batch after
  reconnecting (250 ms → 5 s backoff). When the hub has been unreachable for more than 5 s, queued events are discarded.
- A reader goroutine parses `ack` lines and wakes the matching step-mode request, or queues the ack for llama-server.
- `t_ns` is the tracer clock (`time.Since(processStart)`). On Windows its resolution is coarse, so order by `seq`.

### Where the basic-tier events come from

| Event | Emitted in | Source of data |
|---|---|---|
| `request.received` | `ChatHandler` / `GenerateHandler` (`startTrace`) | `model`, `prompt_chars`, raw `options`, `detail`, `capabilities` (table above) |
| `model.load` start/done | `scheduleRunnerForModel` | GGUF `block_count`, `num_ctx` / `ContextLength()`, VRAM, scheduling wall time |
| `tokenize.done` | `llm` before the completion request | `/tokenize` with `with_pieces`; native path renders with `/apply-template` first; skipped for multimodal prompts |
| `prefill.batch` + `prefill.done` (synthesised) | `llm`, first streamed chunk | one batch `cache_n…cache_n+prompt_n`, `ms` = `prompt_ms` |
| `decode.step.start` | `llm`, per streamed chunk (+ EOS) | `kv_used` = prompt tokens + `predicted_n` before the chunk |
| `decode.logits` | same | `top_logprobs` of the chunk's first token, `p = exp(logprob)`, entropy over the top 10 |
| `decode.sample` (synthesised) | same | `chosen_id` = reported token id; `temp/top_p/top_k` from options; `method` greedy/dist |
| `decode.token` / `stream.chunk` | `server` (`traceChunk`) | post-parser text; `stream.chunk` at `full` only |
| `request.done` / `request.error` | final chunk / error paths / `endTrace` fallback | |

Response-text invariant: `concat(decode.token.text)` == `concat(thinking + content)` of the streamed response.
The app must include `thinking`. Text that the parsers turn into `tool_calls` doesn't appear.

Client isolation: tracing always asks llama-server for 10 logprobs. `userLogprobs` drops them unless the client
set `logprobs`, and truncates `top_logprobs` to the client's value.

## Performance

Setup: RTX 4060 8 GB, CUDA 12.3, llama3.2 3B Q4_K_M fully offloaded, native payload from `build.md`
(`C:\ollama-trace-full`), `/api/chat`, temperature 0, `num_predict: 300` (300 tokens each run),
1 warm-up + 5 runs. The table gives the median of Ollama's `eval_count / eval_duration`. Recording
hub on localhost. The user's other Ollama (port 11434) was running at the same time, with no requests from this benchmark.

| Mode | decode tok/s | vs off | TTFT | events / request |
|---|---|---|---|---|
| tracing off (`OLLAMA_TRACE_URL` unset) | **108.1** (108.2 / 108.0 / 108.1 / 108.3 / 107.9) | — | 21.5 ms | — |
| `basic` | **98.3** | −9.1 % | 23.1 ms | ~1.2 k |
| `layers` | **92.9** | −14.1 % (−5.5 % vs basic) | 25.3 ms | 3,447 |
| `full` (stride 4, attention every 4th layer) | **64.0** | −40.8 % (−35 % vs basic) | 27.7 ms | 26,295 (0 dropped) |
| `full`, `OLLAMA_TRACE_LAYER_STRIDE=1` | **34.4** | −68 % | 28.7 ms | 89,148 emitted |

- `basic` costs as before: about 7 of its 9 points are llama-server computing `n_probs` (softmax +
  full-vocab sort per token), the rest is `timings_per_token` and per-chunk JSON.
- `layers`/`full` add llama-server's own cost (llama.cpp README §8: −6 % / −35 % on CUDA vs its
  no-trace run) plus Go-side parsing and re-emitting. `full` at stride 4 is about 87 events per token.
  The Go side keeps up, and the hub received every event (no `seq` gaps).
- At stride 1, ~300 events per token, the emitter's 10 000-event queue overflowed and ~42 % of the
  droppable events (op/attn/residual/kv) were dropped. Every critical event arrived. The recording
  hub appends each line to a file synchronously, so it was the slow consumer. Use stride 1 for short
  runs or step mode, which forces stride 1 anyway.
- With tracing off, `ollama.exe` sends byte-identical llama-server requests (unit-tested) and every
  trace hook is a nil no-op (0 allocations).

Earlier Go-only numbers (stock 0.40.6 payload): off 105.1, `basic` 94.7 tok/s.

## Known app-side observations (not changed here)

- In the live Full-detail view at GPU speed, the app's playback clock falls behind (~87 events per token at
  60–110 tok/s) and degrades by design. It skips full-tier events while lagging, so the forward-pass
  and sampling panels can be empty at the end of a live run. Replaying the run (seek or step) shows
  all the data.
- `Session.continue()` resumes the clock but doesn't clear a pending *Step token/layer* stop
  (`clock.stopWhen`). If Continue is clicked while a step is still pending, playback pauses again at
  the next matching event, and the run stays in "decoding" until *Play* is pressed.

## Deep tier (commits 0011–0012)

- `trace.Deep` (> `Full`): `ParseDetail("deep")`, `String() == "deep"`, so `request.received.detail`
  is `"deep"`. `eventDetail` adds `decode.node`, `decode.lens`, `decode.attn.heads` at `Deep`;
  everything allowed at `Full` is allowed at `Deep`.
- The runner request is `detail: "full", deep: true` (`runnerTraceRequest.Deep`, omitted below
  Deep, so `full` requests are byte-identical to before). `lens_topk` / `attn_window` use the
  llama-server defaults (5 / 512); there are no env vars for them.
- Steps: the generic step shift in `parseRunnerEvent` applies, so the new events are 0-based like
  `decode.layer` (first decode pass = step 1).
- Unit tests: `TestAllowedFollowsDetail` (deep column), `TestRunnerTraceDeep` (request field, step
  shift, forwarded at deep, dropped at full).

Measured through `C:\ollama-trace-deep` (CUDA, `/api/chat`, llama3.2, temperature 0, 107 tokens,
3 runs, `OLLAMA_TRACE_LAYER_STRIDE=1`, recording hub):

| Detail | tok/s | events / request |
|---|---|---|
| `basic` | 99.0 | – |
| `full` | 49.5 | – |
| `deep` | 14.0 (15.30 / 13.54 / 13.21) | 94,173, 0 dropped, `seq` contiguous |

With the default stride 4, 40 tokens: `basic` 100.7, `full` 69.2, `deep` 37.6 tok/s. Text md5
identical across all of them. Layer step mode at deep: 3 decode passes × 28 layer pauses acked,
each pause flushes that layer's nodes and lens. Deep events are droppable; at ~880 events per
token the emitter kept up at 14 tok/s, but a slow hub can make it drop them (see Performance).
