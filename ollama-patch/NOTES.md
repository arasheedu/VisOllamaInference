# Ollama instrumentation notes

Checkout: `ollama/` at v0.40.13 + 2 commits (`8861adaa`), `LLAMA_CPP_VERSION = b11963`, `go.mod` requires Go 1.26.10.

## Architecture (differs from the original plan)

- There is **no `runner/`, no Go `ollamarunner`, no `ml/backend/ggml`, no `sample/`**. No Go-native GGML engine remains.
- Every GGUF model runs in an **upstream llama.cpp `llama-server` subprocess**. Ollama talks to it over HTTP on `127.0.0.1:<random port>`.
- `mlxrunner/` is a Go-native MLX engine, used only for MLX-format models. It is not relevant here.
- llama.cpp is **not vendored**. `llama/server/CMakeLists.txt` uses FetchContent at configure time (`GIT_TAG` from `LLAMA_CPP_VERSION`) and applies `llama/compat/*.patch`. You can point it at a local source tree with `OLLAMA_LLAMA_CPP_SOURCE`, which skips the compat patches, or with `-DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=...`, which keeps them.

## Hook points (Go side)

| Concern | Location |
|---|---|
| Route registration | `server/routes.go:2427` (`/api/generate`), `:2428` (`/api/chat`) |
| Header access / middleware pattern | `server/inference_request_log.go:49` (`withInferenceRequestLogging`) |
| GenerateHandler / ChatHandler entry | `routes.go:308` / `routes.go:3019` (`checkpointStart`) |
| Model scheduling (load) | `scheduleRunnerForModel` `routes.go:257`; `Scheduler.load` `server/sched.go:539`; spawn `llm.NewLlamaServer` `sched.go:622`; `llama.Load` `sched.go:672` |
| Load finished | `checkpointLoaded` `routes.go:560` (generate), `:3239` (chat) |
| llama-server spawn / flags | `llm/llama_server.go:375–480` |
| Rendered path → `POST /completion` | `llamaServerRunner.Completion` `llama_server.go:1667`; request struct `:1450` (has `n_probs`, unused `timings_per_token`) |
| Per-token callback (completion) | `llama_server.go:1856`; final timings `:1868` |
| Native chat path → `POST /v1/chat/completions` | `llamaServerRunner.Chat` `llama_server.go:2038`; request map `:2291`; stream parse `:2112–2198` |
| Logprobs | `NProbs = max(TopLogprobs,1)` `:1718`; `completion_probabilities` struct `:1659` (ids/bytes not unmarshalled) |
| Timings | `llamaServerTimings` `:1639` (`cache_n, prompt_n, prompt_ms, predicted_n, predicted_ms`) |
| Total/load duration | `routes.go:803`, `:3373`, `:3540` |
| Tokenize | `r.Tokenize` → llama-server `POST /tokenize` `llama_server.go:2656–2742`; upstream supports `with_pieces: true` |
| Streaming to client | `streamResponse` `routes.go:2625`; `writeChatResponse` `routes.go:2966` |

## Event feasibility

| Event | Go-only patch (stock llama-server) | Needs patched llama.cpp |
|---|---|---|
| request.received, model.load, request.done, request.error | ✅ routes.go / sched.go | |
| tokenize.done | ✅ via `/tokenize` with `with_pieces` on the rendered prompt | |
| prefill.batch / prefill.done | ⚠️ synthesized: one batch, duration from `timings.prompt_ms` (first-token latency) | real ubatch loop: `tools/server` slot loop |
| decode.step.start, decode.token | ✅ per streamed piece; kv_used = prompt_n + step | |
| decode.logits | ✅ from `n_probs` / `top_logprobs` (up to 20) | |
| decode.sample | ✅ params from request options; chosen = streamed token | |
| decode.layer, decode.op, prefill.layer | ❌ | `ggml_backend_sched_set_eval_callback` in `src/llama-context.cpp` |
| decode.residual, decode.attn, decode.kv.write | ❌ | eval callback + tensor readback |
| decode.sample.stage | ❌ | sampler chain `src/llama-sampling.cpp`, `common/sampling.cpp` |
| stream.chunk | ✅ in the Go streaming callback | |
| Step mode (token granularity) | ✅ block in Go per-token callback; llama-server keeps generating into the HTTP stream buffer. TCP backpressure eventually stalls it. | layer/sample granularity needs C++ |

## Build prerequisites (Windows)

The C++ payload build needs Go 1.26.10, CMake ≥ 3.24, Visual Studio 2022 C++ tools (Native Desktop), and Ninja. Use `cmake -B build . && cmake --build build --parallel 8`.

For Go-only changes, `go build .` produces `ollama.exe`, which reuses an existing native payload. `FindLlamaServer` searches `<exe dir>\lib\ollama` (`llm/llama_binary.go:50–146`). The stock install ships `lib\ollama\llama-server.exe` plus `libllama-server-impl.dll`.

## Implementation (branch `visualizer-trace`, 3 commits, see `*.patch`)

| Commit | Files |
|---|---|
| `trace: add env-gated inference trace emitter` | `trace/emitter.go` (transport), `trace/tracer.go` (per-request API + payload structs), `trace/trace_test.go`, `envconfig/config.go` (4 vars), `go.mod` (+`github.com/coder/websocket v1.8.15`, no transitive deps) |
| `llm: emit decode trace events from the llama-server stream` | `llm/llama_server_trace.go` (new), `llm/llama_server.go` (+~45 lines in `Completion` / `Chat`, `id` added to `llamaServerTokenProb`) |
| `server: trace chat and generate requests` | `server/trace.go` (new), `server/routes.go` (+~35 lines) |

### Emitter (`trace/`)

- `trace.NewRequest(id, detail, step)` returns `nil` when `OLLAMA_TRACE_URL` is unset. Every
  `*RequestTracer` method starts with `if t == nil { return }`. `testing.AllocsPerRun` checks that the disabled path makes 0 allocations.
  The only per-request cost when tracing is off is one `ctx.Value` lookup at each hook.
- One ordered queue (`chan []byte`, cap 10 000) and one writer goroutine that owns the socket.
  When the queue is full, **droppable** events are counted and dropped. **Critical** events (same split as
  `PRIORITY` in trace-schema) go to an overflow list instead. Once the overflow is non-empty, all new events go
  there too until it drains, so order is preserved. The overflow is capped at 200 000 events as a memory guard. Callers never block.
- The writer batches up to 256 queued events into one newline-delimited text frame. If a write
  fails, it keeps the batch and resends it after reconnecting. Reconnect uses backoff from 250 ms to 5 s. Only the first
  failure is logged. When the hub has been unreachable for more than 5 s, queued events are discarded,
  so a missing app cannot cause unbounded growth or a stale flood later.
- A reader goroutine parses `ack` lines (single or NDJSON) and wakes the matching step-mode request.
- `t_ns` is `time.Since(processStart)`, which is monotonic. **On Windows its resolution is coarse.** The events of
  one decode step often share a `t_ns` value, and so do prefill.* and step 0. Order by `seq`, which is contiguous from 0 per `req`.
- The JSON envelope is built by hand (`{"v":1,"req":…,"seq":…,"t_ns":…,"type":…`) and payload
  structs are spliced in. Field names are pinned by `TestSchemaFieldNames`. Every recorded event passes `parseEvent`.

### Where each event comes from

| Event | Emitted in | Source of data |
|---|---|---|
| `request.received` | `ChatHandler` / `GenerateHandler` right after the body is bound (`startTrace`) | `model`; `prompt_chars` = runes of prompt / all message contents; `options` = the request's raw `options` map; `detail`; static `capabilities` |
| `model.load` start/done | `scheduleRunnerForModel` around `sched.getRunner` | `layers` = GGUF `<arch>.block_count`; `ctx` = requested `num_ctx` (0 = auto) on start, `ContextLength()` on done; `vram_mb` = `MemorySize()` VRAM on done (0 on start); `ms` = scheduling wall time (≈0–3 ms when already loaded) |
| `tokenize.done` | `llm` before the completion request | `POST /tokenize {content, add_special:true, with_pieces:true}` on the exact prompt sent to `/completion`. On the native path, `/apply-template` runs first. Skipped for multimodal prompts. |
| `prefill.batch` + `prefill.done` | `llm`, on the first streamed chunk | One synthesized batch: `from` = `cache_n`, `to` = `cache_n+prompt_n`, `ms` = `prompt_ms` (from `timings_per_token`). Falls back to tokenize count + first-chunk latency. |
| `decode.step.start` | `llm`, per streamed llama-server chunk (and for the EOS token) | `kv_used` = prompt tokens + `predicted_n` before this chunk, `kv_total` = `num_ctx` |
| `decode.logits` | same | `top_logprobs` of the chunk's first token (n_probs = max(client, 10)), `p = exp(logprob)` from the **pre-sampling** distribution; `entropy` in nats over the top-10 only |
| `decode.sample` | same | `chosen_id` = token id llama-server reports (`-1` if unknown; GGUF `eos_token_id` for the EOS step); `temp/top_p/top_k` from resolved options; `method` = `greedy` if temp ≤ 0 else `dist` |
| `decode.token` | `server`, immediately before each chunk is sent to the client (`traceChunk`) | `text` = `thinking + content` **after** Ollama's parsers; `is_eos` on the final chunk when llama-server stopped on EOS (`stop_type:"eos"` / `finish_reason:"stop"`) |
| `stream.chunk` | same, detail = `full` only | `bytes` = UTF-8 length of the chunk text, `utf8_complete` = `utf8.ValidString` (always true; llama-server never splits code points in content) |
| `request.done` | final chunk (`traceChunk`); fallback in a deferred `endTrace` | Response metrics in ms. The fallback (e.g. `done_reason:"load"` with an empty prompt) carries only `total_ms`. |
| `request.error` | `streamResponse` / non-stream loops / `handleScheduleError`; fallback `endTrace` | Error text sent to the client, or `request failed with HTTP status N` / `request canceled` |
| `step.timeout` | `RequestTracer.StepStart` | `granularity:"token"` |

Client isolation: tracing always asks llama-server for logprobs. The `userLogprobs` function drops them unless
the client set `logprobs`, and truncates `top_logprobs` to the client's value. Verified live: traced responses contain
no `logprobs` unless requested, and `top_logprobs:3` returns exactly 3 (unit test `TestUserLogprobsHidesTraceProbs`).

### Response-text invariant

`concat(decode.token.text)` == `concat(message.thinking + message.content)` over the streamed chunks
(`response`/`thinking` for `/api/generate`). This was verified on every recorded run: rendered and native chat,
generate, a thinking model through the builtin parser, emoji (multi-token chunks), and `stream:false`.
**The app must include `thinking`** in its comparison. Text Ollama's parsers turn into structured
`tool_calls` does not appear in `decode.token`, and neither do `<think>` tags the parser strips.

Granularity caveats:
- A step is one llama-server **stream chunk**, which is normally one token. When llama-server holds text back
  (incomplete UTF-8, partial stop-string match), one chunk carries several tokens. `kv_used` then
  jumps by more than 1, and logits/sample describe the chunk's *first* token only (llama-server reports
  one probability entry for the held-back run; its `token` text can be `"�"` for byte tokens).
  So `#steps ≤ eval_count`.
- While a parser buffers (e.g. `<think>`), that step gets `decode.token` with `text:""`. The buffered
  text arrives with a later step. Every step has ≥ 1 `decode.token`. Rarely, a final flush adds a
  second `decode.token` to the last step, so the app should concatenate in `seq` order.
- The EOS step has an empty `topk` (llama-server sends no probabilities for it) and `text:""`, `is_eos:true`.
  Requests that stop on `num_predict` or a stop word have no EOS step and no `is_eos:true`.

### Step mode

- Enabled by `X-Trace-Step` (any value except empty/`0`/`false`/`off`) or `OLLAMA_TRACE_STEP`.
  Only one request at a time holds step mode (an atomic owner). Others run freely and log a warning.
- Before each `decode.step.start` the request blocks until (a) an `ack` for its `req`, (b) `ctx`
  cancellation, or (c) `OLLAMA_TRACE_STEP_TIMEOUT`, which emits `step.timeout` and continues stepping.
  An ack with `continue:true` releases the request permanently. An ack that arrives early is buffered (one credit).
  Prefill events are emitted before the first wait.
- If the hub socket is disconnected, step waits are skipped, so a crashed app never stalls Ollama.
- Granularity is **token only**. `layer`/`sample` are accepted and behave as `token`, with `step_token_only` added
  to `capabilities`.
- **Step mode invalidates all timing stats.** llama-server keeps generating into the HTTP stream
  while Go blocks in the per-token callback (TCP backpressure stalls it eventually). So
  `eval_ms`/`prompt_eval_ms` in `request.done` describe llama-server's own run, and the `t_ns` gaps
  between steps measure the user's clicks.
- Verified live with a stub hub: an 8-step response took 964 ms with 100 ms-delayed acks, versus 164 ms free-running. With no acks
  and a 300 ms timeout it took 2498 ms and emitted 8 `step.timeout` events.

### Capability gaps (always reported in `request.received.capabilities`)

`["no_layers","no_ops","no_residual","no_attn","no_kv_writes","no_sample_stages","no_tokenize_merge","prefill_synthesized"]`
(+ `"step_token_only"` when `X-Trace-Step` is `layer`/`sample`). The list is the same at every detail level,
because the emitter cannot produce any of these with stock llama-server. `layers` detail is accepted and
echoed, but adds nothing over `basic`. `full` adds only `stream.chunk`.

### TODO: C++ hooks for the layers/full tiers (needs patched llama.cpp, `LLAMA_CPP_VERSION=b11963`)

These require building the native payload (VS2022 C++ tools + CMake ≥ 3.24 + Ninja,
`cmake -B build . && cmake --build build`). Put the changes in a new `llama/compat/*.patch` so FetchContent applies them.

1. **Eval callback: `decode.layer`, `prefill.layer`, `decode.op`, `decode.residual`, `decode.attn`, `decode.kv.write`.**
   llama.cpp already threads `ggml_backend_sched_eval_callback` through `common_params::cb_eval` →
   `llama_context_params::cb_eval` → `llama_context` constructor, which calls
   `ggml_backend_sched_set_eval_callback(sched, cparams.cb_eval, cparams.cb_eval_user_data)` in
   `src/llama-context.cpp`. `examples/eval-callback` is the reference user. In `tools/server/server.cpp`
   (server context load), set `params_base.cb_eval` to a trace callback when a new `--trace` flag is given.
   The callback is called with `ask=true` per node (return true only for nodes of interest) and then
   with `ask=false` after the node is computed:
   - timing: record a timestamp when a layer's first node is asked/computed and when its last node is computed. The layer index
     is in the tensor name suffix (`attn_norm-12`, `Qcur-12`, `kq_soft_max-12`, `ffn_up-12`,
     `ffn_out-12`, `l_out-12`). Classify into the op enum: `attn_norm`/`ffn_norm`→norm, `Qcur/Kcur/Vcur`→attn_qkv,
     `kq`/`kq_soft_max`→attn_scores, `kqv_out`/`attn_out`→attn_out, `ffn_up`/`ffn_gate`→mlp_up,
     `ffn_gelu`/`ffn_silu`→mlp_act, `ffn_down`/`ffn_out`→mlp_down. Note: asking for a node forces a scheduler
     split and sync, which is the main `layers`/`full` cost. Batch per-layer timings with `OLLAMA_TRACE_LAYER_STRIDE`.
   - residual: read back `l_out-N` for the last token only (`n_embd` floats), compute L2 norm and
     delta vs. the previous layer on CPU.
   - attention: read the last row of `kq_soft_max-N` (n_kv × n_head) for every 4th layer and reduce top-8 on CPU.
     This is unavailable with flash attention (no materialized softmax), so emit `no_attn` when `-fa` is on.
   - kv writes: the position is known from the ubatch (`llama_context::decode`, `ubatch.pos`), so emit one per layer per step.
   - prefill.layer: the same callback during prompt ubatches (`n_tokens > 1`), indexed by ubatch.
2. **Real prefill batches**: emit per ubatch from the decode loop in `llama_context::decode`
   (`src/llama-context.cpp`) or per `n_batch` slice in `tools/server` `update_slots`.
3. **Sampler stages: `decode.sample.stage`.** `common_sampler_sample` (`common/sampling.cpp`) applies the chain
   built in `common_sampler_init`. Each stage is a `llama_sampler` applied by
   `llama_sampler_chain_apply` in `src/llama-sampling.cpp`. Snapshot `cur_p` (top-64 by logit, plus the
   `selected`/cutoff index) after each `llama_sampler_apply`, mapping sampler names (`penalties`, `top-k`, `top-p`,
   `min-p`, `temp-ext`, `dist`) to the stage enum, and record the RNG draw in `llama_sampler_dist_apply`.
4. **Transport from C++**: the simplest option is an extra per-token field in the existing SSE chunk
   (e.g. `"trace":[…]` when `timings_per_token`-style flag `trace:true` is set). The Go side
   (`llm/llama_server_trace.go`) would forward the field verbatim as events. That keeps one WebSocket owner and the
   same `req`/`seq`. Alternatively, the callback could write NDJSON to a pipe that Ollama passes at spawn
   (`llm/llama_server.go` spawn, ~:375–480).
5. **Step mode at layer/sample granularity** has to block inside the callback (C++), with a condition variable
   released by an ack that Go forwards (e.g. a new `POST /trace/ack` on llama-server).
6. `tokenize.merge`: left to the app (JS BPE). A runner-side alternative would be in `llama_vocab::impl::tokenize`
   (`src/llama-vocab.cpp`, BPE session `llm_tokenizer_bpe_session::tokenize`).

## Performance

Setup: RTX 4060 8 GB, llama3.2 3B Q4_K_M fully offloaded, `/api/chat`, temperature 0, `num_predict: 300`,
1 warm-up and 7 runs (5 for the logprobs runs), median of llama-server's `eval_count/eval_duration`. The stub hub is on localhost.

| Mode | decode tok/s | vs off | TTFT |
|---|---|---|---|
| tracing off (`OLLAMA_TRACE_URL` unset) | 105.1 (104.6 on a rerun) | — | 21.5 ms |
| tracing `basic` | 94.7 | **−9.9 %** | 25.3 ms |
| off, client asks `logprobs` + `top_logprobs:10` | 97.8 | −6.9 % | 22.2 ms |
| off, client asks `logprobs` + `top_logprobs:1` | 97.9 | −6.8 % | 22.4 ms |

`basic` misses the < 2 % target. About 7 of the 10 points come from llama-server computing token
probabilities at all. `n_probs > 0` makes it softmax and sort the full 128k vocab on the CPU for every token, whatever the
k. The rest (~3 %) is `timings_per_token` plus per-chunk JSON on both sides. The Go emitter's own cost
is small: it never blocks, and the hub received all 9 656 events of the 8 traced 300-token runs (1 207 per run, nothing dropped).
The extra `/tokenize` (+`/apply-template`) call adds ~3–4 ms to TTFT.
Options if this matters: (a) make `decode.logits` optional (an env/header that skips `n_probs`, leaving
basic at ~−3 %), or (b) in C++, compute top-k with a partial sort over the logits (llama.cpp's
`n_probs` path sorts the full vocab). `layers`/`full` were not benchmarked: without the C++ hooks they
cost the same as `basic` (`full` adds only `stream.chunk`).

With tracing off, the patched build matched the off numbers above, and all trace calls take the nil fast path (0 allocations, unit-tested).
