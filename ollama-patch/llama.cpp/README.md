# llama.cpp `visualizer-trace` patch

This patch makes `llama-server` emit per-layer, per-op, residual, attention, KV-write, real
prefill-ubatch and sampler-stage trace events for one request at a time. Ollama runs every GGUF
model in a `llama-server` subprocess, forwards the events and adds `v`, `req` and `seq`.

- Base: llama.cpp tag **b11963** (`39febd2ad`), the tag in `ollama/LLAMA_CPP_VERSION`.
- Source: `C:\BitsPlus\VisOllamInference\llama.cpp`, branch `visualizer-trace`.
- Patch series: `000{1..6}-*.patch` in this directory (`git format-patch b11963..visualizer-trace`,
  for `git am`).
- For Ollama's build: `ollama-compat/100-visualizer-trace.patch`, the same change squashed into one
  patch (see §7 for why). 15 files, +1358/−3.

| Patch | What it does |
|---|---|
| 0001 `llama:` | `llama_set_eval_callback(ctx, cb, ud)`: installs or removes the scheduler eval callback at runtime |
| 0002 `common:` | `common_sampler_set_stage_cb()`: optional observer that runs the sampler chain one sampler at a time |
| 0003 `server:` | `tools/server/server-trace.{h,cpp}` (collector, step gate) plus small hooks in the server |
| 0004 `server:` | Step mode ends when the client disconnects; sample pauses report the real step |
| 0005 `server:` | `step: "sample"` captures sampler stages at any detail level |
| 0006 `server:` | Fewer GPU syncs (skip view nodes), single-transfer reads, layer device taken from its weights |

Without a `trace` field in the request, nothing changes. No eval callback is installed, each hook
costs one branch, and the sampler chain runs as one `llama_sampler_apply`. The output is
byte-identical with and without tracing (verified, see *Verification*).

---

## 1. Request

The optional field `trace` is accepted on every completion-style endpoint. That includes `/completion`,
`/completions`, `/v1/completions` and `/v1/chat/completions`. Chat requests pass unknown fields
through to the completion params.

```json
"trace": {
  "detail": "layers" | "full",   // required in practice; "off"/"basic" or null = no tracing
  "layer_stride": 4,             // layer-level events on every step < 8, then on every layer_stride-th step
  "attn_layers": 4,              // decode.attn on layers where layer % attn_layers == 0
  "attn_topk": 8,                // positions per decode.attn (1..256)
  "attn_heads": false,           // also one decode.attn per head (with "head")
  "sample_topn": 64,             // candidates per decode.sample.stage (1..1024)
  "step": null | "layer" | "sample",   // "token" is accepted and ignored (Go handles it)
  "step_timeout_ms": 60000       // per pause
}
```

All fields except `detail` are optional, and the values above are the defaults. A bad `detail` or
`step` value returns HTTP 400 with `Field 'trace': ...`.

What each level emits:

| Event | `layers` | `full` |
|---|---|---|
| `prefill.batch`, `prefill.done`, `decode.sample` | ✓ | ✓ |
| `decode.layer` (without `norm` at `layers`) | ✓ | ✓ (with `norm`) |
| `prefill.layer`, `decode.op`, `decode.residual`, `decode.kv.write`, `decode.attn`, `decode.moe` | | ✓ |
| `decode.sample.stage` | | ✓ (every step) |
| `step.timeout` | step mode | step mode |

Downsampling applies to everything that comes from the forward pass (`decode.layer/op/residual/kv.write/attn/moe`).
These are emitted for all layers on steps < 8 and on steps where `step % layer_stride == 0`. On
other steps the eval callback is **not installed at all**. Use `"layer_stride": 1` to get every
step. `step: "layer"` forces stride 1. Sampler events (`decode.sample`, `decode.sample.stage`) are
emitted on every step.

## 2. Response

Events are plain objects `{"type", "t_ns", ...payload}`. `v`, `req` and `seq` are **not** set;
Go adds them. Payloads follow `packages/trace-schema/src/index.ts`, and all durations are in ms.

Fields added to a response chunk (top level of the JSON object in the SSE `data:` line):

| Field | When | Meaning |
|---|---|---|
| `__trace` | every chunk that has events | array of events, in emission order |
| `__trace_t_ns` | with `__trace` | llama-server `std::chrono::steady_clock` ns when the chunk was built. Use it to rebase `t_ns` onto the Go clock: `offset = go_now_ns_at_receive - __trace_t_ns`, keep the smallest offset seen in the request, `t_go = t_ns + offset`. |
| `__trace_caps` | first chunk with trace data | array of missing capabilities, see §4. `[]` = everything requested is provided |
| `__trace_slot` | same chunk as `__trace_caps` | server slot id, for `POST /trace/ack` |

**Which chunk carries what.** At each `send_partial_response()` the collector hands over everything
collected since the previous chunk:

- The prefill events and the sampling events of step 1 ride on the chunk of the first token.
- The forward-pass events of step *s* (s ≥ 2) and the sampling events of step *s* ride on the chunk
  of token *s*.
- If a token produces no text (a stop-word prefix is being held back, or an incomplete UTF-8
  sequence), its events ride on the next chunk. Every event has its own `step`, so never assume
  that chunk index equals step.
- Anything still pending rides on the final chunk (`stop: true`; for OAI chat, the last chunk of
  the final array).
- `/v1/chat/completions` stream: if a token produces no delta (e.g. the EOS token), the events
  come in a chunk with `"choices": []` (plus `object` and, for trace-only chunks, `id`, `model`
  and `created`). The Go parser must accept such chunks.
- Non-streaming requests: all events are in the single response object.
- `/v1/responses` and `/v1/messages` (Anthropic) are not instrumented. Events are collected but not
  attached.

**Step numbering** matches the Go tracer: `step` is the 1-based index of the generated token.
The forward pass that produced token *s*'s logits belongs to step *s*. Token 1 comes from the
prefill, so `decode.layer` starts at step 2. `prefill.*` events have no `step`.

**Ordering.** `prefill.batch` events are held back until the prefill finishes, because `of` is
only known then. They come right before `prefill.done`, after the `prefill.layer` events of all
ubatches. With several ubatches, `t_ns` therefore isn't monotonic inside that chunk. Every other
event is appended in time order. Sort by `t_ns` if you need strict order.

### Events and real samples

All samples come from llama3.2 3B Q4_K_M on CPU.

```jsonc
// one per real ubatch (llama_decode splits the server batch into n_ubatch pieces); from/to are KV positions, to exclusive
{"type":"prefill.batch","t_ns":763895636401800,"batch_i":0,"of":3,"from":0,"to":512,"ms":3731.2409}
{"type":"prefill.batch","t_ns":763899651271100,"batch_i":1,"of":3,"from":512,"to":1024,"ms":4013.5326}
{"type":"prefill.batch","t_ns":763901464415800,"batch_i":2,"of":3,"from":1024,"to":1237,"ms":1812.4958}
// n_tokens = tokens evaluated (cached prefix excluded); ms = wall time from first prefill decode to first sample
{"type":"prefill.done","t_ns":763901468925200,"n_tokens":1237,"ms":9566.3084}
{"type":"prefill.layer","t_ns":763805598390200,"batch_i":0,"layer":0,"ms":2.6302}

// device = ggml device of the layer's weight matmuls ("CPU", "CUDA0", ...); with -ngl 14 on llama3.2: layers 0-14 "CPU", 15-27 "CUDA0"
{"type":"decode.layer","t_ns":763805658394600,"step":2,"layer":0,"ms":1.1186,"device":"CPU","norm":1.4332201480865479}
// one per (layer, op) that had time attributed, emitted at the end of the layer
{"type":"decode.op","t_ns":763805658415800,"step":2,"layer":0,"op":"norm","ms":0.0213}
{"type":"decode.op","t_ns":763805658418100,"step":2,"layer":0,"op":"attn_qkv","ms":0.2143}
{"type":"decode.op","t_ns":763805658419900,"step":2,"layer":0,"op":"attn_scores","ms":0.02}
{"type":"decode.op","t_ns":763805658422000,"step":2,"layer":0,"op":"attn_out","ms":0.0948}
{"type":"decode.op","t_ns":763805658423600,"step":2,"layer":0,"op":"mlp_up","ms":0.4481}
{"type":"decode.op","t_ns":763805658425100,"step":2,"layer":0,"op":"mlp_act","ms":0.0092}
{"type":"decode.op","t_ns":763805658426600,"step":2,"layer":0,"op":"mlp_down","ms":0.3041}
// norm = ||l_out|| of the current token; delta_norm = ||l_out - layer input|| (attention + MLP contribution)
{"type":"decode.residual","t_ns":763805658413500,"step":2,"layer":0,"norm":1.4332201480865479,"delta_norm":1.6644227504730225}
// heads averaged; pos = KV cell index (see limitations); "head" only with attn_heads: true
{"type":"decode.attn","t_ns":763805657511300,"step":2,"layer":0,"top":[{"pos":0,"w":0.6539716720581055},{"pos":5,"w":0.13259482383728027},{"pos":1,"w":0.0820300430059433}]}
// position of the token whose K/V this pass wrote into every layer
{"type":"decode.kv.write","t_ns":763805658428500,"step":2,"layer":0,"pos":5}

// sampler chain in the real llama.cpp order (temperature 0.8, defaults), sample_topn: 3
{"type":"decode.sample.stage","t_ns":763805654915400,"step":1,"stage":"raw","candidates":[{"id":11,"v":21.01323890686035},{"id":304,"v":19.705520629882812},{"id":1131,"v":16.39820671081543}],"cutoff":128256}
{"type":"decode.sample.stage","t_ns":763805655001300,"step":1,"stage":"top_k","candidates":[...],"cutoff":40}
{"type":"decode.sample.stage","t_ns":763805655012200,"step":1,"stage":"top_p","candidates":[{"id":11,"v":21.01323890686035},{"id":304,"v":19.705520629882812}],"cutoff":2}
{"type":"decode.sample.stage","t_ns":763805655017200,"step":1,"stage":"min_p","candidates":[...],"cutoff":2}
{"type":"decode.sample.stage","t_ns":763805655022300,"step":1,"stage":"temperature","candidates":[{"id":11,"v":26.26654815673828},{"id":304,"v":24.631900787353516}],"cutoff":2}
{"type":"decode.sample.stage","t_ns":763805655101400,"step":1,"stage":"softmax","candidates":[{"id":11,"v":0.8368052840232849},{"id":304,"v":0.1631947010755539}],"cutoff":2}
{"type":"decode.sample.stage","t_ns":763805655104800,"step":1,"stage":"draw","candidates":[{"id":11,"v":0.8368052840232849}],"cutoff":0,"draw":0.22735595703125}
{"type":"decode.sample","t_ns":763805655109800,"step":1,"chosen_id":11,"temp":0.800000011920929,"top_p":0.949999988079071,"top_k":40,"method":"dist"}

// step mode, no ack before step_timeout_ms
{"type":"step.timeout","t_ns":...,"step":2,"granularity":"layer"}
// MoE models only (best effort, untested: no MoE model available locally)
{"type":"decode.moe","t_ns":...,"step":2,"layer":3,"experts":[{"id":17,"w":0.41},{"id":3,"w":0.22}]}
```

#### `decode.op` mapping

At `full` detail the eval callback observes the op-boundary nodes of each layer. The graph segment
that ends at an observed node is charged to that node's op:

| Tensor (`cb(cur, name, il)`) | op |
|---|---|
| `attn_norm`, `ffn_norm` | `norm` |
| `Qcur`, `Kcur`, `Vcur` (Q/K/V matmuls and RoPE; the pre-RoPE reshapes are views and not observed, so V's matmul is charged to the next boundary, the post-RoPE `Kcur` in llama-family graph order) | `attn_qkv` |
| `kq_soft_max` (non-FA: KQ matmul + mask + softmax, plus the K/V cache store) or the `FLASH_ATTN_EXT` node (FA: scores, softmax and the weighted V in one kernel) | `attn_scores` |
| `kqv_out` (non-FA: the V-weighted sum), `attn_out` (the `wo` projection) | `attn_out` |
| `ffn_up`, `ffn_gate`, `ffn_moe_up/gate/gate_up` | `mlp_up` |
| `ffn_silu/gelu/swiglu/geglu/act/relu/reglu/gelu_erf/geglu_erf`, `ffn_moe_silu/swiglu/gelu/geglu` | `mlp_act` |
| `ffn_down`, `ffn_moe_down`, `ffn_out` (models rename the `ffn_down` output to `ffn_out`, and also the residual add) | `mlp_down` |

`l_out` ends the layer. The tiny segment from the last op to `l_out` (control vector, no-op) is not
attributed, so the sum of `decode.op` can be slightly below `decode.layer.ms`. These names are
llama.cpp's own and are shared by the llama-family graph builders (`src/models/*.cpp` and
`src/llama-graph.cpp`). Architectures that don't name a node simply don't get that op.

#### `decode.sample.stage`

The stages come from the **real chain**, in order. llama.cpp sampler names map to schema stages like this:

| llama.cpp sampler | `stage` | `v` |
|---|---|---|
| (logits before the chain) | `raw` | logit |
| `logit-bias` | **`logit_bias`** (new) | logit |
| `penalties` | `rep_penalty` | logit |
| `dry` | `dry` | logit |
| `top-n-sigma` | `top_n_sigma` | logit |
| `top-k` | `top_k` | logit |
| `typical` | `typical_p` | logit |
| `top-p` | `top_p` | logit |
| `min-p` | `min_p` | logit |
| `xtc` | `xtc` | logit |
| `temp`, `temp-ext` | `temperature` | logit (scaled) |
| `dist` | `softmax` then `draw` | p |
| `greedy` (temperature 0) | `draw` only | logit |
| `mirostat`, `mirostat-v2`, `adaptive-p` | `softmax` then `draw` | p |
| reasoning budget / grammar (when the request uses them) | **`reasoning_budget`**, **`grammar`** (new) | logit |
| `infill` | **`infill`** (new) | logit |

- `candidates` holds the top `sample_topn` by `v`, with `-inf` (masked) entries dropped.
- `cutoff` = number of finite candidates left after the stage. For `draw` it is the index of the
  chosen token in the candidate array (`cur_p->selected`).
- `draw` is the real uniform random number `u ∈ [0,1)` that `dist` used. It's recovered by applying
  a clone of the `dist` sampler (same RNG state) to 8192 equal candidates, to ±6e-5. The chosen
  token is where the running CDF over the candidates, in chain order, crosses `u`. The real RNG is
  not advanced, and outputs are unchanged.
- Samplers that the request disabled (llama.cpp turns them into no-op `?name` samplers, e.g. `?dry`,
  `?xtc`, `?typical`, `?penalties` at their defaults) are **not** emitted.
- If grammar rejection makes llama.cpp resample, a second `raw ... draw` sequence follows for the
  same step.
- `decode.sample.method` = the mapped name of the last sampler in the chain (`dist`, `greedy`,
  `mirostat`, …). It is emitted at both detail levels. If Go synthesises its own
  `decode.sample`, it should prefer this one.

## 3. Step mode and `POST /trace/ack`

`trace.step = "layer"` blocks inside the eval callback **after** each decode layer's `l_out` has been
computed (i.e. before the next layer runs, and after the last layer before sampling).
`"sample"` blocks after each emitted sampler stage. It turns on `decode.sample.stage` even at
`detail: "layers"`. Before every pause the server sends a **trace-only
chunk** with the events collected so far. For `/completion` that is
`{"content":"","tokens":[],"stop":false,...,"__trace":[...]}`; for chat it is a chunk with
`"choices":[]`. The client sees each layer or stage as it happens. The prefill isn't stepped.

```
POST /trace/ack
{"id_slot": 0}            // optional; omitted = the request currently in step mode
{"continue": true}        // leave step mode for the rest of this request
```

Responses:

```json
200 {"ok":true,"id_slot":0,"waiting":true,"at":{"granularity":"layer","step":2,"layer":9}}
200 {"ok":true,"id_slot":0,"waiting":true,"at":{"granularity":"sample","step":1,"stage":"top_k"}}
409 {"ok":false,"error":"no request in step mode"}
409 {"ok":false,"error":"slot is not in step mode","id_slot":0}
```

- Acks count like a semaphore. An ack that arrives before the pause isn't lost, it releases the next pause.
- On timeout (`step_timeout_ms`, per pause) the server emits `step.timeout {step, granularity}` and
  continues, staying in step mode.
- If the client disconnects, the server leaves step mode within 250 ms
  (`server_response::is_waiting()`).
- `/trace/ack` is answered directly on the HTTP thread through a process-wide condition variable.
  It doesn't use the task queue, which is blocked during the pause.
- **The slot loop is single-threaded, so a pause blocks the whole server**, not just this request.
  Other requests queue up until the ack. Only one request at a time can be in step mode. A second
  step request launched while the first holds the gate runs without pauses and gets
  `"no_step"` in `__trace_caps`. Step mode makes all timings meaningless.
- Step mode with `stream: false` still blocks, but the events only arrive in the final response.

## 4. Capabilities (`__trace_caps`)

| Value | Meaning |
|---|---|
| `no_attn` | `detail: full`, but the model graph has neither a `kq_soft_max` nor a flash-attention node (e.g. recurrent / SSM models) |
| `no_sample_stages` | backend (GPU) sampling is active (`--backend-sampling`), so there is no CPU chain to observe |
| `no_step` | step mode was requested, but another request holds the step gate |

**Flash attention is supported.** It can't be switched per request: the KV-cache layout is fixed at
context creation and a quantised V cache requires FA. So the collector handles both graphs. Without
FA it reads the real `kq_soft_max-<il>` row of the current token. With FA (the default:
`--flash-attn auto` turns it on for CPU and CUDA) it rebuilds the softmaxed KQ row on the CPU from
the FA node's inputs: the current token's `q`, the K-cache view (F16/F32/quantised via
`to_float`), the mask, and the scale/softcap op params. Checked on llama3.2: `-fa on` and `-fa off`
give the same top-5 positions with weights within 0.001 (`0:0.823 4:0.041 14:0.029 13:0.022 10:0.017`
vs `0:0.824 4:0.041 14:0.029 13:0.022 10:0.017`). Attention sinks and ALiBi are ignored in the
rebuild. Neither is used by llama-family models.

## 5. Hook locations

| Where | Hook |
|---|---|
| `include/llama.h`, `src/llama-context.{h,cpp}` | `llama_set_eval_callback()`: sets `cparams.cb_eval` (re-applied when a graph is rebuilt) and the scheduler's callback (kept across reused graphs) |
| `common/sampling.{h,cpp}` | `common_sampler_set_stage_cb()`; `common_sampler_sample()` reports `raw` / `reasoning_budget` / `grammar` and applies the chain sampler by sampler when observed (`apply_chain()`) |
| `tools/server/server-trace.{h,cpp}` | collector, eval callback, attention/residual reductions, sampler snapshots, step gate, `server_trace_ack()` |
| `server-schema.cpp` `make_llama_cmpl_schema` | parses `"trace"` into `task_params.trace` |
| `server-task.{h,cpp}` | `task_params.trace`; `server_trace_payload trace` on partial/final results; `to_json()` attaches it; `is_trace_only` chunks |
| `server-context.cpp` `server_slot` | `server_trace trace` member; `can_batch_with()` keeps a traced slot alone in its batch; `release()` stops the trace |
| `launch_slot_with_task()` | `trace.start()` with the flush (trace-only chunk) and alive (`queue_results.is_waiting`) callbacks |
| `decode()` | `trace.before_decode()` installs the eval callback for this `llama_process` only (if the step is sampled); `after_decode()` removes it |
| `post_decode()` | `prefill_done()` at the prompt→generating transition; `sample_begin()/sample_end()` around `common_sampler_sample()` |
| `send_partial_response()` / `send_final_response()` / `send_trace_chunk()` | `res->trace = slot.trace.take()` |
| `server-queue.{h,cpp}` | `server_response::is_waiting(id_task)` |
| `server.cpp` | `POST /trace/ack` route |

How the eval callback works. In the *ask* phase the callback parses `name-<il>`. It tracks the
current layer and the ubatch start (the first layer-0 node), and answers `true` only for the
tensors listed above (and only at the detail level that needs them). Observing a node makes the
scheduler compute up to it and synchronise. The *observe* phase then timestamps it
(`steady_clock`), reads back at most one row for the current token (`ggml_backend_tensor_get`),
and reduces on the CPU. The collector's own readback and reduction time is excluded from the
reported durations.

## 6. Limitations

- **Batching:** a traced request never shares a batch (`can_batch_with`). With `-np > 1`, other
  slots run in separate batches while it is active. That costs aggregate throughput, not
  correctness (verified: the untraced output of a concurrent request is unchanged). A traced slot
  that isn't first in the slot order can be delayed by generating slots ahead of it. Ollama uses
  `-np 1` by default (`OLLAMA_NUM_PARALLEL=1`).
- **GPU timing is coarse:** every observed node is a `ggml_backend_synchronize()`, and it splits the
  CUDA graph into many small launches (no CUDA-graph capture, no fusion across observed nodes).
  Per-op ms on GPU therefore include launch and sync latency. Per-layer ms are reasonable, and
  absolute decode speed drops more at `full` than on CPU (see benchmarks).
- `decode.attn.pos` is the **KV cell index**. With one sequence per slot (`-np 1`, or a non-unified
  KV cache) and no context shift, that equals the token position. After a context shift, or with
  `--kv-unified` and several slots, cells and positions can differ. The KQ mask still zeroes other
  sequences' cells, so weights are correct.
- `decode.attn` / `decode.residual` assume one KV stream (a solo batch) and read the last row of
  the observed tensors.
- `decode.moe`: reads `ffn_moe_topk-<il>` (ids) and the last of `ffn_moe_weights[_softmax|_norm|_scaled]-<il>`
  (weights). Untested, because no MoE model is available locally.
- Speculative decoding: forward-pass events are produced, but sampler stages aren't captured on the
  draft-acceptance path (`common_sampler_sample_and_accept_n`).
- If something else set `cb_eval` on the server context (not done by `llama-server`), traced passes
  override it.
- `prefill.done.ms` is wall time, including the collector's overhead. `prefill.batch.ms` excludes it.

## 7. Build (Windows)

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Program Files\CMake\bin;%PATH%
cd /d C:\BitsPlus\VisOllamInference

rem CPU
cmake -S llama.cpp -B build-llama -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=OFF ^
  -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_HTML=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build-llama --config Release --target llama-server -- /m:16
rem -> build-llama\bin\Release\llama-server.exe (+ llama.dll, ggml*.dll, llama-server-impl.dll, ...)

rem CUDA (12.3 with MSVC 14.44 needs both overrides; CUDA >= 12.4 needs neither)
set CL=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
set NVCC_APPEND_FLAGS=-allow-unsupported-compiler
cmake -S llama.cpp -B build-cuda2 -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_HTML=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build-cuda2 --config Release --target llama-server -- /m:16
```

Run: `llama-server -m <gguf blob> --port 11437 --host 127.0.0.1 -c 4096 -np 1 --no-webui`.

### Building through Ollama

`ollama/llama/server/CMakeLists.txt` fetches b11963 and runs `cmake/apply-git-patches.cmake`. That
script applies every `*.patch` under `llama/compat/`: the glob is recursive and sorted by **file
name**, and a patch is skipped when `git apply --reverse --check` says it's already applied. Our
change touches none of the compat patches' files (`src/llama-model-loader.cpp`,
`tools/mtmd/clip.cpp`, `ggml-metal/kernels/unary.metal`), so order doesn't matter.

**Use the squashed patch, not the series, in `llama/compat/`.** With the 6-patch series, the first
configure works, but every re-configure fails. Patches 0004–0006 modify hunks of 0003, so 0003 no
longer reverse-applies on the final tree. The applier then tries to apply it forward and aborts
(reproduced with the real `apply-git-patches.cmake`). `ollama-compat/100-visualizer-trace.patch`
(= `git diff b11963 visualizer-trace`) has no such problem.

Verified on a fresh b11963 tree:
- `git am 0001..0006` → identical tree to the `visualizer-trace` branch.
- `001-llama-cpp-hooks.patch`, `002-metal-unary-cast-constexpr.patch`, then `100-visualizer-trace.patch`
  through `apply-git-patches.cmake`: run 1 applies all three, runs 2 and 3 report "already applied,
  skipping".
- `cmake -S ollama/llama/server` (VS generator, CPU) with that tree via
  `-DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=<tree> -DOLLAMA_LLAMA_CPP_SKIP_COMPAT_PATCH=ON` builds
  `llama-server` (with Ollama's compat sources linked in). Its `full` trace validates with 0 errors.

**Option A (recommended, no CMake change):** copy the squashed patch into the compat directory:

```
copy ollama-patch\llama.cpp\ollama-compat\100-visualizer-trace.patch ollama\llama\compat\
```

Keep it LF: add `*.patch text eol=lf` to `.gitattributes`. **Note:** in the current `ollama/`
working copy on this machine, the existing compat patches are checked out with **CRLF**, and
`git apply` rejects them ("patch does not apply"), so a native-payload build would fail at
configure regardless of our patch. The LF blobs from `git show HEAD:llama/compat/<file>` apply
fine. Fix the checkout (`git config core.autocrlf false` + re-checkout, or the gitattributes rule).

**Option B:** point FetchContent at a prepared tree:

```
git -C llama.cpp worktree add ../llama.cpp-ollama visualizer-trace
cmake -S ollama/llama/server -B build-ollama -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=C:/BitsPlus/VisOllamInference/llama.cpp-ollama
```

With `FETCHCONTENT_SOURCE_DIR_LLAMA_CPP`, Ollama **applies the compat patches into that tree** (it
modifies the working copy), hence the throw-away worktree. Alternatively, apply them yourself and
pass `-DOLLAMA_LLAMA_CPP_SKIP_COMPAT_PATCH=ON`. `OLLAMA_LLAMA_CPP_SOURCE` (env) skips the compat
patches entirely, and Ollama's compat sources then fail to link, so don't use it for this.

## 8. Benchmarks

Decode tok/s computed from the server's `timings` (`predicted_n / predicted_ms`) on llama3.2 3B
Q4_K_M. The prompt is about 17 tokens, with `n_predict: 128`, `ignore_eos`, temperature 0,
`cache_prompt: false`, 3 runs after a warm-up, and server defaults otherwise (`--flash-attn auto`,
`-np 1`, `-c 4096`). `full` uses the request defaults (`layer_stride 4`, `attn_layers 4`,
`sample_topn 64`). `full, stride 1` traces every step. The CPU runs use 10 threads; the GPU is an
RTX 4060 8 GB with CUDA 12.3 and `-ngl 99`.

| Build | no trace | `layers` | `full` | `full`, stride 1 |
|---|---|---|---|---|
| CPU, stock b11963 | 30.38 (30.48 / 30.35 / 30.30) | n/a | n/a | n/a |
| CPU, patched | 30.45 (30.48 / 30.61 / 30.27) | 30.03 (−1.4 %) | 26.49 (−13 %) | 21.00 (−31 %) |
| CUDA, patched | 109.5 (109.34 / 109.66 / 109.40) | 102.9 (−6.0 %) | 71.1 (−35 %) | 41.9 (−62 %) |

Overhead is measured against the patched build's no-trace run. That run matches stock (CPU:
30.45 vs 30.38 tok/s; no stock CUDA build was made). Targets from the plan are < 10 % at `layers`
and < 40 % at `full`, and both are met. The plan's fallback is already the default for attention
(every 4th layer). Dropping `sample_topn` to 32 only moves 94.1 → 95.7 tok/s on CUDA, so 64 stays.

Where the CUDA cost goes (128 tokens):

| Variant | tok/s |
|---|---|
| no trace | 109.5 |
| `full` without forward-pass steps after 8 (`layer_stride: 100000`, sampler stages only) | 94.1 |
| `full` with no attention capture (`attn_layers: 100000`) | 81.4 |
| `full` | 71.1 |

Sampler stages cost about 0.5 ms per token, mostly the `raw` scan over 128k logits. The rest is
the ~11 synchronisations per observed layer, which also disable CUDA-graph replay, plus the FA
attention rebuild. Before commit 0006 (one sync more per Q/K/V view, one transfer per head),
`full` was 66.8 tok/s (−39 %).

## 9. Verification done

- `__trace` events from `/completion` and `/v1/chat/completions` (stream and non-stream) parse with
  `validateEventObject` after adding `v/req/seq`. 0 errors over 2,246 events (12-token `full`
  run) and 8,563 events (30 tokens, `full`, stride 1).
- Output text is byte-identical (md5) for no trace / `layers` / `full` / `full, stride 1, attn every layer,
  per-head` at temperature 0 **and** at temperature 0.8 with seed 42 (the `draw` probe doesn't
  disturb the RNG).
- `-np 2`: a traced and an untraced request running concurrently both produce the same text as a
  solo untraced run.
- Multi-ubatch prefill (1,237 tokens, `n_ubatch` 512) → 3 real `prefill.batch` events 0–512–1024–1237.
- Step mode: `layer` (56 pauses for 2 traced steps × 28 layers, all acked), `sample` (6 stages × 3 steps),
  timeout (28 × `step.timeout`), `continue`, client disconnect (server free again in < 250 ms),
  ack with a wrong `id_slot` → 409.
- FA vs non-FA attention agree (see §4).
- CUDA build (`-ngl 99`): events validate, text md5 identical across no trace / `full` / `full,
  stride 1, attn every layer` at temperature 0.8. `-ngl 14`: `decode.layer.device` is `CPU` for
  layers 0–14 and `CUDA0` for layers 15–27.
- Final build: 28,690 events (30 tokens, stride 1, attention on every layer, per head), 0 schema errors.
- Built through Ollama's `llama/server` CMake project (see §7).
- MoE (`decode.moe`) is **not** verified, because no MoE model is available locally.
