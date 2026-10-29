# LLM Inference Visualizer — Implementation Plan

An Electron app that animates every stage of LLM inference (prompt → tokenize → prefill → decode/sample → streamed response) in real time, fed by an instrumented Ollama build.

This plan is written for execution by Claude Code. Each phase is self-contained, has explicit deliverables and acceptance criteria, and should be completed (and committed) before starting the next. Phases 0–4 do **not** require touching Ollama at all; they build against a recorded/mocked trace so the UI can be iterated quickly. Ollama instrumentation lands in Phase 5.

---

## Guiding decisions (do not re-litigate during implementation)

| Concern | Decision |
|---|---|
| Transport Ollama → app | WebSocket, newline-delimited JSON events. App hosts the server; Ollama connects as a client when `OLLAMA_TRACE_URL` is set. |
| Token/response data | App calls Ollama's public `/api/chat` with `stream: true` itself. The trace socket only adds what the public API cannot see. |
| Correlation | App generates a UUID per request, sends it as header `X-Trace-Request-Id`; patched Ollama echoes it in every trace event. |
| Renderer ingestion | Renderer connects to the trace WebSocket directly (localhost). No per-event IPC through main. |
| Animation model | Playback clock with per-stage minimum durations + catch-up. Visuals are driven by a state machine fed from a time-ordered event buffer, never from raw socket callbacks. |
| Rendering | DOM/SVG + Motion for phase pipeline, token stream, sampling bars. Canvas 2D (PixiJS) for layer grid / KV-cache / attention. One `requestAnimationFrame` loop owns per-frame drawing. |
| Stack | Electron + Vite + TypeScript + React (controls/layout only). `electron-vite` scaffold. Zustand for app state. Vitest for unit tests, Playwright for e2e. |
| Schema | Versioned (`v: 1`). Defined once in a shared package consumed by both the TS app and (mirrored by hand) the Go emitter. |

---

## Phase 0 — Repository scaffold

**Goal:** A running Electron window with the toolchain, lint, test and CI skeleton in place.

### Tasks
1. Create a monorepo with pnpm workspaces:
   ```
   /apps/visualizer          # Electron app (electron-vite, React, TS)
   /packages/trace-schema    # Event types + zod validators + fixtures
   /packages/trace-mock      # Mock emitter (Node) that replays/generates traces
   /ollama-patch/            # Patch files + build notes (filled in Phase 5)
   /traces/                  # Recorded .ndjson traces (fixtures)
   ```
2. Scaffold `apps/visualizer` with `electron-vite` (React + TS template). Enable `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true`.
3. Add ESLint + Prettier + `tsconfig` strict mode. Add Vitest. Add a `pnpm dev`, `pnpm test`, `pnpm lint`, `pnpm build` at root.
4. Add a GitHub Actions workflow running lint + test on push.
5. Write `README.md` with the architecture summary from this plan and a "how to run with mock trace" section (fill in as phases complete).

### Acceptance
- `pnpm dev` opens an Electron window showing "LLM Inference Visualizer".
- `pnpm test` and `pnpm lint` pass with zero findings.

---

## Phase 1 — Trace schema and fixtures

**Goal:** A single source of truth for every event the system can emit, plus realistic recorded fixtures to develop against.

### Tasks
1. In `packages/trace-schema`, define the event union with zod and export inferred TS types. Every event carries:
   ```ts
   { v: 1, req: string, seq: number, t_ns: number, type: string, ...payload }
   ```
2. Event types (initial set — keep extensible):
   - `request.received` `{ model, prompt_chars, options }`
   - `model.load` `{ phase: "start" | "done", layers, ctx, vram_mb, ms? }`
   - `tokenize.done` `{ n, tokens: Array<{ id: number, text: string }> }`
   - `prefill.batch` `{ batch_i, of, from, to, ms }`
   - `prefill.done` `{ n_tokens, ms }`
   - `decode.step.start` `{ step, kv_used, kv_total }`
   - `decode.layer` `{ step, layer, ms, norm? }` *(optional, downsampled)*
   - `decode.logits` `{ step, topk: Array<{ id, text, p }>, entropy }`
   - `decode.sample` `{ step, chosen_id, temp, top_p, top_k, method }`
   - `decode.token` `{ step, text, is_eos }`
   - `request.done` `{ stats: { prompt_eval_count, prompt_eval_ms, eval_count, eval_ms, load_ms, total_ms } }`
   - `request.error` `{ message }`

   **Detailed-tier events** (consumed by Phase 4b, produced by Phase 5 when `OLLAMA_TRACE_DETAIL=full`; all `droppable` except `decode.sample.stage`):
   - `tokenize.merge` `{ step, span: [from, to], pieces: string[], result_id }` *(may be synthesised in-app, see 4b)*
   - `prefill.layer` `{ batch_i, layer, ms }`
   - `decode.op` `{ step, layer, op: "attn_qkv"|"attn_scores"|"attn_out"|"mlp_up"|"mlp_act"|"mlp_down"|"norm", ms }`
   - `decode.residual` `{ step, layer, norm, delta_norm }`
   - `decode.attn` `{ step, layer, head?: number, top: Array<{ pos, w }> }` *(top-k attended positions, heads averaged unless `head` given)*
   - `decode.kv.write` `{ step, layer, pos }`
   - `decode.sample.stage` `{ step, stage: "raw"|"rep_penalty"|"temperature"|"top_k"|"top_p"|"min_p"|"softmax"|"draw", candidates: Array<{ id, v }>, cutoff?: number, draw?: number }`
   - `stream.chunk` `{ step, bytes, utf8_complete, text }`
3. Assign each type a **priority** (`critical` for phase boundaries and tokens, `droppable` for `decode.layer`) and export a `PRIORITY` map. Export `parseEvent(line: string)` that validates and returns a typed event or an error.
4. Write a `traces/README.md` documenting the NDJSON format.
5. Create a hand-authored fixture `traces/fixtures/short-answer.ndjson` (~20 tokens, 8 layers, with realistic `t_ns` spacing: load 800 ms, tokenize 2 ms, prefill 3 batches × 40 ms, decode ~12 ms/token, layer ~0.3 ms). Write a small generator script in `trace-mock` to produce it rather than typing it by hand, and commit the output.
6. Unit tests: every fixture line parses; invalid lines produce typed errors; `seq` is strictly increasing per `req`.

### Acceptance
- `pnpm --filter trace-schema test` passes.
- The fixture replays through `parseEvent` with zero errors.

---

## Phase 2 — Mock trace emitter

**Goal:** A standalone Node process that behaves exactly like instrumented Ollama from the app's point of view, so all UI work can proceed without Ollama.

### Tasks
1. `packages/trace-mock` CLI with two modes:
   - `replay <file.ndjson> [--speed 1.0] [--loop]` — connects to `ws://127.0.0.1:4711`, replays events honoring the original `t_ns` deltas (scaled by `--speed`).
   - `generate --model llama3 --layers 32 --prompt-tokens 120 --response-tokens 200 --tok-per-sec 60` — synthesizes a plausible trace live, including top-k logits with a realistic probability distribution and an occasional low-confidence step.
2. Emit a fresh UUID `req` per run unless `--req <id>` is given (the app will pass this when it also drives a real/mocked chat request).
3. Implement reconnect with backoff; if the app is not listening, log once and keep retrying quietly.
4. Also provide `mock-ollama-http`: a tiny HTTP server on `:11434` that implements just enough of `POST /api/chat` (streaming) to return the same tokens the generator emits, reading `X-Trace-Request-Id` and using it as `req`. This lets the app exercise its full real-world path (public API + side channel) end to end against mocks.
5. Tests: generated traces validate against the schema; replay timing error < 5 ms at speed 1.0.

### Acceptance
- `pnpm mock generate` + `pnpm mock-ollama-http` run together; a throwaway WS client receives a valid, correlated stream.

---

## Phase 3 — Ingestion, buffering and the playback clock

**Goal:** The core engine: events in, a deterministic, time-dilated, state-machine-driven view model out. No visuals yet beyond a debug panel.

### Tasks
1. **Main process**: host the trace WebSocket server on `127.0.0.1:4711` (port configurable). Accept connections from both Ollama/mock (producers) and renderer windows (consumers); fan out producer messages to consumers verbatim. Survive renderer reloads. Expose connection status via a single IPC channel (`trace:status`) — this is the *only* trace-related IPC.
2. **Renderer `TraceClient`**: connects to the server, parses lines with `parseEvent`, pushes into a per-request ring buffer (cap e.g. 50k events). On overflow, drop oldest `droppable` events first, never `critical` ones. Count drops for the debug panel.
3. **`PlaybackClock`**:
   - Maintains `tp` (playback time, in trace ns) advanced on each rAF tick by `dt * speed`.
   - Per-stage minimum on-screen durations (configurable, defaults): `model.load` 600 ms, `tokenize` 400 ms, `prefill.batch` 150 ms, `decode.step` 80 ms, `decode.layer` 8 ms.
   - Holds `tp` back so no stage completes before its minimum has elapsed on screen.
   - Catch-up: if `(now_ns - tp) > lagThreshold` (default 2 s), increase `speed` smoothly up to `maxSpeed` (default 8×) and mark state `catchingUp`; return to 1× when lag < 250 ms. Expose current dilation factor.
   - Supports `pause()`, `resume()`, `seek(t)`, `setSpeed(x)` for the scrubber.
4. **`InferenceStateMachine`**: consumes events whose `t_ns <= tp` and produces an immutable `ViewModel`:
   ```ts
   {
     phase: 'idle'|'loading'|'tokenizing'|'prefill'|'decoding'|'done'|'error',
     model, promptTokens[], responseTokens[], currentStep,
     prefill: { batch, of, progress },
     kv: { used, total },
     layers: Array<{ idx, active, lastMs, norm }>,
     sampling: { topk[], chosen?, temp, top_p, top_k, entropy },
     stats, timeline: Array<{ phase, from, to }>,
     clock: { tp, speed, lagMs, catchingUp, dropped }
   }
   ```
   Must be tolerant of missing `droppable` events and of out-of-order arrival within a small window (sort by `seq` on insert).
5. Wire into a Zustand store but update it at most once per rAF (batch), not per event.
6. **Debug panel** (temporary React component): raw event log tail, phase, `tp` vs real time, speed, drop count.
7. Tests (Vitest, no DOM): feed the fixture through clock + state machine with a simulated rAF; assert the phase sequence, that each stage is on screen ≥ its minimum, that catch-up engages when a burst of 2000 layer events arrives, and that dropping all `decode.layer` events does not change the phase sequence or token output.

### Acceptance
- With `mock generate` running, the debug panel shows the phase progressing through all stages, the token list growing, and the dilation factor rising during the decode burst and settling back to 1×.
- Engine tests pass and run in < 2 s.

---

## Phase 4 — Visualization

**Goal:** The actual animated experience. Build it against the mock, iterate freely.

### Layout (single window, 16:9)
```
┌──────────────────────────────────────────────────────────────┐
│ [Model ▾] [Prompt input ........................] [Run] [⏸ ⏮ ⏭ 1×]│
├───────────────┬──────────────────────────────┬───────────────┤
│ Phase pipeline│  Layer stack / KV cache      │ Sampling      │
│ (vertical)    │  (Canvas)                    │ top-k bars    │
├───────────────┴──────────────────────────────┴───────────────┤
│ Prompt tokens (chips)  →  Response stream (typewriter)       │
├──────────────────────────────────────────────────────────────┤
│ Timeline scrubber with phase bands · stats (tok/s, TTFT …)   │
└──────────────────────────────────────────────────────────────┘
```

### Tasks
1. **Phase pipeline** (SVG + Motion): nodes for Load → Tokenize → Prefill → Decode → Done. Active node pulses; completed nodes show measured duration; a flowing particle/line indicates data movement. Decode node shows `step N / kv used`.
2. **Token chips**: prompt tokens appear one by one during `tokenizing` (staggered), then light up batch-by-batch during `prefill`. Response tokens append during `decode.token` with a short pop animation; EOS gets a distinct marker. Hovering a token shows its id and (for response tokens) the top-k it beat.
3. **Layer stack** (PixiJS canvas): one row per layer. On each `decode.layer`, the row flashes and fades over its minimum duration; colour intensity = normalised `ms` or `norm`. Below it, a KV-cache bar fills from `kv_used/kv_total`. Must sustain 60 fps with 80 layers and 4k context (verify with the Chromium FPS meter; keep draw calls batched).
4. **Sampling panel**: horizontal bars for the top-k candidates, animated with spring transitions between steps; the chosen token highlights; show `temperature / top_p / top_k` and entropy as a small gauge. Low-entropy vs high-entropy steps should be visually obvious.
5. **Response stream**: typewriter output that is *exactly* the concatenated `decode.token` texts; add a blinking caret while `decoding`.
6. **Timeline scrubber**: horizontal bands per phase proportional to real duration; playhead at `tp`; drag to seek (pauses live mode; a "Live" button re-attaches). Show stats: TTFT, prompt tok/s, gen tok/s, total, plus the current dilation factor ("showing at 0.25×").
7. **Theme and motion**: dark theme, one accent colour per phase, easing curves consistent across components. Respect `prefers-reduced-motion` by swapping animations for instant state changes.
8. Remove/hide the Phase 3 debug panel behind a `Ctrl+Shift+D` toggle.
9. Playwright e2e: launch app + mock, click Run, assert the phase pipeline reaches Done and the response text equals the mock's generated text.

### Acceptance
- Full run against `mock generate --response-tokens 300` looks continuous, never freezes, and the final response text matches the mock byte-for-byte.
- 60 fps sustained in the layer canvas under the 80-layer / 4k-context scenario.

---

## Phase 4b — Detailed visualization tier

**Goal:** A "Full detail" mode that exposes the inside of each stage: tokenizer merges, the sub-layer forward pass with attention and residual stream, the complete sampling chain, and byte-level streaming. Built against the mock (extend `trace-mock` first) so it doesn't block on Phase 5.

### Detail levels
Add a global **Detail** toggle: `Basic` (Phase 4 as-is) · `Layers` (Phase 4 + per-layer timing/norm) · `Full` (everything below). The level drives three things: which events the app asks Ollama to emit (sent as header `X-Trace-Detail`), which components mount, and the playback clock's stage minimums.

### Pacing rules for Full detail
- Sub-step minimums: `decode.op` 25 ms, `decode.attn` 60 ms, `decode.sample.stage` 250 ms, `tokenize.merge` 60 ms, `stream.chunk` 40 ms. One token ≈ 32 layers × 7 ops + 8 sampling stages ≈ 5–8 s on screen. This is a slow-motion mode by design; display the dilation factor prominently.
- **Auto-degrade**: when live lag exceeds `fullDetailLagThreshold` (default 4 s), the clock drops to `Layers` detail for the backlog, shows a "fast-forwarding" badge, and returns to `Full` once caught up. Full-detail events in the backlog are skipped, not played faster.
- **Step mode is the primary way to use Full detail.** Add "Step token" / "Step layer" / "Step sampling stage" / "Continue" controls; in replay they just advance `tp` to the next event boundary of that kind; in live mode they also send `ack` (see Phase 5).

### Tasks
1. **Mock support**: extend `trace-mock generate` to emit all detailed-tier events with plausible values (attention concentrating on recent tokens plus a few "anchor" positions, residual norms growing roughly linearly with layer index, sampling stages that actually prune a synthetic logit vector). Add a `traces/fixtures/full-detail.ndjson`.
2. **Tokenizer merge animation** (DOM/SVG + Motion): the raw prompt text sits as individual characters; merges play as adjacent pieces sliding together into a chip, with the merged token id appearing. Byte-fallback and special/chat-template tokens get distinct styling. Implement merges **in-app** with a JS BPE tokenizer loaded from the model's vocab/merges (fetched via `/api/show` or shipped per-model), validated against the `tokenize.done` ids; fall back to `tokenize.merge` events if present. Mismatch → show a warning badge and skip merge animation.
3. **Forward-pass view** (WebGL, PixiJS; optional Three.js 3D tower behind a flag):
   - Each layer expands into sub-blocks: `norm → attn_qkv → attn_scores → attn_out → (+residual) → norm → mlp_up → mlp_act → mlp_down → (+residual)`. The active op glows for its minimum duration; op timing is shown numerically on hover.
   - A **residual stream** column runs the full height; its width/brightness at each layer = `decode.residual.norm`, with `delta_norm` shown as the layer's contribution pulse.
   - **Attention arcs**: on `decode.attn`, draw arcs from the current position to the top-k attended context tokens (rendered as a token strip along the bottom of the canvas), arc opacity = weight. Per-head mode when `head` is present (colour per head, max 32). Must handle 4k context by rendering the token strip virtually (only visible range) and limiting arcs to top-8 per layer.
   - **KV cache**: a per-layer slot grid; `decode.kv.write` lights the newly written slot.
   - Performance: ≥ 60 fps at 80 layers, 4k context, 8 arcs/layer, measured on the dev machine; keep all per-frame geometry in preallocated typed arrays / Pixi `Graphics` batches, no per-frame object allocation.
4. **Sampling funnel** (SVG + Motion): a vertical pipeline of stages driven by `decode.sample.stage`:
   - `raw`: histogram of the top-N logits (N = 64 by default) plus a sparkline of the full distribution shape.
   - `rep_penalty`: bars for tokens already in the output shrink, with lines linking them to their occurrences in the response.
   - `temperature`: bars stretch/flatten; show the formula with the current T.
   - `top_k` / `top_p` / `min_p`: a cutoff line sweeps in; excluded bars fade. For `top_p`, animate the cumulative-probability curve crossing p.
   - `softmax`: bars morph to probabilities, with entropy gauge updating.
   - `draw`: a marker lands at `draw` in the cumulative bar; the selected token highlights and flies into the response stream.
   - Any stage absent from the trace (e.g. sampler not configured with min_p) is shown collapsed and greyed.
5. **Stream view**: show `stream.chunk` bytes arriving; incomplete UTF-8 sequences render as pending byte boxes until `utf8_complete`; a small inter-chunk latency sparkline runs alongside.
6. **Inspector**: clicking any token, layer, op, or sampling stage pauses playback and opens a side panel with the raw event JSON and derived numbers (e.g. this layer's share of step time, this token's probability at each sampling stage).
7. **State machine / view model extensions**: add `ops[]`, `residual[]`, `attn[]`, `kvWrites[]`, `samplingStages[]`, `streamChunks[]` per step; ensure Basic/Layers mode ignores these events without cost (filter at ingestion by detail level).
8. Tests: engine tests for auto-degrade (full-detail backlog skipped, phase/token output unchanged); JS tokenizer matches `tokenize.done` ids on the fixtures; Playwright e2e for Full detail in step mode stepping through one complete token.

### Acceptance
- In Full detail + step mode, a single token can be stepped through from residual input to drawn sample with every sub-step readable, and the inspector shows matching raw data.
- Switching to live mode with the mock at 60 tok/s auto-degrades to Layers, shows the badge, and recovers to Full after the run ends.
- Frame rate targets met; Basic mode performance is unchanged from Phase 4.

---

## Phase 5 — Ollama instrumentation

**Goal:** Replace the mock with real Ollama events. Keep the patch small, env-gated and rebasable.

### Prerequisites
- Go toolchain, Ollama source checked out at a pinned tag (record the tag in `ollama-patch/VERSION`).
- Read the runner code first (`runner/` — the llama.cpp-backed runner and/or the Go `ollamarunner` engine, depending on the pinned version) and identify the exact functions for: tokenize, prefill batch loop, decode step, sampling, detokenize/stream, completion stats. Write these findings into `ollama-patch/NOTES.md` before coding.

### Tasks
1. **Emitter package** `trace/` inside Ollama (new, ~150 lines):
   - `trace.Init()` reads `OLLAMA_TRACE_URL`; if unset, every `Emit` is a no-op (one branch, no allocation).
   - Buffered `chan []byte` (cap 10k) + single writer goroutine owning a WebSocket (`nhooyr.io/websocket`). Reconnect with backoff. **Never block the caller**: on a full channel, drop `droppable` events, keep `critical` ones (two channels or a priority flag).
   - `t_ns` from a monotonic clock; per-request `seq` via an atomic counter in a `RequestTracer` struct.
2. **Request correlation**: in the HTTP server handler for `/api/chat` and `/api/generate`, read `X-Trace-Request-Id` (fallback: generate one) and thread it to the runner. If the runner runs as a subprocess, pass it in the existing request payload.
3. **Hook points** (emit the Phase 1 events):
   - server: `request.received`, `model.load start/done`, `request.done` (from existing timing stats), `request.error`.
   - runner: `tokenize.done`, `prefill.batch` per batch, `prefill.done`, `decode.step.start`, `decode.logits` (top-k from the sampler's candidate list — grab before the sampler commits), `decode.sample`, `decode.token`.
   - `decode.layer`: at detail `layers` or above. Implement via the backend eval callback (`ggml_backend_sched_set_eval_callback` / the Go engine's equivalent). Emit per-layer timing only. Downsample: all layers for steps < 8, then every `OLLAMA_TRACE_LAYER_STRIDE` (default 4).
4. **Detail levels**: `OLLAMA_TRACE_DETAIL=basic|layers|full` (default `basic`), overridable per request by the `X-Trace-Detail` header the app sends. `full` enables the Phase 4b events:
   - `decode.op` / `prefill.layer`: op-level timing from the eval callback, classified by tensor name (`attn_q`, `kq`, `ffn_up`, …) into the op enum. Timing only; no tensor reads.
   - `decode.residual`: L2 norm of the residual stream after each layer, computed on-device where the backend allows (a reduce op appended to the graph) and read back as one float per layer; otherwise skipped with a `capabilities` note in `request.received`.
   - `decode.attn`: top-k (default 8) attended positions per layer from the softmaxed KQ tensor, heads averaged. Reduce on-device (top-k op on the averaged row) before copying; if the backend can't, fall back to copying the single current-token row (ctx × heads floats) for a subset of layers (`OLLAMA_TRACE_ATTN_LAYERS`, default every 4th) and reducing on CPU. Per-head output only when `OLLAMA_TRACE_ATTN_HEADS=1`.
   - `decode.kv.write`: emitted alongside the cache append (position only).
   - `decode.sample.stage`: snapshot the candidate list after each sampler stage (raw → rep_penalty → temperature → top_k → top_p → min_p → softmax → draw), truncated to top-64 by value, plus the cutoff index and the random draw. Hook into the sampler chain; stages that aren't configured are simply not emitted.
   - `stream.chunk`: in the streaming response writer, with the pending-byte count from the UTF-8 assembler.
   - `tokenize.merge`: **not** implemented in the runner initially (the app synthesises merges in JS, Phase 4b task 2); leave a TODO with the hook location.
5. **Step mode** (`OLLAMA_TRACE_STEP=1` or header `X-Trace-Step: token|layer|sample`): before each decode step (or layer / sampler stage, per granularity), the runner blocks on an `ack` message from the trace socket, with a timeout (`OLLAMA_TRACE_STEP_TIMEOUT`, default 60 s) after which it continues and emits `step.timeout`. Only one request may be in step mode at a time; others proceed normally. Document clearly that step mode invalidates all timing stats.
6. **Performance guard**: benchmark tokens/sec with trace off, `basic`, `layers`, `full`. Record numbers in `NOTES.md`. Targets: < 2 % overhead at `basic`, < 10 % at `layers`, < 40 % at `full` (full is a visualization mode, not a production mode). If `full` exceeds target, reduce attention capture to every 4th layer and sampling snapshots to top-32 before anything else.
7. Produce `ollama-patch/*.patch` via `git format-patch` against the pinned tag, plus `build.md` with exact build/run steps and the env vars.
8. **App changes**: the Run button now really calls `http://127.0.0.1:11434/api/chat` (streaming) with the trace, detail and step headers; the model dropdown is populated from `/api/tags`; the app sends `ack` messages when the user presses Step/Continue. Keep the mock path selectable via a settings toggle (`Settings → Source: Ollama | Mock`).
9. **Record real traces** for three models (small/medium/large) at `basic` and `full` detail into `traces/real/` and add them to the replay fixtures.
10. Validate: response text from `/api/chat` stream equals the concatenated `decode.token` texts (assert in the app, surface a warning badge if not); the JS tokenizer's ids match `tokenize.done` for each recorded model.

### Acceptance
- A real prompt against patched Ollama animates end-to-end at every detail level with no code path differences from the mock run.
- Step mode steps a real token through layers and sampling stages from the app.
- Overhead numbers documented and within target.
- Patch applies cleanly to the pinned tag with `git am`.

---

## Phase 6 — Polish, robustness and packaging

### Tasks
1. **Guided tour mode**: a scripted walkthrough that runs a canned prompt in Full detail + step mode with captions explaining each stage (tokenization, prefill, attention, MLP, sampling), aimed at demos and teaching.
2. **Error and edge handling**: Ollama not running, model not pulled, trace socket connected but no events (unpatched Ollama → show "running without instrumentation; token stream only"), app closed mid-request, very long prompts (collapse token chips beyond N with a count).
3. **Session recording**: save every run as `.ndjson` + the chat response; "Open trace…" to replay offline with the scrubber.
4. **Settings**: port, stage and sub-step minimum durations, max speed, auto-degrade threshold, layer/attention stride, per-head attention, reduced motion.
5. **Packaging**: `electron-builder` configs for macOS, Windows, Linux; CI builds artifacts on tag.
6. **Docs**: finalize `README.md` (quick start, architecture diagram, event reference auto-generated from the zod schema, Ollama patch instructions, troubleshooting).

### Acceptance
- Packaged app runs on a clean machine with stock Ollama (degraded mode) and with patched Ollama (full mode).
- All tests green in CI; release artifacts produced on tag.

---

## Phase ordering and parallelism

- Phases 0 → 1 → 2 → 3 → 4 → 4b are strictly sequential.
- Phase 5 can begin as soon as Phase 1 (schema, including the detailed-tier events) is frozen, in parallel with 3–4b, since it only depends on the event contract. If parallelising, freeze the schema with a `v: 1` tag and treat changes as breaking. Within Phase 5, land tasks 1–3 (basic/layers) first and verify against Phase 4 before starting task 4 (full detail).
- Phase 6 last.

## Definition of done (whole project)

- A user types a prompt, presses Run, and watches load → tokenize → prefill → decode/sample → response stream animate continuously, with real timings and a visible time-dilation factor, for any model installed in a patched Ollama.
- In Full detail, the user can step a single real token through tokenizer merges, every layer's attention and MLP sub-blocks with attention arcs and residual-stream growth, and each sampling stage down to the random draw, with the inspector showing the underlying numbers.
- The same app works in degraded mode against stock Ollama and in offline mode against recorded traces.
- Engine logic is unit-tested, the UI is e2e-tested, and the Ollama patch is small, documented and benchmarked.
