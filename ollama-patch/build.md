# Building and running traced Ollama (Windows)

Base: see `VERSION` (v0.40.13-2-g8861adaa, llama.cpp b11963). The patch series has two parts:

- **Go** (`trace/`, `llm/`, `server/`, `envconfig/`): the WebSocket emitter, the basic tier, and the
  forwarding of llama-server's own trace events.
- **Native**: `llama/compat/100-visualizer-trace.patch`, the squashed llama.cpp change
  (`llama.cpp/README.md` in this directory). Ollama's CMake applies it when it fetches llama.cpp.

With the native payload built from this tree you get the full tier: real per-layer, per-op,
residual, attention, KV-write, prefill-ubatch and sampler-stage events, plus step mode at layer and
sample granularity. With a stock payload (for example a junction to an installed Ollama's `lib`),
the same `ollama.exe` falls back to the basic tier and reports `no_layers`, `no_ops` and so on.

## 1. Apply the patch

```powershell
git clone https://github.com/ollama/ollama.git
cd ollama
git checkout 8861adaa77839c961d3c574232a8865d65a426b6
git am ..\ollama-patch\00*.patch        # 10 commits, applies cleanly
```

The checkout at `ollama/` already has these commits on branch `visualizer-trace`.

### CRLF gotcha (read this before building)

With `core.autocrlf=true` (the Git for Windows default), Git checks out
`llama/compat/*.patch` with CRLF. `git apply` then rejects them ("patch does not apply") and the
native build fails at configure. Patch 0004 adds `*.patch text eol=lf` to `.gitattributes`.
However, `git am` doesn't re-checkout files that were already in the working copy, so after
applying the series, re-checkout the compat patches once:

```bash
rm llama/compat/001-llama-cpp-hooks.patch llama/compat/002-metal-unary-cast-constexpr.patch
git checkout -- llama/compat
git ls-files --eol llama/compat/*.patch   # every line must show w/lf
```

## 2. Toolchain

| Tool | Version used | Notes |
|---|---|---|
| Go | 1.26.10 | `go.mod` requires it. `GOTOOLCHAIN=auto` downloads it if an older Go is installed. |
| C compiler for cgo | TDM-GCC-64 10.3 | Compiling works. Go 1.26 *external* linking with it produces a PE that Windows refuses to run, so link with `-ldflags=-linkmode=internal`. `CGO_ENABLED=0` fails (`mlx/`). |
| Visual Studio 2022 Community | MSVC 14.44 | "Desktop development with C++" |
| CMake | ≥ 3.24 (`C:\Program Files\CMake\bin`) | |
| CUDA | 12.3 | MSVC 14.44 needs two workarounds, see below. CUDA ≥ 12.4 needs neither. |
| Git + network | | CMake clones llama.cpp b11963 from GitHub (shallow) |

Ninja isn't needed. The Visual Studio generator works. Ollama passes `-T cuda=<toolkit>` to the CUDA
sub-build by itself.

## 3. Build the native payload (CPU + CUDA) through Ollama's CMake

Run this from `cmd.exe` (`ollama-patch\build-native.bat` does the same):

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Program Files\CMake\bin;%PATH%
rem CUDA 12.3 + MSVC 14.44 workarounds (both are inherited by the nested CUDA build)
set CL=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
set NVCC_APPEND_FLAGS=-allow-unsupported-compiler

cd /d C:\BitsPlus\VisOllamInference
cmake -S ollama -B build-ollama-full -G "Visual Studio 17 2022" -A x64 ^
  -DOLLAMA_LLAMA_BACKENDS=cuda_v12 -DCMAKE_CUDA_ARCHITECTURES=89 -DOLLAMA_BUILD_PARALLEL=16
cmake --build build-ollama-full --config Release ^
  --target ollama-llama-server-local ollama-llama-server-cuda_v12 -- /m:2
```

- `CMAKE_CUDA_ARCHITECTURES=89` is the RTX 4060 only, which keeps the build short. Drop it to get
  Ollama's default Windows list (`50-virtual … 120`).
- Configure clones llama.cpp into `build-ollama-full/_deps/llama_cpp-src` and applies every
  `llama/compat/*.patch`. The log must show
  `llama/compat: applied 100-visualizer-trace.patch`. A re-configure reports
  `already applied, skipping` for all three.
- Only the two payload targets are built. The root project's `ollama-go` target builds `ollama.exe`
  with external linking, which hits the TDM-GCC problem above. Build Go separately (step 4).
- `warning MSB8027: Two or more files with the name of llama.cpp` is upstream noise and harmless.

Output (staged install), `build-ollama-full/lib/ollama/`:

```
llama-server.exe  llama-server-impl.dll  llama.dll  llama-common.dll  mtmd.dll
ggml.dll  ggml-base.dll  ggml-cpu-{x64,sse42,sandybridge,haswell,skylakex,icelake,alderlake,cannonlake,cascadelake}.dll
llama-quantize.exe  llama-quantize-impl.dll  vcomp140.dll  msvcp140*.dll  vcruntime140*.dll  concrt140.dll  *_LICENSE
cuda_v12/ggml-cuda.dll  cuda_v12/cublas64_12.dll  cuda_v12/cublasLt64_12.dll  cuda_v12/cudart64_12.dll  cuda_v12/msvc runtime dlls
```

That's the same layout as the stock `%LOCALAPPDATA%\Programs\Ollama\lib\ollama`, which is built with
clang/MinGW and therefore names the files `libllama*.dll`. `ggml-cpu-*` variants are loaded
dynamically (`GGML_BACKEND_DL`) and `cuda_v12\ggml-cuda.dll` by GPU discovery. The log shows
`inference compute … library=CUDA … libdirs=ollama,cuda_v12`. There is no `cuda_v13`, `rocm` or
`vulkan` here, so add them to `OLLAMA_LLAMA_BACKENDS` if needed.

## 4. Build ollama.exe and assemble `C:\ollama-trace-full`

```bash
# Git Bash
cd ollama
go build -ldflags="-linkmode=internal -X github.com/ollama/ollama/version.Version=0.40.13-trace-full" \
  -o C:/ollama-trace-full/ollama.exe .
cp -r ../build-ollama-full/lib C:/ollama-trace-full/
```

```
C:\ollama-trace-full\
  ollama.exe
  lib\ollama\llama-server.exe, *.dll           (CPU payload, patched llama-server)
  lib\ollama\cuda_v12\ggml-cuda.dll, cublas…   (CUDA 12 backend)
```

`FindLlamaServer` (`llm/llama_binary.go`) looks in `<exe dir>\lib\ollama`, and GPU discovery uses
`lib\ollama\cuda_v12`. `C:\ollama-trace\` (Go-only patch + junction to the stock payload) is a
separate install. Leave it alone if it's still in use.

## 5. Run it with the visualizer

1. Start the visualizer. Its hub listens on `127.0.0.1:4711` by default (Settings → *Trace socket
   port*). Producers connect to `/ingest`.
2. Start Ollama. The Ollama port is your choice; use 11434 only if nothing else is on it:

   ```powershell
   $env:OLLAMA_HOST = "127.0.0.1:11436"
   $env:OLLAMA_TRACE_URL = "ws://127.0.0.1:4711/ingest"
   C:\ollama-trace-full\ollama.exe serve
   ```

   The log shows `inference tracing enabled`, then `trace hub connected`, and on the first traced
   request `llama-server trace support patched=true source="binary scan"`, then `source=probe status=409`.
3. In the app: Settings → *Ollama URL* = `http://127.0.0.1:11436`, choose a model, set detail to
   **Full** (or Layers), and run. *Step* + *Step-mode runner granularity* = `layer`/`sample` makes
   llama-server pause per layer / per sampler stage. `token` keeps the Go token gate.

Models come from `%USERPROFILE%\.ollama\models` (`OLLAMA_MODELS` overrides it).

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_TRACE_URL` | unset (tracing off) | Hub WebSocket, e.g. `ws://127.0.0.1:4711/ingest`. Unset: every trace call is a nil no-op, and llama-server requests are unchanged. |
| `OLLAMA_TRACE_DETAIL` | `basic` | Default when a request has no `X-Trace-Detail`. `layers`/`full` ask llama-server for its own events. |
| `OLLAMA_TRACE_STEP` | unset | Step mode for every request (`token`, `layer`, `sample`). Normally the header is used. |
| `OLLAMA_TRACE_STEP_TIMEOUT` | `60s` | Wait per pause (token gate in Go; passed to llama-server as `step_timeout_ms`). |
| `OLLAMA_TRACE_LAYER_STRIDE` | `4` | llama-server `layer_stride`. Forward-pass events on steps < 8, then every Nth step. `1` = every step. Layer step mode forces 1. |
| `OLLAMA_TRACE_ATTN_LAYERS` | `4` | llama-server `attn_layers`: `decode.attn` on layers where `layer % N == 0`. |
| `OLLAMA_TRACE_ATTN_HEADS` | off | llama-server `attn_heads`: also one `decode.attn` per head. |

## Request headers (sent by the app)

| Header | Meaning |
|---|---|
| `X-Trace-Request-Id` | Correlation id, echoed as `req`. If missing, Ollama generates one and returns it (CORS-exposed). |
| `X-Trace-Detail` | `basic` \| `layers` \| `full` |
| `X-Trace-Step` | `token` (Go gate, llama-server keeps generating), `layer` or `sample` (llama-server pauses; acks are forwarded to `POST /trace/ack`). |

## Tests

```bash
go test ./trace/ ./envconfig/ ./llm/ ./server/...
go test -race ./trace/
```

`llm/llama_server_runner_trace_test.go` runs a fake llama-server and a real WebSocket test hub:
chunk parsing, step shift, clock rebasing, de-duplication, detail filtering, the chat `choices:[]`
and final-array chunks, and ack/continue forwarding.

## Benchmarks

See NOTES.md § Performance.
