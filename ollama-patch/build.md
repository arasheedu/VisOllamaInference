# Building and running traced Ollama (Windows)

Base: see `VERSION` (v0.40.13-2-g8861adaa). The patch is Go-only. It reuses the stock
llama.cpp payload (`lib\ollama\llama-server.exe` + DLLs) from an installed Ollama, so it
does not need CMake or Visual Studio.

## 1. Apply the patch

```powershell
git clone https://github.com/ollama/ollama.git
cd ollama
git checkout 8861adaa77839c961d3c574232a8865d65a426b6
git am ..\ollama-patch\*.patch          # 3 commits, applies cleanly
```

The checkout at `ollama/` already has these commits on branch `visualizer-trace`.

## 2. Toolchain

- **Go 1.26.10**: `go.mod` requires it. If an older Go is installed (for example 1.23.5 in
  `C:\Program Files\Go`), `GOTOOLCHAIN=auto` (the default) downloads 1.26.10 on the first
  `go` command.
- **A C compiler for cgo**: `go-sqlite3` and `mlx` need cgo. TDM-GCC-64 10.3 (`C:\TDM-GCC-64\bin\gcc`)
  works for compiling. **However, Go 1.26 external linking with that toolchain produces a PE that
  Windows refuses to run** ("Exec format error" / "not a valid application for this OS platform").
  Build with the internal linker instead (see below). With a newer MinGW (for example
  llvm-mingw or MSYS2 UCRT64 gcc ≥ 13), plain `go build .` should work as well.
- `CGO_ENABLED=0` does not work, because `mlx/` fails to compile without cgo.

## 3. Build

```bash
# Git Bash
export PATH="/c/Program Files/Go/bin:$PATH"
cd ollama
go build -ldflags=-linkmode=internal -o C:/ollama-trace/ollama.exe .
```

Optional: set the version string with
`-ldflags="-linkmode=internal -X github.com/ollama/ollama/version.Version=0.40.13-trace"`.

`go build ./...` also fails on `app/ui` (`pattern app/dist: no matching files found`) because
the desktop UI bundle is not built. This happens on the base commit too and does not affect `ollama.exe`.

## 4. Give it the native payload

`FindLlamaServer` (`llm/llama_binary.go`) searches `<exe dir>\lib\ollama` (and `..\lib\ollama`).
Point that at the stock install with a directory junction, which needs no admin rights:

```powershell
cmd /c mklink /J C:\ollama-trace\lib "$env:LOCALAPPDATA\Programs\Ollama\lib"
```

The tested stock install was 0.40.6. Its `llama-server.exe` accepted everything this patch
sends: `n_probs`, `timings_per_token`, `/tokenize` `with_pieces`, `/apply-template`.

## 5. Run

```bash
# stop the tray app, or use another port
OLLAMA_HOST=127.0.0.1:11434 \
OLLAMA_TRACE_URL=ws://127.0.0.1:4711/ingest \
C:/ollama-trace/ollama.exe serve
```

Models come from the default `%USERPROFILE%\.ollama\models` (`OLLAMA_MODELS` overrides it).
The log shows `inference tracing enabled` and, once the app is listening, `trace hub connected`.
Ollama connects at startup and reconnects with backoff (250 ms → 5 s), so the app can start
before or after Ollama.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_TRACE_URL` | unset (tracing off) | Hub WebSocket, e.g. `ws://127.0.0.1:4711/ingest`. When unset, every trace call is a nil-receiver no-op. |
| `OLLAMA_TRACE_DETAIL` | `basic` | `basic` \| `layers` \| `full`. Default when a request has no `X-Trace-Detail`. Only `full` changes output (adds `stream.chunk`); `layers` is accepted but has no extra events (see NOTES.md). |
| `OLLAMA_TRACE_STEP` | unset | `token` (or `1`) puts every request in step mode. Normally you'd use the header instead. |
| `OLLAMA_TRACE_STEP_TIMEOUT` | `60s` | Step-mode wait per token. Takes a Go duration (`500ms`, `2m`) or seconds. |

## Request headers (sent by the app)

| Header | Meaning |
|---|---|
| `X-Trace-Request-Id` | Correlation id echoed as `req` in every event. If missing, Ollama generates a UUID and returns it in the `X-Trace-Request-Id` response header. That header is CORS-exposed. |
| `X-Trace-Detail` | `basic` \| `layers` \| `full`; overrides `OLLAMA_TRACE_DETAIL`. |
| `X-Trace-Step` | `token` (also `1`/`true`/`layer`/`sample`; all behave as `token`). |

All three are in the CORS allow-list, so the Electron renderer can send them directly.

## Pointing it at the app

1. Start the visualizer. It hosts the hub on `127.0.0.1:4711`, with producers on `/ingest`.
2. Start Ollama with `OLLAMA_TRACE_URL=ws://127.0.0.1:4711/ingest`.
3. The app calls `POST /api/chat` (or `/api/generate`) with the headers above and receives
   events for that `req` through its `/view` socket.
4. In step mode the app sends `{"v":1,"type":"ack","req":"<id>","granularity":"token"}` over the
   **same producer socket** (hub → Ollama) to release one token, or `{"continue":true}` to let the
   request run freely.

## Tests

```bash
go test ./trace/ ./envconfig/ ./llm/ ./server/...
go test -race ./trace/
```

## Benchmarks

See NOTES.md § Performance.
