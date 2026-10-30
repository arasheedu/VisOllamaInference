<#
.SYNOPSIS
  Starts the deep-tier traced Ollama build for the LLM Inference Visualizer.

.DESCRIPTION
  Sets the tracing environment variables and runs `ollama serve` from the
  deep build. Trace events go to the visualizer's hub, so start the app too
  (pnpm dev). Keep this window open; Ctrl+C stops Ollama.

.EXAMPLE
  .\scripts\start-deep-ollama.ps1
  .\scripts\start-deep-ollama.ps1 -Port 11438 -LayerStride 1
#>
param(
  # where Ollama listens (the app's Ollama URL must match)
  [int]$Port = 11434,
  # the visualizer's trace hub port (Settings in the app, default 4711)
  [int]$HubPort = 4711,
  # trace layers on every Nth token outside step mode (1 = every token, slower)
  [int]$LayerStride = 0,
  [string]$OllamaExe = 'C:\ollama-trace-deep\ollama.exe'
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $OllamaExe)) {
  Write-Error "Deep Ollama build not found at $OllamaExe"
}

$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
  $owner = Get-Process -Id $busy[0].OwningProcess -ErrorAction SilentlyContinue
  Write-Error ("Port $Port is already in use by {0} (PID {1}). Quit it first (e.g. the Ollama tray app) or pass -Port." -f $owner.Path, $busy[0].OwningProcess)
}

$env:OLLAMA_HOST = "127.0.0.1:$Port"
$env:OLLAMA_TRACE_URL = "ws://127.0.0.1:$HubPort/ingest"
if ($LayerStride -gt 0) { $env:OLLAMA_TRACE_LAYER_STRIDE = "$LayerStride" }

Write-Host "Deep Ollama  : $OllamaExe"
Write-Host "Listening on : http://127.0.0.1:$Port   (app: Settings > Ollama URL)"
Write-Host "Trace hub    : $env:OLLAMA_TRACE_URL"
if ($LayerStride -gt 0) { Write-Host "Layer stride : $LayerStride" }
Write-Host "Press Ctrl+C to stop."
Write-Host ''

& $OllamaExe serve
