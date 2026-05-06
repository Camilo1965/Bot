# Limpia logs y estado local del bot (Windows). Ejecutar desde la raíz del repo:
#   powershell -ExecutionPolicy Bypass -File scripts/clean_local_runtime.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$logs = Join-Path $root "logs"
if (Test-Path $logs) {
  Get-ChildItem -Path $logs -File -Recurse -Filter "*.log" | Remove-Item -Force
  # Estado paper / journal / métricas (no modelo ML)
  @(
    "state.json",
    "trade_journal.csv",
    "trade_events.jsonl",
    "runtime_metrics.jsonl",
    "bot_startup_snapshot.json",
    "diagnostic_bundle_FOR_REVIEW.md"
  ) | ForEach-Object {
    $p = Join-Path $logs $_
    if (Test-Path $p) { Remove-Item -Force $p }
  }
}

# Logs en raíz del repo (main.py escribe aquí)
@("bot_debug.log", "bot_debug.log.1", "bot_debug.log.2", "bot_debug.log.3", "audit.log") | ForEach-Object {
  $p = Join-Path $root $_
  if (Test-Path $p) { Remove-Item -Force $p }
}

Get-ChildItem -Path $root -File -Filter "bot_state*.json" -ErrorAction SilentlyContinue | Remove-Item -Force

Write-Host "OK: logs/*.log, logs/state.json, trade journal, bot_debug.log, audit.log limpiados bajo $root"
