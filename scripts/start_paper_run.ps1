# scripts/start_paper_run.ps1
# Start ClawdBot in paper trading mode for 30-day validation run.
# Usage: .\scripts\start_paper_run.ps1
#
# What it does:
#   1. Validates .env has EXECUTION_MODE or sets it to "paper"
#   2. Starts main.py in background, logging to logs/paper_run_<date>.log
#   3. Registers a daily Task Scheduler job to run shadow_run_compare.py at 08:00
#
# Stop the run: Stop-Process -Name python (or CTRL+C in the window)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path $PSScriptRoot -Parent
$LOG_DIR = "$ROOT\logs"
$DATE = (Get-Date -Format "yyyy-MM-dd")
$LOG_FILE = "$LOG_DIR\paper_run_$DATE.log"

if (-not (Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR | Out-Null }

Write-Host "=== ClawdBot Paper Run ==="
Write-Host "Log: $LOG_FILE"
Write-Host "Root: $ROOT"

# Ensure EXECUTION_MODE=paper
$env:EXECUTION_MODE = "paper"

# Validate .env exists
if (-not (Test-Path "$ROOT\.env")) {
    Write-Error ".env not found at $ROOT\.env"
    exit 1
}

# Register daily shadow compare at 08:00 local
$taskName = "ClawdBot_ShadowCompare"
$action = New-ScheduledTaskAction `
    -Execute "python" `
    -Argument "$ROOT\scripts\shadow_run_compare.py --summary --days 1" `
    -WorkingDirectory $ROOT
$trigger = New-ScheduledTaskTrigger -Daily -At "08:00AM"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
try {
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
        -Settings $settings -Force | Out-Null
    Write-Host "[OK] Scheduled daily shadow compare: $taskName at 08:00"
} catch {
    Write-Warning "Could not register shadow compare task: $_"
}

# Start bot
Write-Host "[START] EXECUTION_MODE=paper python main.py >> $LOG_FILE"
Write-Host "Press CTRL+C to stop."
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "python"
$psi.Arguments = "main.py"
$psi.WorkingDirectory = $ROOT
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.EnvironmentVariables["EXECUTION_MODE"] = "paper"

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi
$proc.Start() | Out-Null

Write-Host "[PID $($proc.Id)] Bot running. Tail log: Get-Content $LOG_FILE -Wait"
$proc.WaitForExit()
Write-Host "[EXIT $($proc.ExitCode)] Bot stopped."
