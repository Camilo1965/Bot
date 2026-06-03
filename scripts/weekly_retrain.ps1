# scripts/weekly_retrain.ps1
# Weekly retrain + auto-rollback. Designed for Windows Task Scheduler (Sundays 00:05 UTC).
# Registration command (run once, manually):
#   schtasks /Create /TN "ClawdBot Weekly Retrain" `
#     /TR "powershell -File C:\Users\WinterOS\Desktop\projectos\Bot\scripts\weekly_retrain.ps1" `
#     /SC WEEKLY /D SUN /ST 00:05 /F

param(
    [string]$RepoRoot = "C:\Users\WinterOS\Desktop\projectos\Bot",
    [int]$RegressionThresholdPct = -30,
    [switch]$NoTelegram
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

$ts = Get-Date -Format "yyyy-MM-dd_HHmm"
$logDir = Join-Path $RepoRoot "logs\weekly_retrain\$ts"
$archive = Join-Path $RepoRoot "models\archive\$ts"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $archive | Out-Null

function Log($msg) {
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path (Join-Path $logDir "weekly_retrain.log") -Value $line -Encoding utf8
}

Log "=== Weekly retrain $ts ==="

# Pre-flight: disk space
$free = (Get-PSDrive C).Free / 1GB
if ($free -lt 0.5) { Log "FATAL: disk <500MB free ($($free.ToString('F2'))GB). Aborting."; exit 3 }
Log "Pre-flight OK. Free disk: $($free.ToString('F2')) GB"

# Archive current models
Log "Archiving current models to $archive ..."
Get-ChildItem -Path (Join-Path $RepoRoot "models") -Filter "*_v2.json"          | Copy-Item -Destination $archive
Get-ChildItem -Path (Join-Path $RepoRoot "models") -Filter "*_v2.meta.json"     | Copy-Item -Destination $archive
Get-ChildItem -Path (Join-Path $RepoRoot "models") -Filter "*_regime.json"      | Copy-Item -Destination $archive -ErrorAction SilentlyContinue
Get-ChildItem -Path (Join-Path $RepoRoot "models") -Filter "*_regime.meta.json" | Copy-Item -Destination $archive -ErrorAction SilentlyContinue
Get-ChildItem -Path (Join-Path $RepoRoot "models") -Filter "*_calibration.json" | Copy-Item -Destination $archive
$baseline = Join-Path $RepoRoot "logs\symbol_config_baseline.json"
if (Test-Path $baseline) { Copy-Item $baseline $archive }
Log "Archive complete."

# Pre-retrain backtest
Log "Running pre-retrain backtest (30d disk-loaded)..."
$prePath = Join-Path $logDir "pre.json"
python scripts\backtest_disk_loaded.py --report --all --days 30 *> (Join-Path $logDir "pre_backtest.log")
$reportSrc = Join-Path $RepoRoot "logs\final_verification_report.json"
if (Test-Path $reportSrc) { Copy-Item $reportSrc $prePath -Force } else { Log "WARN: pre backtest report missing." }

# Retrain + refit
Log "Running retrain_all.py ..."
python scripts\retrain_all.py *> (Join-Path $logDir "retrain.log")
$retrainExit = $LASTEXITCODE
Log "retrain_all exit code: $retrainExit"

Log "Running refit_calibrations.py --all ..."
python scripts\refit_calibrations.py --all *> (Join-Path $logDir "refit.log")
$refitExit = $LASTEXITCODE
Log "refit_calibrations exit code: $refitExit"

# Post-retrain backtest
Log "Running post-retrain backtest (30d disk-loaded)..."
$postPath = Join-Path $logDir "post.json"
python scripts\backtest_disk_loaded.py --report --all --days 30 *> (Join-Path $logDir "post_backtest.log")
if (Test-Path $reportSrc) { Copy-Item $reportSrc $postPath -Force } else { Log "WARN: post backtest report missing." }

# Compare + rollback if regression
Log "Comparing pre vs post composite scores (threshold $RegressionThresholdPct%)..."
python scripts\compare_retrain_diff.py --pre $prePath --post $postPath --threshold-pct $RegressionThresholdPct --rollback-from $archive *> (Join-Path $logDir "compare.log")
$cmpExit = $LASTEXITCODE
Log "compare_retrain_diff exit code: $cmpExit"

# Telegram notify (depende de [I-15]; soft fail si modulo no presente)
if (-not $NoTelegram) {
    try {
        $notifyArgs = @{
            "ts" = $ts
            "retrain_exit" = $retrainExit
            "refit_exit" = $refitExit
            "compare_exit" = $cmpExit
        }
        python -c "import sys, json; sys.path.insert(0, '.'); from vibe.notifier import notify_retrain; import asyncio; asyncio.run(notify_retrain($($notifyArgs | ConvertTo-Json -Compress)))" *> (Join-Path $logDir "notify.log")
    } catch {
        Log "Telegram notify skipped (module not available or failed)."
    }
}

Log "=== Weekly retrain finished. Result code: $cmpExit ==="
exit $cmpExit
