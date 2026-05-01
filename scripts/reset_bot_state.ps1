#Requires -Version 5.1
<#
.SYNOPSIS
  Reset local bot state + PostgreSQL so Dashboard / DB start "from zero".

.DESCRIPTION
  - Removes state.json (open positions + session PnL on disk)
  - Removes trade_journal.csv (CSV journal; recreated on next trade)
  - TRUNCATE trades_history, CLEAR market/news telemetry, commands, health_events
  - Sets htf_trend_status trends to 'neutral'

  Does NOT touch MetaTrader: close positions on the broker separately.

.PARAMETER ProjectRoot
  Repo root (folder containing main.py). Default: parent of scripts/

.PARAMETER DryRun
  Print actions only; no deletes / no SQL.

.PARAMETER SkipDatabase
  Skip PostgreSQL (files only).

.PARAMETER IncludeLogs
  Delete logs\*.log (bot_debug, audit, last_session, etc.).

.EXAMPLE
  .\scripts\reset_bot_state.ps1

.EXAMPLE
  .\scripts\reset_bot_state.ps1 -DryRun

.EXAMPLE
  .\scripts\reset_bot_state.ps1 -SkipDatabase -IncludeLogs
#>

param(
    [string]$ProjectRoot = "",
    [switch]$DryRun,
    [switch]$SkipDatabase,
    [switch]$IncludeLogs
)

$ErrorActionPreference = "Stop"

function Resolve-ProjectRoot {
    if ($ProjectRoot -and (Test-Path $ProjectRoot)) {
        return (Resolve-Path $ProjectRoot).Path
    }
    $candidate = Resolve-Path (Join-Path $PSScriptRoot "..")
    if (-not (Test-Path (Join-Path $candidate "main.py"))) {
        Write-Error "main.py not found under $candidate. Pass -ProjectRoot explicitly."
    }
    return $candidate.Path
}

function Import-DotEnv {
    param([string]$EnvPath)
    $map = @{}
    if (-not (Test-Path $EnvPath)) {
        Write-Warning ".env not found at $EnvPath - use DB_* in environment or .env"
        return $map
    }
    Get-Content $EnvPath -Encoding UTF8 | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) { return }
        $eq = $line.IndexOf("=")
        if ($eq -lt 1) { return }
        $k = $line.Substring(0, $eq).Trim()
        $v = $line.Substring($eq + 1).Trim()
        if (($v.StartsWith('"') -and $v.EndsWith('"')) -or ($v.StartsWith("'") -and $v.EndsWith("'"))) {
            $v = $v.Substring(1, $v.Length - 2)
        }
        $map[$k] = $v
    }
    return $map
}

function Invoke-FileReset {
    param(
        [string]$Root,
        [switch]$Dry,
        [switch]$LogsToo
    )

    $statePath = Join-Path $Root "state.json"
    $journalPath = Join-Path $Root "logs\trade_journal.csv"

    if ($Dry) {
        Write-Host "(dry-run) Would remove if present: $statePath"
        Write-Host "(dry-run) Would remove if present: $journalPath"
        if ($LogsToo) {
            Write-Host "(dry-run) Would remove:" (Join-Path $Root 'logs\*.log')
        }
        return
    }

    if (Test-Path $statePath) {
        Remove-Item -LiteralPath $statePath -Force
        Write-Host "Removed: state.json"
    } else {
        Write-Host "Skip: state.json (not present)"
    }

    if (Test-Path $journalPath) {
        Remove-Item -LiteralPath $journalPath -Force
        Write-Host "Removed: logs\trade_journal.csv"
    } else {
        Write-Host "Skip: logs\trade_journal.csv (not present)"
    }

    if ($LogsToo) {
        $logDir = Join-Path $Root "logs"
        if (Test-Path $logDir) {
            Get-ChildItem -Path $logDir -Filter "*.log" -File -ErrorAction SilentlyContinue | ForEach-Object {
                Remove-Item $_.FullName -Force
                Write-Host "Removed: logs\$($_.Name)"
            }
        }
        $auditRoot = Join-Path $Root "audit.log"
        if (Test-Path $auditRoot) {
            Remove-Item -LiteralPath $auditRoot -Force
            Write-Host "Removed: audit.log"
        }
    }
}

function Find-Psql {
    $cmd = Get-Command psql -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @(
        "C:\Program Files\PostgreSQL\17\bin\psql.exe",
        "C:\Program Files\PostgreSQL\16\bin\psql.exe",
        "C:\Program Files\PostgreSQL\15\bin\psql.exe"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Get-EnvOrDefault {
    param([string]$Key, [string]$Default)
    $p = [Environment]::GetEnvironmentVariable($Key, "Process")
    if ($p) { return $p }
    return $Default
}

function Invoke-DatabaseReset {
    param(
        [switch]$Dry
    )

    $hostDb = Get-EnvOrDefault "DB_HOST" "localhost"
    $port = Get-EnvOrDefault "DB_PORT" "5432"
    $user = Get-EnvOrDefault "DB_USER" "clawdbot"
    $pass = Get-EnvOrDefault "DB_PASSWORD" "clawdbot_secret"
    $name = Get-EnvOrDefault "DB_NAME" "clawdbot"

    # Requires tables from init_db / first main.py connect (health_events may be missing on very old DBs).
    $sql = @'
BEGIN;
TRUNCATE TABLE trades_history RESTART IDENTITY CASCADE;
DELETE FROM market_data;
DELETE FROM news_sentiment;
TRUNCATE TABLE commands RESTART IDENTITY CASCADE;
UPDATE htf_trend_status SET trend = 'neutral', updated_at = NOW();
TRUNCATE TABLE health_events;
COMMIT;
'@

    if ($Dry) {
        $target = '{0}@{1}:{2}/{3}' -f $user, $hostDb, $port, $name
        Write-Host "(dry-run) Would run SQL on" $target
        Write-Host $sql
        return
    }

    $psql = Find-Psql
    if (-not $psql) {
        Write-Error "psql not found. Install PostgreSQL client tools or add psql to PATH. Or use -SkipDatabase."
    }

    $env:PGPASSWORD = $pass
    try {
        & $psql -h $hostDb -p $port -U $user -d $name -v ON_ERROR_STOP=1 -c $sql 2>&1 | Write-Host
        if ($LASTEXITCODE -ne 0) {
            Write-Error "psql exited with code $LASTEXITCODE"
        }
        Write-Host "Database reset OK: trades cleared, HTF trends set to neutral."
    } finally {
        Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
    }
}

# --- main ---
$root = Resolve-ProjectRoot
$envFile = Join-Path $root ".env"
$vars = Import-DotEnv -EnvPath $envFile

foreach ($k in @("DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME")) {
    if ($vars.ContainsKey($k) -and -not [string]::IsNullOrEmpty($vars[$k])) {
        [Environment]::SetEnvironmentVariable($k, $vars[$k], "Process")
    }
}

Write-Host "Project: $root"
Write-Host "Stop the bot (main.py) before running. MT5 positions are NOT closed by this script."
Write-Host ""

Invoke-FileReset -Root $root -Dry:$DryRun -LogsToo:$IncludeLogs

if (-not $SkipDatabase) {
    if ($DryRun) {
        Invoke-DatabaseReset -Dry
    } else {
        Invoke-DatabaseReset
    }
} else {
    Write-Host "Skip: database (-SkipDatabase)"
}

Write-Host ""
Write-Host "Done. Start main.py; Dashboard PnL / trade list should be empty until new trades."
