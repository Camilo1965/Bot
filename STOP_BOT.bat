@echo off
setlocal
cd /d "%~dp0"

REM === Para ClawdBot y cierra el tunel Cloudflare ===

REM ── Para el bot ──────────────────────────────────────────────────────────────
if not exist "logs\bot.pid" (
    echo No hay bot.pid — bot no iniciado con START_BOT.bat o ya parado.
) else (
    set /p PID=<"logs\bot.pid"
    echo Parando bot (PID=!PID!)...
    powershell -NoProfile -Command ^
        "$p = Get-Process -Id !PID! -EA SilentlyContinue; if ($p) { $p.Kill(); $p.WaitForExit(5000); Write-Host 'Bot detenido.' } else { Write-Host 'Bot ya estaba parado.' }"
    del "logs\bot.pid" 2>nul
)

REM ── Cierra el tunel Cloudflare ────────────────────────────────────────────────
if exist "logs\cloudflared.pid" (
    set /p CFPID=<"logs\cloudflared.pid"
    echo Cerrando tunel Cloudflare (PID=!CFPID!)...
    powershell -NoProfile -Command ^
        "$p = Get-Process -Id !CFPID! -EA SilentlyContinue; if ($p) { $p.Kill(); Write-Host 'Tunel cerrado.' } else { Write-Host 'Tunel ya estaba cerrado.' }"
    del "logs\cloudflared.pid" 2>nul
    del "logs\tunnel_url.txt" 2>nul
)

echo.
echo Todo detenido. Para exportar logs: doble clic en EXPORT_DAY_REVIEW.bat
echo.
pause
endlocal
