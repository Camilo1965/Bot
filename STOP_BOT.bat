@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

REM === Para ClawdBot ===

if not exist "logs\bot.pid" (
    echo No hay bot.pid — bot no iniciado con START_BOT.bat o ya parado.
    pause & exit /b 0
)

set /p PID=<"logs\bot.pid"
echo Parando bot (PID=!PID!)...
powershell -NoProfile -Command ^
    "$p = Get-Process -Id !PID! -EA SilentlyContinue; if ($p) { $p.Kill(); $p.WaitForExit(5000); Write-Host 'Bot detenido.' } else { Write-Host 'Bot ya estaba parado.' }"
del "logs\bot.pid" 2>nul

echo.
echo Listo. Para exportar logs del dia: doble clic en EXPORT_DAY_REVIEW.bat
echo.
pause
endlocal
