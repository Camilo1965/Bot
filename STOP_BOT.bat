@echo off
setlocal
cd /d "%~dp0"

REM === Para ClawdBot (proceso en segundo plano) ===

if not exist "logs\bot.pid" (
    echo No hay bot.pid — el bot no fue iniciado con START_BOT.bat o ya fue parado.
    pause
    exit /b 0
)

set /p PID=<"logs\bot.pid"
echo Parando bot (PID=%PID%)...

powershell -NoProfile -Command ^
    "$p = Get-Process -Id %PID% -ErrorAction SilentlyContinue; if ($p) { $p.Kill(); $p.WaitForExit(5000); Write-Host 'Bot detenido.' } else { Write-Host 'Proceso no encontrado (ya parado).' }"

del "logs\bot.pid" 2>nul
echo.
echo Listo. Para exportar el log del dia: doble clic en EXPORT_DAY_REVIEW.bat
echo.
pause
endlocal
