@echo off
setlocal
cd /d "%~dp0"

REM === Arranca ClawdBot en segundo plano (sin ventana) ===
REM Stdout/stderr -> logs\bot_stdout.log / logs\bot_stderr.log
REM PID guardado en logs\bot.pid para poder pararlo con STOP_BOT.bat

if not exist "logs" mkdir logs

REM Detecta Python
set "PYEXE="
if exist ".venv312\Scripts\python.exe" set "PYEXE=%~dp0.venv312\Scripts\python.exe"
if not defined PYEXE if exist ".venv\Scripts\python.exe" set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not defined PYEXE (
    echo ERROR: No encuentro .venv312\Scripts\python.exe
    pause
    exit /b 1
)

REM Verifica que no haya una instancia corriendo
if exist "logs\bot.pid" (
    set /p OLDPID=<"logs\bot.pid"
    powershell -NoProfile -Command ^
        "if (Get-Process -Id %OLDPID% -ErrorAction SilentlyContinue) { Write-Host 'AVISO: Bot ya esta corriendo (PID %OLDPID%). Usa STOP_BOT.bat primero.'; exit 1 }"
    if %ERRORLEVEL% equ 1 (
        echo.
        pause
        exit /b 1
    )
)

REM Lanza proceso oculto, captura PID
echo Iniciando ClawdBot en segundo plano...
powershell -NoProfile -Command ^
    "$p = Start-Process -FilePath '%PYEXE%' -ArgumentList 'main.py' -WorkingDirectory '%~dp0' -WindowStyle Hidden -RedirectStandardOutput 'logs\bot_stdout.log' -RedirectStandardError 'logs\bot_stderr.log' -PassThru; $p.Id | Out-File -FilePath 'logs\bot.pid' -Encoding ascii -NoNewline; Write-Host ('Bot iniciado. PID=' + $p.Id)"

echo.
echo Logs en:
echo   logs\bot_stdout.log  (stdout)
echo   logs\bot_stderr.log  (stderr)
echo   bot_debug.log        (debug completo)
echo   audit.log            (telemetria de riesgo)
echo.
echo Para parar: doble clic en STOP_BOT.bat
echo Para revisar: doble clic en EXPORT_DAY_REVIEW.bat
echo.
pause
endlocal
