@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

REM === Arranca ClawdBot en segundo plano (sin ventana) ===
REM Acceso remoto via Tailscale — URL permanente, funciona desde cualquier red.

if not exist "logs" mkdir logs

REM ── Detecta Python ───────────────────────────────────────────────────────────
set "PYEXE="
if exist ".venv312\Scripts\python.exe" set "PYEXE=%~dp0.venv312\Scripts\python.exe"
if not defined PYEXE if exist ".venv\Scripts\python.exe" set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not defined PYEXE (
    echo ERROR: No encuentro .venv312\Scripts\python.exe
    pause & exit /b 1
)

REM ── Verifica instancia existente ─────────────────────────────────────────────
if exist "logs\bot.pid" (
    set /p OLDPID=<"logs\bot.pid"
    powershell -NoProfile -Command ^
        "if (Get-Process -Id !OLDPID! -EA SilentlyContinue) { Write-Host 'AVISO: Bot ya corre (PID !OLDPID!). Usa STOP_BOT.bat primero.'; exit 1 }"
    if !ERRORLEVEL! equ 1 ( echo. & pause & exit /b 1 )
)

REM ── IP local (LAN) ───────────────────────────────────────────────────────────
for /f "tokens=*" %%i in ('powershell -NoProfile -Command ^
    "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notmatch '^(127\.|169\.)' } | Sort-Object PrefixLength -Desc | Select-Object -First 1).IPAddress"') do set "LOCAL_IP=%%i"
if not defined LOCAL_IP set "LOCAL_IP=localhost"

REM ── IP Tailscale ─────────────────────────────────────────────────────────────
set "TS_IP="
set "TS_EXE="
if exist "C:\Program Files\Tailscale\tailscale.exe" set "TS_EXE=C:\Program Files\Tailscale\tailscale.exe"
if not defined TS_EXE (
    for /f "tokens=*" %%i in ('where tailscale 2^>nul') do set "TS_EXE=%%i"
)

if defined TS_EXE (
    for /f "tokens=*" %%i in ('"%TS_EXE%" ip -4 2^>nul') do set "TS_IP=%%i"
    if not defined TS_IP (
        echo AVISO: Tailscale instalado pero no conectado. Ejecuta Tailscale y conectate primero.
        echo        Descarga: https://tailscale.com/download
        echo.
    )
) else (
    echo.
    echo  Tailscale NO esta instalado.
    echo  Para acceso remoto desde cualquier lugar ^(celular, trabajo, etc^):
    echo    1. Descarga e instala: https://tailscale.com/download
    echo    2. Instala tambien en tu celular y conecta con la misma cuenta.
    echo    3. Vuelve a abrir este bat.
    echo.
    echo  Por ahora solo funciona en red local.
    echo.
)

REM ── Firewall (puertos 3000 y 8000) ───────────────────────────────────────────
powershell -NoProfile -Command ^
    "if (-not (Get-NetFirewallRule -DisplayName 'ClawdBot-Dashboard' -EA SilentlyContinue)) { New-NetFirewallRule -DisplayName 'ClawdBot-Dashboard' -Direction Inbound -Protocol TCP -LocalPort 3000,8000 -Action Allow | Out-Null; Write-Host 'Firewall: reglas creadas.' } else { Write-Host 'Firewall: ya configurado.' }"

REM ── Lanza bot oculto ─────────────────────────────────────────────────────────
echo Iniciando ClawdBot...
powershell -NoProfile -Command ^
    "$p = Start-Process -FilePath '%PYEXE%' -ArgumentList 'main.py' -WorkingDirectory '%~dp0' -WindowStyle Hidden -RedirectStandardOutput 'logs\bot_stdout.log' -RedirectStandardError 'logs\bot_stderr.log' -PassThru; $p.Id | Out-File 'logs\bot.pid' -Encoding ascii -NoNewline; Write-Host ('Bot PID=' + $p.Id)"

REM ── Espera dashboard en puerto 3000 ──────────────────────────────────────────
echo Esperando dashboard...
powershell -NoProfile -Command ^
    "$i=0; Write-Host 'Esperando' -NoNewline; while($i -lt 30){ try{ Invoke-WebRequest -Uri 'http://localhost:3000' -UseBasicParsing -TimeoutSec 2 -EA Stop | Out-Null; Write-Host ' listo.'; break }catch{ Write-Host '.' -NoNewline; Start-Sleep 2; $i++ } }"

REM ── Abre navegador con mejor URL disponible ───────────────────────────────────
if defined TS_IP (
    start "" "http://%TS_IP%:3000"
) else (
    start "" "http://localhost:3000"
)

echo.
echo ================================================================
echo   LOCAL:     http://localhost:3000
echo   RED LAN:   http://%LOCAL_IP%:3000
if defined TS_IP (
echo   TAILSCALE: http://%TS_IP%:3000   ^<-- usa esta en tu celular
echo.
echo   La URL de Tailscale es PERMANENTE. Siempre la misma.
) else (
echo   TAILSCALE: no disponible ^(instala en https://tailscale.com/download^)
)
echo ================================================================
echo.
echo Para parar: doble clic en STOP_BOT.bat
echo Para revisar logs: doble clic en EXPORT_DAY_REVIEW.bat
echo.
pause
endlocal
