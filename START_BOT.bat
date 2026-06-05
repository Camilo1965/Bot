@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

REM === Arranca ClawdBot en segundo plano (sin ventana) ===
REM Bot + dashboard proxied a traves de un solo puerto (3000).
REM Cloudflare Quick Tunnel expone ese puerto al mundo sin port forwarding.

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
        "if (Get-Process -Id !OLDPID! -ErrorAction SilentlyContinue) { Write-Host 'AVISO: Bot ya corre (PID !OLDPID!). Usa STOP_BOT.bat primero.'; exit 1 }"
    if !ERRORLEVEL! equ 1 ( echo. & pause & exit /b 1 )
)

REM ── IP local ─────────────────────────────────────────────────────────────────
for /f "tokens=*" %%i in ('powershell -NoProfile -Command ^
    "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notmatch '^(127\.|169\.)' } | Sort-Object PrefixLength -Desc | Select-Object -First 1).IPAddress"') do set "LOCAL_IP=%%i"
if not defined LOCAL_IP set "LOCAL_IP=localhost"

REM ── Firewall (puertos 3000 y 8000) ───────────────────────────────────────────
powershell -NoProfile -Command ^
    "if (-not (Get-NetFirewallRule -DisplayName 'ClawdBot-Dashboard' -EA SilentlyContinue)) { New-NetFirewallRule -DisplayName 'ClawdBot-Dashboard' -Direction Inbound -Protocol TCP -LocalPort 3000,8000 -Action Allow | Out-Null; Write-Host 'Firewall: reglas creadas.' } else { Write-Host 'Firewall: ya configurado.' }"

REM ── Descarga cloudflared si no existe ────────────────────────────────────────
if not exist "cloudflared.exe" (
    echo Descargando cloudflared.exe ...
    powershell -NoProfile -Command ^
        "Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile 'cloudflared.exe' -UseBasicParsing; Write-Host 'cloudflared descargado.'"
)

REM ── Lanza bot oculto ─────────────────────────────────────────────────────────
echo Iniciando ClawdBot...
powershell -NoProfile -Command ^
    "$p = Start-Process -FilePath '%PYEXE%' -ArgumentList 'main.py' -WorkingDirectory '%~dp0' -WindowStyle Hidden -RedirectStandardOutput 'logs\bot_stdout.log' -RedirectStandardError 'logs\bot_stderr.log' -PassThru; $p.Id | Out-File 'logs\bot.pid' -Encoding ascii -NoNewline; Write-Host ('Bot PID=' + $p.Id)"

REM ── Espera dashboard en puerto 3000 ──────────────────────────────────────────
echo Esperando dashboard...
powershell -NoProfile -Command ^
    "$i=0; Write-Host 'Esperando' -NoNewline; while($i -lt 30){ try{ Invoke-WebRequest -Uri 'http://localhost:3000' -UseBasicParsing -TimeoutSec 2 -EA Stop|Out-Null; Write-Host ''; break }catch{ Write-Host '.' -NoNewline; Start-Sleep 2; $i++ } }"

REM ── Lanza tunel Cloudflare (fondo, guarda URL en logs\tunnel_url.txt) ─────────
echo Iniciando tunel Cloudflare...
powershell -NoProfile -Command ^
    "del 'logs\tunnel_url.txt' -EA SilentlyContinue; $p = Start-Process -FilePath 'cloudflared.exe' -ArgumentList 'tunnel','--url','http://localhost:3000','--logfile','logs\cloudflared.log' -WindowStyle Hidden -PassThru; $p.Id | Out-File 'logs\cloudflared.pid' -Encoding ascii -NoNewline; Write-Host ('cloudflared PID=' + $p.Id)"

REM ── Espera URL publica del tunel (aparece en log) ─────────────────────────────
echo Esperando URL publica...
set "PUBLIC_URL="
powershell -NoProfile -Command ^
    "$i=0; while($i -lt 20){ Start-Sleep 2; if(Test-Path 'logs\cloudflared.log'){ $m = Select-String -Path 'logs\cloudflared.log' -Pattern 'https://[a-z0-9\-]+\.trycloudflare\.com' | Select-Object -First 1; if($m){ $url = ([regex]'https://[a-z0-9\-]+\.trycloudflare\.com').Match($m.Line).Value; $url | Out-File 'logs\tunnel_url.txt' -Encoding ascii -NoNewline; Write-Host $url; exit 0 } }; Write-Host '.' -NoNewline; $i++ }; Write-Host 'Timeout esperando URL tunel.'"

REM ── Lee URL y abre navegador ──────────────────────────────────────────────────
set "PUBLIC_URL="
if exist "logs\tunnel_url.txt" set /p PUBLIC_URL=<"logs\tunnel_url.txt"

if defined PUBLIC_URL (
    start "" "%PUBLIC_URL%"
) else (
    start "" "http://localhost:3000"
)

echo.
echo ================================================================
echo   LOCAL:    http://localhost:3000
echo   RED LAN:  http://%LOCAL_IP%:3000
if defined PUBLIC_URL (
echo   INTERNET: %PUBLIC_URL%
echo.
echo   Abre esa URL desde tu celular, tablet o cualquier red.
echo   NOTA: La URL cambia cada vez que reinicias el bot.
echo         Para URL fija instala Tailscale (gratis):
echo         https://tailscale.com/download
) else (
echo   INTERNET: tunel no disponible - revisa logs\cloudflared.log
)
echo ================================================================
echo.
echo Para parar todo: doble clic en STOP_BOT.bat
echo Para revisar logs: doble clic en EXPORT_DAY_REVIEW.bat
echo.
pause
endlocal
