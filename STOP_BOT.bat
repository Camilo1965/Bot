@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

REM === Para ClawdBot y cierra el tunel ===

REM Detecta modo SQLite (sin Docker) desde .env
set "DB_BACKEND_VAL="
if exist ".env" (
    for /f "tokens=*" %%i in ('powershell -NoProfile -Command "$m = Select-String -Path '.env' -Pattern '^\s*DB_BACKEND\s*=\s*(.+?)\s*$' | Select-Object -First 1; if ($m) { $m.Matches[0].Groups[1].Value.Trim().ToLower() }"') do set "DB_BACKEND_VAL=%%i"
)

REM Para el bot
if not exist "logs\bot.pid" goto :stop_tunnel
set /p PID=<"logs\bot.pid"
echo Parando bot ^(PID=!PID!^)...
powershell -NoProfile -Command "$p=Get-Process -Id %PID% -EA SilentlyContinue; if($p){ $p.Kill(); $p.WaitForExit(5000); Write-Host 'Bot detenido.' } else { Write-Host 'Bot ya estaba parado.' }"
del "logs\bot.pid" 2>nul

REM Kill huerfanos: cualquier python.exe que sirva uvicorn dashboard.api.main, y node.exe del dashboard web
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'dashboard\.api\.main' -or $_.CommandLine -match 'dashboard\\web' } | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -EA SilentlyContinue } catch {} }"

:stop_tunnel
REM Cierra tunel Cloudflare
if not exist "logs\cloudflared.pid" goto :stop_docker
set /p CFPID=<"logs\cloudflared.pid"
echo Cerrando tunel ^(PID=!CFPID!^)...
powershell -NoProfile -Command "$p=Get-Process -Id %CFPID% -EA SilentlyContinue; if($p){ $p.Kill(); Write-Host 'Tunel cerrado.' } else { Write-Host 'Tunel ya cerrado.' }"
del "logs\cloudflared.pid" 2>nul
del "logs\tunnel_url.txt" 2>nul

:stop_docker
REM Modo SQLite no tiene Docker: salta
if /i "%DB_BACKEND_VAL%"=="sqlite" goto :done

REM Modo Postgres: baja Docker solo si esta corriendo (no prompt interactivo)
where docker >nul 2>&1
if %ERRORLEVEL% neq 0 goto :done
docker info >nul 2>&1
if %ERRORLEVEL% neq 0 goto :done
echo Bajando Docker (db + redis)...
docker compose stop db redis >nul 2>&1
if %ERRORLEVEL% neq 0 ( docker-compose stop db redis >nul 2>&1 )

:done
echo.
echo Todo detenido.
echo.
endlocal
