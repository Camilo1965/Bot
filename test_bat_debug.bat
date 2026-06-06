@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
echo SECTION: python
set "PYEXE="
if exist ".venv312\Scripts\python.exe" set "PYEXE=%~dp0.venv312\Scripts\python.exe"
echo PYEXE=%PYEXE%
echo SECTION: pg with escaped dollar signs
powershell -NoProfile -Command "$i=0; while($i -lt 3){ Start-Sleep 1; \$r = docker exec clawdbot_db pg_isready -U clawdbot 2>\$null; if(\$LASTEXITCODE -eq 0){ Write-Host 'PG OK'; exit 0 }; Write-Host '.' -NoNewline; \$i++ }; Write-Host ' timeout'"
echo SECTION: launch bot hidden
powershell -NoProfile -Command ^
    "$p = Start-Process -FilePath '%PYEXE%' -ArgumentList 'main.py' -WorkingDirectory '%~dp0' -WindowStyle Hidden -RedirectStandardOutput 'logs\bot_stdout.log' -RedirectStandardError 'logs\bot_stderr.log' -PassThru; $p.Id | Out-File 'logs\bot.pid' -Encoding ascii -NoNewline; Write-Host ('Bot PID=' + $p.Id)"
echo SECTION: done
endlocal
