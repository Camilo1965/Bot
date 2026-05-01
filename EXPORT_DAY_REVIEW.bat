@echo off
setlocal
REM Doble clic: export dia completo + abre esta carpeta en el Explorador.
REM No hace falta activar el venv en PowerShell.

cd /d "%~dp0"
set ERR=1

echo.
echo === Export revision dia completo para IA ===

if exist ".venv312\Scripts\python.exe" (
  ".venv312\Scripts\python.exe" scripts\export_diagnostic_bundle.py --full-day
  set ERR=%ERRORLEVEL%
  goto :done_run
)
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\export_diagnostic_bundle.py --full-day
  set ERR=%ERRORLEVEL%
  goto :done_run
)

where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
  py -3.12 scripts\export_diagnostic_bundle.py --full-day
  set ERR=%ERRORLEVEL%
  goto :done_run
)
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
  python scripts\export_diagnostic_bundle.py --full-day
  set ERR=%ERRORLEVEL%
  goto :done_run
)

echo No encuentro Python: crea .venv312 en esta carpeta o instala Python 3.12 en PATH.
pause
exit /b 1

:done_run
echo.
echo Busca el archivo DIAGNOSTIC_DAY_YYYY-MM-DD.md en esta carpeta.
echo.

explorer "%~dp0"
echo.
if %ERR% equ 0 (echo Listo.) else (echo Hubo un error arriba.)
pause

endlocal
