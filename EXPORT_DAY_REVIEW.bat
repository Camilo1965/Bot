@echo off
setlocal
REM === Export paquete diagnostico (full-day) — doble clic o cmd con argumentos extra ===
REM
REM   Doble clic     -> hoy (REPORT_TIMEZONE en .env)
REM
REM   Otro dia:
REM     EXPORT_DAY_REVIEW.bat --date 2026-05-01
REM
REM   Bundle mas chico (solo ERROR/WARNING en bot_debug):
REM     EXPORT_DAY_REVIEW.bat --bot-debug-errors-only
REM
REM   Salida: DIAGNOSTIC_DAY_YYYY-MM-DD.md en esta carpeta
REM
REM No hace falta activar venv a mano.

cd /d "%~dp0"
set ERR=1

echo.
echo === Export revision dia completo (IA) ===
if not "%~1"=="" echo Args: %*
echo.

set "PYEXE="
if exist ".venv312\Scripts\python.exe" set "PYEXE=%~dp0.venv312\Scripts\python.exe"
if not defined PYEXE if exist ".venv\Scripts\python.exe" set "PYEXE=%~dp0.venv\Scripts\python.exe"

if defined PYEXE (
  "%PYEXE%" scripts\export_diagnostic_bundle.py --full-day %*
  set ERR=%ERRORLEVEL%
  goto :done_run
)

where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
  py -3.12 scripts\export_diagnostic_bundle.py --full-day %*
  set ERR=%ERRORLEVEL%
  goto :done_run
)
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
  python scripts\export_diagnostic_bundle.py --full-day %*
  set ERR=%ERRORLEVEL%
  goto :done_run
)

echo No encuentro Python: crea .venv o .venv312 en esta carpeta o instala Python 3.12 en PATH.
pause
exit /b 1

:done_run
echo.
echo Archivo: DIAGNOSTIC_DAY_YYYY-MM-DD.md (en esta carpeta, salvo --output)
echo.

explorer "%~dp0"
echo.
if %ERR% equ 0 (echo Listo.) else (echo Hubo un error arriba.)
pause

endlocal
