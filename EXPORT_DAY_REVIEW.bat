@echo off
setlocal
REM Ejecuta export día completo y abre la carpeta del repo en el Explorador.
REM Doble clic en este archivo (desde la raíz del proyecto).

cd /d "%~dp0"

if exist ".venv312\Scripts\python.exe" (
  set "PY=.venv312\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  echo No encuentro .venv312 ni .venv con python.exe
  echo Instala el venv o edita este .bat para apuntar a tu Python.
  pause
  exit /b 1
)

echo.
echo === Export revision dia completo para IA ===
"%PY%" scripts\export_diagnostic_bundle.py --full-day
set ERR=%ERRORLEVEL%

echo.
echo Archivo generado en esta carpeta: DIAGNOSTIC_DAY_YYYY-MM-DD.md
echo.

explorer "%~dp0"
echo.
if %ERR% equ 0 (echo Listo. Busca DIAGNOSTIC_DAY_*.md en esta carpeta.) else (echo Hubo un error arriba.)
pause

endlocal
