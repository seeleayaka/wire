@echo off
setlocal
set "WIREMIND_PROJECT_ROOT=%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" -B "%~dp0research\experiments\launch_llm_recheck_window_20261008.py"
) else (
  py -3.11 "%~dp0research\experiments\launch_llm_recheck_window_20261008.py"
)
if errorlevel 1 pause
