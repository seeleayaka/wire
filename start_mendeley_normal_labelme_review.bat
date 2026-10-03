@echo off
setlocal
set "ROOT=E:\PythonProject10"
set "REVIEW=%ROOT%\output\mendeley_cable_labelme_review_20260826"
set "LABELME_RUNTIME=%TEMP%\labelme_pythonproject10"

if not exist "%ROOT%\.venv\Scripts\labelme.exe" (
  echo Labelme was not found: %ROOT%\.venv\Scripts\labelme.exe
  pause
  exit /b 1
)
if not exist "%REVIEW%" (
  echo Review directory was not found: %REVIEW%
  pause
  exit /b 1
)
if not exist "%LABELME_RUNTIME%" mkdir "%LABELME_RUNTIME%"

rem Keep Labelme's log/config in a writable per-project temp directory.
set "LOCALAPPDATA=%LABELME_RUNTIME%"

rem Open the migrated JSON/image set directly; no manual import is needed.
start "Labelme polygon review" "%ROOT%\.venv\Scripts\labelme.exe" "%REVIEW%"
endlocal
