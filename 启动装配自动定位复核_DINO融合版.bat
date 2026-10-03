@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
set "PORTABLE_PYTHON=%ROOT%runtime\windows\python\python.exe"
if exist "%PORTABLE_PYTHON%" goto :portable_runtime
set "MAIN_PYTHON=%ROOT%.venv\Scripts\python.exe"
set "SAM3_PYTHON=%ROOT%runtime\sam3\.venv\Scripts\python.exe"
set "MAIN_PYTHONPATH=%ROOT%models\dinov2;%ROOT%prototype;%PYTHONPATH%"
set "SAM3_PYTHONPATH=%ROOT%runtime\sam3\source;%PYTHONPATH%"
goto :runtime_selected

:portable_runtime
set "MAIN_PYTHON=%PORTABLE_PYTHON%"
set "SAM3_PYTHON=%PORTABLE_PYTHON%"
set "PYTHONHOME=%ROOT%runtime\windows\python"
set "MAIN_PYTHONPATH=%ROOT%runtime\windows\main_site_packages;%ROOT%models\dinov2;%ROOT%prototype"
set "SAM3_PYTHONPATH=%ROOT%runtime\sam3\source;%ROOT%runtime\windows\sam3_site_packages;%ROOT%runtime\windows\sam3_base_site_packages"

:runtime_selected
pushd "%ROOT%"
set "PYTHONPATH=%MAIN_PYTHONPATH%"
"%MAIN_PYTHON%" -B -c "import cv2, numpy, torch; from PyQt5 import QtCore" >nul 2>&1
if errorlevel 1 goto :runtime_missing
set "PYTHONPATH=%SAM3_PYTHONPATH%"
"%SAM3_PYTHON%" -B -c "import cv2, numpy, torch, sam3" >nul 2>&1
if errorlevel 1 goto :runtime_missing
set "PYTHONPATH=%MAIN_PYTHONPATH%"
"%MAIN_PYTHON%" "%ROOT%prototype\assembly_auto_review_dino_v2.py"
set "EXIT_CODE=%ERRORLEVEL%"
popd
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%

:runtime_missing
echo Project runtime is not ready for this computer.
echo Run "%ROOT%0_first_setup_and_start.bat" once, then start this file again.
popd
pause
exit /b 1
