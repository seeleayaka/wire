@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
pushd "%ROOT%"
echo Verifying the bundled Windows runtime for this computer.
echo This package normally needs no Python installation or package download.
echo If the bundled runtime is absent or damaged, Python 3.11 and Internet package access are required to rebuild it.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ROOT%runtime\bootstrap.ps1"
set "SETUP_EXIT_CODE=%ERRORLEVEL%"
if not "%SETUP_EXIT_CODE%"=="0" goto :setup_failed
call "%ROOT%启动装配自动定位复核_DINO融合版.bat"
set "START_EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %START_EXIT_CODE%

:setup_failed
echo Setup did not finish. Check that Python 3.11 is installed and that package downloads are allowed.
popd
pause
exit /b %SETUP_EXIT_CODE%
