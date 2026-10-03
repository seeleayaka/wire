@echo off
setlocal
set "ROOT=E:\PythonProject10"
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\tools\convert_mendeley_cable_labelme.py" --labelme-root "%ROOT%\output\mendeley_cable_labelme_pilot_20260825" --output "%ROOT%\output\mendeley_cable_yoloseg_pilot_20260825"
pause
endlocal
