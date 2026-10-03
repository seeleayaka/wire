@echo off
setlocal
set "ROOT=E:\PythonProject10"
set "NORMAL=%ROOT%\output\mendeley_cable_labelme_normal_addon_20260825"
"%ROOT%\.venv\Scripts\labelme.exe" "%NORMAL%"
endlocal
