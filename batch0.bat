@echo off
cd /d "%~dp0"
echo === kagome_ed batch 0: setup + smoke + estimate ===
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 goto fail
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" smoke 1
if errorlevel 1 goto fail
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" estimate 1
if errorlevel 1 goto fail
echo.
echo === DONE. Please zip the whole "output" folder and send it. ===
pause
exit /b 0
:fail
echo.
echo === Something failed. Please zip the whole "output" folder and send it anyway. ===
pause
exit /b 1
