@echo off
setlocal EnableExtensions
cd /d "%~dp0"
rem NEXUS Code - start the interactive web interface.
rem Usage: gui.bat                      (open a fresh general chat)
rem        gui.bat "C:\path\to\project" (open a workspace directly)
rem        gui.bat "C:\path" --port 9000 --no-browser

set "PYCMD="
if defined NEXUS_PYTHON set "PYCMD=%NEXUS_PYTHON%"
if defined PYCMD goto havepy
py -3 --version >nul 2>nul
if not errorlevel 1 set "PYCMD=py -3"
if defined PYCMD goto havepy
python --version >nul 2>nul
if not errorlevel 1 set "PYCMD=python"
if defined PYCMD goto havepy
echo Python 3 was not found. Install it from python.org (tick "Add to PATH"),
echo or set NEXUS_PYTHON to the full path of python.exe.
pause
exit /b 1

:havepy
if not exist gui.py goto nogui
if not exist ui\index.html goto nogui
if not exist providers\__init__.py goto noprov

:run
set "FIRST=%~1"
if "%FIRST%"=="" goto nows
if "%FIRST:~0,2%"=="--" goto nows
shift
%PYCMD% gui.py --workspace "%FIRST%" %1 %2 %3 %4 %5 %6
goto end

:nows
%PYCMD% gui.py %1 %2 %3 %4 %5 %6
goto end

:nogui
echo gui.py or ui\index.html is missing. Keep all NEXUS files together.
pause
exit /b 1

:noprov
echo The providers folder is missing. Keep it beside agent.py and gui.py.
pause
exit /b 1

:end
if errorlevel 1 pause
