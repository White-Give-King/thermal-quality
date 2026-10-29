@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%USERPROFILE%\.conda\envs\thermal-focus-quality\python.exe"
REM Do not write .pyc bytecode caches (__pycache__) to the project tree.
set "PYTHONDONTWRITEBYTECODE=1"
set "PORT=7860"
set "URL=http://127.0.0.1:%PORT%"

REM Reuse an already-running WebUI instead of starting a second server.
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -Uri '%URL%/' -TimeoutSec 2 -UseBasicParsing; if ($r.StatusCode -eq 200 -and $r.Content -match 'gradio') { exit 0 } } catch {}; exit 1" >nul 2>nul
if not errorlevel 1 (
    echo Gradio is already running at %URL%
    start "" "%URL%"
    exit /b 0
)

REM If the port is held by another application, show its PID and command.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$c = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue; if ($c) { $c | ForEach-Object { $p = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $_.OwningProcess); Write-Output ('Port %PORT% is occupied by PID ' + $_.OwningProcess + ': ' + $p.Name + ' ' + $p.CommandLine) }; exit 1 }; exit 0"
if errorlevel 1 (
    echo.
    echo The port is occupied, and the service did not respond as this Gradio UI.
    echo Stop the listed application or configure a different port.
    pause
    exit /b 1
)

if not exist "%PYTHON%" (
    echo [ERROR] Conda environment Python was not found:
    echo   %PYTHON%
    echo Create the thermal-focus-quality environment as described in README.md.
    pause
    exit /b 1
)

echo Starting Thermal Focus Quality Gradio UI (reload mode)...
echo URL: %URL%
start "" "%URL%"
REM `gradio app.py` runs the app with hot reload: saving a .py file restarts the UI.
"%PYTHON%" -m gradio app.py

if errorlevel 1 (
    echo.
    echo [ERROR] Gradio failed to start. Check the environment dependencies and port %PORT%.
    pause
    exit /b 1
)

endlocal
