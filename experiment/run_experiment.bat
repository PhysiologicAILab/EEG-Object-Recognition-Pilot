@echo off
setlocal
cd /d "%~dp0"
if not defined PSYCHOPY_PYTHON set "PSYCHOPY_PYTHON=python"
"%PSYCHOPY_PYTHON%" -c "import psychopy" >nul 2>&1
if errorlevel 1 (
  echo PsychoPy could not be imported with: %PSYCHOPY_PYTHON%
  echo Set PSYCHOPY_PYTHON to the PsychoPy Python executable,
  echo or open main_experiment.py in PsychoPy Coder and press Run.
  pause
  exit /b 1
)
"%PSYCHOPY_PYTHON%" main_experiment.py
if errorlevel 1 pause
endlocal
