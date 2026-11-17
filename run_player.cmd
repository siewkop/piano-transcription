@echo off
setlocal
pushd "%~dp0" || exit /b 1
set "PYTHON=python"
if defined VIRTUAL_ENV set "PYTHON=%VIRTUAL_ENV%\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" -m piano_transcription.player --checkpoint "models\maestro_600_events\best.pt" %*
set "RESULT=%ERRORLEVEL%"
popd
exit /b %RESULT%
