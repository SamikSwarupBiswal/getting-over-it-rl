@echo off
setlocal
rem This viewer reads the learner's shared weights; it does not start training.
if exist "%~dp0runtime\PREVIEW_STOP" del "%~dp0runtime\PREVIEW_STOP"
wsl.exe -d kali-linux --cd "%~dp0." --exec bash -c "JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 exec ~/.venvs/goirl-gpu/bin/python -u scripts/watch_training.py \"$@\"" preview %*
if errorlevel 1 pause
