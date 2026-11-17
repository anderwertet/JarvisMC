@echo off
setlocal EnableExtensions

rem Everything is resolved relative to this file, so the folder can live anywhere.
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "PY=%ROOT%\venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

rem CTranslate2 loads cuBLAS/cuDNN dynamically; keep the pip-shipped copies on PATH.
if exist "%ROOT%\venv\Lib\site-packages\nvidia\cublas\bin" set "PATH=%ROOT%\venv\Lib\site-packages\nvidia\cublas\bin;%PATH%"
if exist "%ROOT%\venv\Lib\site-packages\nvidia\cudnn\bin" set "PATH=%ROOT%\venv\Lib\site-packages\nvidia\cudnn\bin;%PATH%"
if exist "%ROOT%\venv\Lib\site-packages\nvidia\cuda_runtime\bin" set "PATH=%ROOT%\venv\Lib\site-packages\nvidia\cuda_runtime\bin;%PATH%"

set "PYTHONPATH=%ROOT%"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "JARVIS_ROOT=%ROOT%"

rem Speech model: "turbo" (large-v3-turbo, best accuracy on GPU) or "small" (CPU-friendly).
if "%JARVIS_STT_MODEL%"=="" set "JARVIS_STT_MODEL=turbo"

echo [jarvis] starting bridge on http://127.0.0.1:8765
"%PY%" -m app.server
