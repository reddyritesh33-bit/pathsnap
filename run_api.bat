@echo off
REM Laptop 1 - start the PathSnap API.
REM Usage:  run_api.bat "C:\path\to\evaluator_test"
REM The folder must contain the 10 trip folders (each with .track .nodes .arcs .route).
if "%~1"=="" (
  echo Usage: run_api.bat "C:\path\to\evaluator_test"
  exit /b 1
)
set PATHSNAP_EVAL_DIR=%~1
cd /d "%~dp0"
echo Serving trips from %PATHSNAP_EVAL_DIR%
echo Other laptops connect to http://^<this-laptop-IP^>:8000   (find the IP with: ipconfig)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --no-use-colors
