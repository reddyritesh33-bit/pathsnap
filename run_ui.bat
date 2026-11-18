@echo off
REM Laptop 2 - start the PathSnap evaluator UI on port 8501.
REM Then open http://localhost:8501 and enter http://<laptop-1-IP>:8000 as the API URL.
cd /d "%~dp0"
start "" http://localhost:8501
python ui\serve.py 8501
