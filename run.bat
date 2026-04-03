@echo off
cd /d "%~dp0"
echo Starting Body Measure MVP at http://127.0.0.1:8000
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
