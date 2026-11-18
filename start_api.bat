@echo off
cd /d "%~dp0"
if not exist logs mkdir logs
.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload > logs\server.txt 2>&1
echo Server terminato. Apri logs\server.txt per i dettagli.
pause
