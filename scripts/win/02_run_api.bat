@echo off
setlocal
cd /d %~dp0\..\..
conda activate vibrant-rag
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
endlocal
