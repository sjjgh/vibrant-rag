@echo off
setlocal
cd /d %~dp0\..\..
conda activate vibrant-rag
python -m pip install -r requirements.txt
python ingest_lightrag.py
endlocal
