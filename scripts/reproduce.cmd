@echo off
setlocal
cd /d "%~dp0.."
python -m pytest tests -q
if errorlevel 1 exit /b 1
python scripts\sanity_check.py
if errorlevel 1 exit /b 1
python scripts\train.py --policy band --seeds 42 43 44
if errorlevel 1 exit /b 1
python scripts\train.py --policy direct --seeds 42 43 44
if errorlevel 1 exit /b 1
python scripts\train.py --policy band --blind-cost --seeds 42 43 44
if errorlevel 1 exit /b 1
python scripts\evaluate.py
if errorlevel 1 exit /b 1
python scripts\analyze_results.py
if errorlevel 1 exit /b 1
endlocal
