@echo off
echo === VLM Smoke Test ===
python -u src\vlm_smoketest.py %*
if %ERRORLEVEL% NEQ 0 (
    echo === Smoke Test Failed ===
    exit /b %ERRORLEVEL%
)
echo === Done ===

