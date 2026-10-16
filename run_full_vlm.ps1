# VLM nagyméretű (4000 kép / teljes adathalmaz) futtató PowerShell szkript
# 4-bites kvantálás, egyetlen szerepkör, 4-es kötegméret, naplózás a vlm_run.log fájlba
Write-Host "=== VLM Teljes Futás Indítása Háttérben (PowerShell) ===" -ForegroundColor Cyan

$process = Start-Process -FilePath "python" `
    -ArgumentList "-u src/vlm_inference.py --coco_json data/ExDark/coco_annotations_all.json --limit 4000 --batch_size 4 --single_role --load_in_4bit --max_tokens 128 --max_pixels 401408" `
    -RedirectStandardOutput "vlm_run.log" `
    -RedirectStandardError "vlm_run_err.log" `
    -PassThru

Write-Host "A folyamat sikeresen elindult a háttérben. PID: $($process.Id)" -ForegroundColor Green
Write-Host "Naplófájl valós idejű követése: Get-Content -Wait vlm_run.log" -ForegroundColor Yellow

