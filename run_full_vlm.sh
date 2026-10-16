#!/usr/bin/env bash
# VLM nagyméretű (4000 kép / teljes adathalmaz) futtató szkript nohup segítségével
# 4-bites kvantálás, egyetlen szerepkör, 4-es kötegméret, naplózás a vlm_run.log fájlba
set -e
cd "$(dirname "$0")"

echo "=== VLM Teljes Futás Indítása Háttérben (nohup) ==="
nohup python -u src/vlm_inference.py \
    --coco_json data/ExDark/coco_annotations_all.json \
    --limit 4000 \
    --batch_size 4 \
    --single_role \
    --load_in_4bit \
    --max_tokens 128 \
    --max_pixels 401408 \
    > vlm_run.log 2>&1 &

PID=$!
echo "A folyamat sikeresen elindult a háttérben. PID: $PID"
echo "Naplófájl valós idejű követése: tail -f vlm_run.log"

