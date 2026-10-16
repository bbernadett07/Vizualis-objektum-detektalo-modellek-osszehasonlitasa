"""Run YOLO11 detector on ExDark test images, measure speed, and save unified predictions."""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Any
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))              # Forráskönyvtár hozzáadása az útvonalhoz
from utils import (
    TARGET_CLASSES,                                                     # 4 célosztály listája
    YOLO_COCO_CLASS_MAP,                                                # YOLO osztályleképező szótár
    set_seed,                                                           # Determinisztikus seed beállító
    get_device                                                          # Eszközválasztó segédfüggvény
)


def run_yolo_detection(
    weights_path: str,
    coco_json_path: str,
    images_dir: str,
    output_pred_path: str,
    output_timing_path: str,
    conf_thresh: float = 0.25,
    device: str = "cuda:0",
    max_images: int = 0,
    benchmark_cpu: bool = True
) -> Dict[str, Any]:
    """Run YOLO11 model on test images, recording predictions and latency."""
    print(f"\n[YOLO11] Modell betöltése innen: {weights_path} az eszközre: {device}...")  # Modellbetöltési infó
    model = YOLO(weights_path)                                          # YOLO11 súlyok betöltése a memóriába

    with open(coco_json_path, 'r', encoding='utf-8') as f:             # COCO teszt annotáció megnyitása
        coco_data = json.load(f)                                        # Annotációs fa beolvasása JSON-ből

    images = coco_data['images']                                        # Képek listájának kinyerése
    if max_images > 0 and len(images) > max_images:                     # Opcionális képszám-korlát kezelése
        images = images[:max_images]                                    # Képlista levágása a megadott korlátra
    print(f"[YOLO11] Kiértékelés {len(images)} képen...")               # Képszám kiírása a konzolra

    predictions = {}                                                    # Predikciók tároló szótára
    inference_times_ms = []                                             # Késleltetések tároló listája (ms)

    if "cuda" in device and torch.cuda.is_available():                  # GPU warmup CUDA környezetben
        dummy = torch.zeros((1, 3, 640, 640), device=device)            # Dummy bemeneti tenzor létrehozása
        for _ in range(5):                                              # 5 bemelegítő előrecsatolás
            _ = model(dummy, verbose=False)                             # CUDA kernel fordítások lefuttatása

    for img_info in tqdm(images, desc="YOLO11 Inferencia"):             # Képek bejárása haladási sávval
        img_id = img_info['id']                                         # Kép azonosítója
        img_rel_path = img_info['file_name']                            # Relatív képútvonal
        img_full_path = os.path.join(images_dir, img_rel_path)          # Teljes elérési út

        if not os.path.exists(img_full_path):                           # Hiányzó képfájl ellenőrzése
            continue                                                    # Nem létező kép átugrása

        if "cuda" in device and torch.cuda.is_available():              # Időmérés előtti CUDA szinkronizáció
            torch.cuda.synchronize()                                    # Aszinkron műveletek befejezésének megvárása
        start_t = time.perf_counter()                                   # Nagy felbontású időbélyegző indítása

        results = model.predict(                                        # YOLO11 inferencia futtatása
            source=img_full_path,                                       # Bemeneti kép elérési útja
            conf=conf_thresh,                                           # Konfidencia szűrési küszöbérték
            device=device,                                              # Hardvereszköz (GPU vagy CPU)
            verbose=False                                               # Részletes konzolnaplózás kikapcsolása
        )

        if "cuda" in device and torch.cuda.is_available():              # Időmérés utáni CUDA szinkronizáció
            torch.cuda.synchronize()                                    # Kernel-számítások lezárásának megvárása
        latency_ms = (time.perf_counter() - start_t) * 1000.0          # Eltelt idő milliszekundumban
        inference_times_ms.append(latency_ms)                           # Futásidő rögzítése a listában

        boxes = results[0].boxes                                        # Detektált határolókeretek kinyerése
        img_preds = []                                                  # Képhez tartozó detekciók listája

        if boxes is not None and len(boxes) > 0:                        # Ha van találat a képen
            xyxy = boxes.xyxy.cpu().numpy()                             # Dobozkoordináták átmásolása CPU-ra
            confs = boxes.conf.cpu().numpy()                            # Konfidenciák CPU tömbbé alakítása
            cls_ids = boxes.cls.cpu().numpy().astype(int)               # Osztály azonosítók egész számmá alakítása

            for box, conf, cls_id in zip(xyxy, confs, cls_ids):         # Dobozok egyenkénti bejárása
                label = YOLO_COCO_CLASS_MAP.get(cls_id)                 # Osztály leképezése célosztályra
                if label in TARGET_CLASSES:                             # Csak a 4 vizsgált kategória megtartása
                    x1, y1, x2, y2 = [round(float(coord), 2)            # Koordináták kerekítése 2 tizedesre
                                      for coord in box]                 # List comprehension a 4 koordinátára
                    img_preds.append({                                  # Detekciós elem felépítése
                        "bbox": [x1, y1, x2, y2],                       # Határolókeret sarokpontjai
                        "label": label,                                 # Célosztály neve
                        "score": round(float(conf), 4)                  # Konfidencia pontszám
                    })

        predictions[str(img_id)] = img_preds                            # Predikciók mentése kép ID szerint

    avg_gpu_latency = (float(np.mean(inference_times_ms))               # Átlagos GPU késleltetés számítása
                       if inference_times_ms else 0.0)                  # Üres lista védelem
    gpu_fps = 1000.0 / avg_gpu_latency if avg_gpu_latency > 0 else 0.0  # GPU FPS kiszámítása

    print(f"[YOLO11 GPU] Átlagos késleltetés: {avg_gpu_latency:.2f} ms | FPS: {gpu_fps:.2f}")  # GPU eredmény kiírás

    cpu_stats = {}                                                      # CPU statisztikák szótára
    if benchmark_cpu:                                                   # Ha kértek CPU sebességmérést
        print("[YOLO11] CPU sebességmérés (20 képes részhalmazon)...")  # CPU teszt jelzése
        cpu_times = []                                                  # CPU futásidők listája
        cpu_subset = images[:min(20, len(images))]                      # 20 képes reprezentatív részhalmaz
        cpu_model = YOLO(weights_path)                                  # Külön CPU modell példányosítás
        for img_info in cpu_subset:                                     # Részhalmaz bejárása
            img_full_path = os.path.join(images_dir, img_info['file_name'])  # Kép elérési útja
            t0 = time.perf_counter()                                    # CPU időmérés kezdete
            _ = cpu_model.predict(source=img_full_path, conf=conf_thresh,  # CPU inferencia futtatása
                                  device="cpu", verbose=False)          # Explicit CPU eszköz
            cpu_times.append((time.perf_counter() - t0) * 1000.0)       # CPU késleltetés hozzáadása
        avg_cpu_latency = float(np.mean(cpu_times)) if cpu_times else 0.0  # Átlagos CPU késleltetés
        cpu_fps = 1000.0 / avg_cpu_latency if avg_cpu_latency > 0 else 0.0  # CPU FPS számítása
        print(f"[YOLO11 CPU] Átlagos késleltetés: {avg_cpu_latency:.2f} ms | FPS: {cpu_fps:.2f}")  # CPU eredmény
        cpu_stats = {                                                   # CPU összegző rekord
            "num_images": len(cpu_subset),                              # Tesztelt képek száma
            "avg_latency_ms": round(avg_cpu_latency, 2),                # Átlagos késleltetés ms-ben
            "fps": round(cpu_fps, 2)                                    # Áteresztőképesség FPS-ben
        }

    os.makedirs(os.path.dirname(output_pred_path), exist_ok=True)        # Kimeneti könyvtár létrehozása
    with open(output_pred_path, 'w', encoding='utf-8') as f:           # Predikciós JSON megnyitása írásra
        json.dump(predictions, f, indent=2)                             # Predikciók lemezre írása
    print(f"[YOLO11] Predikciók elmentve ide: {output_pred_path}")      # Sikeres mentés kiírása

    timing_summary = {                                                  # Időzítési összegző rekord
        "model_name": "YOLO11",                                         # Modell megnevezése
        "device": device,                                               # Futtatási hardvereszköz
        "num_images": len(images),                                      # Feldolgozott képek száma
        "avg_latency_ms": round(avg_gpu_latency, 2),                    # Átlagos GPU késleltetés
        "fps": round(gpu_fps, 2),                                       # GPU képkockasebesség
        "cpu_benchmark": cpu_stats                                      # CPU mérés eredményei
    }
    with open(output_timing_path, 'w', encoding='utf-8') as f:         # Időzítési JSON fájl megnyitása
        json.dump(timing_summary, f, indent=2)                          # Időzítési adatok mentése
    print(f"[YOLO11] Futásidő-metrikák elmentve ide: {output_timing_path}")  # Mentési siker

    return predictions                                                  # Predikciók visszaadása memóriában


def main():
    parser = argparse.ArgumentParser(description="YOLO11 detektor futtatása az ExDark adathalmazon.")  # Parser
    parser.add_argument("--weights", type=str, default="models/yolo11/yolo11n.pt",  # Súlyfájl elérési útja
                        help="YOLO súlyfájl elérési útja.")             # Súgó
    parser.add_argument("--coco_json", type=str,                        # COCO teszt JSON fájl
                        default="data/ExDark/coco_annotations_test.json",  # Alapértelmezett COCO tesztfájl
                        help="COCO teszt annotációs fájl.")             # Súgó
    parser.add_argument("--images_dir", type=str, default="data/ExDark/images",  # Képek könyvtára
                        help="Képek gyökérkönyvtára.")                  # Súgó
    parser.add_argument("--output_pred", type=str,                      # Kimeneti predikciók JSON
                        default="results/metrics/yolo11_predictions.json",  # Predikció mentési helye
                        help="Kimeneti predikciós JSON fájl.")          # Súgó
    parser.add_argument("--output_timing", type=str,                    # Kimeneti időzítés JSON
                        default="results/metrics/yolo11_timing.json",    # Időzítés mentési helye
                        help="Kimeneti időmérési JSON fájl.")           # Súgó
    parser.add_argument("--conf_thresh", type=float, default=0.25,      # Konfidenciaküszöb
                        help="Konfidencia küszöbérték a szűréshez.")    # Súgó
    parser.add_argument("--device", type=str,                           # Hardver eszköz választás
                        default="cuda:0" if torch.cuda.is_available() else "cpu",  # Automatikus CUDA/CPU
                        help="Számítási eszköz (GPU vagy CPU).")        # Súgó
    parser.add_argument("--max_images", type=int, default=0,            # Képszám korlát
                        help="Értékelendő képek maximális száma.")      # Súgó
    parser.add_argument("--no_cpu_bench", action="store_true",          # CPU teszt kihagyása
                        help="CPU mérés kihagyása.")                    # Súgó
    parser.add_argument("--seed", type=int, default=42,                 # Véletlenszám mag
                        help="Véletlenszám-generátor kezdőértéke.")     # Súgó
    args = parser.parse_args()                                          # Argumentumok feldolgozása

    set_seed(args.seed)                                                 # Determinisztikus seed rögzítése
    run_yolo_detection(                                                 # Detekciós függvény indítása
        weights_path=args.weights,                                      # Súlyfájl átadása
        coco_json_path=args.coco_json,                                  # COCO JSON átadása
        images_dir=args.images_dir,                                     # Képkönyvtár átadása
        output_pred_path=args.output_pred,                              # Predikció kimenet
        output_timing_path=args.output_timing,                          # Időzítés kimenet
        conf_thresh=args.conf_thresh,                                  # Küszöbérték
        device=args.device,                                             # Hardvereszköz
        max_images=args.max_images,                                     # Képszám limit
        benchmark_cpu=not args.no_cpu_bench                             # CPU teszt kapcsoló
    )


if __name__ == "__main__":
    main()                                                              # Szkript futtatása
