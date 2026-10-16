"""Run RT-DETR detector on ExDark test images, measure speed, and save unified predictions."""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Any
import numpy as np
import torch
from ultralytics import RTDETR
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))              # Forráskönyvtár hozzáadása az elérési úthoz
from utils import (
    TARGET_CLASSES,                                                     # 4 balesetveszélyes célosztály
    YOLO_COCO_CLASS_MAP,                                                # Osztályleképező szótár
    set_seed,                                                           # Determinisztikus seed beállító
    get_device                                                          # Hardvereszköz lekérdező
)


def run_rtdetr_detection(
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
    """Run RT-DETR model on test images, recording predictions and latency."""
    print(f"\n[RT-DETR] Modell betöltése innen: {weights_path} az eszközre: {device}...")  # Tájékoztatás
    model = RTDETR(weights_path)                                        # RT-DETR transformer modell betöltése

    with open(coco_json_path, 'r', encoding='utf-8') as f:             # COCO teszt JSON megnyitása olvasásra
        coco_data = json.load(f)                                        # Annotációk beolvasása JSON-ből

    images = coco_data['images']                                        # Képek listájának kinyerése
    if max_images > 0 and len(images) > max_images:                     # Opcionális részhalmazos tesztelés
        images = images[:max_images]                                    # Képlista levágása a megadott korlátra
    print(f"[RT-DETR] Kiértékelés {len(images)} képen...")              # Képszám kiírása a konzolra

    predictions = {}                                                    # Predikciók tároló szótára
    inference_times_ms = []                                             # Késleltetések tároló listája (ms)

    if "cuda" in device and torch.cuda.is_available():                  # GPU bemelegítés CUDA környezetben
        dummy = torch.zeros((1, 3, 640, 640), device=device)            # Dummy bemeneti tenzor létrehozása
        for _ in range(5):                                              # 5 bemelegítő előrecsatolás futtatása
            _ = model(dummy, verbose=False)                             # Figyelmi rétegek és memóriák allokálása

    for img_info in tqdm(images, desc="RT-DETR Inferencia"):            # Képek bejárása haladási sávval
        img_id = img_info['id']                                         # Kép egyedi azonosítója
        img_rel_path = img_info['file_name']                            # Relatív képútvonal
        img_full_path = os.path.join(images_dir, img_rel_path)          # Teljes elérési út

        if not os.path.exists(img_full_path):                           # Hiányzó képfájl ellenőrzése
            continue                                                    # Nem létező kép átugrása

        if "cuda" in device and torch.cuda.is_available():              # Időmérés előtti CUDA szinkronizáció
            torch.cuda.synchronize()                                    # Aszinkron GPU műveletek megvárása
        start_t = time.perf_counter()                                   # Pontos időmérés indítása

        results = model.predict(                                        # RT-DETR következtetés végrehajtása
            source=img_full_path,                                       # Bemeneti kép útvonala
            conf=conf_thresh,                                           # Konfidenciaküszöb
            device=device,                                              # Hardvereszköz
            verbose=False                                               # Részletes logolás tiltása
        )

        if "cuda" in device and torch.cuda.is_available():              # Időmérés utáni CUDA szinkronizáció
            torch.cuda.synchronize()                                    # Számítások befejezésének megvárása
        latency_ms = (time.perf_counter() - start_t) * 1000.0          # Eltelt idő milliszekundumban
        inference_times_ms.append(latency_ms)                           # Futásidő mentése a statisztikához

        boxes = results[0].boxes                                        # Transformer kimeneti dobozok kinyerése
        img_preds = []                                                  # Detekciók tárolója

        if boxes is not None and len(boxes) > 0:                        # Ha talált dobozokat a modell
            xyxy = boxes.xyxy.cpu().numpy()                             # Koordináták átmásolása CPU memóriába
            confs = boxes.conf.cpu().numpy()                            # Konfidenciák CPU tömbbé alakítása
            cls_ids = boxes.cls.cpu().numpy().astype(int)               # Osztály azonosítók kinyerése

            for box, conf, cls_id in zip(xyxy, confs, cls_ids):         # Detektált dobozok bejárása
                label = YOLO_COCO_CLASS_MAP.get(cls_id)                 # Célosztálynév azonosítása
                if label in TARGET_CLASSES:                             # Szigorú szűrés a 4 célkategóriára
                    x1, y1, x2, y2 = [round(float(coord), 2)            # Koordináták 2 tizedesre kerekítése
                                      for coord in box]                 # Tömör koordináta feldolgozás
                    img_preds.append({                                  # Predikciós rekord létrehozása
                        "bbox": [x1, y1, x2, y2],                       # Határolókeret csúcsai
                        "label": label,                                 # Célosztály címke
                        "score": round(float(conf), 4)                  # Konfidencia érték
                    })

        predictions[str(img_id)] = img_preds                            # Eredmények tárolása kép ID alapján

    avg_gpu_latency = (float(np.mean(inference_times_ms))               # Átlagos GPU késleltetés kiszámítása
                       if inference_times_ms else 0.0)                  # Üres lista kezelése
    gpu_fps = 1000.0 / avg_gpu_latency if avg_gpu_latency > 0 else 0.0  # GPU képkockasebesség (FPS)

    print(f"[RT-DETR GPU] Átlagos késleltetés: {avg_gpu_latency:.2f} ms | FPS: {gpu_fps:.2f}")  # Eredmény kiírása

    cpu_stats = {}                                                      # CPU statisztikai rekord
    if benchmark_cpu:                                                   # Ha kértek CPU sebességmérést
        print("[RT-DETR] CPU sebességmérés (20 képes részhalmazon)...")  # CPU teszt jelzése
        cpu_times = []                                                  # CPU időmérések listája
        cpu_subset = images[:min(20, len(images))]                      # 20 képes reprezentatív minta
        cpu_model = RTDETR(weights_path)                                # Külön CPU modell példány
        for img_info in cpu_subset:                                     # Részhalmaz bejárása
            img_full_path = os.path.join(images_dir, img_info['file_name'])  # Kép elérési útja
            t0 = time.perf_counter()                                    # CPU időmérés kezdete
            _ = cpu_model.predict(source=img_full_path, conf=conf_thresh,  # CPU inferencia futtatása
                                  device="cpu", verbose=False)          # Explicit CPU eszköz megadása
            cpu_times.append((time.perf_counter() - t0) * 1000.0)       # CPU futásidő mentése
        avg_cpu_latency = float(np.mean(cpu_times)) if cpu_times else 0.0  # Átlagos CPU idő
        cpu_fps = 1000.0 / avg_cpu_latency if avg_cpu_latency > 0 else 0.0  # CPU FPS számítása
        print(f"[RT-DETR CPU] Átlagos késleltetés: {avg_cpu_latency:.2f} ms | FPS: {cpu_fps:.2f}")  # Eredmény
        cpu_stats = {                                                   # CPU összegző szótár
            "num_images": len(cpu_subset),                              # Képek száma
            "avg_latency_ms": round(avg_cpu_latency, 2),                # Átlagos idő ms-ben
            "fps": round(cpu_fps, 2)                                    # Áteresztőképesség FPS
        }

    os.makedirs(os.path.dirname(output_pred_path), exist_ok=True)        # Kimeneti könyvtár létrehozása
    with open(output_pred_path, 'w', encoding='utf-8') as f:           # JSON fájl megnyitása írásra
        json.dump(predictions, f, indent=2)                             # Predikciók kiírása lemezre
    print(f"[RT-DETR] Predikciók elmentve ide: {output_pred_path}")     # Mentési siker kiírása

    timing_summary = {                                                  # Időzítési összegzés
        "model_name": "RT-DETR",                                        # Modell neve
        "device": device,                                               # Futtató hardver
        "num_images": len(images),                                      # Képek száma
        "avg_latency_ms": round(avg_gpu_latency, 2),                    # Átlagos GPU késleltetés
        "fps": round(gpu_fps, 2),                                       # GPU sebesség FPS
        "cpu_benchmark": cpu_stats                                      # CPU mérések adatai
    }
    with open(output_timing_path, 'w', encoding='utf-8') as f:         # Időzítési JSON fájl megnyitása
        json.dump(timing_summary, f, indent=2)                          # Időzítési adatok mentése
    print(f"[RT-DETR] Futásidő-metrikák elmentve ide: {output_timing_path}")  # Mentési siker

    return predictions                                                  # Predikciók visszaadása


def main():
    parser = argparse.ArgumentParser(description="RT-DETR detektor futtatása az ExDark adathalmazon.")  # Parser
    parser.add_argument("--weights", type=str, default="models/rtdetr/rtdetr-l.pt",  # Súlyfájl elérési útja
                        help="RT-DETR súlyfájl elérési útja.")          # Súgó
    parser.add_argument("--coco_json", type=str,                        # COCO teszt JSON fájl
                        default="data/ExDark/coco_annotations_test.json",  # Alapértelmezett fájl
                        help="COCO teszt annotációk helye.")            # Súgó
    parser.add_argument("--images_dir", type=str, default="data/ExDark/images",  # Képek könyvtára
                        help="Képek gyökérkönyvtára.")                  # Súgó
    parser.add_argument("--output_pred", type=str,                      # Kimeneti predikciók JSON
                        default="results/metrics/rtdetr_predictions.json",  # Predikció mentési helye
                        help="Kimeneti predikciós JSON fájl.")          # Súgó
    parser.add_argument("--output_timing", type=str,                    # Kimeneti időzítés JSON
                        default="results/metrics/rtdetr_timing.json",   # Időzítés mentési helye
                        help="Kimeneti időmérési JSON fájl.")           # Súgó
    parser.add_argument("--conf_thresh", type=float, default=0.25,      # Konfidenciaküszöb
                        help="Konfidencia küszöbérték a szűréshez.")    # Súgó
    parser.add_argument("--device", type=str,                           # Hardver eszköz választás
                        default="cuda:0" if torch.cuda.is_available() else "cpu",  # Automatikus CUDA/CPU
                        help="Számítási eszköz (GPU vagy CPU).")        # Súgó
    parser.add_argument("--max_images", type=int, default=0,            # Képszám korlát
                        help="Értékelendő képek maximális száma.")      # Súgó
    parser.add_argument("--no_cpu_bench", action="store_true",          # CPU teszt kihagyása
                        help="CPU tesztelés kihagyása.")                # Súgó
    parser.add_argument("--seed", type=int, default=42,                 # Véletlenszám mag
                        help="Véletlenszám-generátor kezdőértéke.")     # Súgó
    args = parser.parse_args()                                          # Argumentumok feldolgozása

    set_seed(args.seed)                                                 # Determinisztikus seed rögzítése
    run_rtdetr_detection(                                               # Detekció futtatása
        weights_path=args.weights,                                      # Súlyfájl
        coco_json_path=args.coco_json,                                  # COCO JSON
        images_dir=args.images_dir,                                     # Képkönyvtár
        output_pred_path=args.output_pred,                              # Predikció kimenet
        output_timing_path=args.output_timing,                          # Időzítés kimenet
        conf_thresh=args.conf_thresh,                                  # Küszöbérték
        device=args.device,                                             # Hardvereszköz
        max_images=args.max_images,                                     # Képszám limit
        benchmark_cpu=not args.no_cpu_bench                             # CPU teszt kapcsoló
    )


if __name__ == "__main__":
    main()                                                              # Belépési pont meghívása
