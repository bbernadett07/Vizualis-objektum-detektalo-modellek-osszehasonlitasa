"""Run Faster R-CNN detector on ExDark test images, measure speed, and save unified predictions."""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))              # Forráskönyvtár hozzáadása az elérési úthoz
from utils import (
    TARGET_CLASSES,                                                     # 4 balesetveszélyes célosztály
    TORCHVISION_COCO_CLASS_MAP,                                         # TorchVision 91-osztályos leképező
    set_seed,                                                           # Determinisztikus véletlenszám-kezdőérték
    get_device                                                          # Hardvereszköz lekérdező segédfüggvény
)


def load_fasterrcnn_model(
    weights_path: Optional[str] = None,                                 # Lokális súlyfájl elérési útja
    device: str = "cuda:0"                                              # Céleszköz (GPU vagy CPU)
) -> torch.nn.Module:
    """Load Faster R-CNN ResNet-50 FPN model."""
    if weights_path and os.path.exists(weights_path):                   # Helyi súlyfájl meglétének ellenőrzése
        print(f"[Faster R-CNN] Súlyok betöltése fájlból: {weights_path}")# Tájékoztató üzenet a fájl alapú betöltésről
        model = fasterrcnn_resnet50_fpn(weights=None)                   # Üres modell architektúra inicializálása
        state_dict = torch.load(                                        # Súlyok beolvasása CPU memóriába
            weights_path,                                               # Súlyfájl útvonala
            map_location="cpu"                                          # CPU memória felesleges GPU allokáció nélkül
        )
        model.load_state_dict(state_dict)                               # Állapot-szótár betöltése a modellbe
    else:
        print("[Faster R-CNN] Súlyok betöltése torchvision-ből...")     # Tájékoztató torchvision letöltésről
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT               # TorchVision hivatalos előtanított súlyai
        model = fasterrcnn_resnet50_fpn(weights=weights)                # Modell példányosítása gyári súlyokkal

    model.to(device)                                                    # Hálózat áthelyezése a számítási eszközre
    model.eval()                                                        # Kiértékelési mód (eval) bekapcsolása
    return model                                                        # Betöltött modell visszaadása


def run_fasterrcnn_detection(
    weights_path: str,                                                  # Súlyfájl elérési útja
    coco_json_path: str,                                                # COCO formátumú annotációk elérési útja
    images_dir: str,                                                    # Tesztképek gyökérkönyvtára
    output_pred_path: str,                                              # Kimeneti predikciós JSON fájl
    output_timing_path: str,                                            # Kimeneti futásidő JSON fájl
    conf_thresh: float = 0.25,                                          # Konfidencia szűrési küszöbérték
    device: str = "cuda:0",                                             # Számítási eszköz megnevezése
    max_images: int = 0,                                                # Maximálisan feldolgozandó képek száma
    benchmark_cpu: bool = True                                          # CPU referencia mérés futtatása
) -> Dict[str, Any]:
    """Run Faster R-CNN model on test images, recording predictions and latency."""
    print(f"\n[Faster R-CNN] Inicializálás az eszközön: {device}...")   # Tájékoztató kiírása konzolra
    model = load_fasterrcnn_model(weights_path, device=device)          # Modell betöltése és felkészítése

    with open(coco_json_path, 'r', encoding='utf-8') as f:             # COCO annotációs fájl megnyitása
        coco_data = json.load(f)                                        # JSON adatok strukturált beolvasása

    images = coco_data['images']                                        # Képek metaadatainak kinyerése
    if max_images > 0 and len(images) > max_images:                     # Opcionális képszám-korlátozás
        images = images[:max_images]                                    # Képlista levágása a limitre
    print(f"[Faster R-CNN] Kiértékelés {len(images)} képen...")         # Feldolgozandó képek száma

    predictions = {}                                                    # Predikciók tároló szótára
    inference_times_ms = []                                             # Latencia értékek listája ms-ban

    if "cuda" in device and torch.cuda.is_available():                  # CUDA bemelegítés inicializálása
        dummy = [torch.zeros((3, 640, 640), device=device)]             # Dummy kép tenzor létrehozása GPU-n
        with torch.no_grad():                                           # Gradiensmentes bemelegítő kontextus
            for _ in range(5):                                          # 5 bemelegítő lépés futtatása
                _ = model(dummy)                                        # CUDA kernelek és RoI belső memória előkészítése

    with torch.no_grad():                                               # Gradiens-számítás letiltása inferenciánál
        for img_info in tqdm(images, desc="Faster R-CNN Inferencia"):   # Végighaladás a képeken haladási sávval
            img_id = img_info['id']                                     # Kép egyedi azonosítója
            img_rel_path = img_info['file_name']                        # Relatív képútvonal
            img_full_path = os.path.join(images_dir, img_rel_path)      # Teljes képelérési útvonal

            if not os.path.exists(img_full_path):                       # Kép létezésének vizsgálata
                continue                                                # Hiányzó kép átugrása

            try:                                                        # Képbeolvasás védett blokkban
                pil_img = Image.open(img_full_path).convert('RGB')      # Kép megnyitása és RGB konverzió
                tensor_img = TF.to_tensor(pil_img).to(device)           # Tenzorrá alakítás és áttöltés az eszközre
            except Exception as e:                                      # Beolvasási hiba kezelése
                print(f"[Figyelmeztetés] Hiba a képnél {img_full_path}: {e}")  # Figyelmeztetés kiírása
                continue                                                # Hibás kép átugrása

            if "cuda" in device and torch.cuda.is_available():          # Időmérés előtti szinkronizáció
                torch.cuda.synchronize()                                # Befejezetlen CUDA műveletek megvárása
            start_t = time.perf_counter()                               # Időmérés indítása nanoszekundumos pontossággal

            outputs = model([tensor_img])[0]                            # Kétlépcsős detektálás és előrecsatolás

            if "cuda" in device and torch.cuda.is_available():          # Időmérés utáni szinkronizáció
                torch.cuda.synchronize()                                # GPU feladatok végének megvárása
            latency_ms = (time.perf_counter() - start_t) * 1000.0       # Futási idő kiszámítása milliszekundumban
            inference_times_ms.append(latency_ms)                       # Késleltetés elmentése a listába

            boxes = outputs['boxes'].cpu().numpy()                      # Határolókeretek mozgatása CPU-ra
            scores = outputs['scores'].cpu().numpy()                    # Konfidencia értékek másolása CPU-ra
            labels = outputs['labels'].cpu().numpy()                    # Osztálycímkék másolása CPU-ra

            img_preds = []                                              # Képhez tartozó detekciók listája
            for box, score, label_id in zip(boxes, scores, labels):     # Detekciók egyesével történő bejárása
                if score < conf_thresh:                                 # Konfidencia küszöbérték vizsgálata
                    continue                                            # Alacsony konfidenciájú doboz elvetése
                label_name = TORCHVISION_COCO_CLASS_MAP.get(            # TorchVision címke leképezése névre
                    int(label_id)                                       # Egész típusú osztályazonosító
                )
                if label_name in TARGET_CLASSES:                        # Célosztályok ellenőrzése
                    x1, y1, x2, y2 = [                                  # Koordináták kerekítése
                        round(float(coord), 2) for coord in box         # Float típus és 2 tizedes kerekítés
                    ]
                    img_preds.append({                                  # Egységes formátumú detekció hozzáadása
                        "bbox": [x1, y1, x2, y2],                       # Határolókeret pixel koordináták
                        "label": label_name,                            # Felismert osztály megnevezése
                        "score": round(float(score), 4)                 # Kerekített konfidencia érték
                    })

            predictions[str(img_id)] = img_preds                        # Képhez tartozó találatok rögzítése

    avg_gpu_latency = (                                                 # Átlagos GPU késleltetés kiszámítása
        float(np.mean(inference_times_ms))                              # Késleltetések átlaga
        if inference_times_ms else 0.0                                  # Védelem üres lista esetén
    )
    gpu_fps = 1000.0 / avg_gpu_latency if avg_gpu_latency > 0 else 0.0  # Képkockasebesség (FPS) kiszámítása

    print(f"[Faster R-CNN GPU] Késleltetés: {avg_gpu_latency:.2f} ms | FPS: {gpu_fps:.2f}")  # Eredmény kiírása

    cpu_stats = {}                                                      # CPU referencia statisztikák
    if benchmark_cpu:                                                   # Ha a CPU mérés engedélyezett
        print("[Faster R-CNN] CPU sebességmérés (20 képes mintán)...")  # Tájékoztatás CPU mérésről
        cpu_times = []                                                  # CPU futásidők gyűjtője
        cpu_subset = images[:min(20, len(images))]                      # Első 20 kép kiválasztása mintának
        cpu_model = load_fasterrcnn_model(weights_path, device="cpu")   # Külön CPU modell példány betöltése
        with torch.no_grad():                                           # Gradiensmentes kiértékelés CPU-n
            for img_info in cpu_subset:                                 # CPU tesztképek bejárása
                img_full_path = os.path.join(                           # Kép elérési útjának összeállítása
                    images_dir, img_info['file_name']                   # Képek könyvtára és fájlneve
                )
                pil_img = Image.open(img_full_path).convert('RGB')      # Kép megnyitása és RGB konverzió
                tensor_img = TF.to_tensor(pil_img).to("cpu")            # Átalakítás tenzorrá CPU-n
                t0 = time.perf_counter()                                # CPU időmérés indítása
                _ = cpu_model([tensor_img])                             # CPU előrecsatolás futtatása
                cpu_times.append((time.perf_counter() - t0) * 1000.0)   # Eltelt idő hozzáadása ms-ban
        avg_cpu_latency = (                                             # Átlagos CPU késleltetés
            float(np.mean(cpu_times)) if cpu_times else 0.0             # CPU idők átlagolása
        )
        cpu_fps = 1000.0 / avg_cpu_latency if avg_cpu_latency > 0 else 0.0  # CPU FPS számítása
        print(f"[Faster R-CNN CPU] Késleltetés: {avg_cpu_latency:.2f} ms | FPS: {cpu_fps:.2f}")  # CPU eredmény
        cpu_stats = {                                                   # CPU összegző szótár készítése
            "num_images": len(cpu_subset),                              # Feldolgozott képek száma
            "avg_latency_ms": round(avg_cpu_latency, 2),                # Kerekített CPU késleltetés
            "fps": round(cpu_fps, 2)                                    # Kerekített CPU képkockasebesség
        }

    os.makedirs(os.path.dirname(output_pred_path), exist_ok=True)       # Célmappa létrehozása ha szükséges
    with open(output_pred_path, 'w', encoding='utf-8') as f:             # Predikciós JSON fájl megnyitása
        json.dump(predictions, f, indent=2)                             # Eredmények kiírása formázott JSON-ként
    print(f"[Faster R-CNN] Predikciók elmentve: {output_pred_path}")    # Sikeres mentés visszaigazolása

    timing_summary = {                                                  # Időzítési összegzés szótár
        "model_name": "Faster R-CNN",                                   # Modell megnevezése
        "device": device,                                               # Használt hardvereszköz
        "num_images": len(images),                                      # Feldolgozott képek száma
        "avg_latency_ms": round(avg_gpu_latency, 2),                    # Kerekített GPU latencia
        "fps": round(gpu_fps, 2),                                       # Kerekített GPU FPS
        "cpu_benchmark": cpu_stats                                      # CPU mérés részletei
    }
    with open(output_timing_path, 'w', encoding='utf-8') as f:           # Időzítési JSON fájl megnyitása
        json.dump(timing_summary, f, indent=2)                          # Időzítési adatok mentése
    print(f"[Faster R-CNN] Időzítési adatok elmentve: {output_timing_path}")  # Mentés megerősítése

    return predictions                                                  # Predikciók visszaadása


def main():
    parser = argparse.ArgumentParser(                                   # Parancssori argumentumkezelő
        description="Faster R-CNN futtatása az ExDark adathalmazon."     # Leírás
    )
    parser.add_argument(                                                # Súlyfájl paraméter
        "--weights", type=str,                                          # Típus string
        default="models/faster_rcnn/fasterrcnn_resnet50_fpn.pth",       # Alapértelmezett modell útvonal
        help="Faster R-CNN súlyfájl útvonala."                          # Segítő szöveg
    )
    parser.add_argument(                                                # COCO JSON paraméter
        "--coco_json", type=str,                                        # Típus string
        default="data/ExDark/coco_annotations_test.json",               # Alapértelmezett teszt JSON
        help="COCO teszt annotációk útvonala."                          # Segítő leírás
    )
    parser.add_argument(                                                # Képkönyvtár paraméter
        "--images_dir", type=str,                                       # Típus string
        default="data/ExDark/images",                                   # Alapértelmezett képmappa
        help="Képek gyökérkönyvtára."                                   # Segítő leírás
    )
    parser.add_argument(                                                # Kimeneti predikció paraméter
        "--output_pred", type=str,                                      # Típus string
        default="results/metrics/faster_rcnn_predictions.json",         # Alapértelmezett kimenet
        help="Predikciók kimeneti JSON fájlja."                         # Segítő leírás
    )
    parser.add_argument(                                                # Kimeneti időzítés paraméter
        "--output_timing", type=str,                                    # Típus string
        default="results/metrics/faster_rcnn_timing.json",              # Alapértelmezett kimenet
        help="Időzítési adatok kimeneti JSON fájlja."                   # Segítő leírás
    )
    parser.add_argument(                                                # Küszöbérték paraméter
        "--conf_thresh", type=float, default=0.25,                      # Alapértelmezett küszöbérték 0.25
        help="Konfidencia szűrési küszöbérték."                         # Segítő leírás
    )
    parser.add_argument(                                                # Számítási eszköz paraméter
        "--device", type=str,                                           # Típus string
        default="cuda:0" if torch.cuda.is_available() else "cpu",       # Automatikus GPU választás
        help="Számítási eszköz (cuda:0 vagy cpu)."                      # Segítő leírás
    )
    parser.add_argument(                                                # Képszám korlát paraméter
        "--max_images", type=int, default=0,                            # Alapértelmezett 0 (összes kép)
        help="Értékelendő képek maximális száma."                       # Segítő leírás
    )
    parser.add_argument(                                                # CPU benchmark elhagyása paraméter
        "--no_cpu_bench", action="store_true",                          # Logikai kapcsoló
        help="CPU tesztelés kihagyása."                                 # Segítő leírás
    )
    parser.add_argument(                                                # Véletlenszám seed paraméter
        "--seed", type=int, default=42,                                 # Alapértelmezett seed 42
        help="Véletlenszám generátor kezdőértéke."                       # Segítő leírás
    )
    args = parser.parse_args()                                          # Argumentumok feldolgozása

    set_seed(args.seed)                                                 # Véletlenszám mag beállítása
    run_fasterrcnn_detection(                                           # Detektáló függvény indítása
        weights_path=args.weights,                                      # Modell útvonal átadása
        coco_json_path=args.coco_json,                                  # Annotáció útvonal átadása
        images_dir=args.images_dir,                                     # Képek mappájának átadása
        output_pred_path=args.output_pred,                              # Predikciós kimenet átadása
        output_timing_path=args.output_timing,                          # Időzítési kimenet átadása
        conf_thresh=args.conf_thresh,                                   # Küszöbérték átadása
        device=args.device,                                             # Hardvereszköz átadása
        max_images=args.max_images,                                     # Képszám limit átadása
        benchmark_cpu=not args.no_cpu_bench                             # CPU mérés logikai értéke
    )


if __name__ == "__main__":
    main()
