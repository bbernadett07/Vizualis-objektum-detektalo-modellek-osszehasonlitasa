"""Standalone VLM Smoke Test module for ExDark benchmark.

Runs multi-stage timing and safety gates:
  Stage 1: 1 image   (Budget: < 15 seconds)
  Stage 2: 5 images  (Budget: < 120 seconds)
  Stage 3: 20 images (Budget: < 300 seconds)

Enforces:
- Shared loader and inference logic from vlm_inference.py
- torch.inference_mode()
- Max 768x768 aspect-preserving resize
- Processor min_pixels / max_pixels caps
- max_new_tokens = 128 (Role 1), 256 (Role 2)
- 60s hard per-image timeout wrapper
- torch.cuda.empty_cache() between images
- Peak VRAM tracking via torch.cuda.max_memory_allocated()
- Comprehensive failure diagnosis & remediation in results/smoketest_failure.json
- Final benchmark audit in results/smoketest_report.json
"""

import os
import sys
import json
import time
import datetime
import argparse
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))              # Forráskönyvtár hozzáadása az importálási úthoz
from vlm_inference import (
    load_vlm,                                                           # Megosztott egységes VLM modellbetöltő
    run_vlm_on_image,                                                   # Két szerepkörös inferenciát végrehajtó függvény
    resize_image_safe,                                                  # 768px-es képarányt megőrző biztonsági átméretező
    log_msg                                                             # Időbélyeges pufferürítő naplózó segédfüggvény
)


def get_vram_peak_mb() -> float:                                        # Csúcs VRAM lekérdező segédfüggvény
    """Return peak GPU VRAM allocated in MB if CUDA is available."""
    if torch.cuda.is_available():                                       # Ha a CUDA rendelkezésre áll
        return torch.cuda.max_memory_allocated() / (1024 * 1024)        # Csúcs memória átváltása bájtból megabájtba
    return 0.0                                                          # CPU futás esetén 0.0 érték


def reset_vram_peak():                                                  # VRAM csúcsérték nullázó és gyorsítótár ürítő
    """Reset peak VRAM tracking and flush CUDA cache."""
    if torch.cuda.is_available():                                       # Csak GPU esetén releváns
        torch.cuda.reset_peak_memory_stats()                            # Statisztikai csúcsszámláló visszaállítása
        torch.cuda.empty_cache()                                        # Nem használt gyorsítótárazott memória elengedése


def print_startup_diagnostics(args: argparse.Namespace):                # Részletes indítási diagnosztikát kiíró függvény
    """Print comprehensive hardware and library diagnostics."""
    print("=" * 70, flush=True)                                         # Elválasztó vonal
    print("              VLM INFERENCIA CSŐVEZETÉK FÜSTTESZT (SMOKE TEST)", flush=True)  # Fejléc cím
    print("=" * 70, flush=True)                                         # Elválasztó vonal

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")     # Dátum és idő formázása
    print(f"Időbélyeg:              {now_str}", flush=True)             # Időbélyeg kiírása
    print(f"Python verzió:          {sys.version.split()[0]}", flush=True)  # Python főverzió kiírása
    print(f"PyTorch verzió:         {torch.__version__}", flush=True)   # Telepített PyTorch verzió

    cuda_avail = torch.cuda.is_available()                              # CUDA elérhetőség lekérdezése
    cuda_version = torch.version.cuda if cuda_avail else "Nem elérhető" # CUDA runtime verzió
    print(f"CUDA elérhetőség:       {cuda_avail} (CUDA verzió: {cuda_version})", flush=True)

    if cuda_avail:                                                      # GPU jelenléte esetén
        device_name = torch.cuda.get_device_name(0)                     # Videokártya pontos típusa
        total_vram_mb = (                                               # Összes fizikai VRAM megabájtban
            torch.cuda.get_device_properties(0).total_memory
            / (1024 * 1024)
        )
        print(f"GPU eszköz:             {device_name} ({total_vram_mb:.0f} MB VRAM)", flush=True)
    else:                                                               # CPU esetén figyelmeztetés
        print("[FIGYELMEZTETÉS] CUDA NEM elérhető! CPU futás nagyon lassú lesz.", flush=True)

    import transformers                                                 # Transformers modul importálása
    print(f"Transformers verzió:    {transformers.__version__}", flush=True)  # Verziószám megjelenítése

    bnb_avail = False                                                   # BitsAndBytes jelenlét logikai jelzője
    try:                                                                # BitsAndBytes import ellenőrzése
        import bitsandbytes                                             # Kvantáló csomag importálása
        bnb_avail = True                                                # Sikeres import jelzése
    except ImportError:                                                 # Hiányzó csomag kezelése
        pass
    print(f"bitsandbytes (4-bit):   {bnb_avail}", flush=True)           # Kvantálási képesség állapota

    api_keys_found = []                                                 # Megtalált API kulcsok listája
    if os.getenv("OPENAI_API_KEY"):                                     # OpenAI kulcs vizsgálata
        api_keys_found.append("OPENAI_API_KEY")                         # Kulcs regisztrálása
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):      # Google kulcs vizsgálata
        api_keys_found.append("GOOGLE_API_KEY/GEMINI_API_KEY")          # Kulcs regisztrálása
    if os.getenv("ANTHROPIC_API_KEY"):                                  # Anthropic kulcs vizsgálata
        api_keys_found.append("ANTHROPIC_API_KEY")                      # Kulcs regisztrálása

    if api_keys_found:                                                  # Ha van elérhető API kulcs
        print(f"Talált API kulcsok:     {', '.join(api_keys_found)} (API tartalék kész)", flush=True)
    else:                                                               # Ha nincs kulcs
        print("Talált API kulcsok:     Nincs (Helyi modell módban üzemel)", flush=True)

    print("\n[Aktív Biztonsági Limitek]", flush=True)                   # Biztonsági limitek fejléce
    print(f"- Backend:              {args.backend} ({args.model})", flush=True)  # Modell és backend
    print(f"- Cél hardver:          {args.device}", flush=True)         # Hardver
    print("- Képfelbontás limit:   768x768 (képarány megőrzésével)", flush=True)  # Felbontási limit
    print("- Processzor pixelek:   min = 256*28*28, max = 768*28*28", flush=True)  # Pixelhatárok
    print("- Max új tokenek:       1. szerepkör <= 128, 2. szerepkör <= 256", flush=True)  # Tokenlimitek
    print("- Képenkénti időkorlát: 60 másodperc szigorú limit (timeout)", flush=True)  # Időkorlát
    print("- Memóriakezelés:       torch.cuda.empty_cache() képenként", flush=True)  # Memóriaürítés
    print("- Következtetési mód:   torch.inference_mode() bekapcsolva", flush=True)  # Inference mód
    print("=" * 70 + "\n", flush=True)                                  # Záró vonal


def generate_recommendations(                                           # Diagnosztikai javaslatokat készítő függvény
    failing_reason: str,                                                # Hibajelenség vagy hibaüzenet
    elapsed_time: float,                                                # Eltelt futásidő másodpercben
    vram_peak_mb: float                                                 # Csúcs VRAM fogyasztás megabájtban
) -> List[str]:
    """Generate diagnostic recommendations based on failure symptoms."""
    recs = []                                                           # Javaslatok gyűjtőlistája
    total_vram = (                                                      # Teljes VRAM lekérése vagy 8000 MB alapérték
        torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        if torch.cuda.is_available() else 8000
    )

    if elapsed_time > 15.0:                                             # Ha az egyedi futás lassabb mint 15 másodperc
        recs.append("Csökkentse a max_new_tokens-t, vagy aktiválja a 4-bites módot (--load_in_4bit).")
    if vram_peak_mb > (0.90 * total_vram):                              # Ha a VRAM elérte a kapacitás 90%-át
        recs.append("A VRAM elérte a 90%-ot. Kapcsolja be a 4-bites kvantálást, vagy csökkentse a képméretet.")
    if "timeout" in failing_reason.lower():                             # Időtúllépés gyanúja esetén
        recs.append("A kép túllépte a 60s korlátot. Ellenőrizze a max_pixels és átméretezési limiteket.")
    if "hang" in failing_reason.lower() or "loop" in failing_reason.lower():  # Beragadó generálás gyanúja
        recs.append("Ellenőrizze az inference_mode() bekapcsolását és a modell lezáró tokenjeit.")
    if not torch.cuda.is_available() and elapsed_time > 10.0:           # CPU lassúság esetén
        recs.append("A CPU túl lassú. Használjon API kulcsot és állítsa be a --backend api kapcsolót.")

    if not recs:                                                        # Ha nem aktiválódott speciális szabály
        recs.append("Vizsgálja meg a hibaláncot (traceback) és a csomagfüggőségek épségét.")

    return recs                                                         # Javaslatlista visszaadása


def execute_image_timed(                                                # Egyetlen kép mérése és végrehajtása
    backend: Any,                                                       # VLM modell backend
    img_info: Dict[str, Any],                                           # Kép COCO metaadatai
    images_dir: str,                                                    # Képek könyvtára
    timeout_sec: int = 60                                               # Képenkénti időtúllépési limit
) -> Dict[str, Any]:
    """Run single image through both Role 1 and Role 2 with timestamped logging and peak VRAM tracking."""
    img_id = str(img_info['id'])                                        # Kép azonosítója
    img_rel_path = img_info['file_name']                                # Relatív elérési út
    img_full_path = os.path.join(images_dir, img_rel_path)              # Teljes képútvonal

    if not os.path.exists(img_full_path):                               # Képfájl létezésének vizsgálata
        raise FileNotFoundError(f"A képfájl nem található: {img_full_path}")  # Hiba dobása

    now_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")     # Kezdési időpont formázása
    print(                                                              # Indítási üzenet kiírása
        f"[{now_start}] [INDÍTÁS] Kép: {img_rel_path} (ID: {img_id}, "
        f"Megvilágítás: {img_info.get('light_condition', 'N/A')})",
        flush=True
    )

    reset_vram_peak()                                                   # VRAM számláló nullázása a tiszta méréshez
    t_start = time.perf_counter()                                       # Időmérés indítása

    resized_img, orig_w, orig_h = resize_image_safe(                    # Biztonsági átméretezés 768px-re
        img_full_path, max_dim=768
    )

    with ThreadPoolExecutor(max_workers=1) as executor:                 # Végrehajtó szál 60s védelemmel
        future = executor.submit(                                       # Feladat elküldése a szálnak
            run_vlm_on_image,                                           # Végrehajtandó függvény
            backend=backend,                                            # VLM modell példány
            img_full_path=img_full_path,                                # Kép útvonala
            orig_w=orig_w, orig_h=orig_h,                               # Eredeti méretek
            resized_img=resized_img,                                    # Átméretezett kép
            detector_summary_text=(                                     # Minta detektor bemenetek
                "- YOLO11: person, car\n"
                "- RT-DETR: person, car, bus\n"
                "- Faster R-CNN: person"
            ),
            role1_tokens=128, role2_tokens=256                          # Token költségkeretek
        )
        try:                                                            # Eredmény lekérdezése határidővel
            detections, role1_raw, role2_critique = future.result(      # Várakozás a szál befejezésére
                timeout=timeout_sec                                     # Időtúllépési limit
            )
        except TimeoutError:                                            # Ha túllépi a 60 másodpercet
            elapsed_fail = time.perf_counter() - t_start                # Eltelt idő kiszámítása
            now_err = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Hiba időpontja
            print(                                                      # Időtúllépés kiírása
                f"[{now_err}] [IDŐTÚLLÉPÉS] Kép: {img_rel_path} túllépte "
                f"a(z) {timeout_sec}s határt! Eltelt: {elapsed_fail:.2f}s",
                flush=True
            )
            raise TimeoutError(                                         # Kivétel dobása a szakasz leállításához
                f"A képenkénti {timeout_sec}s időkorlát túllépve: {img_rel_path}"
            )

    elapsed = time.perf_counter() - t_start                             # Mért futási idő másodpercben
    vram_peak = get_vram_peak_mb()                                      # Csúcs VRAM érték lekérdezése
    now_end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")       # Befejezési időpont
    print(                                                              # Befejezési státusz kiírása
        f"[{now_end}] [BEFEJEZVE] Kép: {img_rel_path} | Futásidő: {elapsed:.2f}s "
        f"| VRAM csúcs: {vram_peak:.1f} MB | Talált objektumok: {len(detections)}",
        flush=True
    )

    reset_vram_peak()                                                   # Gyorsítótár ürítése a következő kép előtt

    return {                                                            # Képfutási adatok strukturált szótára
        "image_id": img_id,                                             # Képazonosító
        "file_name": img_rel_path,                                      # Fájlnév
        "light_condition": img_info.get(                                # Megvilágítási kategória
            'light_condition', 'Ismeretlen'
        ),
        "original_resolution": f"{orig_w}x{orig_h}",                    # Eredeti felbontás szövegként
        "processed_resolution": (                                       # Feldolgozott felbontás
            f"{resized_img.size[0]}x{resized_img.size[1]}"
        ),
        "elapsed_seconds": round(elapsed, 2),                           # Futásidő kerekítve
        "vram_peak_mb": round(vram_peak, 1),                            # VRAM csúcs kerekítve
        "objects_detected": len(detections),                            # Találatok darabszáma
        "detections": detections,                                       # Detekciók listája
        "role1_sample": role1_raw[:120],                                # 1. szerepkör minta szöveg
        "role2_sample": role2_critique[:120]                            # 2. szerepkör minta szöveg
    }


def run_stage(                                                          # Egyetlen tesztszakasz végrehajtó függvény
    stage_num: int,                                                     # Szakasz sorszáma (1, 2 vagy 3)
    stage_name: str,                                                    # Szakasz megnevezése
    target_count: int,                                                  # Feldolgozandó képek száma
    time_budget_sec: float,                                             # Megengedett időkeret másodpercben
    all_images: List[Dict[str, Any]],                                   # Képek teljes listája
    backend: Any,                                                       # VLM modell backend
    images_dir: str,                                                    # Képek könyvtára
    results_dir: str,                                                   # Eredmények könyvtára
    model_config: Dict[str, Any]                                        # Modell konfigurációs szótár
) -> Tuple[bool, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Execute a smoke test stage. Returns (passed, stage_results, failure_info)."""
    print(                                                              # Szakasz belépési üzenet kiírása
        f"\n>>> BELÉPÉS A(Z) {stage_num}. SZAKASZBA: {stage_name} "
        f"({target_count} kép, Időkeret: < {time_budget_sec}s)",
        flush=True
    )
    stage_images = all_images[:target_count]                            # Szakaszhoz tartozó képek levágása
    stage_results = []                                                  # Szakaszeredmények gyűjtője
    t_stage_start = time.perf_counter()                                 # Szakasz időmérés indítása

    for idx, img_info in enumerate(stage_images, 1):                    # Képek bejárása sorrendben
        try:                                                            # Képfuttatás védett blokkban
            res = execute_image_timed(backend, img_info, images_dir)     # Időzített végrehajtás
            stage_results.append(res)                                   # Eredmény mentése
        except Exception as e:                                          # Hiba elkapása
            elapsed_tot = time.perf_counter() - t_stage_start           # Eddig eltelt idő
            vram_peak = get_vram_peak_mb()                              # Aktuális VRAM csúcs
            failure_info = {                                            # Hibajelentés összeállítása
                "stage": stage_num,                                     # Szakasz száma
                "stage_name": stage_name,                               # Szakasz neve
                "failing_image": img_info['file_name'],                 # Hibás kép neve
                "failing_image_id": img_info['id'],                     # Hibás kép azonosítója
                "error": str(e),                                        # Hibaüzenet
                "traceback": traceback.format_exc(),                    # Részletes hibalánc
                "elapsed_time_seconds": round(elapsed_tot, 2),          # Eltelt idő
                "time_budget_seconds": time_budget_sec,                 # Megengedett idő
                "vram_peak_mb": round(vram_peak, 1),                    # Csúcs VRAM
                "model_config": model_config,                           # Konfiguráció
                "recommendations": generate_recommendations(            # Javítási javaslatok
                    str(e), elapsed_tot, vram_peak
                )
            }
            return False, stage_results, failure_info                   # Sikertelen visszatérés

    stage_duration = time.perf_counter() - t_stage_start                # Szakasz teljes ideje
    times = [r['elapsed_seconds'] for r in stage_results]               # Futási idők tömbje
    mean_time = float(np.mean(times)) if times else 0.0                 # Átlagos futásidő
    med_time = float(np.median(times)) if times else 0.0                # Medián futásidő
    max_time = float(np.max(times)) if times else 0.0                   # Maximális futásidő

    print(f"\n--- {stage_num}. Szakasz Összegzés ---", flush=True)      # Összegző fejléc
    print(f"Feldolgozott képek:     {len(stage_results)}/{target_count}", flush=True)  # Képszám
    print(f"Szakasz teljes ideje:  {stage_duration:.2f}s (Időkeret: {time_budget_sec}s)", flush=True)
    print(f"Átlagos késleltetés:   {mean_time:.2f}s | Medián: {med_time:.2f}s | Max: {max_time:.2f}s", flush=True)

    if stage_duration >= time_budget_sec:                               # Időkeret túllépésének szigorú vizsgálata
        print(                                                          # Hibaüzenet kiírása
            f"[SIKERTELEN] A(z) {stage_num}. szakasz TÚLLÉPTE az időkeretet! "
            f"({stage_duration:.2f}s >= {time_budget_sec}s)",
            flush=True
        )
        slowest_item = max(stage_results, key=lambda x: x['elapsed_seconds'])  # Leglassabb kép megkeresése
        failure_info = {                                                # Időtúllépési hibajelentés összeállítása
            "stage": stage_num,                                         # Szakasz száma
            "stage_name": stage_name,                                   # Szakasz neve
            "failing_image": slowest_item['file_name'],                 # Leglassabb képfájl
            "failing_image_id": slowest_item['image_id'],               # Képazonosító
            "error": (                                                  # Hiba leírása
                f"A(z) {stage_num}. szakasz időkerete túllépve: "
                f"{stage_duration:.2f}s >= {time_budget_sec}s"
            ),
            "elapsed_time_seconds": round(stage_duration, 2),           # Mért idő
            "time_budget_seconds": time_budget_sec,                     # Megengedett idő
            "vram_peak_mb": round(                                      # Csúcs VRAM
                max(r['vram_peak_mb'] for r in stage_results), 1
            ),
            "model_config": model_config,                               # Modell adatok
            "recommendations": generate_recommendations(                # Javaslatok
                "timeout", stage_duration,
                max(r['vram_peak_mb'] for r in stage_results)
            )
        }
        return False, stage_results, failure_info                       # Sikertelen visszatérés

    print(f"[SIKERES] A(z) {stage_num}. szakasz sikeresen teljesítve!\n", flush=True)  # Siker visszaigazolása
    return True, stage_results, None                                    # Sikeres szakasz lezárása


def run_smoketest_cli(cli_args: Optional[List[str]] = None) -> int:    # Parancssori belépési pont
    parser = argparse.ArgumentParser(                                   # Argumentumkezelő
        description="VLM Többlépcsős Diagnosztikai Füstteszt (Smoke Test)."
    )
    parser.add_argument(                                                # COCO teszt annotációk útvonala
        "--coco_json", type=str,
        default="data/ExDark/coco_annotations_test.json",
        help="COCO teszt annotációk."
    )
    parser.add_argument(                                                # Képek könyvtára
        "--images_dir", type=str,
        default="data/ExDark/images",
        help="Képek könyvtára."
    )
    parser.add_argument(                                                # Backend típus
        "--backend", type=str, default="local",
        choices=["local", "api", "openai", "gemini", "heuristic"],
        help="VLM backend."
    )
    parser.add_argument(                                                # Modell azonosító
        "--model", type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="VLM modell ID."
    )
    parser.add_argument(                                                # Szolgáltató
        "--provider", type=str, default="openai",
        choices=["openai", "gemini"],
        help="API szolgáltató."
    )
    parser.add_argument(                                                # Számítási eszköz
        "--device", type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Számítási eszköz."
    )
    parser.add_argument(                                                # 4-bites kvantálás
        "--load_in_4bit", action="store_true",
        help="4-bites NF4 kvantálás engedélyezése."
    )
    parser.add_argument(                                                # Kimeneti könyvtár
        "--results_dir", type=str, default="results",
        help="Eredmények mentési könyvtára."
    )
    parser.add_argument(                                                # Opcionális nagy szakasz
        "--include_large_stage", action="store_true",
        help="4000 képes nagyméretű szakasz hozzáadása."
    )
    parser.add_argument(                                                # Kompatibilitási kapcsoló
        "--smoketest", action="store_true",
        help="Kompatibilitási kapcsoló."
    )

    args = parser.parse_args(cli_args)                                  # Argumentumok feldolgozása
    print_startup_diagnostics(args)                                     # Rendszerdiagnosztika megjelenítése

    model_config = {                                                    # Modell beállítások gyűjtője
        "backend": args.backend,                                        # Backend típus
        "model": args.model,                                            # Modell azonosító
        "device": args.device,                                          # Hardvereszköz
        "load_in_4bit": args.load_in_4bit,                              # 4-bites kvantálás
        "provider": args.provider                                       # Szolgáltató
    }

    if not os.path.exists(args.coco_json):                              # Annotációk létezésének ellenőrzése
        print(f"[Hiba] COCO fájl nem található: {args.coco_json}.", flush=True)  # Hibaüzenet
        return 1                                                        # Hibakóddal kilépés

    with open(args.coco_json, 'r', encoding='utf-8') as f:             # Annotációk megnyitása
        coco_data = json.load(f)                                        # JSON adatok beolvasása

    all_images = coco_data.get('images', [])                            # Képlista kinyerése
    if not all_images:                                                  # Üres képlista ellenőrzése
        print("[Hiba] Nincsenek képek a COCO fájlban.", flush=True)     # Hibaüzenet
        return 1                                                        # Kilépés hibakóddal

    all_stage_records = []                                              # Összes szakaszeredmény gyűjtője
    failure_file = os.path.join(args.results_dir, "smoketest_failure.json")  # Hiba JSON fájl
    report_file = os.path.join(args.results_dir, "smoketest_report.json")    # Jelentés JSON fájl
    os.makedirs(args.results_dir, exist_ok=True)                        # Eredmény könyvtár biztosítása

    log_msg(f"VLM backend betöltése ({args.backend})...")               # Betöltés naplózása
    t0_load = time.perf_counter()                                       # Betöltés időmérése
    try:                                                                # Modell betöltése védett blokkban
        backend = load_vlm(                                             # Modellbetöltő hívása
            backend_type=args.backend, model_id=args.model,
            device=args.device, load_in_4bit=args.load_in_4bit,
            provider=args.provider,
            strict=True                                                 # Füsttesztnél szigorú hibaellenőrzés
        )
    except Exception as e:                                              # Betöltési hiba elkapása
        load_fail_time = time.perf_counter() - t0_load                  # Eltelt idő
        vram_peak = get_vram_peak_mb()                                  # Csúcs VRAM
        print("\n" + "!" * 70, flush=True)                              # Figyelmeztető fejléc
        print("                 A MODELL INICIALIZÁLÁSA MEGHIÚSULT (< 30s)", flush=True)
        print("!" * 70, flush=True)
        print(f"Hibaüzenet:       {e}", flush=True)                     # Hiba kiírása
        print(f"Eltelt idő:       {load_fail_time:.2f}s", flush=True)   # Futásidő kiírása
        print(f"Csúcs VRAM:       {vram_peak:.1f} MB", flush=True)      # VRAM kiírása

        recs = generate_recommendations(str(e), load_fail_time, vram_peak)  # Javítási javaslatok generálása
        print("\n[JAVÍTÁSI JAVASLATOK]", flush=True)                    # Javaslatok fejléce
        for rec in recs:                                                # Javaslatok felsorolása
            print(f" * {rec}", flush=True)                              # Javaslat kiírása
        print("!" * 70, flush=True)

        failure_payload = {                                             # Betöltési hibariport összeállítása
            "stage": 0,                                                 # 0. szakasz (inicializálás)
            "stage_name": "Modell Inicializálás",                       # Megnevezés
            "failing_image": "N/A", "failing_image_id": "N/A",          # Kép nem értelmezett
            "error": str(e),                                            # Hibaüzenet
            "traceback": traceback.format_exc(),                        # Hibalánc
            "elapsed_time_seconds": round(load_fail_time, 2),           # Mért idő
            "time_budget_seconds": 30.0,                                # 30 másodperces korlát
            "vram_peak_mb": round(vram_peak, 1),                        # Csúcs VRAM
            "model_config": model_config,                               # Modell adatok
            "recommendations": recs                                     # Javaslatok
        }
        with open(failure_file, 'w', encoding='utf-8') as f:            # Hibafájl megnyitása írásra
            json.dump(failure_payload, f, indent=2)                     # Mentés JSON formátumban
        print(f"Részletes hibariport elmentve ide: {failure_file}", flush=True)  # Mentés igazolása
        return 1                                                        # Kilépés hibakóddal

    load_time = time.perf_counter() - t0_load                           # Sikeres betöltési idő
    log_msg(f"A VLM backend sikeresen betöltve {load_time:.2f}s alatt.")  # Tájékoztatás

    stages = [                                                          # 3 kapuőr szakasz definíciója
        (1, "Egyképes Ellenőrzés (Single-Image)", 1, 15.0),             # 1. szakasz: 1 kép < 15s
        (2, "Kiscsomagos Robusztusság (5 Kép)", 5, 120.0),              # 2. szakasz: 5 kép < 120s
        (3, "Középcsomagos Skálateszt (20 Kép)", 20, 300.0)             # 3. szakasz: 20 kép < 300s
    ]

    if args.include_large_stage:                                        # Opcionális nagyméretű szakasz
        print("\n" + "#" * 70, flush=True)                              # Figyelmeztető sáv
        print(" [FIGYELMEZTETÉS] EZ MÁR NEM FÜSTTESZT (SMOKE TEST)!", flush=True)
        print(" 4000 képes nagyméretű szakasz hozzáadva (Időkeret: 40000s).", flush=True)
        print("#" * 70 + "\n", flush=True)
        stages.append((4, "Nagyméretű Kiértékelés (4000 Kép)", 4000, 40000.0))  # 4. szakasz hozzáadása

    for s_num, s_name, count, budget in stages:                         # Szakaszok végigjárása
        passed, stage_res, failure = run_stage(                         # Szakasz futtatása
            stage_num=s_num, stage_name=s_name, target_count=count,
            time_budget_sec=budget, all_images=all_images,
            backend=backend, images_dir=args.images_dir,
            results_dir=args.results_dir, model_config=model_config
        )

        all_stage_records.extend(stage_res)                             # Eredmények hozzáfűzése

        if not passed:                                                  # Ha a szakasz elbukott
            print("\n" + "!" * 70, flush=True)                          # Elválasztó
            print("                 FÜSTTESZT ELBUKOTT - A VÉGREHAJTÁS LEÁLLT", flush=True)
            print("!" * 70, flush=True)
            print(f"Hiba helye:       {s_num}. Szakasz: {s_name}", flush=True)  # Szakasz neve
            print(f"Érintett kép:     {failure['failing_image']}", flush=True)  # Kép neve
            print(f"Mért futásidő:    {failure['elapsed_time_seconds']}s (Megengedett: {budget}s)", flush=True)
            print(f"Csúcs VRAM:       {failure['vram_peak_mb']} MB", flush=True)  # VRAM
            print(f"Hibaüzenet:       {failure['error']}", flush=True)  # Hiba szövege
            print("\n[JAVÍTÁSI JAVASLATOK]", flush=True)
            for rec in failure['recommendations']:                      # Javaslatok felsorolása
                print(f" * {rec}", flush=True)
            print("!" * 70, flush=True)

            with open(failure_file, 'w', encoding='utf-8') as f:        # Hibafájl írása
                json.dump(failure, f, indent=2)                         # Strukturált kiírás
            print(f"Részletes hibariport elmentve ide: {failure_file}", flush=True)  # Mentés
            print("MEGTAGADVA: A teljes futtatás nem lehetséges a hiba javításáig.", flush=True)
            return 1                                                    # Kilépés hibakóddal

    print("\n" + "=" * 70, flush=True)                                  # Siker sáv
    print("                 FÜSTTESZT SIKERESEN TELJESÍTVE (MINDKÉT SZAKASZ ÁTMENT)", flush=True)
    print("=" * 70, flush=True)

    stage3_records = all_stage_records[-20:]                            # 3. szakasz 20 képének kinyerése
    times = [r['elapsed_seconds'] for r in stage3_records]              # Futási idők listája
    vrams = [r['vram_peak_mb'] for r in stage3_records]                 # VRAM értékek listája
    mean_sec = float(np.mean(times))                                    # Átlagos másodperc
    total_sec = sum(times)                                              # Összes idő

    print("\n[Minta Inferenciás Eredménytábla (3. Szakasz)]", flush=True)  # Táblázat fejléce
    header = f"{'Kép ID':<8} {'Fájlnév':<28} {'Megvilágítás':<14} {'Futásidő':<10} {'Csúcs VRAM':<12} {'Objektumok':<10}"
    print(header, flush=True)                                           # Oszlopfejlécek kiírása
    print("-" * len(header), flush=True)                                # Elválasztó vonal
    for r in stage3_records[:8]:                                        # Első 8 sor kiírása
        fname = os.path.basename(r['file_name'])                        # Fájlnév kinyerése
        print(f"{r['image_id']:<8} {fname:<28} {r['light_condition']:<14} {r['elapsed_seconds']:<5.2f}s     {r['vram_peak_mb']:<6.1f} MB   {r['objects_detected']:<10}", flush=True)
    if len(stage3_records) > 8:                                         # További képek jelzése
        print(f"... és további {len(stage3_records) - 8} kép sikeresen kiértékelve.", flush=True)

    full_dataset_images = len(all_images)                               # Összes kép száma
    est_full_seconds = mean_sec * full_dataset_images                   # Becsült teljes idő másodpercben
    est_full_minutes = est_full_seconds / 60.0                          # Becsült teljes idő percben
    est_full_hours = est_full_minutes / 60.0                            # Becsült teljes idő órában

    print("\n[Benchmark Átviteli Sebesség és Teljes Adatkészlet Előrejelzés]", flush=True)
    print(f"- Átlagos idő képenként:            {mean_sec:.2f} másodperc", flush=True)
    print(f"- Csúcs memóriafogyasztás:          {max(vrams):.1f} MB", flush=True)
    print(f"- ExDark tesztkészlet mérete:       {full_dataset_images} kép", flush=True)
    print(f"- Becsült teljes futásidő:          {est_full_minutes:.1f} perc ({est_full_hours:.2f} óra)", flush=True)
    print(f"- Csak 1. szerepkör (becsült):      {est_full_minutes * 0.45:.1f} perc", flush=True)
    print(f"- 1. + 2. szerepkör együtt (becs.): {est_full_minutes:.1f} perc", flush=True)

    print("\n[Végleges Ajánlás a Teljes Futtatáshoz]", flush=True)      # Ajánlások fejléce
    if mean_sec <= 6.0:                                                 # 6 másodperc alatt kiváló
        print(">>> STÁTUSZ: FOLYTATHATÓ. A VLM csővezeték stabil, gyors és biztonságos határok között mozog.", flush=True)
    elif mean_sec <= 12.0:                                              # 6-12 másodperc között optimalizálás javasolt
        print(">>> STÁTUSZ: TOVÁBBI OPTIMALIZÁLÁS JAVASOLT. Érdemes bekapcsolni a 4-bites módot.", flush=True)
    else:                                                               # 12 másodperc felett API javasolt
        print(">>> STÁTUSZ: VÁLTÁS API-RA JAVASOLT. A helyi inferencia ideje meghaladja a kívánt keretet.", flush=True)

    report_payload = {                                                  # Záró jelentés adatcsomagja
        "status": "PASSED",                                             # Sikeres státusz
        "timestamp": datetime.datetime.now().isoformat(),               # Időbélyeg
        "model_config": model_config,                                   # Modell konfiguráció
        "stages_passed": [1, 2, 3],                                     # Sikeresen teljesített szakaszok
        "total_images_tested": len(stage3_records),                     # Tesztelt képek száma
        "mean_latency_seconds": round(mean_sec, 2),                     # Átlagos latencia
        "max_vram_peak_mb": round(max(vrams), 1),                       # Legmagasabb VRAM érték
        "full_dataset_size": full_dataset_images,                       # Teljes méret
        "projected_full_runtime_minutes": round(est_full_minutes, 2),  # Becsült teljes idő
        "stage_records": stage3_records                                 # Részletes rekordok
    }

    with open(report_file, 'w', encoding='utf-8') as f:                 # Záró jelentés mentése
        json.dump(report_payload, f, indent=2)                          # JSON kiírás
    print(f"\nVégleges Füstteszt Jelentés elmentve ide: {report_file}", flush=True)  # Mentés igazolása
    print("=" * 70, flush=True)                                         # Záró vonal

    return 0                                                            # Sikeres 0-s hibakód visszaadása


def main():                                                             # Belépési pont
    sys.exit(run_smoketest_cli(sys.argv[1:]))                           # CLI futtatása és kilépés


if __name__ == "__main__":
    main()
