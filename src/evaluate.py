"""Evaluation module for YOLO11, RT-DETR, and Faster R-CNN on ExDark benchmark.

Calculates:
- mAP@0.5 and mAP@0.5:0.95 (pycocotools COCOeval)
- Precision, Recall, F1 per category at IoU=0.5
- Lighting condition breakdown across the 10 illumination regimes
- Inference speed (GPU & CPU FPS / latency)
Generates high-resolution figures in results/figures/.
"""

import os
import sys
import json
import argparse
import io
import contextlib
import time
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.append(os.path.dirname(os.path.abspath(__file__)))              # Forráskönyvtár hozzáadása az importálási úthoz
from utils import (
    TARGET_CLASSES,                                                     # 4 balesetveszélyes célosztály listája
    TARGET_CLASS_TO_ID,                                                 # Osztálynév -> COCO azonosító leképezés
    TARGET_ID_TO_CLASS,                                                 # COCO azonosító -> osztálynév leképezés
    LIGHT_CONDITIONS,                                                   # 10 ExDark megvilágítási kategória
    calculate_iou                                                       # Két határolókeret metszet-unió aránya
)

# Publikációs minőségű stílusbeállítások
plt.style.use(                                                          # Diagramstílus kiválasztása
    'seaborn-v0_8-whitegrid'                                             # Fehér rácsos letisztult téma
    if 'seaborn-v0_8-whitegrid' in plt.style.available                  # Ellenőrzés elérhető-e a seaborn stílus
    else 'default'                                                      # Tartalék alapértelmezett matplotlib stílus
)
plt.rcParams['font.sans-serif'] = 'Arial'                               # Publikációs betűtípus beállítása
plt.rcParams['axes.edgecolor'] = '#cccccc'                              # Tengelyszegélyek halványszürke színe
plt.rcParams['axes.linewidth'] = 0.8                                    # Tengelyvonalak vastagsága pontban


def safe_savefig(
    fig: plt.Figure,                                                    # Mentendő matplotlib ábra objektum
    output_path: str,                                                   # Kimeneti képfájl elérési útja
    dpi: int = 300,                                                     # Felbontás (300 DPI nyomdai minőség)
    **kwargs                                                            # További opcionális mentési paraméterek
):
    """Safely save figure on Windows handling transient file lock issues."""
    output_path = os.path.abspath(output_path)                          # Abszolút elérési út normalizálása
    os.makedirs(os.path.dirname(output_path), exist_ok=True)            # Célkönyvtár automatikus létrehozása
    for attempt in range(6):                                            # 6 próbálkozás Windows fájlzárolási hiba esetén
        try:                                                            # Képmentés védett blokkban
            fig.savefig(output_path, dpi=dpi, **kwargs)                 # Ábra lemezre mentése megadott felbontással
            return                                                      # Sikeres mentés után azonnali visszatérés
        except OSError:                                                 # Átmeneti zárolási hiba elkapása
            time.sleep(0.4)                                             # 400 ms várakozás a zárolás feloldására
    fig.savefig(output_path, dpi=dpi, **kwargs)                         # Végső mentési kísérlet


def convert_predictions_to_coco_results(
    pred_dict: Dict[str, List[Dict[str, Any]]],                         # Modell predikciós szótára képenként
    coco_gt: COCO                                                       # COCO formátumú valós adatobjektum
) -> List[Dict[str, Any]]:
    """Convert unified predictions to COCOeval result format [image_id, category_id, bbox, score]."""
    coco_results = []                                                   # COCOeval formátumú lista inicializálása
    for str_img_id, preds in pred_dict.items():                         # Iteráció az összes kép találatain
        img_id = int(str_img_id)                                        # Képazonosító egésszé alakítása
        for p in preds:                                                 # Detekciók egyesével történő bejárása
            label = p['label']                                          # Prediktált osztálynév kinyerése
            if label not in TARGET_CLASS_TO_ID:                         # Nem célosztály kiszűrése
                continue                                                # Lépés a következő detekcióra
            cat_id = TARGET_CLASS_TO_ID[label]                          # Számszerű COCO kategória azonosító
            x1, y1, x2, y2 = p['bbox']                                  # Szélső koordináták kibontása
            w = max(0.0, x2 - x1)                                       # Szélesség számítása COCO formátumhoz
            h = max(0.0, y2 - y1)                                       # Magasság számítása COCO formátumhoz
            score = float(p.get('score', 1.0))                          # Detekciós konfidencia float típusra
            coco_results.append({                                       # Eredmény rekord összeállítása
                "image_id": img_id,                                     # Kép azonosítója
                "category_id": cat_id,                                  # Kategória azonosítója
                "bbox": [round(x1, 2), round(y1, 2),                    # Kerekített [x, y, w, h] formátumú doboz
                         round(w, 2), round(h, 2)],                     # Szélesség és magasság kerekítve
                "score": round(score, 4)                                # Konfidencia 4 tizedesre kerekítve
            })
    return coco_results                                                 # Konvertált eredmények visszaadása


def evaluate_coco_map(
    coco_gt: COCO,                                                      # Ground truth COCO API objektum
    coco_results: List[Dict[str, Any]]                                  # Modell detekciók COCO formátumban
) -> Tuple[float, float, Dict[str, float]]:
    """Evaluate mAP@0.5 and mAP@0.5:0.95 using pycocotools."""
    if not coco_results:                                                # Ha nincsenek predikciók a listában
        return 0.0, 0.0, {c: 0.0 for c in TARGET_CLASSES}              # Nullákkal térünk vissza a hiba elkerülésére

    coco_dt = coco_gt.loadRes(coco_results)                             # Predikciók betöltése a COCO struktúrába
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')                      # COCOeval kiértékelő motor inicializálása
    coco_eval.params.catIds = list(TARGET_CLASS_TO_ID.values())         # Értékelés szűkítése a 4 célosztályra
    coco_eval.evaluate()                                                # Találatok és valós adatok összevetése
    coco_eval.accumulate()                                              # Statisztikák összegzése és felhalmozása
    coco_eval.summarize()                                               # Szabványos COCO összefoglaló számítása

    map_50_95 = float(coco_eval.stats[0])                              # mAP@[0.50:0.95] metrika kinyerése
    map_50 = float(coco_eval.stats[1])                                 # mAP@0.50 metrika kinyerése

    per_class_map50 = {}                                                # Kategóriánkénti mAP50 szótár
    precisions = coco_eval.eval['precision']                            # Teljes precíziós tenzor kinyerése
    for idx, (cat_name, cat_id) in enumerate(                           # Kategóriák index szerinti bejárása
        TARGET_CLASS_TO_ID.items()                                      # Célosztály név és azonosító párok
    ):
        pr = precisions[0, :, idx, 0, 2]                                # Precíziós értékek IoU=0.50 mellett
        valid_pr = pr[pr > -1]                                          # Érvénytelen -1 kitöltő értékek szűrése
        per_class_map50[cat_name] = (                                   # Kategória mAP50 értékének kiszámítása
            round(float(np.mean(valid_pr)), 4)                          # Érvényes precíziók átlagolása
            if len(valid_pr) > 0 else 0.0                               # Nullázás ha nem volt találat
        )

    return round(map_50_95, 4), round(map_50, 4), per_class_map50      # Kerekített metrikák visszaadása


def evaluate_prf1_per_category(
    gt_annotations: List[Dict[str, Any]],                               # Valós annotációk listája
    predictions: Dict[str, List[Dict[str, Any]]],                       # Predikciók szótára képenként
    iou_thresh: float = 0.5                                             # Pozitív döntési IoU küszöbérték
) -> pd.DataFrame:
    """Calculate Precision, Recall, and F1 score per category at IoU threshold."""
    gt_by_img_cat = {}                                                  # Valós dobozok kép és kategória szerint
    for ann in gt_annotations:                                          # Annotációk egyenkénti feldolgozása
        img_id = str(ann['image_id'])                                   # Képazonosító szöveggé alakítása
        cat = TARGET_ID_TO_CLASS[ann['category_id']]                    # Kategórianév visszakeresése azonosítóból
        x, y, w, h = ann['bbox']                                        # Befoglaló keret koordináták
        box = [x, y, x + w, y + h]                                      # Átváltás [x1, y1, x2, y2] formátumra
        gt_by_img_cat.setdefault((img_id, cat), []).append(             # Doboz elhelyezése a szótárban
            {"bbox": box, "matched": False}                             # Párosítatlannak jelölve kezdetben
        )

    stats = {                                                           # Statisztikai gyűjtő inicializálása
        c: {"tp": 0, "fp": 0, "total_gt": 0}                            # TP, FP és összes valós számlálók
        for c in TARGET_CLASSES                                         # Célosztályonként
    }
    for (img_id, cat), boxes in gt_by_img_cat.items():                  # Valós dobozok összegzése osztályonként
        stats[cat]["total_gt"] += len(boxes)                            # Összes valós példányszám növelése

    for str_img_id, preds in predictions.items():                       # Képenkénti predikciók bejárása
        sorted_preds = sorted(                                          # Rendezés csökkenő konfidencia szerint
            preds,                                                      # Predikciók listája
            key=lambda x: x.get('score', 1.0),                          # Rendezési kulcs a konfidencia
            reverse=True                                                # Legmagasabb elöl a mohó párosításhoz
        )
        for p in sorted_preds:                                          # Predikciók vizsgálata sorrendben
            cat = p['label']                                            # Prediktált osztálynév
            if cat not in TARGET_CLASSES:                               # Ha nem tartozik a 4 célosztályhoz
                continue                                                # Nem releváns osztály átugrása
            pred_box = p['bbox']                                        # Prediktált doboz koordináták
            gt_candidates = gt_by_img_cat.get((str_img_id, cat), [])   # Adott képhez tartozó valós jelöltek

            best_iou = 0.0                                              # Legnagyobb megtalált IoU kezdetben 0
            best_idx = -1                                               # Legjobb GT doboz indexe
            for idx, g in enumerate(gt_candidates):                     # Jelölt valós dobozok átvizsgálása
                if not g["matched"]:                                    # Csak a még szabad dobozokat vizsgáljuk
                    iou = calculate_iou(pred_box, g["bbox"])            # Metszet-unió arány kiszámítása
                    if iou > best_iou:                                  # Ha nagyobb átfedést találtunk
                        best_iou = iou                                  # Legjobb IoU felülírása
                        best_idx = idx                                  # Legjobb index frissítése

            if best_iou >= iou_thresh and best_idx >= 0:                # Sikeres átfedés az IoU küszöb felett
                stats[cat]["tp"] += 1                                   # Valódi pozitív (TP) számláló növelése
                gt_candidates[best_idx]["matched"] = True               # GT doboz zárolása további párosítás elől
            else:
                stats[cat]["fp"] += 1                                   # Téves riasztás (FP) számláló növelése

    records = []                                                        # Metrika táblázat sorainak gyűjtője
    for cat in TARGET_CLASSES:                                          # Célosztályok egyenkénti kiértékelése
        tp = stats[cat]["tp"]                                           # Valódi pozitív találatok száma
        fp = stats[cat]["fp"]                                           # Téves detekciók száma
        total_gt = stats[cat]["total_gt"]                               # Összes valós objektum száma
        fn = total_gt - tp                                              # Detektálatlan objektumok száma (FN)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0                 # Precízió = TP / (TP + FP)
        rec = tp / total_gt if total_gt > 0 else 0.0                    # Visszahívás = TP / (TP + FN)
        f1 = (                                                          # Harmonikus közép (F1-score)
            (2 * prec * rec) / (prec + rec)                             # Képlet számlálója
            if (prec + rec) > 0 else 0.0                                # Védelem nullával osztás ellen
        )

        records.append({                                                # Kategória összefoglaló sora
            "category": cat,                                            # Osztály neve
            "ground_truth_count": total_gt,                             # Valós darabszám
            "true_positives": tp,                                       # Valódi pozitívok száma
            "false_positives": fp,                                      # Téves pozitívok száma
            "false_negatives": fn,                                      # Kihagyott negatívok száma
            "precision": round(prec, 4),                                # Precízió 4 tizedesre
            "recall": round(rec, 4),                                    # Visszahívás 4 tizedesre
            "f1_score": round(f1, 4)                                    # F1-pontszám 4 tizedesre
        })

    return pd.DataFrame(records)                                        # Eredmények pandas DataFrame formátumban


def evaluate_lighting_breakdown(
    coco_data: Dict[str, Any],                                          # Teljes COCO adatstruktúra szótárként
    coco_gt: COCO,                                                      # COCO API segédobjektum
    model_predictions: Dict[str, Dict[str, List[Dict[str, Any]]]]       # Modellek predikciói modellenként
) -> pd.DataFrame:
    """Evaluate performance across the 10 illumination regimes using pycocotools."""
    records = []                                                        # Megvilágítási statisztikai sorok
    light_names = list(LIGHT_CONDITIONS.values())                       # 10 megvilágítási állapot neve

    for light in light_names:                                           # Fényviszonyok bejárása ciklusban
        img_ids = [                                                     # Adott fényviszony képeinek szűrése
            img['id'] for img in coco_data['images']                     # Képek metaadatainak bejárása
            if img.get('light_condition') == light                      # Megegyező fényviszony attribútum
        ]
        if not img_ids:                                                 # Ha egy kép sincs az adott kategóriában
            continue                                                    # Következő megvilágításra ugrás

        gt_subset_annotations = [                                       # Képekhez tartozó valós annotációk
            ann for ann in coco_data['annotations']                     # Annotációk szűrése
            if ann['image_id'] in img_ids                               # Képazonosító alapján
        ]
        num_gt = len(gt_subset_annotations)                             # Valós objektumok száma ebben a fényben

        for model_name, preds in model_predictions.items():             # Modellek értékelése a részhalmazon
            sub_preds = {                                               # Csak az érintett képek detekciói
                str(k): preds.get(str(k), [])                           # Predikciók kinyerése képazonosítóval
                for k in img_ids                                        # Szűrt képazonosítók
            }
            coco_results = convert_predictions_to_coco_results(         # Eredmények COCO formátumba alakítása
                sub_preds, coco_gt                                      # Részhalmaz predikciók és GT objektum
            )

            if coco_results:                                            # Ha vannak érvényes predikciók
                try:                                                    # Kiértékelés védett blokkban
                    with contextlib.redirect_stdout(io.StringIO()):     # Felesleges konzol kimenetek elnyomása
                        dt_obj = coco_gt.loadRes(coco_results)          # Predikciók betöltése a GT környezetbe
                        ev = COCOeval(coco_gt, dt_obj, 'bbox')          # Részhalmazos kiértékelő inicializálása
                        ev.params.imgIds = img_ids                      # Csak az aktuális fényviszony képei
                        ev.params.catIds = list(                        # Csak a célosztályok azonosítói
                            TARGET_CLASS_TO_ID.values()
                        )
                        ev.evaluate()                                   # Értékelés futtatása
                        ev.accumulate()                                 # Statisztikák összesítése
                        ev.summarize()                                  # Összefoglaló készítése
                        map50 = round(float(ev.stats[1]), 4)            # mAP@0.50 kinyerése a statisztikából
                        recall50 = round(float(ev.stats[8]), 4)         # Recall@0.50 kinyerése a statisztikából
                except Exception:                                       # Kiértékelési kivétel kezelése
                    map50 = 0.0                                         # Hiba esetén biztonsági 0 mAP érték
                    recall50 = 0.0                                      # Hiba esetén biztonsági 0 visszahívás
            else:
                map50 = 0.0                                             # Üres predikció esetén 0 mAP
                recall50 = 0.0                                          # Üres predikció esetén 0 visszahívás

            records.append({                                            # Rekord összeállítása a táblázathoz
                "light_condition": light,                               # Fényviszony megnevezése
                "image_count": len(img_ids),                            # Képek száma a kategóriában
                "gt_object_count": num_gt,                              # Valós célpontok száma
                "model": model_name,                                    # Kiértékelt modell neve
                "mAP_50": map50,                                        # Elért mAP@0.50 pontosság
                "recall_50": recall50                                   # Elért Recall@0.50 visszahívás
            })

    return pd.DataFrame(records)                                        # Eredménytábla DataFrame formátumban


def plot_map_comparison(
    overall_df: pd.DataFrame,                                           # Összesített modelleredményeket tartalmazó tábla
    per_class_dict: Dict[str, Dict[str, float]],                        # Kategóriánkénti mAP50 értékek modellenként
    output_path: str                                                    # Kimeneti diagram mentési útvonala
):
    """Bar chart comparing mAP@0.5 and mAP@0.5:0.95 overall and per category."""
    fig, axes = plt.subplots(                                           # Két paneles ábra inicializálása
        1, 2, figsize=(14, 5.5), dpi=300                                # 1 sor, 2 oszlop, 300 DPI felbontás
    )

    models = overall_df['model'].tolist()                               # Modellek neveinek listája
    x = np.arange(len(models))                                          # X tengely oszlop pozíciók tömbje
    width = 0.35                                                        # Oszlopok szélessége

    ax = axes[0]                                                        # Első panel kijelölése (összesített mAP)
    b1 = ax.bar(                                                        # mAP@0.50 oszlopok kirajzolása
        x - width/2, overall_df['mAP_50'], width,                       # Balra tolt pozíció és adatok
        label='mAP@0.50', color='#1f77b4',                              # Címke és kék színkód
        edgecolor='black', linewidth=0.7                                # Fekete körvonal a publikációs minőséghez
    )
    b2 = ax.bar(                                                        # mAP@[0.50:0.95] oszlopok kirajzolása
        x + width/2, overall_df['mAP_50_95'], width,                    # Jobbra tolt pozíció és adatok
        label='mAP@0.50:0.95', color='#aec7e8',                         # Világoskék színárnyalat
        edgecolor='black', linewidth=0.7                                # Fekete körvonal
    )

    for bar in b1:                                                      # mAP@0.50 oszlopok feliratozása
        yval = bar.get_height()                                         # Oszlopmagasság lekérdezése
        ax.text(                                                        # Szöveges érték kiírása az oszlop fölé
            bar.get_x() + bar.get_width()/2, yval + 0.01,               # Középre zárt vízszintes és függőleges pozíció
            f"{yval:.3f}", ha='center', va='bottom',                    # 3 tizedesjegy formázás
            fontsize=9, fontweight='bold'                               # Félkövér betűméret
        )
    for bar in b2:                                                      # mAP@[0.50:0.95] oszlopok feliratozása
        yval = bar.get_height()                                         # Oszlopmagasság lekérdezése
        ax.text(                                                        # Szöveges érték kiírása
            bar.get_x() + bar.get_width()/2, yval + 0.01,               # Középre igazított pozíció
            f"{yval:.3f}", ha='center', va='bottom',                    # Formázott érték
            fontsize=9                                                  # Betűméret
        )

    ax.set_ylabel('Mean Average Precision (mAP)',                       # Y tengely felirata
                  fontsize=11, fontweight='bold')
    ax.set_title('Overall Benchmark mAP (ExDark Test Split)',           # Diagram címe
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)                                                    # X tengely osztásközeinek beállítása
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')          # Modellek nevei feliratként
    ax.set_ylim(0, max(overall_df['mAP_50'].max() * 1.25, 0.5))        # Y skála dinamikus felső határa
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)          # Jelmagyarázat fehér háttérrel

    ax2 = axes[1]                                                       # Második panel kijelölése (osztályonként)
    classes = TARGET_CLASSES                                            # 4 célosztály listája
    x2 = np.arange(len(classes))                                        # Kategóriák oszlop pozíciói
    bar_w = 0.25                                                        # Kategória oszlopszélesség
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']                          # Színpaletta a 3 detektorhoz

    for idx, m in enumerate(models):                                    # Modellek bejárása osztályonkénti rajzoláshoz
        m_vals = [per_class_dict.get(m, {}).get(c, 0.0)                 # Adott modell kategóriás mAP értékei
                  for c in classes]
        offset = (idx - 1) * bar_w                                      # Eltolás számítása a csoportosításhoz
        rects = ax2.bar(                                                # Oszlopcsoport kirajzolása modellenként
            x2 + offset, m_vals, bar_w, label=m,                        # Eltolt pozíció és adatok
            color=colors[idx % len(colors)],                            # Modellhez rendelt szín
            edgecolor='black', linewidth=0.7                            # Vékony fekete keret
        )
        for r in rects:                                                 # Értékek ráírása az oszlopokra
            val = r.get_height()                                        # Oszlop magassága
            if val > 0.02:                                              # Csak 2% felett írjuk ki a zsúfoltság elkerülésére
                ax2.text(                                               # Felirat elhelyezése
                    r.get_x() + r.get_width()/2, val + 0.01,            # Középre zárt pozíció
                    f"{val:.2f}", ha='center', va='bottom',             # 2 tizedesjegy formázás
                    fontsize=8                                          # Kis betűméret
                )

    ax2.set_ylabel('mAP@0.50 per Category',                             # Második panel Y tengelycímke
                   fontsize=11, fontweight='bold')
    ax2.set_title('Per-Category mAP@0.50 Comparison',                   # Második panel diagramcímke
                  fontsize=12, fontweight='bold')
    ax2.set_xticks(x2)                                                  # Osztások rögzítése a kategóriáknál
    ax2.set_xticklabels([c.capitalize() for c in classes],              # Kategórianevek nagy kezdőbetűvel
                        fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 0.85)                                               # Rögzített felső skálahatár
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9)          # Jelmagyarázat doboza

    plt.tight_layout()                                                  # Automatikus margóigazítás
    safe_savefig(fig, output_path, dpi=300)                             # Kép lemezre mentése biztonságos függvénnyel
    plt.close()                                                         # Ábra memória felszabadítása
    print(f"[Ábra] mAP összehasonlító diagram elmentve: {output_path}") # Tájékoztató kiírás


def plot_category_prf1(
    prf1_dict: Dict[str, pd.DataFrame],                                 # Osztályonkénti PRF1 táblázatok szótára modellenként
    output_path: str                                                    # Kimeneti kép mentési útvonala
):
    """Grouped chart comparing Precision, Recall, and F1 across categories."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)            # 3 paneles ábra Precízió, Recall és F1-hez
    models = list(prf1_dict.keys())                                     # Modellek nevei
    classes = TARGET_CLASSES                                            # 4 célosztály
    x = np.arange(len(classes))                                         # X pozíciók
    bar_w = 0.25                                                        # Oszlopszélesség
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c']                         # Színpaletta modellenként

    metrics = [                                                         # Ábrázolandó metrikák oszlopai és címei
        ('precision', 'Precision@0.50'),                                # Precízió oszlop
        ('recall', 'Recall@0.50'),                                      # Visszahívás oszlop
        ('f1_score', 'F1-Score@0.50')                                   # F1-pontszám oszlop
    ]

    for i, (col, title) in enumerate(metrics):                          # 3 panel bejárása
        ax = axes[i]                                                    # Aktuális aldiagram kiválasztása
        for idx, m in enumerate(models):                                # Modellek bejárása
            df = prf1_dict[m]                                           # Modellhez tartozó DataFrame lekérése
            vals = [                                                    # Kategóriaértékek összegyűjtése sorrendben
                float(df[df['category'] == c][col].values[0])           # Adott kategória és metrika értéke
                if len(df[df['category'] == c]) > 0 else 0.0            # Védelem hiányzó kategória esetén
                for c in classes                                        # 4 osztályon végighaladva
            ]
            offset = (idx - 1) * bar_w                                  # Vízszintes eltolás a csoportosításhoz
            bars = ax.bar(                                              # Oszlopok felrajzolása
                x + offset, vals, bar_w, label=m,                       # Pozíció, magasság és modellcímke
                color=palette[idx % len(palette)],                      # Szín kiválasztása
                edgecolor='black', linewidth=0.7                        # Körvonal formázása
            )
            for b in bars:                                              # Oszlopok számszerű feliratozása
                h = b.get_height()                                      # Magasság lekérdezése
                if h > 0.03:                                            # Csak 3% felett jelenítjük meg a számot
                    ax.text(                                            # Szöveg elhelyezése
                        b.get_x() + b.get_width()/2, h + 0.01,          # Oszlop közepe felett
                        f"{h:.2f}", ha='center', va='bottom',           # Formázás 2 tizedesre
                        fontsize=7.5                                    # Kisméretű betű
                    )

        ax.set_title(title, fontsize=12, fontweight='bold')             # Panel címének beállítása
        ax.set_xticks(x)                                                # X osztások rögzítése
        ax.set_xticklabels([c.capitalize() for c in classes],          # Kategóriacímkék nagy kezdőbetűvel
                           fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1.05)                                            # Y skála 0-tól 105%-ig
        if i == 0:                                                      # Csak az első panelre írunk Y tengelyfeliratot
            ax.set_ylabel('Metric Score', fontsize=11,                  # Metrika pontszám felirat
                          fontweight='bold')
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)      # Jelmagyarázat elhelyezése

    plt.tight_layout()                                                  # Elrendezés optimalizálása
    safe_savefig(fig, output_path, dpi=300)                             # Ábra biztonságos mentése
    plt.close()                                                         # Memória felszabadítása
    print(f"[Ábra] PRF1 diagram elmentve: {output_path}")               # Sikeres mentés kiírása


def plot_lighting_breakdown(
    lighting_df: pd.DataFrame,                                          # Fényviszonyok szerinti eredménytábla
    output_path: str                                                    # Kimeneti diagram mentési útvonala
):
    """Plot detection robustness across illumination conditions."""
    valid_lights = lighting_df[                                         # Statisztikailag releváns fények szűrése
        lighting_df['image_count'] >= 5                                 # Legalább 5 képpel rendelkező csoportok
    ]['light_condition'].unique()                                       # Egyedi fényviszony nevek
    sub_df = lighting_df[                                               # Adattábla szűrése a reprezentatív fényekre
        lighting_df['light_condition'].isin(valid_lights)
    ]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)                    # Széles ábra a fényviszonyokhoz
    sns.barplot(                                                        # Csoportosított oszlopdiagram Seabornnal
        data=sub_df,                                                    # Szűrt adatforrás
        x='light_condition',                                            # X tengely a megvilágítási kategóriák
        y='mAP_50',                                                     # Y tengely a detektálási pontosság (mAP50)
        hue='model',                                                    # Csoportosítás modellek szerint
        palette=['#1f77b4', '#ff7f0e', '#2ca02c'],                      # Egységes színkódok
        edgecolor='black',                                              # Fekete szegély az oszlopok körül
        linewidth=0.7,                                                  # Szegélyvonal vastagság
        ax=ax                                                           # Cél diagramtengely
    )

    ax.set_title(                                                       # Főcím beállítása
        'Object Detection Robustness Across ExDark Lighting Regimes',
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel('Illumination Condition', fontsize=11,                # X tengelycímke
                  fontweight='bold')
    ax.set_ylabel('mAP@0.50', fontsize=11, fontweight='bold')           # Y tengelycímke
    ax.set_ylim(0, max(sub_df['mAP_50'].max() * 1.25, 0.6))             # Dinamikus felső határ
    ax.legend(title='Model', frameon=True, facecolor='white',           # Modell jelmagyarázat doboz
              framealpha=0.9)

    for p in ax.patches:                                                # Oszlopok egyenkénti feliratozása
        height = p.get_height()                                         # Oszlop magassága
        if height > 0.03:                                               # Csak értelmes magasságú oszlopokra írunk
            ax.text(                                                    # Számkiírás az oszlop felett
                p.get_x() + p.get_width() / 2., height + 0.01,          # Középpont meghatározása
                f"{height:.2f}", ha='center', va='bottom',              # 2 tizedesjegy pontosság
                fontsize=7.5                                            # Betűméret
            )

    plt.xticks(rotation=25, ha='right', fontsize=10,                    # X tengely feliratok elforgatása 25 fokkal
               fontweight='bold')
    plt.tight_layout()                                                  # Elrendezés igazítása
    safe_savefig(fig, output_path, dpi=300)                             # Kép lemezre írása
    plt.close()                                                         # Ábra bezárása
    print(f"[Ábra] Megvilágítási diagram elmentve: {output_path}")      # Megerősítés


def plot_speed_vs_accuracy(
    overall_df: pd.DataFrame,                                           # Összesített pontossági és sebességadatok
    output_path: str                                                    # Mentési kép elérési útvonala
):
    """Plot Speed (FPS) vs Accuracy (mAP@0.5) Pareto curve."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)                     # Két dimenziós pontdiagram létrehozása

    colors = {                                                          # Színek hozzárendelése az egyes modellekhez
        'YOLO11': '#1f77b4',                                            # Kék a YOLO11-hez
        'RT-DETR': '#ff7f0e',                                           # Narancs az RT-DETR-hez
        'Faster R-CNN': '#2ca02c'                                       # Zöld a Faster R-CNN-hez
    }
    markers = {'YOLO11': 'o', 'RT-DETR': 's', 'Faster R-CNN': '^'}      # Különböző formájú jelölők modellekhez

    for _, row in overall_df.iterrows():                                # Adattábla sorainak bejárása modellenként
        m = row['model']                                                # Modell neve
        gpu_fps = row['gpu_fps']                                        # GPU képkockasebesség (FPS)
        gpu_map = row['mAP_50']                                        # mAP@0.50 pontosság
        ax.scatter(                                                     # GPU adatpont kirajzolása nagy méretben
            gpu_fps, gpu_map, s=220, color=colors[m],                   # Méret 220 pont, modell színe
            marker=markers[m], label=f"{m} (GPU)",                      # Egyedi jelölő és felirat
            edgecolor='black', linewidth=1.5, zorder=5                  # Fekete kontúr és legfelső réteg
        )
        ax.annotate(                                                    # Szöveges megjegyzés GPU ponthoz
            f"{m}\n({gpu_fps:.1f} FPS, {gpu_map:.3f} mAP)",             # Név, FPS és mAP kiírása
            (gpu_fps, gpu_map), textcoords="offset points",             # Kapcsolódási pont és relatív eltolás
            xytext=(0, 12), ha='center', fontsize=9,                    # 12 ponttal fentebb, középre igazítva
            fontweight='bold'
        )

        cpu_fps = row['cpu_fps']                                        # CPU képkockasebesség
        ax.scatter(                                                     # CPU referencia pont rajzolása kereszt jelölővel
            cpu_fps, gpu_map, s=120, color=colors[m],                   # Méret 120, modell színe
            marker='x', label=f"{m} (CPU)", linewidth=2.5, zorder=5    # 'x' jelölőforma vastag vonallal
        )
        ax.annotate(                                                    # Szöveges felirat CPU ponthoz
            f"{m} CPU ({cpu_fps:.1f} FPS)", (cpu_fps, gpu_map),         # Modellnév és CPU FPS
            textcoords="offset points", xytext=(0, -18),                # 18 ponttal a jelölő alá
            ha='center', fontsize=8, color='#555555'                    # Szürke kisméretű szöveg
        )

    ax.set_title(                                                       # Diagram főcíme
        'Inference Speed (FPS) vs Detection Performance (mAP@0.50)',
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel(                                                      # X tengely felirata logaritmikus skálával
        'Throughput (Frames Per Second, FPS - Log Scale)',
        fontsize=11, fontweight='bold'
    )
    ax.set_ylabel('Detection Accuracy (mAP@0.50)',                      # Y tengely pontosság felirata
                  fontsize=11, fontweight='bold')
    ax.set_xscale('log')                                                # Logaritmikus skála az X tengelyen
    ax.set_ylim(0, max(overall_df['mAP_50'].max() * 1.3, 0.6))          # Megfelelő margó a pontok felett
    ax.grid(True, which='both', linestyle='--', alpha=0.5)              # Finom kettős rácsozat a log skálához

    handles, labels = ax.get_legend_handles_labels()                    # Jelmagyarázat elemek összegyűjtése
    by_label = dict(zip(labels, handles))                               # Duplikációk kiszűrése szótárral
    ax.legend(                                                          # Jelmagyarázat dobozának kirajzolása
        by_label.values(), by_label.keys(),                             # Egyedi feliratok és jelölők
        frameon=True, facecolor='white', framealpha=0.9,                # Fehér áttetsző keret
        loc='lower right'                                               # Jobb alsó sarokba igazítva
    )

    plt.tight_layout()                                                  # Elrendezés igazítása
    safe_savefig(fig, output_path, dpi=300)                             # Diagram lemezre írása
    plt.close()                                                         # Ábra bezárása
    print(f"[Ábra] Sebesség-pontosság diagram elmentve: {output_path}") # Megerősítés


def evaluate_confusion_matrices(
    coco_data: Dict[str, Any],                                          # Ground truth COCO annotációk
    model_predictions: Dict[str, Dict[str, List[Dict[str, Any]]]],      # Modellek predikciói
    output_fig_path: str,                                               # Kimeneti konfúziós ábra útvonala
    output_json_path: str,                                              # Kimeneti mátrix adatok JSON útvonala
    iou_thresh: float = 0.5,                                            # Átfedési küszöbérték (IoU)
    conf_thresh: float = 0.25                                           # Konfidencia szűrési küszöb
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Compute object-detection confusion matrix for each detector against ExDark ground truth."""
    classes = TARGET_CLASSES                                            # 4 balesetveszélyes célosztály
    cols = classes + ['Background']                                     # Háttér kategória hozzáadása a hibákhoz

    gt_by_img = {}                                                      # Valós dobozok csoportosítása kép szerint
    for ann in coco_data['annotations']:                                # Összes annotáció végigjárása
        cat = TARGET_ID_TO_CLASS.get(ann['category_id'])                # Kategória név kikeresése
        if cat in classes:                                              # Ha célosztály
            x, y, w, h = ann['bbox']                                    # Koordináták
            gt_by_img.setdefault(str(ann['image_id']), []).append({     # Doboz rögzítése a kép azonosítójához
                'bbox': [x, y, x + w, y + h],                           # Koordináták átváltása sarokpontokra
                'label': cat                                            # Célosztály neve
            })

    all_matrices = {}                                                   # Minden modell mátrixát gyűjtő szótár

    fig, axes = plt.subplots(                                           # Több paneles ábra generálása
        1, len(model_predictions),                                      # Modellek száma határozza meg a szélességet
        figsize=(6.5 * len(model_predictions), 5.5), dpi=300            # Dinamikus ábraszélesség
    )
    if len(model_predictions) == 1:                                     # Egyetlen modell esetén listába csomagolás
        axes = [axes]                                                   # Egységes iterálhatóság biztosítása

    for ax_idx, (m_name, preds_dict) in enumerate(                      # Modellek végigjárása
        model_predictions.items()
    ):
        mat = {r: {c: 0 for c in cols} for r in cols}                   # Üres NxN mátrix inicializálása nullákkal

        for img_id, gts in gt_by_img.items():                           # Képek valós dobozainak bejárása
            preds = [                                                   # Konfidencia küszöböt elérő detekciók
                p for p in preds_dict.get(img_id, [])                   # Képhez tartozó predikciók
                if p.get('score', 1.0) >= conf_thresh                   # Szűrés a küszöbértékre
            ]
            used_preds = set()                                          # Már felhasznált predikciók indexei

            for gt in gts:                                              # Valós objektumok párosítása predikciókkal
                best_iou = 0.0                                          # Legnagyobb átfedés kezdetben nulla
                best_p_idx = -1                                         # Legjobb találat indexe
                for p_idx, p in enumerate(preds):                       # Detekciók végigpróbálása
                    if p_idx in used_preds:                             # Ha már párosítva lett másik valós dobozzal
                        continue                                        # Átugrás
                    iou = calculate_iou(gt['bbox'], p['bbox'])          # IoU kiszámítása
                    if iou > best_iou:                                  # Ha jobb az átfedés
                        best_iou = iou                                  # Maximális IoU frissítése
                        best_p_idx = p_idx                              # Index elmentése

                if best_iou >= iou_thresh:                              # Ha eléri a detektálási küszöböt
                    pred_label = preds[best_p_idx]['label']             # Prediktált osztály címkéje
                    if pred_label in cols:                              # Ha ismert kategória
                        mat[gt['label']][pred_label] += 1               # Valós vs Prediktált cella növelése
                        used_preds.add(best_p_idx)                      # Predikció megjelölése felhasználtként
                else:
                    mat[gt['label']]['Background'] += 1                 # Nem talált objektum: kihagyva a háttérbe

            for p_idx, p in enumerate(preds):                           # Fel nem használt detekciók vizsgálata
                if p_idx not in used_preds:                             # Ha nem tartozik valós objektumhoz
                    pred_label = p['label']                             # Hallucinált osztály
                    if pred_label in cols:                              # Ismert kategória ellenőrzése
                        mat['Background'][pred_label] += 1              # Háttérből tévesen jósolt riasztás (FP)

        all_matrices[m_name] = mat                                      # Modell konfúziós mátrixának tárolása

        df_mat = pd.DataFrame(mat).T                                    # Transzponálás a [Valós x Prediktált] nézethez
        ax = axes[ax_idx]                                               # Aktuális panel kiválasztása
        sns.heatmap(                                                    # Hőtérkép rajzolása Seabornnal
            df_mat, annot=True, fmt='d', cmap='Blues',                  # Számszerű értékek és kék színskála
            cbar=False, linewidths=0.5, linecolor='#eeeeee',            # Színsáv elrejtése és szürke elválasztók
            ax=ax                                                       # Cél tengely
        )
        ax.set_title(f'{m_name} Confusion Matrix',                      # Panel címe a modell nevével
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Class', fontsize=10,                   # X tengely a jósolt kategória
                      fontweight='bold')
        ax.set_ylabel('Ground Truth Class', fontsize=10,                # Y tengely a valós kategória
                      fontweight='bold')

    plt.tight_layout()                                                  # Automatikus térközigazítás
    safe_savefig(fig, output_fig_path, dpi=300)                         # Ábra lemezre mentése
    plt.close()                                                         # Ábra memóriafoglalásának lezárása
    print(f"[Ábra] Konfúziós mátrix ábra elmentve: {output_fig_path}")  # Megerősítés kiírása

    with open(output_json_path, 'w', encoding='utf-8') as f:             # Kimeneti JSON fájl megnyitása írásra
        json.dump(all_matrices, f, indent=2)                            # Mátrix adatok strukturált mentése
    print(f"[Metrika] Konfúziós mátrix adatok mentve: {output_json_path}")  # Tájékoztatás

    return all_matrices                                                 # Számított mátrixok visszaadása


def evaluate_worst_failure_cases(
    coco_data: Dict[str, Any],                                          # Teljes annotációs adatbázis
    model_predictions: Dict[str, Dict[str, List[Dict[str, Any]]]],      # Modellek predikciói
    output_json_path: str,                                              # Kimeneti hibalista JSON
    output_csv_path: str,                                               # Kimeneti hibalista CSV
    top_k: int = 10,                                                    # Legrosszabb képek száma modellenként
    iou_thresh: float = 0.5,                                            # Döntési IoU küszöb
    conf_thresh: float = 0.25                                           # Konfidencia küszöbérték
) -> Dict[str, List[Dict[str, Any]]]:
    """Identify and extract the 10 worst performing failure cases for each detector."""
    classes = TARGET_CLASSES                                            # 4 célosztály
    gt_by_img = {}                                                      # Valós annotációk gyűjtője
    img_meta_map = {                                                    # Kép metaadatok gyorskeresője azonosító alapján
        str(im['id']): im for im in coco_data['images']
    }

    for ann in coco_data['annotations']:                                # Annotációk szűrése és csoportosítása
        cat = TARGET_ID_TO_CLASS.get(ann['category_id'])                # Kategórianév feloldása
        if cat in classes:                                              # Csak a célosztályok megtartása
            x, y, w, h = ann['bbox']                                    # Keret koordináták
            gt_by_img.setdefault(str(ann['image_id']), []).append({     # Képhez fűzés
                'bbox': [x, y, x + w, y + h],                           # Befoglaló koordináták
                'label': cat                                            # Címke
            })

    failure_report = {}                                                 # Legrosszabb képek szótára modellenként
    all_worst_rows = []                                                 # Összesített hibasorok CSV exporthoz

    for m_name, preds_dict in model_predictions.items():                 # Hibaelemzés detektoronként
        img_records = []                                                # Képenkénti hibaösszesítő lista
        for img_id, gts in gt_by_img.items():                           # Valós objektumot tartalmazó képek bejárása
            preds = [                                                   # Konfidencia szűrt detekciók
                p for p in preds_dict.get(img_id, [])                   # Adott kép detekciói
                if p.get('score', 1.0) >= conf_thresh                   # Küszöb felett
            ]
            used_preds = set()                                          # Használt detekciók halmaza
            tp = 0                                                      # Valódi pozitívok számlálója

            for gt in gts:                                              # Valós dobozok átvizsgálása
                best_iou = 0.0                                          # Legnagyobb IoU
                best_p_idx = -1                                         # Legjobb találat indexe
                for p_idx, p in enumerate(preds):                       # Detekciók végigpróbálása
                    if p_idx in used_preds:                             # Ha már párosítva van
                        continue                                        # Átugrás
                    iou = calculate_iou(gt['bbox'], p['bbox'])          # IoU számítása
                    if iou > best_iou:                                  # Jobb átfedés vizsgálata
                        best_iou = iou                                  # Frissítés
                        best_p_idx = p_idx                              # Frissítés
                if best_iou >= iou_thresh and (                         # Átfedés és azonos kategória vizsgálata
                    preds[best_p_idx]['label'] == gt['label']
                ):
                    tp += 1                                             # Találat elkönyvelése
                    used_preds.add(best_p_idx)                          # Zárolás

            fp = len(preds) - len(used_preds)                           # Téves riasztások száma (FP)
            fn = len(gts) - tp                                          # Elmulasztott objektumok száma (FN)
            f1 = (                                                      # Kép-szintű F1 pontszám
                (2 * tp) / (2 * tp + fp + fn)                           # F1 számláló és nevező
                if (2 * tp + fp + fn) > 0 else 0.0                      # Nullával való osztás megelőzése
            )

            meta = img_meta_map.get(img_id, {})                         # Kép metaadatok lekérése
            img_records.append({                                        # Képrekord létrehozása
                "model": m_name,                                        # Modell neve
                "image_id": img_id,                                     # Kép azonosítója
                "file_name": meta.get('file_name', 'N/A'),              # Fájlnév és relatív útvonal
                "light_condition": meta.get(                            # Megvilágítási kategória
                    'light_condition', 'Unknown'
                ),
                "gt_count": len(gts),                                   # Valós objektumok száma a képen
                "true_positives": tp,                                   # Helyesen detektált objektumok
                "false_positives": fp,                                  # Tévesen jelzett objektumok
                "false_negatives": fn,                                  # Észrevétlen objektumok
                "f1_score": round(f1, 4),                               # Kép pontossági F1-értéke
                "total_errors": fp + fn                                 # Hibák összesített összege (FP + FN)
            })

        worst_list = sorted(                                            # Rendezés a legrosszabb képekre
            img_records,                                                # Képek listája
            key=lambda x: (x['f1_score'], -x['total_errors'],           # Legkisebb F1, majd legtöbb hiba elöl
                           -x['gt_count'])
        )[:top_k]                                                       # Első top_k legrosszabb kiválasztása
        failure_report[m_name] = worst_list                             # Eltárolás a modell szótárában
        all_worst_rows.extend(worst_list)                               # Hozzáadás az összesített sorokhoz

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)       # Célkönyvtár létrehozása ha hiányzik
    with open(output_json_path, 'w', encoding='utf-8') as f:             # JSON fájl megnyitása írásra
        json.dump(failure_report, f, indent=2)                          # Hibalista kiírása JSON-be
    pd.DataFrame(all_worst_rows).to_csv(output_csv_path, index=False)   # Hibalista mentése CSV táblázatként
    print(f"[Metrika] Hibaképek elmentve: {output_json_path} és {output_csv_path}")  # Tájékoztatás

    return failure_report                                               # Eredmények visszaadása


def main():
    parser = argparse.ArgumentParser(                                   # Parancssori kapcsolók inicializálása
        description="Detektorok kiértékelése az ExDark COCO benchmarkon."
    )
    parser.add_argument(                                                # COCO teszt annotációs fájl
        "--coco_json", type=str,
        default="data/ExDark/coco_annotations_test.json",
        help="COCO teszt annotációs fájl útvonala."
    )
    parser.add_argument(                                                # YOLO11 predikciók útvonala
        "--yolo_preds", type=str,
        default="results/metrics/yolo11_predictions.json",
        help="YOLO11 predikciós JSON fájl."
    )
    parser.add_argument(                                                # YOLO11 időzítési adatok
        "--yolo_timing", type=str,
        default="results/metrics/yolo11_timing.json",
        help="YOLO11 időmérési JSON fájl."
    )
    parser.add_argument(                                                # RT-DETR predikciók útvonala
        "--rtdetr_preds", type=str,
        default="results/metrics/rtdetr_predictions.json",
        help="RT-DETR predikciós JSON fájl."
    )
    parser.add_argument(                                                # RT-DETR időzítési adatok
        "--rtdetr_timing", type=str,
        default="results/metrics/rtdetr_timing.json",
        help="RT-DETR időmérési JSON fájl."
    )
    parser.add_argument(                                                # Faster R-CNN predikciók útvonala
        "--fasterrcnn_preds", type=str,
        default="results/metrics/faster_rcnn_predictions.json",
        help="Faster R-CNN predikciós JSON fájl."
    )
    parser.add_argument(                                                # Faster R-CNN időzítési adatok
        "--fasterrcnn_timing", type=str,
        default="results/metrics/faster_rcnn_timing.json",
        help="Faster R-CNN időmérési JSON fájl."
    )
    parser.add_argument(                                                # Diagramok kimeneti könyvtára
        "--figures_dir", type=str,
        default="results/figures",
        help="Diagramok kimeneti könyvtára."
    )
    parser.add_argument(                                                # Metrikák kimeneti könyvtára
        "--metrics_dir", type=str,
        default="results/metrics",
        help="Metrikák kimeneti könyvtára."
    )
    args = parser.parse_args()                                          # Argumentumok feldolgozása

    with open(args.coco_json, 'r', encoding='utf-8') as f:             # COCO annotációk beolvasása JSON-ből
        coco_data = json.load(f)                                        # JSON adatok parse-olása szótárba

    coco_gt = COCO(args.coco_json)                                      # COCO API inicializálása a hivatkozáshoz

    models_config = {                                                   # Detektorok konfigurációs térképe
        "YOLO11": {"preds": args.yolo_preds, "timing": args.yolo_timing},
        "RT-DETR": {"preds": args.rtdetr_preds, "timing": args.rtdetr_timing},
        "Faster R-CNN": {"preds": args.fasterrcnn_preds, "timing": args.fasterrcnn_timing}
    }

    loaded_predictions = {}                                             # Betöltött predikciók szótára
    loaded_timing = {}                                                  # Betöltött időmérések szótára

    for name, cfg in models_config.items():                             # Modellek bejárása fájlellenőrzéssel
        if os.path.exists(cfg["preds"]):                                # Ha a predikciós fájl létezik a lemezen
            loaded_predictions[name] = json.load(                       # Predikciók beolvasása a memóriába
                open(cfg["preds"], 'r', encoding='utf-8')
            )
        if os.path.exists(cfg["timing"]):                               # Ha az időzítési fájl létezik
            loaded_timing[name] = json.load(                            # Időzítési adatok beolvasása
                open(cfg["timing"], 'r', encoding='utf-8')
            )

    overall_metrics_list = []                                           # Összesített benchmark eredménylista
    per_class_maps = {}                                                 # Osztályonkénti mAP tároló
    prf1_dict = {}                                                      # PRF1 táblázatok tárolója modellenként

    for name, preds in loaded_predictions.items():                     # Kiértékelés modellenként
        print(f"\n================ {name} Kiértékelése ================")  # Fejléc kiírása konzolra
        timing = loaded_timing.get(name, {})                            # Időzítési adatok kinyerése
        coco_res = convert_predictions_to_coco_results(preds, coco_gt)  # Predikciók átalakítása COCO formátumra

        map_50_95, map_50, class_maps = evaluate_coco_map(              # Hivatalos COCO mAP metrikák számítása
            coco_gt, coco_res
        )
        per_class_maps[name] = class_maps                               # Kategória mAP értékek elmentése

        prf1_df = evaluate_prf1_per_category(                           # Precízió, visszahívás és F1 számítása
            coco_data['annotations'], preds, iou_thresh=0.5
        )
        prf1_df['model'] = name                                         # Modell oszlop hozzáadása a táblázathoz
        prf1_dict[name] = prf1_df                                       # Tábla tárolása az ábrázoláshoz

        gpu_latency = timing.get('avg_latency_ms', 0.0)                 # GPU késleltetés lekérése ms-ban
        gpu_fps = timing.get('fps', 0.0)                                # GPU képkockasebesség lekérése
        cpu_bench = timing.get('cpu_benchmark', {})                     # CPU referencia mérés szótára
        cpu_latency = cpu_bench.get('avg_latency_ms', 0.0)              # CPU késleltetés
        cpu_fps = cpu_bench.get('fps', 0.0)                             # CPU képkockasebesség

        overall_metrics_list.append({                                   # Összesített rekord összeállítása
            "model": name,                                              # Modell megnevezése
            "mAP_50": map_50,                                           # mAP@0.50 pontosság
            "mAP_50_95": map_50_95,                                     # mAP@[0.50:0.95] pontosság
            "mean_precision_50": round(                                 # Átlagos precízió kerekítése
                float(prf1_df['precision'].mean()), 4
            ),
            "mean_recall_50": round(                                    # Átlagos visszahívás kerekítése
                float(prf1_df['recall'].mean()), 4
            ),
            "mean_f1_50": round(                                        # Átlagos F1-érték kerekítése
                float(prf1_df['f1_score'].mean()), 4
            ),
            "gpu_latency_ms": gpu_latency,                              # GPU késleltetés (ms)
            "gpu_fps": gpu_fps,                                         # GPU átviteli sebesség (FPS)
            "cpu_latency_ms": cpu_latency,                              # CPU késleltetés (ms)
            "cpu_fps": cpu_fps                                          # CPU átviteli sebesség (FPS)
        })

    overall_df = pd.DataFrame(overall_metrics_list)                     # DataFrame létrehozása az eredményekből
    print("\n--- Átfogó Benchmark Összegzés ---")                       # Összegző fejléc kiírása
    print(overall_df.to_string(index=False))                            # Táblázat szöveges megjelenítése

    os.makedirs(args.metrics_dir, exist_ok=True)                        # Metrika könyvtár létrehozása ha szükséges
    overall_df.to_csv(                                                  # Összesített adatok mentése CSV fájlba
        os.path.join(args.metrics_dir, "overall_metrics.csv"),
        index=False
    )
    with open(                                                          # Összesített adatok mentése JSON fájlba
        os.path.join(args.metrics_dir, "overall_metrics.json"),
        'w', encoding='utf-8'
    ) as f:
        json.dump(overall_metrics_list, f, indent=2)                    # Strukturált mentés

    all_prf1_df = pd.concat(prf1_dict.values(), ignore_index=True)      # Osztályonkénti táblák egyesítése egy táblává
    all_prf1_df.to_csv(                                                 # Egyesített tábla mentése CSV-ként
        os.path.join(args.metrics_dir, "per_category_metrics.csv"),
        index=False
    )

    print("\n--- Megvilágítási Csoportok Részletes Értékelése ---")     # Fényviszony vizsgálat indítása
    lighting_df = evaluate_lighting_breakdown(                          # Megvilágítási robusztusság számítása
        coco_data, coco_gt, loaded_predictions
    )
    lighting_df.to_csv(                                                 # Megvilágítási adatok mentése CSV-be
        os.path.join(args.metrics_dir, "lighting_breakdown.csv"),
        index=False
    )

    os.makedirs(args.figures_dir, exist_ok=True)                        # Ábra könyvtár biztosítása
    plot_map_comparison(                                                # mAP összehasonlító diagram generálása
        overall_df, per_class_maps,
        os.path.join(args.figures_dir, "map_comparison.png")
    )
    plot_category_prf1(                                                 # PRF1 összehasonlító ábra készítése
        prf1_dict,
        os.path.join(args.figures_dir, "category_metrics_comparison.png")
    )
    plot_lighting_breakdown(                                            # Fényviszony robusztussági ábra rajzolása
        lighting_df,
        os.path.join(args.figures_dir, "lighting_condition_breakdown.png")
    )
    plot_speed_vs_accuracy(                                             # Sebesség-pontosság Pareto diagram készítése
        overall_df,
        os.path.join(args.figures_dir, "speed_vs_accuracy.png")
    )

    print("\n--- Detektor Konfúziós Mátrixok Számítása ---")            # Konfúziós mátrix számítás indítása
    evaluate_confusion_matrices(                                        # Konfúziós mátrixok generálása és rajzolása
        coco_data=coco_data,
        model_predictions=loaded_predictions,
        output_fig_path=os.path.join(
            args.figures_dir, "detector_confusion_matrices.png"
        ),
        output_json_path=os.path.join(
            args.metrics_dir, "detector_confusion_matrices.json"
        ),
        iou_thresh=0.5,
        conf_thresh=0.25
    )

    print("\n--- Legrosszabb Hibaképek Azonosítása Detektoronként ---")  # Hibaelemzés indítása
    evaluate_worst_failure_cases(                                       # Legrosszabb esetek kigyűjtése
        coco_data=coco_data,
        model_predictions=loaded_predictions,
        output_json_path=os.path.join(
            args.metrics_dir, "worst_failure_cases.json"
        ),
        output_csv_path=os.path.join(
            args.metrics_dir, "worst_failure_cases.csv"
        ),
        top_k=10,
        iou_thresh=0.5,
        conf_thresh=0.25
    )

    print("\n[Kész] Minden kiértékelés és publikációs ábra sikeresen elkészült!")  # Végső sikerüzenet


if __name__ == "__main__":
    main()
