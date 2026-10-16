"""Comparison module evaluating VLM vs classic detectors (YOLO11, RT-DETR, Faster R-CNN).

Computes:
- Object-level spatial agreement metrics (IoU overlap)
- Image-level class recognition metrics (Precision, Recall, F1 across person, car, bus, motorbike)
- Agreement confusion matrices and heatmaps
- Multi-model visual overlays on diverse samples across all 4 target categories and lighting regimes
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.dirname(os.path.abspath(__file__)))              # Forráskönyvtár hozzáadása az importálási úthoz
from utils import (
    TARGET_CLASSES,                                                     # 4 balesetveszélyes célosztály listája
    TARGET_CLASS_TO_ID,                                                 # Osztálynév -> COCO azonosító leképezés
    TARGET_ID_TO_CLASS,                                                 # COCO azonosító -> osztálynév leképezés
    calculate_iou,                                                      # Határolókeretek metszet-unió arányának számítása
    draw_bounding_boxes                                                 # Többmodell határolókereteket kirajzoló függvény
)

plt.style.use(                                                          # Diagramstílus beállítása
    'seaborn-v0_8-whitegrid'                                             # Fehér rácsos letisztult téma
    if 'seaborn-v0_8-whitegrid' in plt.style.available                  # Elérhetőség vizsgálata
    else 'default'                                                      # Alapértelmezett matplotlib stílus fallback
)
plt.rcParams['font.sans-serif'] = 'Arial'                               # Publikációs betűtípus


def safe_savefig(
    fig: plt.Figure,                                                    # Ábra objektum
    output_path: str,                                                   # Kimeneti fájl útvonala
    dpi: int = 300,                                                     # Képfelbontás pont/hüvelykben
    **kwargs                                                            # Opcionális matplotlib paraméterek
):
    """Safely save figure on Windows handling transient file lock issues."""
    output_path = os.path.abspath(output_path)                          # Abszolút elérési út normalizálása
    os.makedirs(os.path.dirname(output_path), exist_ok=True)            # Célkönyvtár létrehozása ha hiányzik
    for attempt in range(6):                                            # Windows átmeneti fájlzárolási kísérletek
        try:                                                            # Képmentési kísérlet
            fig.savefig(output_path, dpi=dpi, **kwargs)                 # Kép kiírása lemezre
            return                                                      # Sikeres mentés után kilépés
        except OSError:                                                 # Zárolási hiba elkapása
            time.sleep(0.4)                                             # 400 ms várakozás
    fig.savefig(output_path, dpi=dpi, **kwargs)                         # Végső mentési kísérlet


def compute_vlm_detector_agreement(
    gt_annotations: List[Dict[str, Any]],                               # Valós annotációk listája
    vlm_preds: Dict[str, Dict[str, Any]],                               # VLM detekciók szótára képenként
    detector_preds: Dict[str, List[Dict[str, Any]]],                    # Detektor predikciók szótára képenként
    iou_thresh: float = 0.25                                            # VLM térbeli konszenzus IoU küszöbértéke
) -> Dict[str, Any]:
    """Compare VLM detections against a specific detector across images."""
    gt_by_img = {}                                                      # Valós dobozok gyűjtője kép szerint
    for ann in gt_annotations:                                          # Annotációk egyenkénti bejárása
        img_id = str(ann['image_id'])                                   # Képazonosító szöveggé alakítása
        x, y, w, h = ann['bbox']                                        # Befoglaló koordináták
        gt_by_img.setdefault(img_id, []).append({                       # Doboz hozzáadása a képhez
            "bbox": [x, y, x + w, y + h],                               # Átváltás [x1, y1, x2, y2] sarokpontokra
            "label": TARGET_ID_TO_CLASS.get(                            # Számszerű azonosító leképezése névre
                ann['category_id'], 'unknown'
            )
        })

    vlm_found_det_missed = 0                                            # Csak a VLM által talált objektumok száma
    det_found_vlm_missed = 0                                            # Csak a detektor által talált objektumok száma
    both_found = 0                                                      # Mindkét modell által megtalált objektumok
    both_missed = 0                                                     # Kölcsönösen elmulasztott objektumok száma

    class_matrix = {                                                    # Egyetértési mátrix inicializálása
        c1: {c2: 0 for c2 in TARGET_CLASSES + ['Missed']}               # Kategóriák és kihagyott oszlopok
        for c1 in TARGET_CLASSES + ['Missed']                           # Kategóriák és kihagyott sorok
    }

    for img_id, vlm_item in vlm_preds.items():                          # Képek bejárása VLM predikciók alapján
        if img_id not in gt_by_img:                                     # Ha a képhez nincs valós annotáció
            continue                                                    # Átugrás
        vlm_boxes = vlm_item.get('detections', [])                      # VLM határolókeretek lekérése
        det_boxes = detector_preds.get(img_id, [])                      # Detektor határolókeretek lekérése
        gt_boxes = gt_by_img.get(img_id, [])                            # Valós határolókeretek lekérése

        for gt in gt_boxes:                                             # Valós objektumok egyenkénti ellenőrzése
            gt_b = gt['bbox']                                           # Valós keret koordináták
            gt_label = gt['label']                                      # Valós osztálycímke
            if gt_label not in TARGET_CLASSES:                          # Csak a 4 célosztály vizsgálata
                continue                                                # Nem releváns osztály átugrása

            vlm_match = any(                                            # VLM térbeli találat vizsgálata
                calculate_iou(v['approx_bbox'], gt_b) >= iou_thresh     # Engedékenyebb 0.25 IoU token koordinátákhoz
                and v['label'] == gt_label                              # Azonos kategória egyezés
                for v in vlm_boxes                                      # VLM predikciók végigpróbálása
            )

            det_match = any(                                            # Klasszikus detektor találat vizsgálata
                calculate_iou(d['bbox'], gt_b) >= 0.5                   # Standard 0.50 IoU regressziós dobozokhoz
                and d['label'] == gt_label                              # Azonos kategória megléte
                for d in det_boxes                                      # Detektor predikciók végigpróbálása
            )

            if vlm_match and det_match:                                 # Mindkét modell helyesen megtalálta
                both_found += 1                                         # Konszenzus számláló növelése
                class_matrix[gt_label][gt_label] += 1                   # Átlós megegyezés bejegyzése
            elif vlm_match and not det_match:                           # Csak a VLM találta meg
                vlm_found_det_missed += 1                               # VLM-fölény számláló növelése
                class_matrix[gt_label]['Missed'] += 1                   # Detektor kihagyásként könyvelve
            elif not vlm_match and det_match:                           # Csak a detektor találta meg
                det_found_vlm_missed += 1                               # Detektor-fölény számláló növelése
                class_matrix['Missed'][gt_label] += 1                   # VLM kihagyásként könyvelve
            else:                                                       # Egyik modell sem észlelte (mindkettő kihagyta)
                both_missed += 1                                        # Kritikus biztonsági kockázat számláló
                class_matrix['Missed']['Missed'] += 1                   # Kölcsönös kihagyás bejegyzése

    return {                                                            # Összesített egyetértési szótár
        "both_detected": both_found,                                    # Mindkettő detektálta
        "vlm_found_detector_missed": vlm_found_det_missed,              # Detektor kihagyta, VLM észlelte
        "detector_found_vlm_missed": det_found_vlm_missed,              # VLM kihagyta, Detektor észlelte
        "both_missed": both_missed,                                     # Egyik sem látta
        "total_gt_evaluated": (                                         # Összes kiértékelt valós objektum
            both_found + vlm_found_det_missed
            + det_found_vlm_missed + both_missed
        ),
        "agreement_matrix": class_matrix                                # Részletes osztályszintű mátrix
    }


def compute_image_level_recognition(
    coco_data: Dict[str, Any],                                          # COCO formátumú annotációs adatbázis
    vlm_preds: Dict[str, Dict[str, Any]],                               # VLM kimenetek
    detectors: Dict[str, Dict[str, List[Dict[str, Any]]]]               # Klasszikus detektorok predikciói
) -> pd.DataFrame:
    """Compute image-level category recognition metrics (Precision, Recall, F1)."""
    gt_presence = {}                                                    # Képszintű valós jelenlét halmazok
    for ann in coco_data['annotations']:                                # Valós annotációk bejárása
        img_id = str(ann['image_id'])                                   # Képazonosító szövegként
        cat = TARGET_ID_TO_CLASS.get(ann['category_id'])                # Kategórianév feloldása
        if cat:                                                         # Ha létező célkategória
            gt_presence.setdefault(img_id, set()).add(cat)              # Kategória rögzítése a kép jelenléti halmazában

    models_preds = {                                                    # Prediktált kategóriahalmazok modellenként
        'VLM (Qwen2-VL)': {                                             # VLM jelenlét halmazok összeállítása
            img_id: set(d['label'] for d in item.get('detections', [])) # VLM által észlelt címkék egyedi halmaza
            for img_id, item in vlm_preds.items()                       # VLM képek bejárása
            if img_id in gt_presence                                    # Csak a releváns képek megtartása
        }
    }
    for d_name, d_preds in detectors.items():                           # Detektorok jelenléti halmazainak építése
        models_preds[d_name] = {                                        # Adott detektor szótárának készítése
            img_id: set(p['label'] for p in d_preds.get(img_id, []))    # Detektált osztálynevek halmaza
            for img_id in gt_presence                                   # Valós képek szűrése
        }

    rows = []                                                           # Metrika táblázat sorai
    for model_name, preds_dict in models_preds.items():                 # Modellek egyenkénti kiértékelése
        for cat in TARGET_CLASSES:                                      # 4 célosztály vizsgálata
            tp = sum(                                                   # Valódi pozitív képszám
                1 for img_id in gt_presence                             # Képek bejárása
                if cat in gt_presence[img_id]                           # Valójában jelen van
                and cat in preds_dict.get(img_id, set())                # És a modell is jelezte
            )
            fp = sum(                                                   # Tévesen jelzett képszám
                1 for img_id in gt_presence                             # Képek bejárása
                if cat not in gt_presence[img_id]                       # Nincs a képen
                and cat in preds_dict.get(img_id, set())                # De a modell detektálta
            )
            fn = sum(                                                   # Kihagyott képszám
                1 for img_id in gt_presence                             # Képek bejárása
                if cat in gt_presence[img_id]                           # Jelen van a valóságban
                and cat not in preds_dict.get(img_id, set())            # De a modell nem jelezte
            )

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0             # Képszintű precízió képlete
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0              # Képszintű visszahívás képlete
            f1 = (                                                      # Képszintű F1 pontszám
                (2 * prec * rec) / (prec + rec)                         # Harmonikus közép számlálója
                if (prec + rec) > 0 else 0.0                            # Nullával osztás elkerülése
            )

            rows.append({                                               # Eredménysor rögzítése
                "model": model_name,                                    # Modell megnevezése
                "category": cat,                                        # Célosztály neve
                "true_positives": tp,                                   # Helyes jelenlét jelzések
                "false_positives": fp,                                  # Téves jelenlét riasztások
                "false_negatives": fn,                                  # Észrevétlen jelenlétek
                "precision": round(prec, 4),                            # Képszintű precízió 4 tizedesre
                "recall": round(rec, 4),                                # Képszintű visszahívás 4 tizedesre
                "f1_score": round(f1, 4)                                # Képszintű F1-érték 4 tizedesre
            })

    return pd.DataFrame(rows)                                           # Pandas táblázat visszaadása


def plot_image_level_recognition(
    df: pd.DataFrame,                                                   # Képszintű metrikák DataFrame
    output_path: str                                                    # Kimeneti diagram útvonala
):
    """Plot grouped bar chart comparing Image-level Precision and Recall across models and categories."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)          # Két paneles ábra Precízió és Visszahíváshoz
    palette = {                                                         # Harmonikus színpaletta a 4 modellhez
        'YOLO11': '#00A8E8',                                            # Ciánkék a YOLO11-hez
        'RT-DETR': '#FFB703',                                           # Sárga az RT-DETR-hez
        'Faster R-CNN': '#FB8500',                                       # Narancs a Faster R-CNN-hez
        'VLM (Qwen2-VL)': '#8338EC'                                     # Lila a VLM modellhez
    }

    sns.barplot(                                                        # Precízió oszlopdiagram felrajzolása
        data=df, x='category', y='precision', hue='model',              # Kategóriák, precízió és modellek szerinti bontás
        palette=palette, ax=axes[0]                                     # Bal oldali panelre
    )
    axes[0].set_title(                                                  # Bal panel címe
        'Image-Level Category Precision (Adverse Conditions)',
        fontsize=12, fontweight='bold'
    )
    axes[0].set_xlabel('Category', fontsize=11, fontweight='bold')      # X tengely felirat
    axes[0].set_ylabel('Precision', fontsize=11, fontweight='bold')     # Y tengely felirat
    axes[0].set_ylim(0, 1.05)                                           # Skálahatár 0 és 105% között
    axes[0].legend(title='Model', loc='lower right', frameon=True)      # Jelmagyarázat elhelyezése

    sns.barplot(                                                        # Visszahívás oszlopdiagram felrajzolása
        data=df, x='category', y='recall', hue='model',                 # Kategóriák, visszahívás és modellek bontása
        palette=palette, ax=axes[1]                                     # Jobb oldali panelre
    )
    axes[1].set_title(                                                  # Jobb panel címe
        'Image-Level Category Recall (Adverse Conditions)',
        fontsize=12, fontweight='bold'
    )
    axes[1].set_xlabel('Category', fontsize=11, fontweight='bold')      # X tengely felirat
    axes[1].set_ylabel('Recall', fontsize=11, fontweight='bold')        # Y tengely felirat
    axes[1].set_ylim(0, 1.05)                                           # Skálahatár
    axes[1].legend(title='Model', loc='upper right', frameon=True)      # Jelmagyarázat elhelyezése

    plt.tight_layout()                                                  # Elrendezés igazítása
    safe_savefig(fig, output_path, dpi=300)                             # Ábra lemezre mentése
    plt.close()                                                         # Ábra bezárása
    print(f"[Ábra] Képszintű felismerési diagram elmentve: {output_path}")  # Tájékoztatás


def plot_agreement_heatmap(
    agreement_dict: Dict[str, Dict[str, Any]],                          # Egyetértési szótár modellenként
    output_path: str                                                    # Kimeneti hőtérkép útvonala
):
    """Plot agreement heatmap comparing VLM with YOLO11, RT-DETR, Faster R-CNN."""
    models = list(agreement_dict.keys())                                # Modellek neveinek listája
    fig, axes = plt.subplots(                                           # Több paneles ábra a modellekhez
        1, len(models), figsize=(6 * len(models), 5), dpi=300           # Szélesség a modellek számához igazítva
    )
    if len(models) == 1:                                                # Ha egyetlen modell van
        axes = [axes]                                                   # Listává alakítás az egységes iterációhoz

    categories = TARGET_CLASSES + ['Missed']                            # 4 célosztály plusz kihagyott kategória

    for idx, (m_name, data) in enumerate(agreement_dict.items()):       # Modellek végigjárása
        ax = axes[idx]                                                  # Panel kiválasztása
        mat = data['agreement_matrix']                                  # Nyers mátrix kinyerése
        matrix_data = [                                                 # 2D numerikus tömb felépítése
            [mat[r][c] for c in categories] for r in categories
        ]
        df_mat = pd.DataFrame(                                          # DataFrame indexekkel és oszlopokkal
            matrix_data, index=categories, columns=categories
        )

        sns.heatmap(                                                    # Hőtérkép kirajzolása kék árnyalatokkal
            df_mat, annot=True, fmt='d', cmap='Blues',                  # Számszerű értékek egész szám formátumban
            cbar=False, linewidths=0.5, linecolor='#eeeeee',            # Színsáv nélkül, vékony szürke ráccsal
            ax=ax                                                       # Cél tengely
        )
        ax.set_title(f'VLM vs {m_name} Object Agreement',               # Panel címe
                     fontsize=11, fontweight='bold')
        ax.set_xlabel(f'{m_name} Detections', fontsize=10,              # X tengely a detektor predikciói
                      fontweight='bold')
        ax.set_ylabel('VLM Detections', fontsize=10,                    # Y tengely a VLM predikciói
                      fontweight='bold')

    plt.tight_layout()                                                  # Elrendezés igazítása
    safe_savefig(fig, output_path, dpi=300)                             # Hőtérkép lemezre írása
    plt.close()                                                         # Ábra bezárása
    print(f"[Ábra] Egyetértési hőtérkép elmentve: {output_path}")       # Tájékoztatás


def generate_visual_overlays(
    images_dir: str,                                                    # Eredeti képek könyvtára
    coco_data: Dict[str, Any],                                          # COCO annotációk
    vlm_preds: Dict[str, Dict[str, Any]],                               # VLM predikciók
    detector_preds: Dict[str, Dict[str, List[Dict[str, Any]]]],         # Detektorok predikciói
    output_dir: str                                                     # Kimeneti ábrák könyvtára
):
    """Generate overlay images with bounding boxes from Ground Truth, YOLO11, RT-DETR, Faster R-CNN, and VLM."""
    os.makedirs(output_dir, exist_ok=True)                              # Kimeneti mappa biztosítása

    gt_by_img = {}                                                      # Valós dobozok gyűjtője
    for ann in coco_data['annotations']:                                # Annotációk bejárása
        img_id = str(ann['image_id'])                                   # Képazonosító szövegként
        x, y, w, h = ann['bbox']                                        # Befoglaló méretek
        gt_by_img.setdefault(img_id, []).append({                       # Doboz tárolása
            "bbox": [x, y, x + w, y + h],                               # [x1, y1, x2, y2] koordináták
            "label": TARGET_ID_TO_CLASS.get(                            # Címke név feloldása
                ann['category_id'], 'unknown'
            ),
            "format": "xyxy"                                            # Formátum jelölése
        })

    preferred_ids = ['1', '128', '374', '469']                          # 4 reprezentatív kép a 4 osztályhoz
    evaluated_ids = list(vlm_preds.keys())                              # VLM által kiértékelt azonosítók
    sample_ids = [i for i in preferred_ids if i in evaluated_ids]       # Preferált képek szűrése az elérhetőkre
    if len(sample_ids) < 4:                                             # Ha hiányzik valamelyik preferált kép
        for i in evaluated_ids:                                         # Képek pótlása a kiértékeltekből
            if i not in sample_ids:                                     # Duplikáció elkerülése
                sample_ids.append(i)                                    # Hozzáadás a mintákhoz
            if len(sample_ids) == 4:                                    # 4 mintakép elérésekor megállás
                break

    overlay_images = []                                                 # Kirajzolt képek gyűjtője
    titles = []                                                         # Képcímek gyűjtője

    for img_id in sample_ids:                                           # Kiválasztott képek feldolgozása
        vlm_item = vlm_preds[img_id]                                    # VLM adatrekord lekérése
        img_rel_path = vlm_item['file_name']                            # Relatív képútvonal
        img_full_path = os.path.join(images_dir, img_rel_path)          # Teljes képútvonal
        if not os.path.exists(img_full_path):                           # Ha a képfájl hiányzik a lemezről
            continue                                                    # Átugrás

        gt_boxes = gt_by_img.get(img_id, [])                            # Valós dobozok lekérése
        det_dict = {                                                    # Detektált keretek szótára modellenként
            'YOLO11': detector_preds.get('YOLO11', {}).get(img_id, []), # YOLO11 detekciói
            'RT-DETR': detector_preds.get(                              # RT-DETR detekciói
                'RT-DETR', {}
            ).get(img_id, []),
            'Faster R-CNN': detector_preds.get(                         # Faster R-CNN detekciói
                'Faster R-CNN', {}
            ).get(img_id, []),
            'VLM': [                                                    # VLM közelítő koordinátái
                {"bbox": d['approx_bbox'], "label": d['label']}         # VLM doboz és osztálycímke
                for d in vlm_item.get('detections', [])                 # VLM észlelések bejárása
            ]
        }

        out_img_path = os.path.join(                                    # Egyedi overlay kép mentési útvonala
            output_dir, f"overlay_sample_{img_id}.png"
        )
        img_drawn = draw_bounding_boxes(                                # Színes dobozok felrajzolása a képre
            image_path=img_full_path,                                   # Eredeti képfájl
            gt_boxes=gt_boxes,                                          # Zöld valós dobozok
            detections=det_dict,                                        # Színes modell dobozok
            output_path=out_img_path                                    # Kimeneti mentési hely
        )
        overlay_images.append(img_drawn)                                # Rajzolt kép elmentése a memóriában
        light_cond = vlm_item.get('light_condition', 'Unknown')         # Megvilágítás lekérdezése
        titles.append(                                                  # Kép feliratának összeállítása
            f"Image {img_id}: {img_rel_path}\nLighting: {light_cond}"
        )
        print(f"[Vizuális] Overlay kép elkészült: {img_rel_path} -> {out_img_path}")  # Tájékoztatás

    if overlay_images:                                                  # Ha készültek képek, összeállítjuk a tablót
        n = len(overlay_images)                                         # Kész képek száma
        rows = 2 if n >= 2 else 1                                       # Sorok száma a 2x2 elrendezéshez
        cols = 2 if n >= 2 else 1                                       # Oszlopok száma
        fig, axes = plt.subplots(                                       # Több paneles ábra inicializálása
            rows, cols, figsize=(14, 11), dpi=250                       # Nagy felbontás a tablóhoz
        )
        axes = np.array(axes).reshape(-1)                               # 1D tömbbé simítás a könnyű indexeléshez

        for i in range(min(n, 4)):                                      # Maximum 4 kép kirajzolása
            axes[i].imshow(overlay_images[i])                           # Kép megjelenítése az aldiagramon
            axes[i].set_title(titles[i], fontsize=10,                   # Képcím és megvilágítás kiírása
                              fontweight='bold', pad=6)
            axes[i].axis('off')                                         # Tengelyvonalak elrejtése

        for i in range(n, len(axes)):                                   # Maradék üres panelek elrejtése
            axes[i].axis('off')                                         # Üres keret kikapcsolása

        legend_elements = [                                             # Színkódolt jelmagyarázat összeállítása
            plt.Line2D([0], [0], color='#00FF00', lw=3,                 # Zöld: Ground Truth
                       label='Ground Truth (ExDark)'),
            plt.Line2D([0], [0], color='#00E6FF', lw=2.5,               # Cián: YOLO11
                       label='YOLO11 Detection'),
            plt.Line2D([0], [0], color='#FFD700', lw=2.5,               # Sárga: RT-DETR
                       label='RT-DETR Detection'),
            plt.Line2D([0], [0], color='#FF4500', lw=2.5,               # Vörös: Faster R-CNN
                       label='Faster R-CNN Detection'),
            plt.Line2D([0], [0], color='#B400FF', lw=2.5,               # Lila: VLM Zero-Shot
                       label='VLM Zero-Shot (Qwen2-VL)')
        ]
        fig.legend(                                                     # Jelmagyarázat sáv elhelyezése felül
            handles=legend_elements,                                    # Színes vonalelemek
            loc='upper center',                                         # Középre igazítás felül
            bbox_to_anchor=(0.5, 0.99),                                 # Abszolút pozicionálás
            ncol=5,                                                     # 5 oszlopos vízszintes elrendezés
            frameon=True, facecolor='white', framealpha=0.95,           # Fehér keret háttérrel
            fontsize=10                                                 # Betűméret
        )

        plt.subplots_adjust(top=0.92, wspace=0.05, hspace=0.15)         # Margók és térközök finomhangolása
        combined_path = os.path.join(output_dir, "overlay_samples.png") # Összesített tabló elérési útja
        safe_savefig(fig, combined_path, dpi=250, bbox_inches='tight')  # Mentés szoros margókkal
        plt.close()                                                     # Ábra lezárása
        print(f"[Vizuális] Összetett 2x2 tabló elmentve -> {combined_path}")  # Tájékoztatás


def main():
    parser = argparse.ArgumentParser(                                   # Parancssori argumentumok kezelője
        description="VLM és klasszikus detektorok összehasonlítása az ExDark adathalmazon."
    )
    parser.add_argument(                                                # COCO teszt annotációk útvonala
        "--coco_json", type=str,
        default="data/ExDark/coco_annotations_test.json",
        help="COCO teszt annotációk."
    )
    parser.add_argument(                                                # Képek gyökérkönyvtára
        "--images_dir", type=str,
        default="data/ExDark/images",
        help="Képek gyökérkönyvtára."
    )
    parser.add_argument(                                                # VLM predikciós JSON
        "--vlm_preds", type=str,
        default="results/vlm_outputs/vlm_independent_predictions.json",
        help="VLM predikciós JSON fájl."
    )
    parser.add_argument(                                                # VLM szöveges magyarázatok JSON
        "--vlm_explanations", type=str,
        default="results/vlm_outputs/vlm_explanations.json",
        help="VLM magyarázatok JSON fájlja."
    )
    parser.add_argument(                                                # YOLO11 predikciós JSON
        "--yolo_preds", type=str,
        default="results/metrics/yolo11_predictions.json",
        help="YOLO11 predikciós JSON fájl."
    )
    parser.add_argument(                                                # RT-DETR predikciós JSON
        "--rtdetr_preds", type=str,
        default="results/metrics/rtdetr_predictions.json",
        help="RT-DETR predikciós JSON fájl."
    )
    parser.add_argument(                                                # Faster R-CNN predikciós JSON
        "--fasterrcnn_preds", type=str,
        default="results/metrics/faster_rcnn_predictions.json",
        help="Faster R-CNN predikciós JSON fájl."
    )
    parser.add_argument(                                                # Diagramok mappája
        "--figures_dir", type=str,
        default="results/figures",
        help="Diagramok kimeneti könyvtára."
    )
    parser.add_argument(                                                # Metrikák mappája
        "--metrics_dir", type=str,
        default="results/metrics",
        help="Metrikák kimeneti könyvtára."
    )
    args = parser.parse_args()                                          # Argumentumok feldolgozása

    coco_data = json.load(open(args.coco_json, 'r', encoding='utf-8'))  # COCO annotációk beolvasása JSON-ből
    vlm_preds_raw = json.load(                                          # VLM nyers predikciók beolvasása
        open(args.vlm_preds, 'r', encoding='utf-8')
    )

    vlm_by_filename = {                                                 # Gyorskereső fájlnév alapján az illesztéshez
        item['file_name']: item for item in vlm_preds_raw.values()      # Érvényes szótár elemek szűrése
        if isinstance(item, dict) and 'file_name' in item               # Fájlnév mező ellenőrzése
    }
    aligned_vlm_preds = {}                                              # Illesztett VLM predikciók szótára
    for img in coco_data['images']:                                     # COCO képek végigjárása
        str_id = str(img['id'])                                         # Képazonosító
        fn = img['file_name']                                           # Fájlnév
        if fn in vlm_by_filename:                                       # Fájlnév alapú párosítás
            aligned_vlm_preds[str_id] = vlm_by_filename[fn]             # Azonosítóhoz rendelés
        elif str_id in vlm_preds_raw:                                   # Azonosító alapú párosítás
            aligned_vlm_preds[str_id] = vlm_preds_raw[str_id]           # Nyers hozzárendelés

    detectors = {                                                       # Detektor predikciók betöltése szótárba
        "YOLO11": json.load(open(args.yolo_preds, 'r', encoding='utf-8')),
        "RT-DETR": json.load(open(args.rtdetr_preds, 'r', encoding='utf-8')),
        "Faster R-CNN": json.load(
            open(args.fasterrcnn_preds, 'r', encoding='utf-8')
        )
    }

    print("\n--- VLM és Klasszikus Detektorok Egyetértésének Vizsgálata ---")  # Vizsgálat indítása
    agreement_results = {}                                              # Egyetértési eredmények gyűjtője
    for d_name, d_preds in detectors.items():                           # Detektorok végigjárása
        res = compute_vlm_detector_agreement(                           # Konszenzus és egyetértés számítása
            gt_annotations=coco_data['annotations'],
            vlm_preds=aligned_vlm_preds,
            detector_preds=d_preds,
            iou_thresh=0.25
        )
        agreement_results[d_name] = res                                 # Eredmény rögzítése
        print(f"\n[{d_name} vs VLM]")                                   # Detektor név kiírása
        print(f"  Konszenzusos detekciók (Mindkettő megtalálta): {res['both_detected']}")
        print(f"  VLM megtalálta, Detektor kihagyta:           {res['vlm_found_detector_missed']}")
        print(f"  Detektor megtalálta, VLM kihagyta:           {res['detector_found_vlm_missed']}")
        print(f"  Mindkettő kihagyta (Kölcsönös hiba):         {res['both_missed']}")

    os.makedirs(args.metrics_dir, exist_ok=True)                        # Metrika könyvtár biztosítása
    with open(                                                          # Egyetértési adatok kiírása JSON-be
        os.path.join(args.metrics_dir, "vlm_vs_detectors.json"),
        'w', encoding='utf-8'
    ) as f:
        json.dump(agreement_results, f, indent=2)                       # JSON mentés

    matrix_rows = []                                                    # Táblázatos mátrix sorok gyűjtője CSV-hez
    for d_name, data in agreement_results.items():                      # Detektorok bejárása
        mat = data['agreement_matrix']                                  # Mátrix kinyerése
        for r in mat:                                                   # Sorok bejárása
            for c in mat[r]:                                            # Oszlopok bejárása
                matrix_rows.append({                                    # CSV rekord összeállítása
                    "detector": d_name,                                 # Detektor megnevezése
                    "vlm_class": r,                                     # VLM osztály
                    "detector_class": c,                                # Detektor osztály
                    "count": mat[r][c]                                  # Események darabszáma
                })
    pd.DataFrame(matrix_rows).to_csv(                                   # Mátrix mentése CSV táblázatba
        os.path.join(args.metrics_dir, "vlm_agreement_matrix.csv"),
        index=False
    )

    plot_agreement_heatmap(                                             # Egyetértési hőtérkép rajzolása
        agreement_results,
        os.path.join(args.figures_dir, "vlm_detector_agreement_heatmap.png")
    )

    print("\n--- Képszintű Kategória-Felismerési Metrikák Számítása ---")  # Képszintű értékelés
    recognition_df = compute_image_level_recognition(                   # Képszintű jelenlét metrikák számítása
        coco_data, aligned_vlm_preds, detectors
    )
    rec_csv_path = os.path.join(                                        # CSV fájl elérési útja
        args.metrics_dir, "vlm_image_level_recognition.csv"
    )
    recognition_df.to_csv(rec_csv_path, index=False)                    # Képszintű metrikák mentése CSV-be
    print(f"[Metrika] Képszintű felismerési metrikák elmentve: {rec_csv_path}")  # Tájékoztatás

    plot_image_level_recognition(                                       # Képszintű diagram generálása
        recognition_df,
        os.path.join(
            args.figures_dir, "vlm_vs_detectors_class_recognition.png"
        )
    )

    print("\n--- Reprezentatív Vizuális Overlay Ábrák Generálása ---")   # Overlay készítés indítása
    generate_visual_overlays(                                           # Határolókeretes képek és tabló készítése
        images_dir=args.images_dir,
        coco_data=coco_data,
        vlm_preds=aligned_vlm_preds,
        detector_preds=detectors,
        output_dir=args.figures_dir
    )

    print("\n[Kész] VLM és detektor összehasonlítás sikeresen befejeződött!")  # Végső sikerüzenet


if __name__ == "__main__":
    main()
