"""Shared utilities for the ExDark detection and VLM benchmark framework."""

import os
import random
import json
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional, Any

TARGET_CLASSES = ['person', 'car', 'bus', 'motorbike']                  # 4 kiemelt dinamikus célosztály

TARGET_CLASS_TO_ID = {                                                  # COCO szabvány szerinti 1-alapú azonosítók
    'person': 1,                                                        # Személy azonosító
    'car': 2,                                                           # Személyautó azonosító
    'bus': 3,                                                           # Busz azonosító
    'motorbike': 4                                                      # Motorkerékpár azonosító
}
TARGET_ID_TO_CLASS = {v: k for k, v in TARGET_CLASS_TO_ID.items()}      # Visszakódoló szótár ID -> név

EXDARK_CLASS_MAP = {                                                    # ExDark címkenormalizáló szótár
    'People': 'person',                                                 # People -> person
    'People ': 'person',                                                # Szóközzel végződő People javítása
    'people': 'person',                                                 # Kisbetűs people normalizálás
    'person': 'person',                                                 # Standard person megtartása
    'Car': 'car',                                                       # Nagybetűs Car normalizálás
    'car': 'car',                                                       # Standard car megtartása
    'Bus': 'bus',                                                       # Nagybetűs Bus normalizálás
    'bus': 'bus',                                                       # Standard bus megtartása
    'Motorbike': 'motorbike',                                           # Nagybetűs Motorbike normalizálás
    'motorbike': 'motorbike',                                           # Standard motorbike megtartása
    'motorcycle': 'motorbike',                                          # Szinonima motorcycle normalizálása
    'Motorcycle': 'motorbike'                                           # Nagybetűs Motorcycle normalizálása
}

YOLO_COCO_CLASS_MAP = {                                                 # Ultralytics 80-osztályos leképezés
    0: 'person',                                                        # 0-s index: person
    'person': 'person',                                                 # Szöveges person kulcs
    2: 'car',                                                           # 2-es index: car
    'car': 'car',                                                       # Szöveges car kulcs
    3: 'motorbike',                                                     # 3-as index: motorcycle/motorbike
    'motorcycle': 'motorbike',                                          # Szöveges motorcycle kulcs
    'motorbike': 'motorbike',                                           # Szöveges motorbike kulcs
    5: 'bus',                                                           # 5-ös index: bus
    'bus': 'bus'                                                        # Szöveges bus kulcs
}

TORCHVISION_COCO_CLASS_MAP = {                                          # TorchVision 91-osztályos leképezés
    1: 'person',                                                        # 1-es index: person
    3: 'car',                                                           # 3-as index: car
    4: 'motorbike',                                                     # 4-es index: motorcycle/motorbike
    6: 'bus'                                                            # 6-os index: bus
}

LIGHT_CONDITIONS = {                                                    # 10 ExDark megvilágítási kategória
    1: "Low",                                                           # Általános alacsony fény
    2: "Ambient",                                                       # Szórt háttérvilágítás
    3: "Object",                                                        # Célzott tárgyvilágítás
    4: "Single",                                                        # Egyetlen pontszerű fényforrás
    5: "Weak",                                                          # Gyenge világítás
    6: "Strong",                                                        # Erős pontszerű fény
    7: "Screen",                                                        # Kijelző/képernyő fénye
    8: "Window",                                                        # Ablakon beszűrődő éjszakai fény
    9: "Shadow",                                                        # Mély árnyékzónák
    10: "Twilight"                                                      # Alkonyat / szürkület
}

IN_OUT_CONDITIONS = {                                                   # Beltéri / Kültéri környezeti jelzők
    1: "Indoor",                                                        # Beltéri környezet
    2: "Outdoor"                                                        # Kültéri környezet
}

SPLIT_MAP = {                                                           # Hivatalos ExDark adatfelosztás
    1: "train",                                                         # Tanító halmaz
    2: "val",                                                           # Validációs halmaz
    3: "test"                                                           # Teszt halmaz
}


def set_seed(seed: int = 42):
    """Set random seed across random, numpy, and torch for reproducibility."""
    random.seed(seed)                                                   # Python beépített véletlenszám kezdőérték
    np.random.seed(seed)                                                # NumPy véletlenszám-generátor magja
    torch.manual_seed(seed)                                             # PyTorch CPU mag beállítása
    if torch.cuda.is_available():                                       # Ha van elérhető CUDA hardver
        torch.cuda.manual_seed_all(seed)                                # Összes GPU mag kezdőértékének rögzítése
        torch.backends.cudnn.deterministic = True                       # Determinisztikus konvolúciós algoritmusok
        torch.backends.cudnn.benchmark = False                          # Heurisztikus optimalizáló kikapcsolása


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():                                       # CUDA elérhetőség vizsgálata
        return torch.device('cuda:0')                                   # Elsődleges GPU visszaadása
    return torch.device('cpu')                                          # Fallback CPU eszköz visszaadása


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) of two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])                                          # Metszet bal X koordinátája
    y1 = max(box1[1], box2[1])                                          # Metszet felső Y koordinátája
    x2 = min(box1[2], box2[2])                                          # Metszet jobb X koordinátája
    y2 = min(box1[3], box2[3])                                          # Metszet alsó Y koordinátája

    inter_width = max(0.0, x2 - x1)                                     # Metszet szélessége negatív levágással
    inter_height = max(0.0, y2 - y1)                                    # Metszet magassága negatív levágással
    inter_area = inter_width * inter_height                             # Metszet területe négyzetpixelben

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])   # Első doboz területe
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])   # Második doboz területe
    union_area = area1 + area2 - inter_area                             # Unióterület számítása

    if union_area <= 0:                                                 # Nullával való osztás védelme
        return 0.0                                                      # Nulla átfedés visszaadása
    return inter_area / union_area                                      # IoU arányszám visszaadása [0.0, 1.0]


def draw_bounding_boxes(
    image_path: str,
    gt_boxes: Optional[List[Dict[str, Any]]] = None,
    detections: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    output_path: Optional[str] = None
) -> Image.Image:
    """Draw bounding boxes on an image for comparison."""
    palette = {                                                         # Egyedi színpaletta modellenként
        'Ground Truth': (0, 255, 0),                                    # Zöld: valós tények (Ground Truth)
        'YOLO11': (0, 230, 255),                                        # Ciánkék: YOLO11 detekciók
        'RT-DETR': (255, 215, 0),                                       # Sárga: RT-DETR detekciók
        'Faster R-CNN': (255, 69, 0),                                   # Narancssárga: Faster R-CNN
        'VLM': (180, 0, 255)                                            # Lila: VLM határolókeretek
    }

    img = Image.open(image_path).convert('RGB')                         # Kép betöltése és RGB formátumra hozása
    draw = ImageDraw.Draw(img)                                          # PIL rajzoló felület létrehozása
    w, h = img.size                                                     # Kép szélessége és magassága

    try:                                                                # Rendszer betűtípus betöltési kísérlet
        font = ImageFont.truetype(                                      # TrueType Arial betűtípus skálázása
            "arial.ttf", size=max(14, int(min(w, h) * 0.025))           # Relatív betűméret a képfelbontáshoz
        )
    except Exception:                                                   # Ha nem elérhető a betűtípus
        font = ImageFont.load_default()                                 # Alapértelmezett PIL betűtípus fallback

    if gt_boxes:                                                        # Valós dobozok kirajzolása
        color = palette['Ground Truth']                                 # Zöld szín beállítása
        for obj in gt_boxes:                                            # Objektumok bejárása
            box = obj['bbox']                                           # Doboz koordináták kinyerése
            if len(box) == 4:                                           # Érvényes 4 koordinátás doboz
                x1, y1 = box[0], box[1]                                 # Kezdőpontok
                x2 = box[0] + box[2] if obj.get('format') == 'xywh' else box[2]  # Végpont X számítása
                y2 = box[1] + box[3] if obj.get('format') == 'xywh' else box[3]  # Végpont Y számítása
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)  # GT téglalap kirajzolása vastagon
                label = f"GT: {obj['label']}"                           # GT felirat összeállítása
                draw.text((x1, max(0, y1 - 16)), label, fill=color, font=font)  # Szöveg felrajzolása a doboz fölé

    if detections:                                                      # Modell detekciók felrajzolása
        for model_name, preds in detections.items():                    # Modellek szerinti bejárás
            color = palette.get(model_name, (255, 255, 255))            # Modellhez rendelt egyedi szín
            for p in preds:                                             # Predikciók bejárása
                box = p.get('bbox') or p.get('approx_bbox')             # Standard vagy VLM koordináta
                if not box or len(box) != 4:                            # Hibás doboz kihagyása
                    continue                                            # Következő predikcióra lépés
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]         # Koordináták kicsomagolása
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)  # Detekciós téglalap rajzolása
                score_str = f" {p['score']:.2f}" if 'score' in p else ""  # Konfidencia formázása
                label = f"{model_name}: {p['label']}{score_str}"        # Címkeszöveg összeállítása
                draw.text((x1 + 2, y1 + 2), label, fill=color, font=font)  # Címke elhelyezése a dobozban

    if output_path:                                                     # Ha kértek fájlmentést
        os.makedirs(os.path.dirname(output_path), exist_ok=True)         # Mappa létrehozása a képnek
        img.save(output_path)                                           # Kép kiírása a lemezre

    return img                                                          # Rajzolt PIL kép visszaadása
