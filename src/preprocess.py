"""Preprocess ExDark dataset annotations into unified COCO format.

Filters to target categories: person, car, bus, motorbike.
Extracts illumination metadata (10 lighting conditions) and indoor/outdoor flags.
Optionally integrates synthetic adverse condition benchmarks (fog, rain, glare).
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any
from PIL import Image
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))              # src könyvtár felvétele az import útvonalba
from utils import (
    TARGET_CLASSES,                                                     # 4 balesetveszélyes célosztály listája
    TARGET_CLASS_TO_ID,                                                 # Kategórianév -> COCO azonosító szótár
    EXDARK_CLASS_MAP,                                                   # ExDark címkenormalizáló szótár
    LIGHT_CONDITIONS,                                                   # 10 megvilágítási kategória szótára
    IN_OUT_CONDITIONS,                                                  # Beltéri/Kültéri környezeti jelzők
    SPLIT_MAP,                                                          # Adathalmaz-felosztások (train/val/test)
    set_seed                                                            # Determinisztikus véletlenszám-beállító
)


def parse_imageclasslist(list_path: str) -> Dict[str, Dict[str, Any]]:
    """Parse imageclasslist.txt containing illumination and split annotations."""
    if not os.path.exists(list_path):                                   # Metaadat fájl létezésének vizsgálata
        print(f"[Figyelmeztetés] {list_path} nem található. A felosztási metaadatok nem lesznek elérhetők.")  # Figyelmeztetés
        return {}                                                       # Üres szótár visszaadása hiba esetén

    metadata = {}                                                       # Metaadat tároló szótár inicializálása
    with open(list_path, 'r', encoding='utf-8') as f:                   # Szöveges metaadat fájl megnyitása
        lines = f.readlines()                                           # Összes sor beolvasása a memóriába

    header = True                                                       # Fejléc átugrását jelző logikai változó
    for line in lines:                                                  # Soronkénti bejárás a fájlban
        line = line.strip()                                             # Felesleges szóközök és sortörések levágása
        if not line:                                                    # Üres sorok átugrása
            continue                                                    # Következő iterációra lépés
        if header and ('Name' in line or '|' in line):                  # Fejléc sor azonosítása oszlopnevek alapján
            header = False                                              # Fejléc véget ért, innentől adatsorok jönnek
            continue                                                    # Fejléc sor kihagyása
        parts = line.split()                                            # Sor felbontása szóközök mentén
        if len(parts) >= 5:                                             # Legalább 5 oszlop meglétének ellenőrzése
            img_name = parts[0]                                         # 1. oszlop: képfájl neve
            light_id = int(parts[2])                                    # 3. oszlop: megvilágítási kategória ID
            in_out_id = int(parts[3])                                   # 4. oszlop: beltéri/kültéri jelző ID
            split_id = int(parts[4])                                    # 5. oszlop: adatfelosztási split ID
            metadata[img_name] = {                                      # Strukturált metaadat rekord építése
                "light_id": light_id,                                   # Numerikus fény azonosító
                "light_name": LIGHT_CONDITIONS.get(light_id, f"Condition_{light_id}"),  # Fényviszony szöveges neve
                "in_out": IN_OUT_CONDITIONS.get(in_out_id, "Unknown"),  # Beltér/kültér szöveges leképezése
                "split": SPLIT_MAP.get(split_id, "unknown")             # Train/val/test felosztás neve
            }
    return metadata                                                     # Feldolgozott szótár visszaadása


def parse_exdark_annotation_file(txt_path: str) -> List[Dict[str, Any]]:
    """Parse an ExDark annotation txt file."""
    if not os.path.exists(txt_path):                                    # Ha az annotációs txt nem létezik
        return []                                                       # Üres objektumlistával térünk vissza

    objects = []                                                        # Talált objektumok listájának tárolója
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:   # Dollar annotáció megnyitása kódolásvédelemmel
        lines = f.readlines()                                           # Sorok beolvasása a memóriába

    for line in lines:                                                  # Annotációs sorok bejárása
        line = line.strip()                                             # Sorvégi whitespace karakterek eltávolítása
        if not line or line.startswith('%'):                            # Megjegyzések (%) és üres sorok kihagyása
            continue                                                    # Tovább a következő sorra
        parts = line.split()                                            # Szóközök mentén darabolás
        if len(parts) < 5:                                              # Ha hiányzik az osztály vagy a 4 koordináta
            continue                                                    # Érvénytelen sor eldobása
        raw_class = parts[0]                                            # Nyers osztálynév a fájlból
        mapped_class = EXDARK_CLASS_MAP.get(raw_class)                  # Normalizálás standard osztálynévre
        if mapped_class not in TARGET_CLASSES:                          # Csak a 4 vizsgált célosztály szűrése
            continue                                                    # Egyéb kategóriák figyelmen kívül hagyása

        try:
            x = float(parts[1])                                         # X kezdőkoordináta lebegőpontossá alakítása
            y = float(parts[2])                                         # Y kezdőkoordináta lebegőpontossá alakítása
            w = float(parts[3])                                         # Határolókeret szélességének kinyerése
            h = float(parts[4])                                         # Határolókeret magasságának kinyerése
        except ValueError:                                              # Hibás, nem numerikus koordináták elkapása
            continue                                                    # Érvénytelen doboz kihagyása

        if w <= 0 or h <= 0:                                            # Degenerált, nem pozitív területű dobozok
            continue                                                    # Hibás kiterjedésű keret eldobása

        objects.append({                                                # Standardizált objektumrekord hozzáadása
            "class_name": mapped_class,                                 # Normalizált osztálynév
            "category_id": TARGET_CLASS_TO_ID[mapped_class],            # Standard numerikus COCO azonosító
            "bbox": [x, y, w, h]                                        # Dollar formátumú [x, y, w, h] doboz
        })
    return objects                                                      # Kinyert objektumok visszaadása


def apply_synthetic_adverse_condition(img: Image.Image, condition_type: str) -> Image.Image:
    """Synthesize adverse weather conditions (fog/rain)."""
    arr = np.array(img, dtype=np.float32)                               # Kép konvertálása lebegőpontos NumPy tömbbé
    h, w, c = arr.shape                                                 # Magasság, szélesség és csatornaszám

    if condition_type == "fog":                                         # Szintetikus köd szimulálása szórási modellel
        depth = np.linspace(0.4, 0.95, h).reshape(h, 1, 1)              # Vertikális lineáris mélységgradiens
        t = np.exp(-0.8 * depth)                                        # Atmoszférikus áteresztőképesség
        airlight = 200.0                                                # Szórt környezeti fény intenzitásértéke
        arr = arr * t + airlight * (1.0 - t)                            # Koschmieder szórási egyenlet végrehajtása
        arr = np.clip(arr, 0, 255).astype(np.uint8)                     # Érvényes 8-bites képtartományba vágás
        return Image.fromarray(arr)                                     # Visszaalakítás PIL képformátumba

    elif condition_type == "rain":                                      # Szintetikus esősávok szimulációja
        rain_layer = np.zeros((h, w), dtype=np.float32)                 # Üres esőmaszk létrehozása nullákkal
        num_drops = int(w * h * 0.001)                                  # Esőcseppek száma a felbontás arányában
        xs = np.random.randint(0, w, num_drops)                         # Véletlenszerű X koordináták sorsolása
        ys = np.random.randint(0, h, num_drops)                         # Véletlenszerű Y koordináták sorsolása
        lengths = np.random.randint(15, 30, num_drops)                  # Esőcsíkok véletlenszerű hossza
        for x, y, length in zip(xs, ys, lengths):                       # Esőcsíkok kirajzolása diszkrét vonalakként
            for step in range(length):                                  # Lépkedés az esőcsík mentén ferdén
                cur_y = min(h - 1, y + step)                            # Vertikális képhatár védelem
                cur_x = min(w - 1, x + int(step * 0.2))                 # Horizontális enyhe dőlésszög
                rain_layer[cur_y, cur_x] = 210.0                        # Fényes esőnyom pixelek beállítása
        import cv2                                                      # OpenCV modul dinamikus importálása
        rain_layer = cv2.GaussianBlur(rain_layer, (3, 7), 0)            # Optikai elmosás Gauss-szűrővel
        for ch in range(c):                                             # Végighaladunk az RGB színcsatornákon
            arr[:, :, ch] = np.maximum(arr[:, :, ch], rain_layer)       # Maximum-képzés az esőréteggel
        arr = np.clip(arr, 0, 255).astype(np.uint8)                     # 8-bites egésszé alakítás vágással
        return Image.fromarray(arr)                                     # PIL kép visszaadása

    return img                                                          # Alapesetben változatlan kép visszaadása


def build_coco_dataset(
    data_dir: str,
    output_coco_path: str,
    target_split: str = "test",
    max_images: int = 0,
    add_synthetic_adverse: bool = False
) -> Dict[str, Any]:
    """Build unified COCO dataset from ExDark files."""
    images_root = os.path.join(data_dir, "images")                      # Képek könyvtárának abszolút elérési útja
    annotations_root = os.path.join(data_dir, "annotations")            # Annotációk gyökérkönyvtára
    list_path = os.path.join(data_dir, "imageclasslist.txt")            # Hivatalos felosztásleíró fájl

    metadata = parse_imageclasslist(list_path)                          # Metaadatok betöltése a memóriába

    coco_images = []                                                    # COCO képrekordok listája
    coco_annotations = []                                               # COCO annotációs rekordok listája
    annotation_id = 1                                                   # Egyedi annotációszámláló 1-től indítva
    image_id = 1                                                        # Egyedi képszámláló 1-től indítva

    class_counts = {c: 0 for c in TARGET_CLASSES}                       # Osztályonkénti példányszámláló inicializálása
    light_counts = {name: 0 for name in LIGHT_CONDITIONS.values()}      # Fényviszonyonkénti képszámláló

    folders = [f for f in os.listdir(images_root)                       # Képmappák felderítése az images mappában
               if os.path.isdir(os.path.join(images_root, f))]          # Csak a valódi alkönyvtárakat tartjuk meg

    candidate_items = []                                                # Előzetesen szűrt jelölt képek tárolója
    for folder in sorted(folders):                                      # Determinisztikus, ábécé szerinti mappa bejárás
        folder_path = os.path.join(images_root, folder)                 # Osztálymappa elérési útja
        for file_name in sorted(os.listdir(folder_path)):               # Képfájlok rendezett bejárása
            if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):  # Csak a standard képformátumok szűrése
                continue                                                # Nem képfájl átugrása
            txt_name = file_name + ".txt"                               # Hozzá tartozó Dollar annotáció neve
            txt_path = os.path.join(annotations_root, folder, txt_name) # Annotációs fájl elérési útja a mappában
            if not os.path.exists(txt_path):                            # Ha nincs a mappában, fallback keresés
                txt_path = os.path.join(annotations_root, txt_name)     # Gyökér annotációs mappa ellenőrzése
                if not os.path.exists(txt_path):                        # Ha sehol sem található az annotáció
                    continue                                            # Képfájl kihagyása annotáció hiányában

            meta = metadata.get(file_name, {                            # Metaadat kinyerése alapértelmezett fallbackkel
                "split": "test", "light_name": "Ambient",               # Fallback: teszt felosztás, szórt fény
                "light_id": 2, "in_out": "Outdoor"                      # Fallback azonosítók
            })
            if target_split and meta["split"] != target_split:          # Csak a cél splitbe eső képek megtartása
                continue                                                # Nem illeszkedő felosztás átugrása

            candidate_items.append((folder, file_name, txt_path, meta)) # Jelölt rekord felvétele a listába

    print(f"[Infó] Összesen {len(candidate_items)} jelölt kép található a(z) '{target_split}' felosztásban.")  # Infó

    if max_images > 0 and len(candidate_items) > max_images:            # Ha részhalmazos mintavételezést kértek
        step = len(candidate_items) / max_images                        # Mintavételezési lépésköz kiszámítása
        candidate_items = [candidate_items[int(i * step)]               # Egyenletes eloszlású almintavételezés
                           for i in range(max_images)]                  # Cél elemszám legyártása
        print(f"[Infó] Mintavételezés {len(candidate_items)} képre csökkentve.")  # Almintavétel tényének közlése

    for folder, file_name, txt_path, meta in candidate_items:           # Végighaladunk a validált jelölt képeken
        img_rel_path = f"{folder}/{file_name}"                          # Relatív képútvonal a COCO formátumhoz
        img_abs_path = os.path.join(images_root, folder, file_name)     # Abszolút képútvonal a kép megnyitásához

        try:
            with Image.open(img_abs_path) as im:                        # Kép megnyitása a valós méretek olvasására
                width, height = im.size                                 # Valós felbontás (szélesség, magasság)
        except Exception as e:                                          # Sérült képfájl hibakezelése
            print(f"[Figyelmeztetés] Nem sikerült megnyitni a képet: {img_abs_path}: {e}")  # Hiba naplózása
            continue                                                    # Sérült kép eldobása a csővezetékből

        objs = parse_exdark_annotation_file(txt_path)                   # Annotációk beolvasása az adott képhez
        if not objs:                                                    # Ha a kép nem tartalmaz célosztályt
            continue                                                    # Tesztprotokoll: üres képek kihagyása

        light_name = meta["light_name"]                                 # Megvilágítási kategória neve
        light_counts[light_name] = light_counts.get(light_name, 0) + 1  # Megvilágítási statisztika növelése

        img_record = {                                                  # Standard COCO képrekord összeállítása
            "id": image_id,                                             # Egyedi képi azonosító
            "file_name": img_rel_path,                                  # Relatív elérési út
            "width": width,                                             # Kép szélessége pixelben
            "height": height,                                           # Kép magassága pixelben
            "light_condition": light_name,                              # ExDark fényviszony neve
            "light_id": meta["light_id"],                               # Numerikus fény azonosító
            "in_out": meta["in_out"],                                   # Beltér/kültér címke
            "split": meta["split"],                                     # Felosztási split megnevezése
            "adverse_condition": "low_light"                            # Kedvezőtlen fényviszony címke
        }
        coco_images.append(img_record)                                  # Képrekord mentése a COCO listába

        for obj in objs:                                                # Végigiterálunk a kép objektumain
            category_name = obj["class_name"]                           # Objektum kategóriájának neve
            class_counts[category_name] += 1                            # Kategória előfordulási statisztika növelése
            x, y, w, h = obj["bbox"]                                    # Doboz koordináták kicsomagolása
            x = max(0.0, min(float(width - 1), x))                      # X kezdőpont képkerethez vágása (clamping)
            y = max(0.0, min(float(height - 1), y))                     # Y kezdőpont képkerethez vágása (clamping)
            w = max(1.0, min(float(width - x), w))                      # Szélesség vágása a jobb szélhez
            h = max(1.0, min(float(height - y), h))                     # Magasság vágása az alsó szélhez

            ann_record = {                                              # COCO formátumú annotációs rekord
                "id": annotation_id,                                    # Egyedi doboz azonosító
                "image_id": image_id,                                   # Hivatkozott kép azonosítója
                "category_id": obj["category_id"],                      # Célosztály numerikus kódja
                "bbox": [round(x, 2), round(y, 2),                      # Lekerekített doboz koordináták [x, y, w, h]
                         round(w, 2), round(h, 2)],                     # Szélesség és magasság mentése
                "area": round(w * h, 2),                                # Doboz területe négyzetpixelben
                "iscrowd": 0                                            # Standard COCO: egyedi objektum jelölő
            }
            coco_annotations.append(ann_record)                         # Annotáció hozzáadása a listához
            annotation_id += 1                                          # Doboz-azonosító növelése

        image_id += 1                                                   # Kép-azonosító növelése

    categories = [                                                      # 4 standard COCO kategória deklarációja
        {"id": 1, "name": "person", "supercategory": "person"},         # Személy osztály
        {"id": 2, "name": "car", "supercategory": "vehicle"},           # Személyautó osztály
        {"id": 3, "name": "bus", "supercategory": "vehicle"},           # Busz osztály
        {"id": 4, "name": "motorbike", "supercategory": "vehicle"}      # Motorkerékpár osztály
    ]

    coco_data = {                                                       # Teljes COCO JSON fa struktúra
        "info": {                                                       # Metaadat blokk
            "description": "ExDark Adverse Condition Benchmark",        # Benchmark leírása
            "version": "1.0",                                           # Verziószám
            "year": 2026,                                               # Évszám
            "target_classes": TARGET_CLASSES                            # Célosztályok listája
        },
        "licenses": [],                                                 # Standard licenc mező
        "images": coco_images,                                          # Képek listája
        "annotations": coco_annotations,                                # Annotációk listája
        "categories": categories                                        # Kategóriák sémája
    }

    os.makedirs(os.path.dirname(output_coco_path), exist_ok=True)        # Kimeneti könyvtár létrehozása
    with open(output_coco_path, 'w', encoding='utf-8') as f:           # JSON fájl megnyitása írásra UTF-8 kódolással
        json.dump(coco_data, f, indent=2)                               # Adatstruktúra kiírása 2 szóközös behúzással

    print(f"\n[Kész] Az egységesített COCO adathalmaz elmentve ide: {output_coco_path}")  # Mentési siker
    print(f"       Összes kép: {len(coco_images)}")                     # Képszám összegzés
    print(f"       Összes annotáció: {len(coco_annotations)}")          # Annotációszám összegzés
    print(f"       Osztály-eloszlás: {class_counts}")                   # Osztályeloszlás statisztika
    print(f"       Megvilágítási eloszlás: {dict(light_counts)}")       # Megvilágítási statisztika

    return coco_data                                                    # Betöltött adatszerkezet visszaadása


def main():
    parser = argparse.ArgumentParser(                                   # Parancssori értelmező létrehozása
        description="ExDark adathalmaz előfeldolgozása egységes COCO formátumba."  # Parser célleírása
    )
    parser.add_argument("--data_dir", type=str, default="data/ExDark",  # ExDark adatkönyvtár kapcsoló
                        help="ExDark gyökérkönyvtár elérési útja.")     # Súgó leírás
    parser.add_argument("--output_file", type=str,                      # Kimeneti JSON elérési útja
                        default="data/ExDark/coco_annotations_test.json",  # Alapértelmezett kimenet
                        help="Kimeneti COCO JSON fájl helye.")          # Súgó leírás
    parser.add_argument("--split", type=str, default="test",            # Adatfelosztás (test/val/train/all)
                        help="Feldolgozandó adatfelosztás.")            # Súgó leírás
    parser.add_argument("--max_images", type=int, default=0,            # Képszám korlát paraméter
                        help="Maximálisan feldolgozandó képek száma.")  # 0 = mindegyik
    parser.add_argument("--seed", type=int, default=42,                 # Véletlenszám seed
                        help="Véletlenszám-generátor kezdőértéke.")     # Reprodukálhatóság
    args = parser.parse_args()                                          # Argumentumok feldolgozása

    set_seed(args.seed)                                                 # Determinisztikus seed beállítása
    split_to_use = None if args.split == "all" else args.split         # 'all' esetén nincs szűrés felosztásra
    build_coco_dataset(                                                 # COCO adatkészlet generáló hívása
        data_dir=args.data_dir,                                         # Adatkönyvtár átadása
        output_coco_path=args.output_file,                              # Kimeneti fájl átadása
        target_split=split_to_use,                                      # Cél felosztás átadása
        max_images=args.max_images                                      # Képszám korlát átadása
    )


if __name__ == "__main__":
    main()                                                              # Belépési pont futtatása
