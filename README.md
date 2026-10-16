# Gyenge megvilágítású objektumdetektálás és VLM benchmark (ExDark adathalmaz)

Egy Python-alapú kísérleti keretrendszer, amely három előtanított objektumdetektort (**YOLO11**, **RT-DETR**, **Faster R-CNN**) és egy látás-nyelvi modellt (**Qwen2-VL**) hasonlít össze az [ExDark (Exclusively Dark)](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) adathalmazon, kedvezőtlen, gyenge megvilágítási körülmények között.

A benchmark szigorúan négy dinamikus kategóriára összpontosít: **`person`**, **`car`**, **`bus`** és **`motorbike`**.

---

## Projektstruktúra

```
project/
├── data/
│   └── ExDark/                       # Képek, annotációk és egységesített COCO formátum
├── models/
│   ├── yolo11/                       # Előtanított YOLO11 súlyok (yolo11n.pt)
│   ├── rtdetr/                       # Előtanított RT-DETR súlyok (rtdetr-l.pt)
│   └── faster_rcnn/                  # Előtanított Faster R-CNN súlyok (fasterrcnn_resnet50_fpn.pth)
├── src/
│   ├── download_data.py              # ExDark képek és annotációk letöltése és gyorsítótárazása
│   ├── preprocess.py                 # Annotációk konvertálása egységes COCO teszt formátumra
│   ├── detect_yolo.py                # YOLO11 inferencia és késleltetés mérése
│   ├── detect_rtdetr.py              # RT-DETR inferencia és késleltetés mérése
│   ├── detect_fasterrcnn.py          # Faster R-CNN inferencia és késleltetés mérése
│   ├── vlm_inference.py              # Kettős szerepkörű VLM inferencia (Detektor + Magyarázó)
│   ├── vlm_smoketest.py              # Többlépcsős diagnosztikai és biztonsági füstteszt
│   ├── evaluate.py                   # mAP, PRF1, megvilágítási bontás és diagramok számítása
│   ├── compare_vlm_detectors.py      # VLM és detektorok egyetértése, hőtérképek, vizuális átfedések
│   └── utils.py                      # Megosztott segédfüggvények, osztályleképezések, vizualizáció
├── results/
│   ├── figures/                      # Publikációs ábrák és vizuális átfedés minták
│   ├── metrics/                      # JSON és CSV kiértékelési metrikák
│   └── vlm_outputs/                  # Nyers VLM detekciók és minőségi magyarázatok
├── .gitignore                        # Git kizárási szabályok
├── requirements.txt                  # Python függőségek
├── run_full_vlm.ps1                  # Teljes VLM futtató PowerShell szkript
├── run_full_vlm.sh                   # Teljes VLM futtató Bash szkript
├── run_smoketest.bat                 # Füstteszt futtató Windows Batch szkript
├── run_smoketest.sh                  # Füstteszt futtató Bash szkript
└── README.md                         # Dokumentáció és reprodukálási útmutató
```

---

## Gyorsindítás és telepítés

### 1. Klónozás és függőségek telepítése

Győződj meg róla, hogy a Python 3.10+ és a CUDA-támogatással rendelkező PyTorch telepítve van:

```bash
git clone <repo-url>
cd project
pip install -r requirements.txt
```

---

## VLM füstteszt (Smoke Test)

A VLM teljes adathalmazon történő futtatása előtt mindig futtasd le a következőt:

```bash
./run_smoketest.sh
```

Ez a VLM-et 1, 5, majd 20 képen futtatja le, szigorú időkereteket betartva.
Ha bármelyik szakasz elbukik, NE folytasd a teljes adathalmazzal. A részleteket lásd:
`results/smoketest_report.json`.

Opcionális füstteszt konfigurációk:

```bash
# Explicit helyi modell futtatás
./run_smoketest.sh --backend local --model Qwen/Qwen2-VL-2B-Instruct

# API-alapú futtatás (automatikusan észleli a kulcsokat)
./run_smoketest.sh --backend api --provider openai --model gpt-4o-mini

# Egyenértékű Python parancs
python src/vlm_inference.py --smoketest
```

---

### 2. A teljes csővezeték futtatása

A teljes folyamat lépésről lépésre, egymás után futtatható:

```bash
# 1. lépés: Az ExDark adathalmaz letöltése és kicsomagolása
python src/download_data.py

# 2. lépés: Előfeldolgozás egységes COCO formátumba (szűrve person, car, bus, motorbike osztályokra)
python src/preprocess.py

# 3. lépés: YOLO11 detektor futtatása
python src/detect_yolo.py

# 4. lépés: RT-DETR detektor futtatása
python src/detect_rtdetr.py

# 5. lépés: Faster R-CNN detektor futtatása
python src/detect_fasterrcnn.py

# 6. lépés: Látás-nyelvi modell (VLM) inferencia futtatása (Qwen2-VL 4-bites kötegelt vagy teljes adathalmaz)
python -u src/vlm_inference.py --coco_json data/ExDark/coco_annotations_all.json --limit 4000 --batch_size 4 --single_role --load_in_4bit

# 7. lépés: COCO metrikák kiértékelése és összehasonlító ábrák generálása
python src/evaluate.py

# 8. lépés: VLM és a detektorok összehasonlítása, vizuális átfedések generálása
python src/compare_vlm_detectors.py
```

---

## A csővezeték komponensei

### 1. Adathalmaz előfeldolgozása (`src/preprocess.py`)
- A nyers ExDark annotációkat a hivatalos COCO JSON formátumra szabványosítja.
- Az annotációkat szigorúan a következő osztályokra szűri: `person`, `car`, `bus` és `motorbike`.
- Hozzárendeli a megvilágítási metaadatokat a 10 megvilágítási körülmény alapján (`Low`, `Ambient`, `Object`, `Single`, `Weak`, `Strong`, `Screen`, `Window`, `Shadow`, `Twilight`).

### 2. Előtanított detektorok
- **YOLO11 (`src/detect_yolo.py`):** Horgonymentes (anchor-free), egylépcsős CNN-transzformer hibrid.
- **RT-DETR (`src/detect_rtdetr.py`):** Valós idejű látás-transzformer skálán belüli interakcióval (AIFI) és skálák közötti fúzióval (CCFM).
- **Faster R-CNN (`src/detect_fasterrcnn.py`):** Kétlépcsős RPN ResNet-50 Feature Pyramid Networkkel (FPN).
- Minden detekció egységesítve van az alábbi formátumra:
  ```json
  [{"bbox": [x1, y1, x2, y2], "label": "car", "score": 0.91}, ...]
  ```

### 3. VLM integráció (`src/vlm_inference.py`)
A **Qwen2-VL-2B-Instruct** modellt két egymást kiegészítő szerepkörben használja:
- **1. szerepkör (Önálló detektor):** Gyalogosok és járművek zero-shot felismerése és hozzávetőleges határolókeretes lokalizációja.
- **2. szerepkör (Kritikai magyarázó):** Megkapja a klasszikus detektorok előrejelzéseit, és diagnosztizálja a gyenge megvilágítás miatti hibamódokat (kihagyott objektumok, tükröződési műtermékek, árnyéktakarás).
- Szigorú biztonsági korlátok:
  - A felbontás legfeljebb 768x768 pixelre korlátozva (képarány megőrzésével)
  - Korlátozott vizuális tokenek (`min_pixels` / `max_pixels`)
  - Szigorú tokenkeretek (`max_new_tokens <= 128` az 1. szerepkörhöz, `<= 256` a 2. szerepkörhöz)
  - Képenkénti 60 másodperces időkorlát automatikus ellenőrzőpont-mentéssel (checkpointing)

### 4. Metrikák és kiértékelés (`src/evaluate.py`)
Kiszámítja:
- `mAP@0.50` és `mAP@0.50:0.95` a `pycocotools.COCOeval` segítségével
- Kategóriánkénti precízió (Precision), visszahívás (Recall) és F1-pontszám
- Megvilágítási robusztussági bontás mind a 10 megvilágítási körülményre
- GPU és CPU késleltetés (latency, ms) és áteresztőképesség (FPS)

### 5. VLM és a detektorok összehasonlítása (`src/compare_vlm_detectors.py`)
Kiszámítja:
- Egyetértési konfúziós mátrixok (Konszenzus, VLM-exkluzív, Detektor-exkluzív, Kölcsönös kihagyások)
- Egyetértési hőtérkép (`results/figures/vlm_detector_agreement_heatmap.png`)
- Színkódolt vizuális átfedések (`results/figures/overlay_samples.png`):
  - **Zöld:** Valós adat (Ground Truth)
  - **Ciánkék:** YOLO11
  - **Arany/Sárga:** RT-DETR
  - **Narancssárga:** Faster R-CNN
  - **Lila:** VLM

---

## Benchmark összefoglaló

| Modell | mAP@0.50 | mAP@0.50:0.95 | Precízió@0.50 | Visszahívás@0.50 | F1-pontszám | GPU késleltetés | GPU FPS | CPU FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11** | 0.6078 | 0.3710 | **0.7816** | 0.6393 | **0.7007** | **17.55 ms** | **56.98** | **23.34** |
| **RT-DETR** | **0.8157** | **0.4846** | 0.5028 | **0.8784** | 0.6327 | 36.32 ms | 27.53 | 3.01 |
| **Faster R-CNN** | 0.6793 | 0.3560 | 0.4644 | 0.7709 | 0.5705 | 80.84 ms | 12.37 | 0.77 |
