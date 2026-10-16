# Low-Light Object Detection & VLM Benchmark (ExDark Dataset)

A Python-based experimental framework comparing three pretrained object detectors (**YOLO11**, **RT-DETR**, **Faster R-CNN**) and a Vision-Language Model (**Qwen2-VL**) on the [ExDark (Exclusively Dark)](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) dataset under adverse low-light illumination conditions.

The benchmark focuses strictly on four dynamic categories: **`person`**, **`car`**, **`bus`**, and **`motorbike`**.

---

## Project Structure

```
project/
├── data/
│   └── ExDark/                       # Dataset images, annotations, and unified COCO format
├── models/
│   ├── yolo11/                       # Pretrained YOLO11 weights (yolo11n.pt)
│   ├── rtdetr/                       # Pretrained RT-DETR weights (rtdetr-l.pt)
│   └── faster_rcnn/                  # Pretrained Faster R-CNN weights
├── src/
│   ├── download_data.py              # Download & cache ExDark images and annotations
│   ├── preprocess.py                 # Convert annotations to unified COCO test format
│   ├── detect_yolo.py                # YOLO11 inference and latency benchmarking
│   ├── detect_rtdetr.py              # RT-DETR inference and latency benchmarking
│   ├── detect_fasterrcnn.py          # Faster R-CNN inference and latency benchmarking
│   ├── vlm_inference.py              # Dual-role VLM inference (Detector + Explainer)
│   ├── evaluate.py                   # Compute mAP, PRF1, illumination breakdown, and plots
│   ├── compare_vlm_detectors.py      # VLM vs detector agreement, heatmaps, visual overlays
│   └── utils.py                      # Shared utilities, class mappings, visualization
├── results/
│   ├── figures/                      # Publication plots and visual overlay samples
│   ├── metrics/                      # JSON & CSV evaluation metrics
│   └── vlm_outputs/                  # Raw VLM detections and qualitative critiques
├── report/
│   └── report.md                     # Full scientific report with figures and tables
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation and reproduction guide
```

---

## Quickstart & Installation

### 1. Clone & Install Dependencies

Ensure Python 3.10+ and PyTorch with CUDA support are installed:

```bash
git clone <repo-url>
cd project
pip install -r requirements.txt
```

---

## VLM Smoke Test

Before running the VLM on the full dataset, always run:

    ./run_smoketest.sh

This runs the VLM on 1, 5, then 20 images with strict timing budgets.
If any stage fails, DO NOT proceed to the full dataset. See
results/smoketest_report.json for details.

Optional smoke test configurations:

```bash
# Explicit local model execution
./run_smoketest.sh --backend local --model Qwen/Qwen2-VL-2B-Instruct

# API-based execution (auto-detects keys)
./run_smoketest.sh --backend api --provider openai --model gpt-4o-mini

# Equivalent Python command
python src/vlm_inference.py --smoketest
```

---

### 2. Run the Full Pipeline

The complete pipeline can be executed sequentially:

```bash
# Step 1: Download & extract ExDark dataset
python src/download_data.py

# Step 2: Preprocess into unified COCO format (filtered to person, car, bus, motorbike)
python src/preprocess.py

# Step 3: Run YOLO11 detector
python src/detect_yolo.py

# Step 4: Run RT-DETR detector
python src/detect_rtdetr.py

# Step 5: Run Faster R-CNN detector
python src/detect_fasterrcnn.py

# Step 6: Run Vision-Language Model inference (Qwen2-VL 4-bit batched or full dataset)
python -u src/vlm_inference.py --coco_json data/ExDark/coco_annotations_all.json --limit 4000 --batch_size 4 --single_role --load_in_4bit

# Step 7: Evaluate COCO metrics & generate comparison figures
python src/evaluate.py

# Step 8: Compare VLM vs detectors & generate visual overlays
python src/compare_vlm_detectors.py
```

---

## Pipeline Components

### 1. Dataset Preprocessing (`src/preprocess.py`)
- Standardizes raw ExDark annotations into the official COCO JSON format.
- Filters annotations strictly to `person`, `car`, `bus`, and `motorbike`.
- Attaches illumination metadata across the 10 lighting conditions (`Low`, `Ambient`, `Object`, `Single`, `Weak`, `Strong`, `Screen`, `Window`, `Shadow`, `Twilight`).

### 2. Pretrained Detectors
- **YOLO11 (`src/detect_yolo.py`):** Anchor-free single-stage CNN-transformer hybrid.
- **RT-DETR (`src/detect_rtdetr.py`):** Real-time Vision Transformer with intra-scale interaction (AIFI) and cross-scale fusion (CCFM).
- **Faster R-CNN (`src/detect_fasterrcnn.py`):** Two-stage RPN with ResNet-50 Feature Pyramid Network.
- All detections are normalized to:
  ```json
  [{"bbox": [x1, y1, x2, y2], "label": "car", "score": 0.91}, ...]
  ```

### 3. VLM Integration (`src/vlm_inference.py`)
Uses **Qwen2-VL-2B-Instruct** in two complementary roles:
- **Role 1 (Independent Detector):** Zero-shot identification and approximate bounding box localization of pedestrians and vehicles.
- **Role 2 (Critical Explainer):** Receives the classic detectors' predictions and diagnoses low-light failure modes (missed objects, glare artifacts, shadow occlusion).
- Hard safety limits:
  - Resolution capped to max 768x768 (preserves aspect ratio)
  - Bounded visual tokens (`min_pixels` / `max_pixels`)
  - Strict token budgets (`max_new_tokens <= 128` for Role 1, `<= 180` for Role 2)
  - 60s per-image timeout with automatic checkpointing

### 4. Metrics & Evaluation (`src/evaluate.py`)
Computes:
- `mAP@0.50` and `mAP@0.50:0.95` via `pycocotools.COCOeval`
- Per-category Precision, Recall, and F1-score
- Illumination robustness breakdown across 10 lighting conditions
- GPU and CPU latency (ms) and throughput (FPS)

### 5. VLM vs Detector Comparison (`src/compare_vlm_detectors.py`)
Computes:
- Agreement confusion matrices (Consensus, VLM exclusive, Detector exclusive, Mutual misses)
- Agreement heatmap (`results/figures/vlm_detector_agreement_heatmap.png`)
- Color-coded visual overlays (`results/figures/overlay_samples.png`):
  - **Green:** Ground Truth
  - **Cyan:** YOLO11
  - **Gold/Yellow:** RT-DETR
  - **Orange:** Faster R-CNN
  - **Purple:** VLM

---

## Benchmark Summary

| Model | mAP@0.50 | mAP@0.50:0.95 | Precision@0.50 | Recall@0.50 | F1-Score | GPU Latency | GPU FPS | CPU FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11** | 0.6078 | 0.3710 | **0.7816** | 0.6393 | **0.7007** | **17.55 ms** | **56.98** | **23.34** |
| **RT-DETR** | **0.8157** | **0.4846** | 0.5028 | **0.8784** | 0.6327 | 36.32 ms | 27.53 | 3.01 |
| **Faster R-CNN** | 0.6793 | 0.3560 | 0.4644 | 0.7709 | 0.5705 | 80.84 ms | 12.37 | 0.77 |
