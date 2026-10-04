# Road Damage Detection - ICT-4442 Phase 2

This repository contains the reproducible preprocessing and baseline-model work for the ICT-4442 road-damage project.

## Team ownership

- Lakshya Jain: ResNet-50/MobileNetV2 classification model
- Dhruv Maheshwari: YOLOv8 object detector
- Kumar Kshitij: preprocessing, splits, and segmentation investigation
- MANAS LAXMAN KUDIPADY: evaluation, transformer investigation, and report integration

## Dataset

Use the official RDD2022 release. Download it separately and place the extracted dataset under `data/raw/rdd2022/`. Do not commit the raw images or annotations. The preparation script searches recursively for `.jpg` images and matching Pascal VOC `.xml` files, so the original country/train folder layout can be retained.

The dataset contains four labels: D00 (longitudinal crack), D10 (transverse crack), D20 (alligator crack), and D40 (pothole).

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare the detector dataset

```bash
python src/data/prepare_rdd2022.py --input data/raw/rdd2022 --output data/processed/rdd2022_yolo --seed 42
```

This creates a YOLO directory, `data.yaml`, `splits.csv`, and `dataset_stats.csv`. Review the generated sample images before training.

## Reproducibility rules

- Keep seed `42` unless the team documents a change.
- Never place test images in training folders.
- Record Python, PyTorch, Ultralytics, device, image size, epochs, batch size, and training time.
- Do not report metrics until they are produced by a saved run.
