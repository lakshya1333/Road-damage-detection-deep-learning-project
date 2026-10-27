# Road Damage Detection & Classification — ICT-4442 Final

Four deep-learning families compared on **RDD2022** (D00 longitudinal, D10 transverse, D20 alligator, D40 pothole) under **one split and one common protocol**.

| Model | Family | Owner | Script | Kaggle notebook |
|---|---|---|---|---|
| A — ResNet-50 | CNN classifier | Lakshya Jain | `src/models/train_classifier.py --arch resnet50` | `notebooks/01_resnet50_classifier_Lakshya.ipynb` |
| B — YOLOv8s | One-stage detector | Dhruv Maheshwari | `src/models/train_yolo.py` | `notebooks/02_yolov8_detector_Dhruv.ipynb` |
| C — U-Net (ResNet-34) | Encoder–decoder segmentation | Kumar Kshitij | `src/models/train_unet.py` | `notebooks/03_unet_segmentation_Kumar.ipynb` |
| D — ViT-B/16 | Attention / transformer | MANAS LAXMAN KUDIPADY | `src/models/train_classifier.py --arch vit_b_16` | `notebooks/04_vit_transformer_Manas.ipynb` |

## Repository layout

```
src/common/dataset.py        shared data contract: dataset discovery, class mapping, manifest, crops
src/common/metrics.py        common-protocol metrics, IoU, segmentation scores, latency
src/data/build_manifest.py   STEP 1 for everyone: identical split + class map → results/data/
src/models/                  one training script per model family (writes results/<model>/)
src/evaluation/compare_models.py   single comparison table, McNemar tests, error analysis, figures
src/inference/video_infer.py dashcam/video monitoring with the trained detector
notebooks/                   Kaggle notebooks (one per member + final comparison)
reporting/                   builds the Part C report (.docx) and slides (.pptx) from results/
```

## Common evaluation protocol

Every ground-truth damage instance of the shared **test** split gets one predicted class:

* **Classifiers** (ResNet-50, ViT): the GT crop (+10 % context) is classified.
* **YOLOv8**: class of the predicted box with the highest IoU, if IoU ≥ 0.5, else *missed*.
* **U-Net**: majority predicted damage class inside the GT box, if ≥ 5 % of the box is damage, else *missed*.

Missed instances count as false negatives. We report accuracy, macro precision/recall/F1, miss rate, per-class F1, params and batch-1 latency, plus native metrics (mAP for YOLOv8, mIoU/Dice for U-Net).

## Reproduce (Kaggle, GPU T4, dataset `aliabdelmenam/rdd-2022`)

```bash
pip install -r requirements.txt
python -m src.data.build_manifest --out results/data --plot                     # everyone, same output
python -m src.models.train_classifier --arch resnet50 --out results/resnet50     # A
python -m src.models.train_yolo --model yolov8s.pt --epochs 30 --out results/yolov8   # B
python -m src.models.train_unet --epochs 15 --size 512 --out results/unet        # C
python -m src.models.train_classifier --arch vit_b_16 --lr 5e-5 --out results/vit_b16 # D
python -m src.evaluation.compare_models --results results --out results/final --gallery
python reporting/build_report.py  --results results --out ICT4442_Final_Report.docx
python reporting/build_slides.py  --results results --out ICT4442_Final_Presentation.pptx
```

All randomness is seeded (42). Do not commit the dataset or `*.pt` weights (see `.gitignore`); commit `results/**/metrics.json`, CSVs and PNGs.

## Phase-2 history

`src/models/rdd2022-compare-base-models.ipynb` and `yolov8-on-rdd2022.ipynb` are the Phase-2 runs. **Note:** the “Proposed Dual-Branch” row printed in `rdd2022-compare-base-models.ipynb` was *not* produced by a training run (the notebook prints “NO RE-TRAINING PERFORMED”) and must not be reported.
