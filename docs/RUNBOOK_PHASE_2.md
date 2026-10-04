# Phase 2 Runbook

## 1. Download and place the dataset

1. Download RDD2022 from its official source and keep the download/version details.
2. Extract it into `data/raw/rdd2022/`.
3. Confirm that image files and matching Pascal VOC XML files exist.
4. Do not upload raw images, annotations, credentials, or model weights to GitHub.

## 2. Create an isolated environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Prepare data

```powershell
python src/data/prepare_rdd2022.py --input data/raw/rdd2022 --output data/processed/rdd2022_yolo --seed 42
python src/data/make_classification_crops.py --yolo-root data/processed/rdd2022_yolo --output data/processed/rdd2022_crops
```

Check the generated counts and open sample images before training.

## 4. Run the two Phase 2 baselines

YOLOv8:

```powershell
python src/models/train_yolo.py --data data/processed/rdd2022_yolo/data.yaml --epochs 30 --batch 16
```

Classification:

```powershell
python src/models/train_classifier.py --data data/processed/rdd2022_crops --architecture resnet50 --epochs 5 --batch 32
```

Use smaller batch sizes if memory is insufficient. Record the exact command and hardware in the interim report.

## 5. What Lakshya must understand

- Why the classifier receives labelled crops rather than detector mAP inputs.
- What transfer learning and the final classification layer do.
- What cross-entropy loss, precision, recall, and macro F1 mean.
- How the train/validation/test split prevents leakage.
- How to read the confusion matrix and explain errors.

Never copy the metrics from another run or another paper. The viva will test whether the owner understands their model.
