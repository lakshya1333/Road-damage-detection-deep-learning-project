# Phase 2 Interim Report Evidence

Replace every bracketed field with verified information before submission. Do not leave placeholder metrics in the final report.

## Team information

- Team number: [TEAM NUMBER]
- Members: Lakshya Jain [REGISTRATION NUMBER], Dhruv Maheshwari [REGISTRATION NUMBER], Kumar Kshitij [REGISTRATION NUMBER], MANAS LAXMAN KUDIPADY [REGISTRATION NUMBER]
- GitHub repository: [URL]
- Extended submission deadline: [DATE AND TIME]

## 1. Literature review

| Paper (author, year) | Method | Dataset | Key result | Relevance to this project |
|---|---|---|---|---|
| Arya et al. (2022) | RDD2022 dataset | RDD2022 | [Verified result] | Direct benchmark and annotation source |
| Arya et al. (2021) | RDD2020 dataset | RDD2020 | [Verified result] | Earlier road-damage benchmark |
| Zeng and Zhong (2024) | YOLOv8-PD | [Verify dataset] | [Verify result] | Detector comparison |
| Wang et al. (2025) | YOLO-RD | [Verify dataset] | [Verify result] | Recent detection method |
| Lau et al. (2020) | U-Net CNN | [Verify dataset] | [Verify result] | Segmentation baseline |
| Yu et al. (2023) | RUC-Net | [Verify dataset] | [Verify result] | Crack segmentation approach |
| Zhang and Zhang (2024) | Transformer/UNet | [Verify dataset] | [Verify result] | Attention-based model motivation |
| [Paper 8] | [Method] | [Dataset] | [Verified result] | [Relevance] |
| [Paper 9] | [Method] | [Dataset] | [Verified result] | [Relevance] |
| [Paper 10, optional] | [Method] | [Dataset] | [Verified result] | [Relevance] |

## 2. Dataset acquisition and preprocessing

RDD2022 was downloaded from: [OFFICIAL URL]. Download/version date: [DATE]. The original images and Pascal VOC XML annotations are kept outside Git. The preparation command was:

```text
python src/data/prepare_rdd2022.py --input data/raw/rdd2022 --output data/processed/rdd2022_yolo --seed 42
```

- Images discovered: [NUMBER]
- Usable annotated images: [NUMBER]
- Train/validation/test images: [NUMBER] / [NUMBER] / [NUMBER]
- D00/D10/D20/D40 object counts: [NUMBERS]
- Input size: [VALUE]
- Augmentation: [ACTUAL SETTINGS]
- Leakage control: split was performed before crop generation; source grouping: [COUNTRY/FOLDER]

Attach or link to `splits.csv`, `dataset_stats.csv`, and sample annotation visualizations.

## 3. Models implemented so far

### 3.1 Classification baseline

Owner: Lakshya Jain. Architecture: [ResNet-50 or MobileNetV2]. Training command: [COMMAND]. Device: [CPU/GPU]. Epochs: [NUMBER]. Seed: [NUMBER].

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | [VALUE] | [VALUE] |
| Macro precision | [VALUE] | [VALUE] |
| Macro recall | [VALUE] | [VALUE] |
| Macro F1 | [VALUE] | [VALUE] |

Evidence: [model path], [history path], [confusion matrix path].

### 3.2 YOLOv8 detector

Owner: Dhruv Maheshwari. Model: [yolov8n/s]. Training command: [COMMAND]. Device: [CPU/GPU]. Epochs: [NUMBER]. Seed: [NUMBER].

| Metric | Test |
|---|---:|
| Precision | [VALUE] |
| Recall | [VALUE] |
| mAP@0.5 | [VALUE] |
| mAP@0.5:0.95 | [VALUE] |

Evidence: [run directory], [prediction images], [results CSV].

Classification accuracy and detector mAP are reported separately because they measure different tasks.

## 4. Risks and plan to final

| Risk | Impact | Mitigation | Owner | Target date |
|---|---|---|---|---|
| RDD2022 has boxes but no true pixel masks | U-Net cannot claim supervised segmentation without another mask source | Locate a compatible mask dataset or document weak-mask limitation | Kumar | [DATE] |
| Limited GPU time | Fewer tuning runs | Use fixed small baseline first and record hardware | Lakshya/Dhruv | [DATE] |
| Country/domain shift | Generalization may be overstated | Report country/group split and limitation | Manas | [DATE] |
| Video-frame leakage | Inflated test performance | Keep video sequences out of test or split by source video | Team | [DATE] |
| Class imbalance | Poor minority-class recall | Report per-class metrics and use weighted sampling if justified | Lakshya | [DATE] |
