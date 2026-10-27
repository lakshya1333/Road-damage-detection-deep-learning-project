"""Shared report/slide content: team, references, literature table, interim (Phase 2) results.

Every number in INTERIM comes from the team's saved Phase-2 Kaggle runs
(src/models/rdd2022-compare-base-models.ipynb, src/models/yolov8-on-rdd2022.ipynb).
Final numbers are NEVER typed here; they are read from results/ by the build scripts.
"""

TITLE = "Deep Learning-Based Road Damage Detection and Classification for Automated Road Condition Monitoring"
SHORT_TITLE = "Road Damage Detection with Deep Learning"
COURSE = "Deep Learning Project [ICT-4442]"
MONTH_YEAR = "October 2026"
REPO_URL = "https://github.com/lakshya1333/Road-damage-detection-deep-learning-project"

TEAM = [  # name, reg no, model key, model owned, presentation part
    ("Lakshya Jain", "230953334", "resnet50", "ResNet-50 CNN classifier (Model A)"),
    ("Dhruv Maheshwari", "230953526", "yolov8", "YOLOv8 one-stage detector (Model B)"),
    ("Kumar Kshitij", "230953488", "unet", "U-Net encoder-decoder segmentation (Model C)"),
    ("MANAS LAXMAN KUDIPADY", "230953530", "vit_b16", "ViT-B/16 attention-based classifier (Model D)"),
]

KEYWORDS = ["road damage detection", "RDD2022", "convolutional neural networks", "YOLOv8", "U-Net",
            "vision transformer", "transfer learning", "pavement condition monitoring"]

# IEEE-style references, numbered in order of first citation in the report.
REFERENCES = [
    'D. Arya, H. Maeda, S. K. Ghosh, D. Toshniwal, and Y. Sekimoto, "RDD2022: A multi-national image dataset for automatic road damage detection," arXiv:2209.08538, 2022.',
    'H. Maeda, Y. Sekimoto, T. Seto, T. Kashiyama, and H. Omata, "Road damage detection and classification using deep neural networks with smartphone images," Computer-Aided Civil and Infrastructure Engineering, vol. 33, no. 12, pp. 1127-1141, 2018.',
    'D. Arya, H. Maeda, S. K. Ghosh, D. Toshniwal, and Y. Sekimoto, "RDD2020: An annotated image dataset for automatic road damage detection using deep learning," Data in Brief, vol. 36, Art. no. 107133, 2021.',
    'D. Arya, H. Maeda, S. K. Ghosh, D. Toshniwal, A. Mraz, T. Kashiyama, and Y. Sekimoto, "Deep learning-based road damage detection and classification for multiple countries," Automation in Construction, vol. 132, Art. no. 103935, 2021 (preprint arXiv:2008.13101).',
    'L. Zhang, F. Yang, Y. D. Zhang, and Y. J. Zhu, "Road crack detection using deep convolutional neural network," in Proc. IEEE Int. Conf. Image Processing (ICIP), 2016, pp. 3708-3712.',
    'J. Zeng and H. Zhong, "YOLOv8-PD: An improved road damage detection algorithm based on YOLOv8n model," Scientific Reports, vol. 14, Art. no. 12367, 2024.',
    'W. Wang, X. Yu, B. Jing, Z. Tang, W. Zhang, S. Wang, Y. Xiao, S. Li, and L. Yang, "YOLO-RD: A road damage detection method for effective pavement maintenance," Sensors, vol. 25, no. 5, Art. no. 1442, 2025.',
    'S. L. H. Lau, E. K. P. Chong, X. Yang, and X. Wang, "Automated pavement crack segmentation using U-Net-based convolutional neural network," IEEE Access, vol. 8, pp. 114892-114899, 2020.',
    'G. Yu, J. Dong, Y. Wang, and X. Zhou, "RUC-Net: A residual-Unet-based convolutional neural network for pixel-level pavement crack segmentation," Sensors, vol. 23, no. 1, Art. no. 53, 2023.',
    'H. Liu, X. Miao, C. Mertz, C. Xu, and H. Kong, "CrackFormer: Transformer network for fine-grained crack detection," in Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV), 2021, pp. 3783-3792.',
    'Y. Zhang and L. Zhang, "Detection of pavement cracks by deep learning models of Transformer and UNet," IEEE Trans. Intelligent Transportation Systems, vol. 25, no. 11, 2024 (preprint arXiv:2304.12596).',
    'K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778.',
    'M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted residuals and linear bottlenecks," in Proc. IEEE/CVF CVPR, 2018, pp. 4510-4520.',
    'J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You only look once: Unified, real-time object detection," in Proc. IEEE CVPR, 2016, pp. 779-788.',
    'G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," version 8.0, 2023. [Online]. Available: https://github.com/ultralytics/ultralytics',
    'O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in Proc. MICCAI, LNCS vol. 9351, 2015, pp. 234-241.',
    'A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in Proc. Int. Conf. Learning Representations (ICLR), 2021.',
    'J. Chen et al., "TransUNet: Transformers make strong encoders for medical image segmentation," arXiv:2102.04306, 2021.',
    'I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in Proc. ICLR, 2019.',
    'F. Milletari, N. Navab, and S.-A. Ahmadi, "V-Net: Fully convolutional neural networks for volumetric medical image segmentation," in Proc. Int. Conf. 3D Vision (3DV), 2016, pp. 565-571.',
]

# Literature table (Paper, Method, Dataset, Key result, Relevance). Key results verified from the papers.
LIT_TABLE = [
    ("Maeda et al., 2018 [2]", "SSD-Inception / SSD-MobileNet detectors", "9,053 Japanese smartphone images, 8 classes",
     "SSD-MobileNet best; ~1.5 s/image on a smartphone, strong class-wise variation in recall",
     "First large road-damage benchmark; motivates lightweight one-stage detectors"),
    ("Arya et al., 2021 [3]", "RDD2020 dataset", "26,336 images (India, Japan, Czech)",
     "Four-class (D00/D10/D20/D40) labelling scheme reused by RDD2022", "Defines our class taxonomy"),
    ("Arya et al., 2021 [4]", "SSD-MobileNet + transfer learning across countries", "26,620 images, 3 countries",
     "Japan-only model fails on Indian roads (F1 ≤ 0.05 for D00); mixing local data helps", "Explains domain shift in multi-country RDD2022"),
    ("Arya et al., 2022 [1]", "RDD2022 / CRDDC'2022 dataset", "47,420 images, 6 countries, >55,000 instances",
     "Public multi-national benchmark with VOC boxes", "Our primary dataset"),
    ("Zhang et al., 2016 [5]", "Patch-level ConvNet classifier", "Smartphone pavement images (500)",
     "CNN outperformed hand-crafted SVM/boosting features", "Basis for our CNN-classification family"),
    ("Wang et al., 2025 [7]", "YOLO-RD: YOLOv8 + SOM, MAF, WTC modules", "RDD2022 (Japan subset)",
     "mAP 25.75%; small-object mAP +4.93 pts over YOLOv8; 20.89 M params", "Recent YOLOv8-based detector; shows difficulty of small cracks"),
    ("Lau et al., 2020 [8]", "U-Net with pretrained ResNet-34 encoder", "CFD, Crack500",
     "F1 = 96% (CFD), 73% (Crack500)", "Direct template for our U-Net (ResNet-34 encoder)"),
    ("Yu et al., 2023 [9]", "RUC-Net: residual U-Net + scSE attention + focal loss", "CFD, Crack500, DeepCrack",
     "CFD: P 0.713, R 0.768, F1 0.739, IoU 0.586", "Shows benefit of residual encoders and imbalance-aware loss"),
    ("Liu et al., 2021 [10]", "CrackFormer: self-attention encoder-decoder", "Three crack datasets",
     "ODS F-score 0.871 / 0.877 / 0.881", "Evidence that attention captures long, thin crack context"),
    ("Zhang & Zhang, 2024 [11]", "9 models: CNN U-Nets vs TransUNet, SwinUNet, MTUNet", "711 pavement images (224 px)",
     "Transformers best (MTUNet test IoU 80.8%, F1 89.4%) but slower and heavier", "Motivates our attention-based model and the speed trade-off"),
]

# Real Phase-2 results (validation split, Kaggle 2x T4). Copied from the saved notebook outputs.
INTERIM = {
    "note": "Phase-2 runs on Kaggle (2x Tesla T4), validation split of the Kaggle RDD2022 RDD_SPLIT, 30 epochs, crop classification.",
    "classifiers": [  # model, params(M), val acc, val macro P, val macro R, val macro F1, minutes
        ("ResNet-50", 23.52, 78.36, 72.72, 72.16, 72.41, 78.6),
        ("MobileNetV3-Large", 2.98, 76.99, 71.19, 70.90, 71.03, 41.3),
        ("EfficientNet-B0", 4.02, 77.86, 71.89, 71.84, 71.85, 55.3),
        ("DenseNet-121", 6.96, 78.27, 72.75, 73.06, 72.90, 88.9),
        ("ConvNeXt-Tiny", 27.82, 79.39, 74.11, 74.52, 74.29, 85.7),
    ],
    "yolo": {"model": "YOLOv8 (25.8 M params, 640 px, 50 epochs, 5-class export)", "precision": 0.671, "recall": 0.594,
             "mAP50": 0.639, "mAP50_95": 0.355, "hours": 7.4,
             "per_class_mAP50": {"D00": 0.598, "D10": 0.587, "D20": 0.690, "D44 (other)": 0.797, "D40": 0.523}},
}
