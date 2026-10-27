"""Metrics shared by all models, including the common instance-level protocol.

Common protocol (used for the single comparison table)
------------------------------------------------------
Every model must assign a damage class to every ground-truth damage instance
of the shared TEST split:

* Classifiers (ResNet-50, ViT-B/16) receive the ground-truth crop and predict a class.
* YOLOv8 predicts boxes on the full image; a GT instance takes the class of the
  predicted box with the highest IoU, provided IoU >= 0.5, otherwise it is "missed".
* U-Net predicts a pixel mask; a GT instance takes the majority damage class inside
  its box, provided at least 5 % of the box is predicted as damage, otherwise "missed".

A missed instance (pred = -1) counts as a false negative for its class.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .dataset import CLASS_CODES, NUM_CLASSES

LABELS = list(range(NUM_CLASSES))


def classification_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    _, _, wf, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    detected = y_pred >= 0
    cm_labels = LABELS + [-1]
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(mp),
        "macro_recall": float(mr),
        "macro_f1": float(mf),
        "weighted_f1": float(wf),
        "miss_rate": float(1 - detected.mean()) if len(y_true) else 0.0,
        "accuracy_on_detected": float((y_true[detected] == y_pred[detected]).mean()) if detected.any() else 0.0,
        "per_class": {CLASS_CODES[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                                       "support": int(s[i])} for i in LABELS},
        # rows = true class, columns = predicted class (+ last column "missed")
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=cm_labels)[:NUM_CLASSES].tolist(),
    }


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N,4) and b (M,4) in xyxy format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def segmentation_scores(confusion: np.ndarray) -> dict:
    """confusion: (C+1, C+1) pixel confusion with index 0 = background."""
    tp = np.diag(confusion).astype(float)
    fp = confusion.sum(0) - tp
    fn = confusion.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    names = ["background"] + CLASS_CODES
    return {
        "per_class_iou": {n: float(v) for n, v in zip(names, iou)},
        "per_class_dice": {n: float(v) for n, v in zip(names, dice)},
        "mIoU_damage": float(iou[1:].mean()),
        "mDice_damage": float(dice[1:].mean()),
        "mIoU_all": float(iou.mean()),
        "pixel_accuracy": float(tp.sum() / max(confusion.sum(), 1)),
        "binary_damage_iou": float(_binary_iou(confusion)),
    }


def _binary_iou(confusion: np.ndarray) -> float:
    tp = confusion[1:, 1:].sum()
    fp = confusion[0, 1:].sum()
    fn = confusion[1:, 0].sum()
    return tp / max(tp + fp + fn, 1)


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def measure_latency(forward, example, device, warmup: int = 10, iters: int = 50) -> float:
    """Mean milliseconds per forward pass with batch size 1."""
    import torch

    with torch.inference_mode():
        for _ in range(warmup):
            forward(example)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            forward(example)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def save_instance_predictions(df: pd.DataFrame, path: Path) -> None:
    required = {"instance_id", "gt_class", "pred_class"}
    if not required.issubset(df.columns):
        raise ValueError(f"instance prediction file must contain {required}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
