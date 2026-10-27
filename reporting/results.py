"""Load whatever results exist; everything missing is rendered as 'TBD' (never invented)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

TBD = "TBD"
MODEL_ORDER = ["resnet50", "yolov8", "unet", "vit_b16"]
DEFAULT_NAMES = {"resnet50": "ResNet-50", "yolov8": "YOLOv8s", "unet": "U-Net (ResNet-34)", "vit_b16": "ViT-B/16"}
FAMILIES = {"resnet50": "CNN classifier", "yolov8": "One-stage detector", "unet": "Encoder-decoder segmentation",
            "vit_b16": "Attention / Transformer"}


def pct(value, digits=1):
    if value is None or (isinstance(value, float) and value != value):
        return TBD
    return f"{100 * value:.{digits}f}"


def num(value, digits=3):
    if value is None or (isinstance(value, float) and value != value):
        return TBD
    return f"{value:.{digits}f}"


class Results:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.models = {}
        for key in MODEL_ORDER:
            path = self.root / key / "metrics.json"
            if path.exists():
                self.models[key] = json.loads(path.read_text())
        final = self.root / "final" / "final_results.json"
        self.final = json.loads(final.read_text()) if final.exists() else None
        stats = self.root / "data" / "dataset_stats.json"
        self.stats = json.loads(stats.read_text()) if stats.exists() else None
        self.table = {row["key"]: row for row in self.final["table"]} if self.final else {}

    # ------------------------------------------------------------------ #
    @property
    def complete(self) -> bool:
        return self.final is not None and all(k in self.table for k in MODEL_ORDER)

    def name(self, key):
        return self.models.get(key, {}).get("model", DEFAULT_NAMES[key])

    def common(self, key):
        """Common-protocol metrics for a model (from the final comparison, else the model's own file)."""
        if key in self.table:
            return self.final["models"][key]["common"]
        if key in self.models:
            return self.models[key].get("test")
        return None

    def row(self, key):
        c = self.common(key)
        m = self.models.get(key, {})
        native = TBD
        if m.get("detection_test"):
            d = m["detection_test"]
            native = f"mAP50 {num(d['mAP50'])} / mAP50-95 {num(d['mAP50_95'])}"
        elif m.get("segmentation_test"):
            s = m["segmentation_test"]
            native = f"mIoU {num(s['mIoU_damage'])} / Dice {num(s['mDice_damage'])}"
        elif m:
            native = "(same as accuracy)"
        return {
            "Model": self.name(key), "Family": FAMILIES[key],
            "Params (M)": num(m["params"] / 1e6, 1) if m else TBD,
            "Latency (ms)": num(m.get("latency_ms_bs1"), 1) if m else TBD,
            "Acc. (%)": pct(c["accuracy"]) if c else TBD,
            "Macro-P (%)": pct(c["macro_precision"]) if c else TBD,
            "Macro-R (%)": pct(c["macro_recall"]) if c else TBD,
            "Macro-F1 (%)": pct(c["macro_f1"]) if c else TBD,
            "Miss (%)": pct(c["miss_rate"]) if c else TBD,
            "Native metric": native,
        }

    def per_class_f1(self, key):
        c = self.common(key)
        return {k: pct(v["f1"]) for k, v in c["per_class"].items()} if c else {k: TBD for k in ("D00", "D10", "D20", "D40")}

    def best(self):
        rows = [(self.common(k)["macro_f1"], k) for k in MODEL_ORDER if self.common(k)]
        return max(rows)[1] if rows else None

    def figure(self, *parts):
        path = self.root.joinpath(*parts)
        return path if path.exists() else None

    def split_counts(self):
        if not self.stats:
            return None
        return self.stats["images_per_split"], self.stats["instances_per_split"], self.stats["instances_per_class_per_split"]
