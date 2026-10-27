"""Shared RDD2022 data contract used by every model in the project.

All four models read the SAME manifest built here, so the train/val/test split,
the class mapping and the list of test damage instances are identical across
models. This is what makes the final comparison table fair.

Manifest columns (one row per damage instance):
    instance_id, split, image, img_w, img_h, class_id, cx, cy, w, h
(cx, cy, w, h are YOLO-normalised box coordinates in [0, 1]).
Images that contain no damage are listed separately in images.csv
(has_damage = 0) and are used only by the detector and the segmenter.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SEED = 42
CLASS_CODES = ["D00", "D10", "D20", "D40"]
CLASS_NAMES = ["Longitudinal crack", "Transverse crack", "Alligator crack", "Pothole"]
NUM_CLASSES = 4
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")

# keyword -> unified class id; anything not matched (e.g. D44 "other corruption",
# repairs, manholes) is dropped so that all models use the four synopsis classes.
_KEYWORDS = [
    (("d00", "longitudinal"), 0),
    (("d10", "transverse", "lateral"), 1),
    (("d20", "alligator"), 2),
    (("d40", "pothole"), 3),
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def name_to_class(name: str) -> int | None:
    lowered = str(name).lower()
    for keys, class_id in _KEYWORDS:
        if any(key in lowered for key in keys):
            return class_id
    return None


def map_from_names(names) -> dict[int, int]:
    """Build {source_id: unified_id} from a list/dict of class names."""
    if isinstance(names, dict):
        items = names.items()
    else:
        items = enumerate(names)
    mapping = {}
    for source_id, name in items:
        unified = name_to_class(name)
        if unified is not None:
            mapping[int(source_id)] = unified
    return mapping


def parse_class_map(text: str | None) -> dict[int, int] | None:
    if not text:
        return None
    return {int(a): int(b) for a, b in (pair.split(":") for pair in text.split(","))}


# --------------------------------------------------------------------------- #
# Locating the dataset
# --------------------------------------------------------------------------- #
def _split_dirs(root: Path, split: str) -> tuple[Path, Path] | None:
    aliases = [split] + (["valid", "validation"] if split == "val" else [])
    for name in aliases:
        for images, labels in (
            (root / name / "images", root / name / "labels"),
            (root / "images" / name, root / "labels" / name),
        ):
            if images.is_dir() and labels.is_dir():
                return images, labels
    return None


def find_dataset_root(candidates: list[str | Path] | None = None) -> Path:
    """Return the first directory that has train/val/test images+labels."""
    search = [Path(c) for c in (candidates or [])]
    search += [Path("/kaggle/input"), Path("/content"), Path("data/raw"), Path("data")]
    for base in search:
        if not base.exists():
            continue
        if all(_split_dirs(base, s) for s in SPLITS):
            return base
        for path in sorted(base.rglob("*")):
            if path.is_dir() and path.name.lower() in {"train"} and all(_split_dirs(path.parent, s) for s in SPLITS):
                return path.parent
    raise FileNotFoundError(
        "Could not find an RDD2022 folder with train/val/test images and YOLO labels. "
        "Pass --data-root explicitly."
    )


def _read_yaml_names(root: Path):
    import yaml

    for candidate in [root / "data.yaml", root.parent / "data.yaml", *root.glob("*.yaml")]:
        if candidate.exists():
            content = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            if "names" in content:
                return content["names"], candidate
    return None, None


def infer_class_map(root: Path, label_dirs: list[Path]) -> tuple[dict[int, int], str]:
    names, source = _read_yaml_names(root)
    if names is not None:
        mapping = map_from_names(names)
        if len(set(mapping.values())) == NUM_CLASSES:
            return mapping, f"names from {source}"
    max_id = -1
    for label_dir in label_dirs:
        for i, path in enumerate(label_dir.glob("*.txt")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    max_id = max(max_id, int(float(line.split()[0])))
            if i > 3000:
                break
    if max_id <= 3:
        return {0: 0, 1: 1, 2: 2, 3: 3}, "identity (ids 0-3)"
    # The common Kaggle/Ultralytics RDD2022 export: D00, D10, D20, D44(other corruption), D40
    return {0: 0, 1: 1, 2: 2, 4: 3}, "5-class export (D44 'other corruption' id 3 dropped)"


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def build_manifest(data_root: Path, out_dir: Path, class_map: dict[int, int] | None = None,
                   max_images: dict[str, int] | None = None, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Scan the dataset once and write instances.csv, images.csv and dataset_stats.json."""
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = {s: _split_dirs(data_root, s) for s in SPLITS}
    missing = [s for s, d in dirs.items() if d is None]
    if missing:
        raise FileNotFoundError(f"Missing split folders {missing} under {data_root}")
    mapping_note = "user supplied"
    if class_map is None:
        class_map, mapping_note = infer_class_map(data_root, [d[1] for d in dirs.values()])

    instance_rows, image_rows = [], []
    dropped = Counter()
    rng = random.Random(seed)
    for split, (image_dir, label_dir) in dirs.items():
        images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if max_images and max_images.get(split):
            rng.shuffle(images)
            images = sorted(images[: max_images[split]])
        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            boxes = []
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    source_id = int(float(parts[0]))
                    cx, cy, w, h = (float(v) for v in parts[1:])
                    if source_id not in class_map:
                        dropped[source_id] += 1
                        continue
                    # clip to the image and discard degenerate boxes
                    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
                    x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
                    if x2 - x1 <= 1e-4 or y2 - y1 <= 1e-4:
                        dropped["degenerate"] += 1
                        continue
                    boxes.append((class_map[source_id], (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
            with Image.open(image_path) as im:
                img_w, img_h = im.size
            image_rows.append({"split": split, "image": str(image_path), "img_w": img_w, "img_h": img_h,
                               "num_instances": len(boxes), "has_damage": int(bool(boxes))})
            for k, (cid, cx, cy, w, h) in enumerate(boxes):
                instance_rows.append({"instance_id": f"{split}/{image_path.stem}/{k}", "split": split,
                                      "image": str(image_path), "img_w": img_w, "img_h": img_h,
                                      "class_id": cid, "cx": cx, "cy": cy, "w": w, "h": h})
    instances = pd.DataFrame(instance_rows)
    images_df = pd.DataFrame(image_rows)
    stats = {
        "data_root": str(data_root),
        "class_mapping": {str(k): CLASS_CODES[v] for k, v in class_map.items()},
        "class_mapping_source": mapping_note,
        "dropped_labels": {str(k): v for k, v in dropped.items()},
        "images_per_split": images_df.groupby("split").size().to_dict(),
        "damaged_images_per_split": images_df.groupby("split")["has_damage"].sum().astype(int).to_dict(),
        "instances_per_split": instances.groupby("split").size().to_dict(),
        "instances_per_class_per_split": {
            s: {CLASS_CODES[c]: int(n) for c, n in g["class_id"].value_counts().sort_index().items()}
            for s, g in instances.groupby("split")
        },
        "seed": seed,
    }
    instances.to_csv(out_dir / "instances.csv", index=False)
    images_df.to_csv(out_dir / "images.csv", index=False)
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return instances, images_df, stats


def load_manifest(manifest_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_dir = Path(manifest_dir)
    return pd.read_csv(manifest_dir / "instances.csv"), pd.read_csv(manifest_dir / "images.csv")


def crop_box(image: Image.Image, row, pad: float = 0.10, min_size: int = 32) -> Image.Image:
    """Crop a damage instance with a little context around the box."""
    W, H = image.size
    bw, bh = row.w * W, row.h * H
    cx, cy = row.cx * W, row.cy * H
    bw2, bh2 = max(bw * (1 + 2 * pad), min_size), max(bh * (1 + 2 * pad), min_size)
    x1, y1 = int(max(0, cx - bw2 / 2)), int(max(0, cy - bh2 / 2))
    x2, y2 = int(min(W, cx + bw2 / 2)), int(min(H, cy + bh2 / 2))
    return image.crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)))


def xywhn_to_xyxy(cx, cy, w, h, W, H):
    return ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)
