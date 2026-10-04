"""Convert RDD2022 Pascal VOC annotations into a deterministic YOLO dataset."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

LABELS = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def find_annotation(image_path: Path, input_root: Path) -> Path | None:
    candidates = [image_path.with_suffix(".xml"), image_path.parent / "annotations" / f"{image_path.stem}.xml"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(input_root.rglob(f"{image_path.stem}.xml"))
    return matches[0] if matches else None


def read_objects(annotation_path: Path) -> tuple[int, int, list[tuple[int, float, float, float, float]]]:
    root = ET.parse(annotation_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing image size in {annotation_path}")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {annotation_path}")

    objects = []
    for item in root.findall("object"):
        name = item.findtext("name", "").strip()
        if name not in LABELS:
            continue
        box = item.find("bndbox")
        if box is None:
            continue
        xmin = max(0.0, float(box.findtext("xmin", "0")))
        ymin = max(0.0, float(box.findtext("ymin", "0")))
        xmax = min(float(width), float(box.findtext("xmax", "0")))
        ymax = min(float(height), float(box.findtext("ymax", "0")))
        if xmax <= xmin or ymax <= ymin:
            continue
        center_x = ((xmin + xmax) / 2) / width
        center_y = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        objects.append((LABELS[name], center_x, center_y, box_width, box_height))
    return width, height, objects


def assign_splits(records: list[dict], seed: int, val_ratio: float, test_ratio: float) -> None:
    if val_ratio + test_ratio >= 1:
        raise ValueError("Validation and test ratios must sum to less than 1")
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["group"], []).append(record)
    groups = list(grouped)
    random.Random(seed).shuffle(groups)
    test_count = max(1, round(len(groups) * test_ratio))
    val_count = max(1, round(len(groups) * val_ratio))
    test_groups = set(groups[:test_count])
    val_groups = set(groups[test_count:test_count + val_count])
    for record in records:
        record["split"] = "test" if record["group"] in test_groups else "val" if record["group"] in val_groups else "train"


def main() -> None:
    args = parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    records = []
    for image_path in sorted(input_root.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        annotation_path = find_annotation(image_path, input_root)
        if annotation_path is None:
            continue
        try:
            width, height, objects = read_objects(annotation_path)
        except (ET.ParseError, ValueError) as error:
            print(f"Skipping {image_path}: {error}")
            continue
        if not objects:
            continue
        relative = image_path.relative_to(input_root)
        if len(relative.parts) > 1 and relative.parts[0].lower() in {"train", "test", "val", "validation"}:
            group = relative.parts[1] if len(relative.parts) > 2 else relative.parts[0]
        else:
            group = relative.parts[0] if len(relative.parts) > 1 else relative.stem
        records.append({"image": image_path, "relative": relative, "width": width, "height": height, "objects": objects, "group": group})
    if not records:
        raise RuntimeError("No images with usable matching XML annotations were found")
    assign_splits(records, args.seed, args.val_ratio, args.test_ratio)

    for record in records:
        split = record["split"]
        destination_image = output_root / "images" / split / record["relative"].name
        destination_label = output_root / "labels" / split / f"{record['relative'].stem}.txt"
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["image"], destination_image)
        destination_label.write_text("\n".join(" ".join(f"{value:.8f}" if position else str(value) for position, value in enumerate(item)) for item in record["objects"]) + "\n", encoding="utf-8")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: D00\n  1: D10\n  2: D20\n  3: D40\n",
        encoding="utf-8",
    )
    with (output_root / "splits.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "split", "group", "object_count"])
        writer.writeheader()
        writer.writerows({"image": str(record["relative"]), "split": record["split"], "group": record["group"], "object_count": len(record["objects"])} for record in records)
    class_counts = Counter(item[0] for record in records for item in record["objects"])
    with (output_root / "dataset_stats.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class_id", "class_name", "object_count"])
        writer.writerows((class_id, name, class_counts[class_id]) for name, class_id in LABELS.items())
    print(f"Prepared {len(records)} images at {output_root}")


if __name__ == "__main__":
    main()
