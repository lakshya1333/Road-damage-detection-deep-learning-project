"""Create class-labelled image crops from the generated YOLO dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

CLASS_NAMES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    count = 0
    for split in ("train", "val", "test"):
        image_root = args.yolo_root / "images" / split
        label_root = args.yolo_root / "labels" / split
        if not image_root.exists():
            continue
        for image_path in sorted(image_root.glob("*")):
            label_path = label_root / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            with Image.open(image_path) as image:
                width, height = image.size
                for index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                    class_id, center_x, center_y, box_width, box_height = map(float, line.split())
                    left = max(0, int((center_x - box_width / 2) * width))
                    top = max(0, int((center_y - box_height / 2) * height))
                    right = min(width, int((center_x + box_width / 2) * width))
                    bottom = min(height, int((center_y + box_height / 2) * height))
                    if right <= left or bottom <= top:
                        continue
                    class_name = CLASS_NAMES[int(class_id)]
                    destination = args.output / split / class_name
                    destination.mkdir(parents=True, exist_ok=True)
                    image.crop((left, top, right, bottom)).save(destination / f"{image_path.stem}_{index}.jpg", quality=95)
                    count += 1
    print(f"Created {count} classification crops at {args.output}")


if __name__ == "__main__":
    main()
