"""Train and validate the YOLOv8 road-damage detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to generated data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        project=str(args.project),
        name="rdd2022_yolov8n_seed42",
        pretrained=True,
        plots=True,
    )
    metrics = model.val(data=str(args.data), split="test", imgsz=args.imgsz, batch=args.batch)
    print({"precision": metrics.box.mp, "recall": metrics.box.mr, "map50": metrics.box.map50, "map5095": metrics.box.map})


if __name__ == "__main__":
    main()
