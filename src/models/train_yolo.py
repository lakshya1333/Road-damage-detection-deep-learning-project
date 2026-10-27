"""YOLOv8 one-stage detector (detection family). Owner: Dhruv Maheshwari.

Builds a 4-class YOLO dataset from the shared manifest (same split as every other
model, background images kept to teach the detector what "no damage" looks like),
trains YOLOv8, reports detection metrics on the TEST split and writes the common
instance-level predictions used by the comparison table.

Example:
    python -m src.models.train_yolo --manifest results/data --out results/yolov8 --model yolov8s.pt --epochs 30
    # evaluate an already-trained checkpoint without re-training (class names are re-mapped automatically):
    python -m src.models.train_yolo --manifest results/data --out results/yolov8 --weights best.pt --skip-train
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.common.dataset import CLASS_CODES, CLASS_NAMES, NUM_CLASSES, SEED, load_manifest, map_from_names, xywhn_to_xyxy
from src.common.metrics import box_iou, classification_metrics, save_instance_predictions, save_json


def build_yolo_dataset(instances: pd.DataFrame, images: pd.DataFrame, root: Path, background_fraction: float) -> Path:
    if root.exists():
        shutil.rmtree(root)
    rng = np.random.default_rng(SEED)
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    grouped = instances.groupby("image")
    for _, row in images.iterrows():
        if not row.has_damage and row.split == "train" and rng.random() > background_fraction:
            continue
        source = Path(row.image)
        target = root / "images" / row.split / f"{source.stem}{source.suffix.lower()}"
        if not target.exists():
            try:
                os.symlink(source.resolve(), target)
            except OSError:
                shutil.copy2(source, target)
        lines = []
        if row.has_damage:
            for _, inst in grouped.get_group(row.image).iterrows():
                lines.append(f"{inst.class_id} {inst.cx:.6f} {inst.cy:.6f} {inst.w:.6f} {inst.h:.6f}")
        (root / "labels" / row.split / f"{source.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(root.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
        "names": {i: f"{CLASS_CODES[i]}_{CLASS_NAMES[i].split()[0]}" for i in range(NUM_CLASSES)}}, sort_keys=False))
    return data_yaml


def assign_instances(model, instances: pd.DataFrame, name_map: dict[int, int], imgsz: int, conf: float, device) -> pd.DataFrame:
    """Common protocol: each GT test instance takes the class of the best-IoU predicted box."""
    test = instances[instances.split == "test"]
    rows = []
    image_list = sorted(test.image.unique())
    for start in range(0, len(image_list), 32):
        batch = image_list[start:start + 32]
        results = model.predict(batch, imgsz=imgsz, conf=conf, device=device, verbose=False)
        for image_path, result in zip(batch, results):
            gt = test[test.image == image_path]
            W, H = gt.img_w.iloc[0], gt.img_h.iloc[0]
            gt_boxes = np.array([xywhn_to_xyxy(r.cx, r.cy, r.w, r.h, W, H) for r in gt.itertuples()])
            boxes = result.boxes
            pred_xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.zeros((0, 4))
            pred_cls = np.array([name_map.get(int(c), -1) for c in boxes.cls.cpu().numpy()]) if len(boxes) else np.zeros(0, int)
            pred_conf = boxes.conf.cpu().numpy() if len(boxes) else np.zeros(0)
            keep = pred_cls >= 0
            pred_xyxy, pred_cls, pred_conf = pred_xyxy[keep], pred_cls[keep], pred_conf[keep]
            ious = box_iou(gt_boxes, pred_xyxy)
            for k, inst in enumerate(gt.itertuples()):
                if ious.shape[1]:
                    j = int(ious[k].argmax())
                    best_iou, cls, score = float(ious[k, j]), int(pred_cls[j]), float(pred_conf[j])
                else:
                    best_iou, cls, score = 0.0, -1, 0.0
                rows.append({"instance_id": inst.instance_id, "gt_class": inst.class_id,
                             "pred_class": cls if best_iou >= 0.5 else -1, "matched_class": cls,
                             "matched_iou": best_iou, "score": score, "box_area": inst.w * inst.h,
                             "aspect": (inst.w * W) / (inst.h * H)})
    return pd.DataFrame(rows)


def main() -> None:
    from ultralytics import YOLO
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("results/data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--yolo-root", type=Path, default=Path("/tmp/rdd_yolo"))
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--weights", type=Path, default=None, help="evaluate this checkpoint")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--background-fraction", type=float, default=0.3,
                        help="fraction of undamaged TRAIN images kept (all val/test images are kept)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    instances, images = load_manifest(args.manifest)
    data_yaml = build_yolo_dataset(instances, images, args.yolo_root, args.background_fraction)
    print("YOLO dataset:", data_yaml)

    t0 = time.time()
    if args.skip_train:
        if args.weights is None:
            raise ValueError("--skip-train needs --weights")
        model = YOLO(str(args.weights))
        train_minutes = None
    else:
        model = YOLO(str(args.weights) if args.weights else args.model)
        model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                    device=device, seed=args.seed, deterministic=True, patience=args.patience, cos_lr=True,
                    fliplr=0.5, flipud=0.0, degrees=0.0, mosaic=1.0, close_mosaic=5, amp=device != "cpu",
                    project=str(args.out.resolve()), name="train", exist_ok=True, plots=True)
        model = YOLO(str(args.out / "train" / "weights" / "best.pt"))
        train_minutes = (time.time() - t0) / 60

    name_map = map_from_names(model.names)
    if len(set(name_map.values())) != NUM_CLASSES:
        raise RuntimeError(f"Could not map model classes {model.names} to D00/D10/D20/D40")
    # Ultralytics evaluates with its own class indices; when a 5-class checkpoint is reused we
    # report the per-class AP of the four mapped classes only.
    metrics = model.val(data=str(data_yaml), split="test", imgsz=args.imgsz, batch=args.batch, device=device,
                        plots=True, project=str(args.out.resolve()), name="test", exist_ok=True) if len(model.names) == NUM_CLASSES else None
    detection = {}
    if metrics is not None:
        box = metrics.box
        detection = {"precision": float(box.mp), "recall": float(box.mr), "mAP50": float(box.map50), "mAP50_95": float(box.map),
                     "per_class_AP50": {CLASS_CODES[name_map[int(c)]]: float(box.ap50[i]) for i, c in enumerate(box.ap_class_index)},
                     "per_class_AP50_95": {CLASS_CODES[name_map[int(c)]]: float(box.ap[i]) for i, c in enumerate(box.ap_class_index)},
                     "speed_ms": {k: float(v) for k, v in metrics.speed.items()}}
        detection["f1"] = 2 * detection["precision"] * detection["recall"] / max(detection["precision"] + detection["recall"], 1e-9)

    preds = assign_instances(model, instances, name_map, args.imgsz, args.conf, device)
    save_instance_predictions(preds, args.out / "instance_predictions.csv")
    common = classification_metrics(preds.gt_class, preds.pred_class)

    # latency, batch 1, full pipeline (preprocess + inference + NMS)
    sample = sorted(instances[instances.split == "test"].image.unique())[:30]
    model.predict(sample[:3], imgsz=args.imgsz, device=device, verbose=False)
    start = time.perf_counter()
    for path in sample:
        model.predict(path, imgsz=args.imgsz, device=device, verbose=False)
    latency = (time.perf_counter() - start) * 1000 / max(len(sample), 1)

    if not args.skip_train:
        # sample predictions for the report
        model.predict(sample[:12], imgsz=args.imgsz, conf=args.conf, device=device, save=True,
                      project=str(args.out.resolve()), name="samples", exist_ok=True, verbose=False)
    summary = {
        "model": Path(args.model).stem.replace("yolov8", "YOLOv8") if Path(args.model).stem.startswith("yolov8") else "YOLOv8",
        "family": "One-stage object detector",
        "task": "full-image damage detection",
        "params": int(sum(p.numel() for p in model.model.parameters())),
        "latency_ms_bs1": latency,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "train_minutes": train_minutes,
        "config": {"base": args.model, "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
                   "optimizer": "Ultralytics auto (SGD/AdamW)", "schedule": "cosine", "conf": args.conf,
                   "augmentation": "mosaic (off last 5 epochs), HSV, scale/translate, horizontal flip; no vertical flip/rotation",
                   "background_fraction": args.background_fraction, "seed": args.seed},
        "detection_test": detection,
        "test": common,
        "matching": "class-agnostic best IoU >= 0.5 at conf >= %.2f" % args.conf,
    }
    save_json(summary, args.out / "metrics.json")
    print(summary["detection_test"])
    print(f"COMMON PROTOCOL accuracy {common['accuracy']:.4f} macro-F1 {common['macro_f1']:.4f} miss-rate {common['miss_rate']:.3f}")


if __name__ == "__main__":
    main()
