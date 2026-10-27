"""Video / dashcam monitoring with the trained YOLOv8 detector (synopsis video component).

Frames are sampled at a fixed rate, damage is detected per frame and an annotated
video plus a per-frame CSV log (counts per class) are written.

Example:
    python -m src.inference.video_infer --weights results/yolov8/train/weights/best.pt --video road.mp4 --fps 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from src.common.dataset import CLASS_CODES, map_from_names


def main() -> None:
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/video"))
    parser.add_argument("--fps", type=float, default=5.0, help="frames analysed per second of video")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    name_map = map_from_names(model.names)
    capture = cv2.VideoCapture(str(args.video))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(source_fps / args.fps))
    args.out.mkdir(parents=True, exist_ok=True)
    writer, rows, index = None, [], 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            annotated = result.plot()
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(str(args.out / f"{args.video.stem}_annotated.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            writer.write(annotated)
            counts = {code: 0 for code in CLASS_CODES}
            for c in result.boxes.cls.cpu().numpy():
                if int(c) in name_map:
                    counts[CLASS_CODES[name_map[int(c)]]] += 1
            rows.append({"frame": index, "time_s": index / source_fps, **counts,
                         "inference_ms": result.speed.get("inference", float("nan"))})
        index += 1
    capture.release()
    if writer is not None:
        writer.release()
    log = pd.DataFrame(rows)
    log.to_csv(args.out / f"{args.video.stem}_detections.csv", index=False)
    print(f"Analysed {len(log)} frames; damage detected in {(log[CLASS_CODES].sum(1) > 0).sum()} frames.")
    print(log[CLASS_CODES].sum().to_string())


if __name__ == "__main__":
    main()
