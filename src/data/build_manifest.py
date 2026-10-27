"""Step 1 for every member: build the shared manifest (same split for all models).

Example (Kaggle):
    python -m src.data.build_manifest --out results/data
    python -m src.data.build_manifest --data-root /kaggle/input/rdd-2022/RDD_SPLIT --out results/data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.dataset import CLASS_CODES, build_manifest, find_dataset_root, parse_class_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/data"))
    parser.add_argument("--class-map", default=None, help="override, e.g. '0:0,1:1,2:2,4:3'")
    parser.add_argument("--max-train", type=int, default=0, help="optional cap for quick runs")
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--plot", action="store_true", help="save class-distribution and sample figures")
    args = parser.parse_args()

    root = args.data_root or find_dataset_root()
    caps = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    instances, images, stats = build_manifest(root, args.out, parse_class_map(args.class_map), caps)
    print(json.dumps(stats, indent=2))
    if args.plot:
        plot_overview(instances, images, args.out)


def plot_overview(instances, images, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    counts = instances.pivot_table(index="class_id", columns="split", values="instance_id", aggfunc="count").fillna(0)
    counts = counts.reindex(columns=[c for c in ("train", "val", "test") if c in counts.columns])
    ax = counts.plot.bar(figsize=(7, 4), color=["#2a6f97", "#61a5c2", "#a9d6e5"][: len(counts.columns)])
    ax.set_xticklabels([CLASS_CODES[i] for i in counts.index], rotation=0)
    ax.set_ylabel("Damage instances")
    ax.set_title("RDD2022 class distribution per split")
    plt.tight_layout()
    plt.savefig(out / "class_distribution.png", dpi=200)
    plt.close()

    colors = ["#e63946", "#f4a261", "#2a9d8f", "#6a4c93"]
    sample = images[images.has_damage == 1].sample(min(6, int(images.has_damage.sum())), random_state=42)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for axis, (_, row) in zip(axes.flat, sample.iterrows()):
        im = Image.open(row.image).convert("RGB")
        draw = ImageDraw.Draw(im)
        for _, inst in instances[instances.image == row.image].iterrows():
            W, H = im.size
            box = ((inst.cx - inst.w / 2) * W, (inst.cy - inst.h / 2) * H, (inst.cx + inst.w / 2) * W, (inst.cy + inst.h / 2) * H)
            draw.rectangle(box, outline=colors[inst.class_id], width=4)
            draw.text((box[0] + 4, box[1] + 2), CLASS_CODES[inst.class_id], fill=colors[inst.class_id])
        axis.imshow(im)
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(out / "annotated_samples.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
