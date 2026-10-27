"""U-Net with a ResNet-34 encoder (encoder-decoder segmentation family). Owner: Kumar Kshitij.

RDD2022 has bounding boxes, not pixel masks. We therefore train on *box-derived weak
masks*: every pixel inside a damage box gets that box's class (smaller boxes are drawn
last so they stay visible), everything else is background. This is a documented
limitation: the U-Net learns "damage regions", not exact crack outlines, so IoU is
measured against the same box-filled masks on the test split.

Example:
    python -m src.models.train_unet --manifest results/data --out results/unet --epochs 15 --size 512
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.common.dataset import CLASS_CODES, NUM_CLASSES, SEED, load_manifest, set_seed
from src.common.metrics import (classification_metrics, count_parameters, measure_latency, save_instance_predictions,
                                save_json, segmentation_scores)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
N_OUT = NUM_CLASSES + 1  # 0 = background


def box_mask(rows: pd.DataFrame, size: int) -> np.ndarray:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    for row in rows.assign(area=rows.w * rows.h).sort_values("area", ascending=False).itertuples():
        x1, y1 = (row.cx - row.w / 2) * size, (row.cy - row.h / 2) * size
        x2, y2 = (row.cx + row.w / 2) * size, (row.cy + row.h / 2) * size
        draw.rectangle((x1, y1, x2, y2), fill=int(row.class_id) + 1)
    return np.asarray(mask, dtype=np.int64)


class SegDataset(Dataset):
    def __init__(self, image_list, instances: pd.DataFrame, size: int, train: bool):
        self.images = list(image_list)
        self.by_image = {k: g for k, g in instances.groupby("image")}
        self.size, self.train = size, train
        self.empty = instances.iloc[:0]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        path = self.images[i]
        with Image.open(path) as im:
            image = im.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        y = torch.from_numpy(box_mask(self.by_image.get(path, self.empty), self.size))
        if self.train:
            if torch.rand(1) < 0.5:  # horizontal flip only: keeps longitudinal/transverse semantics
                x, y = x.flip(-1), y.flip(-1)
            brightness = 1 + (torch.rand(1) - 0.5) * 0.5
            contrast = 1 + (torch.rand(1) - 0.5) * 0.5
            mean = x.mean()
            x = ((x - mean) * contrast + mean) * brightness
            x = x.clamp(0, 1)
        return (x - MEAN) / STD, y


def build_model(encoder: str, pretrained: bool) -> nn.Module:
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None, classes=N_OUT)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    confusion = torch.zeros(N_OUT, N_OUT, dtype=torch.long, device=device)
    for x, y in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            pred = model(x.to(device, non_blocking=True)).argmax(1)
        y = y.to(device)
        confusion += torch.bincount(y.flatten() * N_OUT + pred.flatten(), minlength=N_OUT * N_OUT).view(N_OUT, N_OUT)
    return segmentation_scores(confusion.cpu().numpy())


@torch.inference_mode()
def assign_instances(model, test_images, instances, size, device, batch, min_fraction=0.05):
    """Common protocol: majority predicted damage class inside each GT box (missed if < 5 % damage pixels)."""
    model.eval()
    test = instances[instances.split == "test"]
    by_image = {k: g for k, g in test.groupby("image")}
    rows, samples = [], []
    dataset = SegDataset([p for p in test_images if p in by_image], test, size, train=False)
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=2)
    k = 0
    for x, y in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            pred = model(x.to(device)).argmax(1).cpu().numpy()
        for b in range(len(pred)):
            path = dataset.images[k]
            k += 1
            if len(samples) < 4:
                samples.append((path, y[b].numpy(), pred[b]))
            for inst in by_image[path].itertuples():
                x1, y1 = int(max(0, (inst.cx - inst.w / 2) * size)), int(max(0, (inst.cy - inst.h / 2) * size))
                x2, y2 = int(min(size, math.ceil((inst.cx + inst.w / 2) * size))), int(min(size, math.ceil((inst.cy + inst.h / 2) * size)))
                region = pred[b, y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
                counts = np.bincount(region.flatten(), minlength=N_OUT)
                damage = counts[1:].sum()
                fraction = damage / max(region.size, 1)
                cls = int(counts[1:].argmax()) if damage else -1
                rows.append({"instance_id": inst.instance_id, "gt_class": inst.class_id,
                             "pred_class": cls if fraction >= min_fraction else -1, "matched_class": cls,
                             "damage_fraction": float(fraction), "box_area": inst.w * inst.h,
                             "aspect": (inst.w * inst.img_w) / (inst.h * inst.img_h)})
    return pd.DataFrame(rows), samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("results/data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--background-fraction", type=float, default=0.2)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    import segmentation_models_pytorch as smp

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    instances, images = load_manifest(args.manifest)
    rng = np.random.default_rng(args.seed)
    lists = {}
    for split in ("train", "val", "test"):
        sub = images[images.split == split]
        if split == "train":
            sub = sub[(sub.has_damage == 1) | (rng.random(len(sub)) < args.background_fraction)]
        lists[split] = sub.image.tolist()
    datasets = {s: SegDataset(lists[s], instances[instances.split == s], args.size, s == "train") for s in lists}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {s: DataLoader(d, batch_size=args.batch, shuffle=s == "train", generator=generator if s == "train" else None,
                             num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=s == "train" and len(d) > args.batch)
               for s, d in datasets.items()}
    print({s: len(d) for s, d in datasets.items()}, "device", device)

    model = build_model(args.encoder, not args.no_pretrained).to(device)
    # class weights against the heavy background imbalance
    weights = torch.tensor([0.5] + [1.0] * NUM_CLASSES, device=device)
    ce = nn.CrossEntropyLoss(weight=weights)
    dice = smp.losses.DiceLoss(mode="multiclass")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * len(loaders["train"]), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history, best, bad = [], -1.0, 0
    best_path = args.out / "best.pt"
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen, start = 0.0, 0, time.time()
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = ce(logits, y) + dice(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(x)
            seen += len(x)
        val = evaluate(model, loaders["val"], device)
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1), "val_mIoU_damage": val["mIoU_damage"],
                        "val_mDice_damage": val["mDice_damage"], "val_binary_iou": val["binary_damage_iou"],
                        "epoch_seconds": time.time() - start})
        improved = val["mIoU_damage"] > best
        if improved:
            best, bad = val["mIoU_damage"], 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
        print(f"[unet] epoch {epoch:02d} loss {history[-1]['train_loss']:.4f} val mIoU(damage) {val['mIoU_damage']:.4f} "
              f"binary IoU {val['binary_damage_iou']:.4f}{' *' if improved else ''}", flush=True)
        if bad >= args.patience:
            print("Early stopping.")
            break
    train_minutes = (time.time() - t0) / 60

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    seg_val = evaluate(model, loaders["val"], device)
    seg_test = evaluate(model, loaders["test"], device)
    preds, samples = assign_instances(model, lists["test"], instances, args.size, device, args.batch)
    save_instance_predictions(preds, args.out / "instance_predictions.csv")
    common = classification_metrics(preds.gt_class, preds.pred_class)
    latency = measure_latency(model, torch.randn(1, 3, args.size, args.size, device=device), device)
    pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)
    summary = {
        "model": f"U-Net ({args.encoder})",
        "family": "Encoder-decoder segmentation",
        "task": "pixel-level damage-region segmentation (box-derived masks)",
        "params": count_parameters(model),
        "latency_ms_bs1": latency,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "train_minutes": train_minutes,
        "epochs_run": len(history),
        "config": {"encoder": args.encoder, "size": args.size, "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                   "optimizer": "AdamW", "schedule": "OneCycle", "loss": "weighted cross-entropy (background weight 0.5) + multiclass Dice",
                   "augmentation": "horizontal flip, brightness/contrast jitter", "background_fraction": args.background_fraction,
                   "pretrained": not args.no_pretrained, "seed": args.seed},
        "segmentation_val": seg_val,
        "segmentation_test": seg_test,
        "test": common,
        "matching": "majority damage class inside GT box, missed if < 5% of box predicted as damage",
    }
    save_json(summary, args.out / "metrics.json")
    plot_samples(samples, args.out)
    print(f"TEST mIoU(damage) {seg_test['mIoU_damage']:.4f} | COMMON accuracy {common['accuracy']:.4f} macro-F1 {common['macro_f1']:.4f}")


def plot_samples(samples, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#000000", "#e63946", "#f4a261", "#2a9d8f", "#6a4c93"])
    fig, axes = plt.subplots(len(samples), 3, figsize=(9, 3 * len(samples)))
    axes = np.atleast_2d(axes)
    for row, (path, gt, pred) in zip(axes, samples):
        with Image.open(path) as im:
            row[0].imshow(im.convert("RGB").resize(gt.shape[::-1]))
        row[1].imshow(gt, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        row[2].imshow(pred, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        for axis, title in zip(row, ("image", "box-derived mask", "U-Net prediction")):
            axis.set_title(title, fontsize=9)
            axis.axis("off")
    plt.tight_layout()
    plt.savefig(out / "unet_samples.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
