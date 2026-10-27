"""Damage-instance classifiers: ResNet-50 / MobileNetV2 (CNN family) and ViT-B/16 (attention family).

Owners: Lakshya Jain (resnet50, mobilenet_v2) and MANAS LAXMAN KUDIPADY (vit_b_16).

The classifier receives each ground-truth damage box (plus 10 % context), resized to
224x224, and predicts one of D00/D10/D20/D40. Crops are cached once as a uint8
memmap so that GPU time is spent on training, not JPEG decoding.

Example:
    python -m src.models.train_classifier --arch resnet50 --manifest results/data --out results/resnet50
    python -m src.models.train_classifier --arch vit_b_16 --manifest results/data --out results/vit_b16 --lr 5e-5
"""
from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2

from src.common.dataset import CLASS_CODES, NUM_CLASSES, SEED, crop_box, load_manifest, set_seed
from src.common.metrics import (classification_metrics, count_parameters, measure_latency, save_instance_predictions,
                                save_json)

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def cache_crops(frame: pd.DataFrame, path: Path, size: int, workers: int = 8) -> np.memmap:
    """Decode every crop once and store it as (N, size, size, 3) uint8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    done_flag = path.with_suffix(".done")
    if path.exists() and done_flag.exists():
        return np.memmap(path, dtype=np.uint8, mode="r", shape=(len(frame), size, size, 3))
    array = np.memmap(path, dtype=np.uint8, mode="w+", shape=(len(frame), size, size, 3))
    groups = list(frame.reset_index(drop=True).groupby("image"))

    def work(item):
        image_path, rows = item
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            for index, row in rows.iterrows():
                array[index] = np.asarray(crop_box(im, row).resize((size, size), Image.BILINEAR))

    start = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for k, _ in enumerate(pool.map(work, groups)):
            if k % 2000 == 0:
                print(f"  cached {k}/{len(groups)} images ({time.time() - start:.0f}s)", flush=True)
    array.flush()
    done_flag.touch()
    return array


class CropDataset(Dataset):
    def __init__(self, array, labels, train: bool, size: int):
        self.array, self.labels = array, np.asarray(labels)
        ops = [v2.ToImage()]
        if train:
            # Horizontal flip keeps crack orientation semantics (longitudinal stays longitudinal).
            # 90-degree rotation would turn D00 into D10, so it is deliberately NOT used.
            ops += [v2.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.8, 1.25), antialias=True),
                    v2.RandomHorizontalFlip(),
                    v2.ColorJitter(0.25, 0.25, 0.15, 0.02)]
        ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]
        self.transform = v2.Compose(ops)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(np.array(self.array[i])), int(self.labels[i])


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_model(arch: str, pretrained: bool) -> nn.Module:
    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    elif arch == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    elif arch == "vit_tiny":  # only for CPU smoke tests
        model = models.VisionTransformer(image_size=224, patch_size=16, num_layers=2, num_heads=2,
                                         hidden_dim=64, mlp_dim=128, num_classes=NUM_CLASSES)
    else:
        raise ValueError(arch)
    return model


def class_weights(labels, device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    return torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    probs, targets = [], []
    for images, labels in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(images.to(device, non_blocking=True))
        probs.append(logits.float().softmax(1).cpu())
        targets.append(labels)
    probs = torch.cat(probs).numpy()
    return np.concatenate([t.numpy() for t in targets]), probs.argmax(1), probs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("resnet50", "mobilenet_v2", "vit_b_16", "vit_tiny"), default="resnet50")
    parser.add_argument("--manifest", type=Path, default=Path("results/data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/rdd_crop_cache"))
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None, help="default 2e-4 for CNNs, 5e-5 for ViT")
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    is_vit = args.arch.startswith("vit")
    lr = args.lr or (5e-5 if is_vit else 2e-4)
    wd = args.weight_decay if args.weight_decay is not None else (0.05 if is_vit else 1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | arch={args.arch} lr={lr} wd={wd}")

    instances, _ = load_manifest(args.manifest)
    frames = {s: instances[instances.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    arrays = {s: cache_crops(f, args.cache_dir / f"{s}_{args.size}_{len(f)}.u8", args.size) for s, f in frames.items()}
    datasets = {s: CropDataset(arrays[s], frames[s].class_id.values, s == "train", args.size) for s in frames}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {s: DataLoader(datasets[s], batch_size=args.batch, shuffle=(s == "train"), generator=generator if s == "train" else None,
                             num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=(s == "train" and len(datasets[s]) > args.batch))
               for s in frames}
    print({s: len(d) for s, d in datasets.items()})

    model = build_model(args.arch, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(frames["train"].class_id.values, device), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = args.epochs * len(loaders["train"])
    warmup = len(loaders["train"]) if is_vit else 0  # one warm-up epoch for the transformer
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / max(warmup, 1) if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(steps - warmup, 1))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history, best_f1, bad_epochs = [], -1.0, 0
    best_path = args.out / "best.pt"
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen, start = 0.0, 0, time.time()
        for images, labels in loaders["train"]:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(labels)
            seen += len(labels)
        y, p, _ = predict(model, loaders["val"], device)
        val = classification_metrics(y, p)
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1), "val_accuracy": val["accuracy"],
                        "val_macro_f1": val["macro_f1"], "epoch_seconds": time.time() - start})
        improved = val["macro_f1"] > best_f1
        if improved:
            best_f1, bad_epochs = val["macro_f1"], 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
        print(f"[{args.arch}] epoch {epoch:02d} loss {history[-1]['train_loss']:.4f} "
              f"val acc {val['accuracy']:.4f} macro-F1 {val['macro_f1']:.4f}{' *' if improved else ''}", flush=True)
        if bad_epochs >= args.patience:
            print("Early stopping.")
            break
    train_minutes = (time.time() - t0) / 60

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    results = {}
    for split in ("val", "test"):
        y, p, probs = predict(model, loaders[split], device)
        results[split] = classification_metrics(y, p)
        if split == "test":
            frame = frames["test"]
            save_instance_predictions(pd.DataFrame({
                "instance_id": frame.instance_id, "gt_class": y, "pred_class": p, "score": probs.max(1),
                "box_area": frame.w * frame.h, "aspect": (frame.w * frame.img_w) / (frame.h * frame.img_h)}),
                args.out / "instance_predictions.csv")
    example = torch.randn(1, 3, args.size, args.size, device=device)
    latency = measure_latency(model, example, device)
    pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)
    summary = {
        "model": {"resnet50": "ResNet-50", "mobilenet_v2": "MobileNetV2", "vit_b_16": "ViT-B/16", "vit_tiny": "ViT-tiny (smoke)"}[args.arch],
        "family": "Attention / Transformer" if is_vit else "CNN classifier",
        "task": "damage-instance classification (GT crop)",
        "arch": args.arch,
        "params": count_parameters(model),
        "latency_ms_bs1": latency,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "train_minutes": train_minutes,
        "epochs_run": len(history),
        "best_val_macro_f1": best_f1,
        "config": {"lr": lr, "weight_decay": wd, "batch": args.batch, "size": args.size, "epochs": args.epochs,
                   "optimizer": "AdamW", "schedule": "cosine" + (" + 1 warm-up epoch" if is_vit else ""),
                   "loss": "class-weighted cross-entropy, label smoothing 0.05", "seed": args.seed,
                   "pretrained": not args.no_pretrained,
                   "augmentation": "RandomResizedCrop(0.75-1), HorizontalFlip, ColorJitter"},
        "samples": {s: len(d) for s, d in datasets.items()},
        "val": results["val"],
        "test": results["test"],
    }
    save_json(summary, args.out / "metrics.json")
    plot_outputs(history, results["test"], args.out, summary["model"])
    print(f"TEST accuracy {results['test']['accuracy']:.4f} macro-F1 {results['test']['macro_f1']:.4f} -> {args.out}")


def plot_outputs(history, test, out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(h.epoch, h.train_loss, marker="o")
    ax[0].set_title(f"{title}: training loss")
    ax[0].set_xlabel("epoch")
    ax[1].plot(h.epoch, h.val_accuracy, marker="o", label="val accuracy")
    ax[1].plot(h.epoch, h.val_macro_f1, marker="s", label="val macro-F1")
    ax[1].legend()
    ax[1].set_xlabel("epoch")
    plt.tight_layout()
    plt.savefig(out / "training_curves.png", dpi=180)
    plt.close()

    cm = np.array(test["confusion_matrix"])[:, :NUM_CLASSES]
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, axis = plt.subplots(figsize=(4.6, 4))
    axis.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            axis.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                      color="white" if cmn[i, j] > 0.5 else "black")
    axis.set_xticks(range(NUM_CLASSES), CLASS_CODES)
    axis.set_yticks(range(NUM_CLASSES), CLASS_CODES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(f"{title} test confusion matrix")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
