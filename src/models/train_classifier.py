"""Train a transfer-learning classifier on generated damage crops."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, class_names: list[str]) -> dict:
    model.eval()
    targets = []
    predictions = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(targets, predictions),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "classification_report": report,
        "confusion_matrix": confusion_matrix(targets, predictions, labels=list(range(len(class_names)))).tolist(),
        "sample_count": len(targets),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Directory containing train/val/test ImageFolder folders")
    parser.add_argument("--architecture", choices=("resnet50", "mobilenet_v2"), default="resnet50")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("runs/classification"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_set = datasets.ImageFolder(args.data / "train", transform=train_transform)
    val_set = datasets.ImageFolder(args.data / "val", transform=eval_transform)
    test_set = datasets.ImageFolder(args.data / "test", transform=eval_transform)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=0)

    if args.architecture == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(train_set.classes))
    else:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_set.classes))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_function = nn.CrossEntropyLoss()

    history = []
    best_val_accuracy = -1.0
    args.output.mkdir(parents=True, exist_ok=True)
    best_path = args.output / f"{args.architecture}_best.pt"
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        val_metrics = evaluate(model, val_loader, device, train_set.classes)
        history.append({"epoch": epoch + 1, "train_loss": train_loss / len(train_set), "val_metrics": val_metrics})
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            torch.save(model.state_dict(), best_path)
        print(history[-1])

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    final_metrics = {
        "validation": evaluate(model, val_loader, device, train_set.classes),
        "test": evaluate(model, test_loader, device, train_set.classes),
    }
    run = {
        "architecture": args.architecture,
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch,
        "classes": train_set.classes,
        "sample_counts": {"train": len(train_set), "validation": len(val_set), "test": len(test_set)},
        "history": history,
        "final_metrics": final_metrics,
    }
    (args.output / "metrics.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    (args.output / "confusion_matrix_test.json").write_text(json.dumps(final_metrics["test"]["confusion_matrix"], indent=2), encoding="utf-8")
    print(json.dumps(final_metrics["test"], indent=2))
    print(f"Saved best model, metrics, and confusion matrix to {args.output}")


if __name__ == "__main__":
    main()
