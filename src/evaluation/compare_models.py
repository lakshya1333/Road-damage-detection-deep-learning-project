"""Single comparison table across all models on the common test instances + error analysis.

Run after every member has produced results/<model>/metrics.json and instance_predictions.csv:
    python -m src.evaluation.compare_models --results results --out results/final --manifest results/data --gallery
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.common.dataset import CLASS_CODES, NUM_CLASSES, crop_box
from src.common.metrics import classification_metrics, save_json

MODEL_DIRS = {  # folder -> owner
    "resnet50": "Lakshya Jain",
    "yolov8": "Dhruv Maheshwari",
    "unet": "Kumar Kshitij",
    "vit_b16": "MANAS LAXMAN KUDIPADY",
}
COLORS = ["#2a6f97", "#e76f51", "#2a9d8f", "#8e5ea2", "#999999"]
SIZE_BINS = [0, 0.005, 0.02, 0.08, 1.01]
SIZE_LABELS = ["tiny (<0.5%)", "small (0.5-2%)", "medium (2-8%)", "large (>8%)"]


def load(results: Path):
    loaded = {}
    for folder, owner in MODEL_DIRS.items():
        metrics_path, preds_path = results / folder / "metrics.json", results / folder / "instance_predictions.csv"
        if metrics_path.exists() and preds_path.exists():
            loaded[folder] = (json.loads(metrics_path.read_text()), pd.read_csv(preds_path), owner)
        else:
            print(f"[warn] {folder}: results missing, skipped")
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/final"))
    parser.add_argument("--manifest", type=Path, default=Path("results/data"))
    parser.add_argument("--gallery", action="store_true", help="save failure-case crops (needs the images)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    loaded = load(args.results)
    if not loaded:
        raise SystemExit("No model results found.")
    common_ids = set.intersection(*(set(p.instance_id) for _, p, _ in loaded.values()))
    print(f"{len(common_ids)} common test instances across {len(loaded)} models")

    rows, per_class, size_rows, summary = [], [], [], {}
    preds_by_model = {}
    for folder, (metrics, preds, owner) in loaded.items():
        preds = preds[preds.instance_id.isin(common_ids)].sort_values("instance_id").reset_index(drop=True)
        preds_by_model[folder] = preds
        m = classification_metrics(preds.gt_class, preds.pred_class)
        native = ""
        if "detection_test" in metrics and metrics["detection_test"]:
            d = metrics["detection_test"]
            native = f"mAP@0.5 = {d['mAP50']:.3f}; mAP@0.5:0.95 = {d['mAP50_95']:.3f}"
        elif "segmentation_test" in metrics:
            s = metrics["segmentation_test"]
            native = f"mIoU(damage) = {s['mIoU_damage']:.3f}; Dice = {s['mDice_damage']:.3f}"
        rows.append({
            "key": folder, "Model": metrics["model"], "Family": metrics["family"], "Owner": owner,
            "Params (M)": metrics["params"] / 1e6, "Latency (ms)": metrics.get("latency_ms_bs1"),
            "Accuracy": m["accuracy"], "Macro P": m["macro_precision"], "Macro R": m["macro_recall"],
            "Macro F1": m["macro_f1"], "Miss rate": m["miss_rate"], "Acc. on located": m["accuracy_on_detected"],
            "Task-native metric": native,
        })
        for code in CLASS_CODES:
            per_class.append({"Model": metrics["model"], "Class": code, **m["per_class"][code]})
        preds["size_bin"] = pd.cut(preds.box_area, SIZE_BINS, labels=SIZE_LABELS, right=False)
        for label, group in preds.groupby("size_bin", observed=False):
            size_rows.append({"Model": metrics["model"], "Box size": label, "n": len(group),
                              "Accuracy": float((group.gt_class == group.pred_class).mean()) if len(group) else np.nan,
                              "Miss rate": float((group.pred_class < 0).mean()) if len(group) else np.nan})
        cm = np.array(m["confusion_matrix"])
        confusions = sorted(((int(cm[i, j]), CLASS_CODES[i], CLASS_CODES[j] if j < NUM_CLASSES else "missed")
                             for i in range(NUM_CLASSES) for j in range(NUM_CLASSES + 1) if i != j), reverse=True)[:4]
        summary[folder] = {"model": metrics["model"], "common": m, "top_confusions": confusions,
                           "native": metrics.get("detection_test") or metrics.get("segmentation_test")}

    table = pd.DataFrame(rows)
    table.to_csv(args.out / "comparison_table.csv", index=False)
    pd.DataFrame(per_class).to_csv(args.out / "per_class_f1.csv", index=False)
    pd.DataFrame(size_rows).to_csv(args.out / "accuracy_by_box_size.csv", index=False)

    # McNemar exact tests (paired, same instances) between every pair of models
    tests = []
    for a, b in itertools.combinations(preds_by_model, 2):
        ca = preds_by_model[a].gt_class == preds_by_model[a].pred_class
        cb = preds_by_model[b].gt_class == preds_by_model[b].pred_class
        n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
        p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        tests.append({"A": summary[a]["model"], "B": summary[b]["model"], "A right/B wrong": n10, "A wrong/B right": n01, "p_value": p})
    pd.DataFrame(tests).to_csv(args.out / "mcnemar_tests.csv", index=False)

    best = table.sort_values("Macro F1", ascending=False).iloc[0]
    save_json({"n_common_instances": len(common_ids), "table": rows, "per_class": per_class, "by_box_size": size_rows,
               "mcnemar": tests, "models": summary, "best_model_macro_f1": best["Model"]}, args.out / "final_results.json")
    print(table.drop(columns=["key"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    plots(table, pd.DataFrame(per_class), pd.DataFrame(size_rows), summary, args.out)
    if args.gallery:
        gallery(preds_by_model, summary, args.manifest, args.out)


def plots(table, per_class, by_size, summary, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = table.Model.tolist()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(CLASS_CODES))
    width = 0.8 / len(names)
    for k, name in enumerate(names):
        vals = per_class[per_class.Model == name].set_index("Class").loc[CLASS_CODES, "f1"]
        ax.bar(x + k * width - 0.4 + width / 2, vals, width, label=name, color=COLORS[k % len(COLORS)])
    ax.set_xticks(x, ["D00 long.", "D10 trans.", "D20 alligator", "D40 pothole"])
    ax.set_ylabel("F1 (common protocol)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=len(names), loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out / "per_class_f1.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for k, row in table.reset_index(drop=True).iterrows():
        if pd.notna(row["Latency (ms)"]):
            ax.scatter(row["Latency (ms)"], row["Macro F1"], s=60 + 6 * row["Params (M)"], color=COLORS[k % len(COLORS)])
            ax.annotate(f"{row.Model}\n{row['Params (M)']:.1f}M", (row["Latency (ms)"], row["Macro F1"]),
                        textcoords="offset points", xytext=(8, -4), fontsize=8)
    ax.set_xlabel("Latency per image, batch 1 (ms)")
    ax.set_ylabel("Macro F1 (common protocol)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out / "f1_vs_latency.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for k, name in enumerate(names):
        sub = by_size[by_size.Model == name]
        ax.plot(range(len(SIZE_LABELS)), sub.Accuracy, marker="o", label=name, color=COLORS[k % len(COLORS)])
    ax.set_xticks(range(len(SIZE_LABELS)), SIZE_LABELS)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("GT box area (fraction of image)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out / "accuracy_by_box_size.png", dpi=200)
    plt.close()

    n = len(summary)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4))
    axes = np.atleast_1d(axes)
    for axis, (key, s) in zip(axes, summary.items()):
        cm = np.array(s["common"]["confusion_matrix"], dtype=float)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        axis.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                axis.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                          color="white" if norm[i, j] > 0.5 else "black")
        axis.set_xticks(range(NUM_CLASSES + 1), CLASS_CODES + ["miss"], fontsize=7)
        axis.set_yticks(range(NUM_CLASSES), CLASS_CODES, fontsize=7)
        axis.set_title(s["model"], fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "confusion_matrices.png", dpi=200)
    plt.close()


def gallery(preds_by_model, summary, manifest: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    instances = pd.read_csv(manifest / "instances.csv").set_index("instance_id")
    for key, preds in preds_by_model.items():
        wrong = preds[preds.gt_class != preds.pred_class].sample(min(12, int((preds.gt_class != preds.pred_class).sum())), random_state=0)
        if wrong.empty:
            continue
        fig, axes = plt.subplots(2, 6, figsize=(14, 5))
        for axis in axes.flat:
            axis.axis("off")
        for axis, row in zip(axes.flat, wrong.itertuples()):
            inst = instances.loc[row.instance_id]
            with Image.open(inst.image) as im:
                axis.imshow(crop_box(im.convert("RGB"), inst, pad=0.3))
            pred = CLASS_CODES[row.pred_class] if row.pred_class >= 0 else "missed"
            axis.set_title(f"GT {CLASS_CODES[row.gt_class]} / pred {pred}", fontsize=8)
        fig.suptitle(f"{summary[key]['model']}: failure cases (test)", fontsize=11)
        plt.tight_layout()
        plt.savefig(out / f"failures_{key}.png", dpi=130)
        plt.close()


if __name__ == "__main__":
    main()
