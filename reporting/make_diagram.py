"""Draw the system pipeline figure used in the report and slides (reporting/assets/pipeline.png)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "assets"
OUT.mkdir(exist_ok=True)
INK, MUTED = "#1d3557", "#5c677d"
C = {"data": "#e9f1f7", "a": "#d6e6f5", "b": "#fbe1d6", "c": "#d4efe9", "d": "#e7dcf1", "eval": "#fff3c4"}


def box(ax, x, y, w, h, text, color, bold=False, size=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec=INK, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.1))


fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 11)
ax.set_ylim(0, 4.6)
ax.axis("off")
box(ax, 0.1, 1.8, 1.7, 1.0, "RDD2022\n47,420 images\n6 countries\nVOC/YOLO boxes", C["data"], size=8.5)
box(ax, 2.2, 1.8, 1.8, 1.0, "Shared manifest\nfixed train/val/test\nD00 D10 D20 D40\nseed 42", C["data"], size=8.5)
arrow(ax, (1.8, 2.3), (2.2, 2.3))
rows = [(3.55, "GT crop 224 px", "A  ResNet-50\nCNN classifier", C["a"]),
        (2.55, "Full image 640 px", "B  YOLOv8s\none-stage detector", C["b"]),
        (1.55, "Image 512 px + box masks", "C  U-Net (ResNet-34)\nsegmentation", C["c"]),
        (0.55, "GT crop 224 px", "D  ViT-B/16\nattention classifier", C["d"])]
for y, inp, model, color in rows:
    box(ax, 4.5, y, 1.75, 0.75, inp, "#ffffff", size=8)
    box(ax, 6.6, y, 1.9, 0.75, model, color, bold=True, size=8.5)
    arrow(ax, (4.0, 2.3), (4.5, y + 0.37))
    arrow(ax, (6.25, y + 0.37), (6.6, y + 0.37))
    arrow(ax, (8.5, y + 0.37), (8.95, 2.3))
box(ax, 8.95, 1.55, 1.95, 1.5, "Common protocol\nclass of every GT\ntest instance\n(miss = FN)\n\nAcc · P · R · F1\n+ mAP / mIoU · ms", C["eval"], size=8.2)
plt.tight_layout()
plt.savefig(OUT / "pipeline.png", dpi=220)
print(OUT / "pipeline.png")
