# Viva preparation: likely questions per member

Each member is questioned on their own model (±20 % of marks). Know your script line by line.

## Everyone
- **Why a common protocol?** Accuracy, mAP and IoU measure different tasks. Assigning a class to every GT test instance puts all four models on the same scale. Missed instances count as errors, so localisation is accounted for.
- **Why macro-F1?** D00 dominates the data. Macro averaging gives the rare pothole class the same weight as D00.
- **Leakage?** The dataset ships with fixed train/val/test folders. One manifest is shared by all models, and crops are generated *after* the split.
- **Why no rotation augmentation?** Rotating a longitudinal crack by 90° turns it into a transverse crack, so the label would be wrong.
- **Why was D44 dropped?** The synopsis defines four classes, and "other corruption" is not one of them.

## Lakshya: ResNet-50
- Residual block: y = F(x) + x. The identity path lets gradients flow, which avoids degradation in deep nets. A bottleneck is 1×1 reduce → 3×3 → 1×1 expand.
- Transfer learning: ImageNet features are reused and the 1000-way FC layer is replaced by a 4-way one. All layers are fine-tuned with AdamW at 2e-4 and a cosine schedule.
- Loss: cross-entropy with inverse-√frequency class weights and label smoothing 0.05, which reduces over-confidence on noisy labels.
- Phase 2: five backbones all land at about 77–79 %, so labels and ambiguity limit accuracy, not model capacity.
- Read your confusion matrix: which pair is confused most (usually D00↔D10 or D20→D00), and why.

## Dhruv: YOLOv8
- One-stage, anchor-free detector. The backbone (CSPDarknet with C2f blocks) feeds a PAN-FPN neck at strides 8/16/32, then a decoupled head.
- Losses: BCE for classes, CIoU + Distribution Focal Loss for boxes. NMS removes duplicates.
- mAP@0.5 vs mAP@0.5:0.95: AP is the area under the precision–recall curve, averaged over classes, and the second averages over IoU thresholds too.
- Why thin cracks are hard: elongated boxes, where a small offset drops IoU below 0.5. Mosaic augmentation and the 640 px input size matter here.
- Video: frames are sampled at 5 fps and damage counts are logged per frame. Mention tracking as future work.

## Kumar: U-Net
- Encoder–decoder with skip connections, which concatenate high-resolution features. The encoder is ResNet-34.
- **No pixel masks in RDD2022**, so masks are box-derived and IoU measures region overlap. Be upfront about this limitation.
- Dice loss = 1 − 2|P∩G|/(|P|+|G|), which handles foreground imbalance. It is combined with weighted cross-entropy (background weight 0.5).
- Instance assignment: majority class inside the GT box, and missed if less than 5 % of the box is predicted as damage.
- The manifest and split design: how the class mapping is inferred and how degenerate boxes are handled.

## Manas: ViT-B/16 + evaluation
- Patch embedding: 224/16 = 14×14 = 196 tokens, plus a CLS token and position embeddings, through 12 layers × 12 heads. Attention = softmax(QKᵀ/√d)V.
- Weaker inductive bias than CNNs (no built-in locality), so ViT needs pretraining and careful optimisation: warm-up, low learning rate, weight decay 0.05.
- Why ViT and not an LSTM: the input is a static image with no sequence, so a recurrent model does not fit.
- McNemar test: a paired comparison on the same instances, counting where model A is right and B wrong versus the reverse.
- How the comparison table and figures are produced (`compare_models.py`).
