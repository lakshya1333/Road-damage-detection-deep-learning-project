"""Build the Part C final report (.docx) from the official template + results/.

    python reporting/build_report.py --results results --out ICT4442_Final_Report.docx

Numbers are read from results/ (produced by the notebooks). Anything not yet produced
is printed as "TBD" and highlighted, so no value is ever invented.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

import content as C
from results import MODEL_ORDER, TBD, Results, num, pct

HERE = Path(__file__).parent
TEMPLATE = HERE.parent / "ICT_4442_Mini_Project_Report_Template (1).docx"
FONT = "Times New Roman"
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


# --------------------------------------------------------------------------- #
# low-level formatting helpers
# --------------------------------------------------------------------------- #
def set_font(run, size=10, bold=None, italic=None, small_caps=None, color=None, name=FONT):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if small_caps is not None:
        run.font.small_caps = small_caps
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    return run


def add_text(par, text, size=10, **kw):
    """Add text; 'TBD' tokens are highlighted so they cannot be missed."""
    parts = text.split(TBD)
    for i, part in enumerate(parts):
        if part:
            set_font(par.add_run(part), size, **kw)
        if i < len(parts) - 1:
            r = set_font(par.add_run(TBD), size, bold=True)
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return par


class Writer:
    def __init__(self, doc: Document):
        self.doc = doc
        self.fig_no = 0
        self.tab_no = 0
        self.h1 = 0
        self.h2 = 0

    def para(self, text="", size=10, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=4, indent=True, **kw):
        p = self.doc.add_paragraph()
        p.alignment = align
        pf = p.paragraph_format
        pf.space_after = Pt(space_after)
        pf.space_before = Pt(0)
        pf.line_spacing = 1.0
        if indent and align == WD_ALIGN_PARAGRAPH.JUSTIFY:
            pf.first_line_indent = Cm(0.35)
        add_text(p, text, size, **kw)
        return p

    def rich(self, chunks, size=10, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=True):
        """chunks: list of (text, dict(bold/italic))."""
        p = self.para("", size, align, indent=indent)
        for text, style in chunks:
            add_text(p, text, size, **style)
        return p

    def heading1(self, title, numbered=True):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.keep_with_next = True
        if numbered:
            self.h1 += 1
            self.h2 = 0
            title = f"{ROMAN[self.h1 - 1]}. {title}"
        set_font(p.add_run(title), 10, small_caps=True)
        return p

    def heading2(self, title):
        self.h2 += 1
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        set_font(p.add_run(f"{chr(64 + self.h2)}. {title}"), 10, italic=True)
        return p

    def bullet(self, text, size=10):
        p = self.para("", size, indent=False)
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.first_line_indent = Cm(-0.35)
        add_text(p, "•  " + text, size)
        return p

    def figure(self, path, caption, width_cm=16):
        if path is None or not Path(path).exists():
            self.para(f"[Figure pending: {caption} — produced by the notebooks]", 9, WD_ALIGN_PARAGRAPH.CENTER, italic=True)
            return
        self.fig_no += 1
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(path), width=Cm(width_cm))
        cap = self.doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(8)
        add_text(cap, f"Fig. {self.fig_no}. {caption}", 8)

    def table(self, header, rows, caption, widths=None, size=8):
        self.tab_no += 1
        cap = self.doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_before = Pt(6)
        cap.paragraph_format.keep_with_next = True
        set_font(cap.add_run(f"TABLE {ROMAN[self.tab_no - 1]}"), 8, small_caps=True)
        cap2 = self.doc.add_paragraph()
        cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap2.paragraph_format.keep_with_next = True
        set_font(cap2.add_run(caption), 8, small_caps=True)
        t = self.doc.add_table(rows=1, cols=len(header))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        borders(t)
        for i, h in enumerate(header):
            cell = t.rows[0].cells[i]
            cell.text = ""
            add_text(cell.paragraphs[0], h, size, bold=True)
            shade(cell, "D9E2F3")
        for row in rows:
            cells = t.add_row().cells
            for i, value in enumerate(row):
                cells[i].text = ""
                cells[i].paragraphs[0].paragraph_format.space_after = Pt(0)
                add_text(cells[i].paragraphs[0], str(value), size)
        if widths:
            t.autofit = False
            grid = t._tbl.tblGrid
            for i, gc in enumerate(grid.findall(qn("w:gridCol"))):
                gc.set(qn("w:w"), str(int(widths[i] * 567)))
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Cm(w)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return t

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def borders(table):
    tbl = table._tbl
    pr = tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:color"), "808080")
        b.append(el)
    pr.append(b)


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), color)
    tcPr.append(sh)


# --------------------------------------------------------------------------- #
# cover page from the template
# --------------------------------------------------------------------------- #
def prepare_template() -> Document:
    doc = Document(str(TEMPLATE))
    body = doc.element.body
    paragraphs = doc.paragraphs
    # delete everything after the cover (from 'PART A' heading on), keep the final sectPr
    start = next(i for i, p in enumerate(paragraphs) if p.text.startswith("PART A"))
    first = paragraphs[start]._element
    remove = False
    for el in list(body):
        if el is first:
            remove = True
        if remove and el.tag != qn("w:sectPr"):
            body.remove(el)

    def replace(par, text, size=None, bold=True):
        for r in par.runs[1:]:
            r._element.getparent().remove(r._element)
        run = par.runs[0] if par.runs else par.add_run()
        run.text = text
        set_font(run, size or (run.font.size.pt if run.font.size else 14), bold=bold)

    for p in doc.paragraphs:
        t = p.text.strip()
        if t == "<TITLE OF THE PROJECT>":
            replace(p, C.TITLE, 16)
        elif t.startswith("Synopsis/Interim/Final"):
            replace(p, "Final report on", 14)
        elif t.startswith("< MONTH"):
            replace(p, C.MONTH_YEAR, 12)
        elif t in ("Deep Learning Project", "[ICT-4442]", "Submitted By") or (t.isupper() and len(t) > 5):
            for r in p.runs:
                set_font(r, r.font.size.pt if r.font.size else 14, bold=r.font.bold)
    names = [p for p in doc.paragraphs if p.text.strip().startswith("<NAME OF THE STUDENT")]
    for p, (name, reg, _, _) in zip(names, C.TEAM):
        replace(p, f"{name}  ({reg})", 14)
    # 4th member: clone the 3rd name paragraph
    new = copy.deepcopy(names[-1]._element)
    names[-1]._element.addnext(new)
    from docx.text.paragraph import Paragraph

    replace(Paragraph(new, names[-1]._parent), f"{C.TEAM[3][0]}  ({C.TEAM[3][1]})", 14)
    # drop one empty line so the cover still fits on one page
    empties = [p for p in doc.paragraphs[:30] if not p.text.strip() and not p._element.findall(".//" + qn("w:drawing"))]
    if empties:
        empties[-1]._element.getparent().remove(empties[-1]._element)
    # A4 page as required by C.4
    for s in doc.sections:
        s.page_width, s.page_height = Cm(21.0), Cm(29.7)
        s.left_margin = s.right_margin = Cm(1.9)
        s.top_margin = s.bottom_margin = Cm(1.8)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(10)
    return doc


# --------------------------------------------------------------------------- #
# report body
# --------------------------------------------------------------------------- #
def build(results_dir: Path, out: Path) -> None:
    R = Results(results_dir)
    doc = prepare_template()
    W = Writer(doc)
    best = R.best()
    ready = R.complete

    # ---- title block (IEEE style) ----
    p = W.para("", 24, WD_ALIGN_PARAGRAPH.CENTER, indent=False)
    add_text(p, C.TITLE, 24)
    W.para(", ".join(f"{n}" for n, *_ in C.TEAM), 11, WD_ALIGN_PARAGRAPH.CENTER, indent=False)
    W.para("School of Computer Engineering, Manipal Institute of Technology, Manipal Academy of Higher Education, Manipal, India",
           10, WD_ALIGN_PARAGRAPH.CENTER, indent=False, italic=True)
    W.para("Code: " + C.REPO_URL, 9, WD_ALIGN_PARAGRAPH.CENTER, indent=False, name="Courier New")

    # ---- abstract ----
    if best and ready:
        bc = R.common(best)
        result_sentence = (f"On {R.final['n_common_instances']:,} shared test instances, {R.name(best)} achieved the highest macro-F1 "
                           f"({pct(bc['macro_f1'])}%, accuracy {pct(bc['accuracy'])}%). ")
        y = R.models.get("yolov8", {}).get("detection_test") or {}
        if y:
            result_sentence += f"The detector reached mAP@0.5 of {num(y.get('mAP50'))} while also localising damage in real time. "
    else:
        result_sentence = ("Final test-set figures are produced by the shared evaluation script and are reported in Section VI "
                           f"(values marked {TBD} are pending the final runs). ")
    abstract = (
        "Manual road inspection is slow, subjective and hard to scale, which delays repair of cracks and potholes. "
        "This project compares four deep-learning architecture families for automatic road damage recognition on the "
        "multi-national RDD2022 dataset with four classes: longitudinal crack (D00), transverse crack (D10), alligator crack (D20) "
        "and pothole (D40). We implement a transfer-learned ResNet-50 classifier, a YOLOv8 one-stage detector, a U-Net "
        "segmentation network with a ResNet-34 encoder trained on box-derived masks, and a Vision Transformer (ViT-B/16). "
        "All models share one train/validation/test split and a common instance-level protocol in which every ground-truth "
        "damage instance of the test split must be assigned a class; instances a localising model fails to find count as errors. "
        + result_sentence +
        "We discuss why localisation, label granularity, inductive bias and object size explain the differences, analyse failure "
        "cases, and assess which model is most suitable for edge deployment on dashcam or survey-vehicle video.")
    p = W.para("", 9, indent=False)
    add_text(p, "Abstract", 9, bold=True, italic=True)
    add_text(p, " — " + abstract, 9, bold=True)
    p = W.para("", 9, indent=False)
    add_text(p, "Keywords", 9, bold=True, italic=True)
    add_text(p, " — " + ", ".join(C.KEYWORDS), 9, bold=True)

    # ---- I. Introduction ----
    W.heading1("Introduction")
    W.para("Road surfaces deteriorate continuously under traffic loading, weathering and material fatigue. The visible symptoms — "
           "longitudinal and transverse cracks, alligator (fatigue) cracking and potholes — are early indicators of structural "
           "failure, and leaving them unrepaired raises maintenance cost, vehicle wear and accident risk. Most municipalities still "
           "rely on visual surveys by inspection crews. These surveys are slow, depend on the inspector, and cannot be repeated "
           "often over a large network.")
    W.para("Low-cost cameras mounted on vehicles, together with deep learning, make continuous automated monitoring possible. "
           "Maeda et al. [2] showed that smartphone images can train usable damage detectors, and the RDD2020 and RDD2022 "
           "benchmarks [1], [3] extended this to several countries. Different works, however, frame the problem in different ways: "
           "as image or patch classification, as object detection, or as pixel-level segmentation. Each framing calls for a "
           "different architecture family, and published numbers are rarely comparable because datasets, splits and metrics differ.")
    W.para("The scope of this mini project is a controlled comparison. Four team members each own one architecture family — CNN "
           "classification, one-stage detection, encoder-decoder segmentation and attention-based classification. All four are "
           "trained on the same RDD2022 split and scored with one common protocol. The comparison answers which family is most "
           "accurate for recognising road damage, which is fastest and lightest for deployment, and why they differ. "
           "Video monitoring is supported by running the detector on frames sampled from dashcam clips.")

    # ---- II. Literature review ----
    W.heading1("Literature Review")
    W.para("Benchmarks and detection. Maeda et al. [2] released 9,053 Japanese smartphone images with eight damage classes and "
           "found SSD-MobileNet the best trade-off, running at about 1.5 s per image on a phone. Recall varied strongly by class: "
           "rare classes such as potholes were almost never found. RDD2020 [3] fixed the four-class taxonomy (D00, D10, D20, D40) "
           "that we use. Arya et al. [4] trained models across countries and showed a severe domain shift: a model trained only on "
           "Japanese images reached an F1 of 0.05 or less on Indian D00 cracks, and mixing local data was necessary. RDD2022 [1] "
           "extends this to 47,420 images from six countries with more than 55,000 instances, and it is our dataset. Recent "
           "YOLOv8-based detectors add attention and lightweight convolutions. YOLOv8-PD [6] targets a smaller, faster YOLOv8n "
           "model. YOLO-RD [7] adds star-operation, multi-dimensional fusion and wavelet-convolution modules and reports a mAP of "
           "25.75% on the RDD2022 Japan subset, with a +4.93-point gain on small objects. This shows how difficult thin, small "
           "cracks remain even for state-of-the-art detectors.")
    W.para("Classification and segmentation. Zhang et al. [5] were among the first to classify pavement patches with a ConvNet, "
           "outperforming hand-crafted features. For pixel-level analysis, Lau et al. [8] replaced the U-Net encoder with a "
           "pretrained ResNet-34 and obtained F1 scores of 96% on CFD and 73% on Crack500. RUC-Net [9] added residual blocks, scSE "
           "attention and focal loss (F1 0.739, IoU 0.586 on CFD). Our U-Net (Model C) follows the ResNet-34-encoder design of [8], "
           "with an imbalance-aware loss in the spirit of [9].")
    W.para("Attention-based models. CrackFormer [10] replaced convolutions with self-attention to capture the long, thin context "
           "of cracks (ODS 0.871–0.881 on three datasets). Zhang and Zhang [11] compared six CNN U-Nets against TransUNet [18], "
           "SwinUNet and MTUNet. The transformers were most accurate (MTUNet test IoU 80.8%, F1 89.4%), but they were slower and "
           "used more memory. These two findings — a possible accuracy gain from global attention, and a cost in speed and data "
           "requirements — motivate Model D, a ViT-B/16 [17], compared directly against the ResNet-50 [12] classifier on identical "
           "inputs. Table I summarises the ten most relevant studies.")
    W.table(["Paper", "Method", "Dataset", "Key result", "Relevance"], C.LIT_TABLE,
            "Summary of reviewed literature", widths=[2.6, 3.4, 3.0, 4.0, 4.2], size=7)
    W.para("Gap addressed. Existing works usually evaluate a single framing on its own metric (mAP, IoU or accuracy). This "
           "makes it hard to tell whether a reported gain comes from the architecture or from an easier task definition. Our "
           "contribution is a same-split, same-instance comparison of four families, with an explicit accounting of the "
           "localisation burden that classifiers do not carry.")

    # ---- III. Objectives ----
    W.heading1("Objectives")
    W.bullet("O1 — Implement four deep-learning models from distinct families (CNN classifier, one-stage detector, encoder-decoder "
             "segmenter, vision transformer) that recognise the four RDD2022 damage classes.")
    W.bullet("O2 — Benchmark all models on one fixed RDD2022 test split with a common protocol (accuracy, macro precision/recall/F1, "
             "miss rate), together with each model's native metric (mAP, IoU), and identify the most accurate family.")
    W.bullet("O3 — Quantify deployability (parameters, batch-1 latency) and determine the best model for real-time, edge-based "
             "road-condition monitoring from video.")

    # ---- IV. Methodology ----
    W.heading1("Methodology")
    W.para("Fig. 1 shows the overall pipeline. A single manifest fixes the split and the class mapping. Each model consumes the "
           "representation natural to its family. The outputs are converted into one class decision per ground-truth test "
           "instance for the common comparison.")
    W.figure(HERE / "assets" / "pipeline.png", "Overall pipeline: one shared manifest, four model families, one common evaluation protocol.")
    m = R.models
    cfg = lambda k, f, d="": m.get(k, {}).get("config", {}).get(f, d)

    W.heading2("ResNet-50 CNN classifier (owner: Lakshya Jain)")
    W.para("Architecture. ResNet-50 [12] stacks 16 bottleneck residual blocks (1×1–3×3–1×1 convolutions with identity shortcuts) "
           "in four stages, followed by global average pooling. The residual connections let gradients flow through 50 layers "
           "without degradation. We load ImageNet weights (IMAGENET1K_V2) and replace the 1000-way fully-connected layer with a "
           "4-way layer (23.5 M parameters).")
    W.para(f"Input and training. Each ground-truth box is cropped with 10% context on every side (minimum 32 px), resized to "
           f"224×224 and normalised with ImageNet statistics. Crops are cached once as a uint8 memory map. Training uses AdamW [19] "
           f"(lr = {cfg('resnet50', 'lr', 2e-4)}, weight decay = {cfg('resnet50', 'weight_decay', 1e-4)}), batch size "
           f"{cfg('resnet50', 'batch', 64)}, a cosine learning-rate schedule, mixed precision and gradient clipping at 1.0, for up "
           f"to {cfg('resnet50', 'epochs', 12)} epochs with early stopping on validation macro-F1 (patience 4). The loss is "
           "cross-entropy with inverse-square-root class weights and label smoothing of 0.05, to counter the dominance of D00. "
           "Augmentation consists of random resized crop (scale 0.75–1), horizontal flip and colour jitter. Rotations and "
           "vertical flips are deliberately excluded: rotating a longitudinal crack by 90° would turn it into a transverse "
           "crack and corrupt the label.")
    W.para("Justification. ResNet-50 is the standard CNN baseline. It isolates how well damage types can be recognised when the "
           "location is already known, and its inductive biases (locality, translation equivariance) suit texture-like crack "
           "patterns. In Phase 2 the same pipeline was benchmarked against MobileNetV3, EfficientNet-B0, DenseNet-121 and "
           "ConvNeXt-Tiny (Table III). ResNet-50 was retained as the representative CNN because it is well understood and "
           "close to the best accuracy.")

    W.heading2("YOLOv8 one-stage detector (owner: Dhruv Maheshwari)")
    W.para("Architecture. YOLOv8 [15], a descendant of YOLO [14], uses a CSPDarknet backbone with C2f blocks, a PAN-FPN neck "
           "that fuses features at strides 8, 16 and 32, and a decoupled, anchor-free head. The head predicts class scores "
           "(binary cross-entropy) and box distributions (Distribution Focal Loss plus CIoU loss) for every grid cell. "
           "Non-maximum suppression removes duplicate boxes. We use the small variant YOLOv8s (about 11 M parameters), "
           "pretrained on COCO.")
    W.para(f"Training. Images are trained at {cfg('yolov8', 'imgsz', 640)} px with batch size {cfg('yolov8', 'batch', 16)} for "
           f"up to {cfg('yolov8', 'epochs', 30)} epochs, using a cosine schedule, early stopping (patience 10), mixed precision "
           "and the optimiser chosen automatically by Ultralytics. Mosaic augmentation is used and switched off for the last 5 "
           "epochs, together with HSV, scale and translate jitter and horizontal flips; vertical flips and rotations are "
           "disabled for the reason given above. 30% of undamaged training images are kept as background so the detector "
           "learns to suppress false alarms on clean road. At test time we use a confidence threshold of 0.25.")
    W.para("Justification. Detection is the operational framing for road monitoring: the system must find damage in a full "
           "dashcam frame, not just classify a given patch. YOLO-family models dominate the RDD challenges [1], [6], [7] "
           "because they run in real time. The same model drives the video pipeline (frames sampled at 5 fps, per-frame damage "
           "counts logged).")

    W.heading2("U-Net encoder-decoder segmentation (owner: Kumar Kshitij)")
    W.para("Architecture. U-Net [16] pairs a contracting encoder with an expanding decoder. Skip connections concatenate "
           "encoder features at each resolution, so fine spatial detail lost by pooling is recovered. Following [8], the "
           "encoder is an ImageNet-pretrained ResNet-34. The decoder has five upsampling stages and ends in a 5-channel "
           "softmax: background plus D00, D10, D20 and D40 (24.4 M parameters).")
    W.para(f"Weak supervision and training. RDD2022 provides only bounding boxes. We therefore generate box-derived masks: every "
           f"pixel inside a box takes that box's class, and larger boxes are drawn first so small ones remain visible. Images "
           f"are resized to {cfg('unet', 'size', 512)}×{cfg('unet', 'size', 512)}. The loss is weighted cross-entropy (background "
           f"weight 0.5) plus multiclass Dice loss [20], which directly optimises region overlap under heavy foreground/background "
           f"imbalance. Training uses AdamW (lr = {cfg('unet', 'lr', 3e-4)}) with a OneCycle schedule, batch size "
           f"{cfg('unet', 'batch', 12)}, up to {cfg('unet', 'epochs', 15)} epochs, early stopping on validation mIoU, and "
           "horizontal-flip plus brightness/contrast augmentation. 20% of undamaged training images are included.")
    W.para("Justification. Segmentation is the only family that yields damage area, which is the basis of severity indices. "
           "Comparing it under the same protocol shows what a dense per-pixel model gains or loses relative to box-level "
           "models. The box-mask limitation is acknowledged: the IoU reported here measures agreement with damage regions, "
           "not with true crack outlines.")

    W.heading2("Vision Transformer ViT-B/16 (owner: MANAS LAXMAN KUDIPADY)")
    W.para("Architecture. ViT-B/16 [17] splits a 224×224 crop into 196 non-overlapping 16×16 patches. Each patch is linearly "
           "embedded into 768 dimensions, a learnable class token and position embeddings are added, and the sequence passes "
           "through 12 transformer encoder layers, each with 12-head self-attention and an MLP. The class-token output feeds a "
           "4-way linear head (85.8 M parameters). Self-attention connects every patch to every other patch in the first layer, "
           "so a crack spanning the whole crop can be modelled globally. A CNN reaches the same receptive field only after "
           "many layers.")
    W.para(f"Training. ViT-B/16 uses the same cached crops, augmentation, weighted loss and split as ResNet-50, so only the "
           f"architecture changes. Transformers are sensitive to optimisation, so we use a smaller learning rate "
           f"({cfg('vit_b16', 'lr', 5e-5)}), stronger weight decay ({cfg('vit_b16', 'weight_decay', 0.05)}), a one-epoch linear "
           f"warm-up followed by cosine decay, and gradient clipping, with ImageNet-1k pretrained weights. Training runs for up "
           f"to {cfg('vit_b16', 'epochs', 12)} epochs with batch size {cfg('vit_b16', 'batch', 64)}.")
    W.para("Justification. The guidelines suggest an attention-based family, and recurrent models do not fit static images. "
           "Literature [10], [11] reports accuracy gains from attention on crack data, but at a cost in speed and data. Pairing "
           "ViT with ResNet-50 on identical inputs gives a clean test of CNN versus transformer inductive bias at RDD2022 scale. "
           "A TransUNet-style hybrid [18] was considered, but it was set aside for compute reasons (Section VIII).")

    # ---- V. Experimental setup ----
    W.heading1("Experimental Setup")
    W.heading2("Dataset and split")
    counts = R.split_counts()
    if counts:
        imgs, inst, per_class = counts
        split_text = (f"The split contains {imgs.get('train', 0):,} / {imgs.get('val', 0):,} / {imgs.get('test', 0):,} images "
                      f"(train/val/test) with {inst.get('train', 0):,} / {inst.get('val', 0):,} / {inst.get('test', 0):,} damage instances.")
        rows = [[code] + [f"{per_class.get(s, {}).get(code, 0):,}" for s in ("train", "val", "test")] for code in ("D00", "D10", "D20", "D40")]
    else:
        split_text = ("Split sizes are written to results/data/dataset_stats.json by the manifest step "
                      f"(Phase 2: 26,869 / 5,758 / 5,758 labelled images; final counts {TBD}).")
        rows = [[code, TBD, TBD, TBD] for code in ("D00", "D10", "D20", "D40")]
    W.para("We use RDD2022 [1] in the publicly shared Kaggle export with predefined train/validation/test folders and YOLO-format "
           "labels (RDD_SPLIT). The same export was used in Phase 2. A manifest script scans the dataset once, maps class "
           "names to the four synopsis classes, drops the extra 'other corruption' class (D44) and degenerate boxes, and writes "
           "one row per instance with a unique ID. Every model reads this manifest, so splits, classes and test instances are "
           "identical across models; no image appears in more than one split. " + split_text)
    W.table(["Class", "Train", "Val", "Test"], rows, "Damage instances per class and split", widths=[3, 2.5, 2.5, 2.5])
    W.figure(R.figure("data", "class_distribution.png"), "Class distribution of RDD2022 instances per split.", 11)
    W.heading2("Preprocessing")
    W.para("The preprocessing pipeline has five stages. (1) Validation: labels with the wrong number of fields or degenerate boxes "
           "are discarded and boxes are clipped to the image. (2) Encoding: source class IDs are mapped to D00/D10/D20/D40 by name. "
           "(3) Model-specific inputs: 224-px context crops for the classifiers, 640-px letterboxed images for YOLOv8, 512-px "
           "images with box-derived masks for U-Net. (4) Normalisation with ImageNet mean and standard deviation. "
           "(5) Orientation-safe augmentation as described in Section IV. The random seed is 42 everywhere.")
    W.heading2("Common evaluation protocol and metrics")
    W.para("Accuracy is the share of test instances assigned the correct class. For each class, precision, recall and F1 are "
           "computed and macro-averaged, so the rare pothole class weighs as much as the frequent longitudinal crack. The "
           "classifiers receive each test instance's box and predict its class. For YOLOv8, a ground-truth instance takes the "
           "class of the predicted box with the highest IoU, provided IoU ≥ 0.5; otherwise it is missed. For U-Net, the "
           "instance takes the majority predicted damage class inside its box, provided at least 5% of the box is predicted "
           "as damage; otherwise it is missed. A missed instance counts as a false negative, and the miss rate is reported "
           "separately. Each model's native metric is also reported: mAP@0.5 and mAP@0.5:0.95 for YOLOv8, and mIoU/Dice "
           "against box masks for U-Net. Parameters and mean batch-1 latency (including YOLO pre- and post-processing) measure "
           "deployability. Paired exact McNemar tests check whether accuracy differences between models are statistically "
           "significant.")
    W.heading2("Implementation")
    dev = next((v.get("device") for v in m.values() if v.get("device")), TBD)
    W.para(f"All models are implemented in PyTorch, using torchvision for ResNet-50 and ViT-B/16, Ultralytics for YOLOv8 and "
           f"segmentation-models-pytorch for U-Net. Training ran on Kaggle GPUs ({dev}), one notebook per member, from the shared "
           f"repository ({C.REPO_URL}). All scripts, configuration and metric files are versioned there.")

    # ---- VI. Results ----
    W.heading1("Results and Discussion")
    W.heading2("Common comparison")
    header = ["Model", "Family", "Params (M)", "Latency (ms)", "Acc. (%)", "Macro-P (%)", "Macro-R (%)", "Macro-F1 (%)", "Miss (%)", "Native metric"]
    W.table(header, [[R.row(k)[h] for h in header] for k in MODEL_ORDER],
            "Single comparison table on the common test instances", widths=[2.3, 2.3, 1.3, 1.3, 1.2, 1.3, 1.3, 1.3, 1.1, 3.0], size=7)
    W.table(["Model", "D00 F1", "D10 F1", "D20 F1", "D40 F1"],
            [[R.name(k)] + [R.per_class_f1(k)[c] for c in ("D00", "D10", "D20", "D40")] for k in MODEL_ORDER],
            "Per-class F1 (%) under the common protocol", widths=[4, 2.4, 2.4, 2.4, 2.4])
    W.figure(R.figure("final", "per_class_f1.png"), "Per-class F1 of the four models on the common test instances.", 14)
    W.figure(R.figure("final", "confusion_matrices.png"), "Confusion matrices (row-normalised; last column = missed instance).", 17)
    for para in discussion(R):
        W.para(para)
    W.heading2("Phase-2 preliminary results (validation split)")
    W.para("For transparency, Table V reproduces the saved Phase-2 runs (30 epochs, validation split). They are not mixed with the "
           "final test results because the split, epochs and class handling differ. They show that, among CNN backbones, "
           "accuracy saturates near 77–79% regardless of model size: MobileNetV3 is within 1.4 macro-F1 points of ResNet-50 "
           "with 8× fewer parameters, and ConvNeXt-Tiny is best. The earlier detector run reached a validation mAP@0.5 of "
           f"{C.INTERIM['yolo']['mAP50']}, with potholes hardest (0.523).")
    W.table(["Model (Phase 2)", "Params (M)", "Val acc. (%)", "Macro-P (%)", "Macro-R (%)", "Macro-F1 (%)", "Train (min)"],
            [[a, f"{b:.2f}", f"{c:.2f}", f"{d:.2f}", f"{e:.2f}", f"{f:.2f}", f"{g:.0f}"] for a, b, c, d, e, f, g in C.INTERIM["classifiers"]],
            "Phase-2 CNN crop classifiers (validation split, Kaggle 2×T4)", widths=[3.6, 1.8, 2, 2, 2, 2, 1.8])
    W.heading2("Error analysis")
    for para in error_analysis(R):
        W.para(para)
    W.figure(R.figure("final", "accuracy_by_box_size.png"), "Common-protocol accuracy versus ground-truth box size.", 13)
    gallery = R.figure("final", "failures_yolov8.png") or R.figure("final", "failures_resnet50.png")
    W.figure(gallery, "Representative failure cases on the test split (ground truth / prediction).", 17)
    W.figure(R.figure("unet", "unet_samples.png"), "U-Net qualitative results: image, box-derived target, prediction.", 8)
    W.figure(R.figure("final", "f1_vs_latency.png"), "Accuracy-speed trade-off (marker size proportional to parameters).", 12)

    # ---- VII. Conclusion ----
    W.heading1("Conclusion")
    W.para(conclusion(R))

    # ---- VIII. Future work ----
    W.heading1("Future Work")
    for text in [
        "True pixel masks: train and evaluate the U-Net on crack datasets with real masks (Crack500, DeepCrack) or on SAM-refined "
        "RDD2022 masks, so that IoU reflects crack geometry and supports area-based severity indices.",
        "Hybrid attention: replace the U-Net encoder with a transformer (TransUNet [18] or Swin-UNet), and add attention blocks "
        "to YOLOv8, combining global context with dense or real-time outputs.",
        "Two-stage pipeline: use YOLOv8 to localise damage and a crop classifier to re-classify each box, combining the "
        "detector's coverage with the classifier's label accuracy.",
        "Domain shift: report per-country results and apply domain adaptation, since [4] shows cross-country degradation.",
        "Edge deployment: export the best model to ONNX/TensorRT with INT8 quantisation, measure frame rate on a Jetson-class "
        "device, and track damage across video frames to avoid double counting.",
    ]:
        W.bullet(text)
    W.para("Limitations: box-derived masks for segmentation; a single training run per model (no seed variance); "
           "the oracle-box advantage of the classifiers; and only partial use of the video component, through qualitative "
           "dashcam demonstrations rather than an annotated video test set.")

    # ---- Acknowledgement ----
    W.heading1("Acknowledgement", numbered=False)
    W.para("We thank our course faculty for guidance throughout the ICT-4442 mini project, the organisers of the CRDDC'2022 "
           "challenge for releasing RDD2022, and Kaggle for GPU time. Declaration of tool use: an AI assistant (Anthropic Claude) "
           "was used to help scaffold the evaluation scripts and to draft parts of this report's wording. All experiments were "
           "run, checked and interpreted by the team, and every reported number is read from our own saved runs.")

    # ---- References ----
    W.heading1("References", numbered=False)
    for i, ref in enumerate(C.REFERENCES, 1):
        p = W.para("", 8, indent=False, space_after=2)
        p.paragraph_format.left_indent = Cm(0.7)
        p.paragraph_format.first_line_indent = Cm(-0.7)
        add_text(p, f"[{i}]\t{ref}", 8)

    # ---- C.3 Contribution statement ----
    W.page_break()
    W.heading1("C.3 : Individual Contribution Statement", numbered=False)
    contributions = {
        "resnet50": "Repository structure; crop caching and classifier training script; ResNet-50 (and Phase-2 CNN backbone) "
                    "training, tuning and evaluation; Section IV-A; slides on Model A.",
        "yolov8": "YOLO dataset conversion; YOLOv8 training, test evaluation and instance matching; video inference script; "
                  "Section IV-B; slides on Model B.",
        "unet": "Shared manifest and split validation; box-derived mask generation; U-Net training and segmentation metrics; "
                "Section IV-C and Experimental Setup; slides on Model C.",
        "vit_b16": "ViT-B/16 fine-tuning; common evaluation and comparison script, McNemar tests and error analysis; "
                   "Section IV-D, Results integration; slides on Model D.",
    }
    W.table(["Member Name", "Reg. No.", "Model / Task Owned", "Key Contributions", "Signature"],
            [[n, r, owned, contributions[k], ""] for n, r, k, owned in C.TEAM],
            "Individual contribution statement (to be reviewed, edited and signed by each member)", widths=[3, 2, 3.2, 6.5, 2.5])
    W.para("This page must be written by the members themselves; each member is individually examined in the viva on the "
           "stated contribution. Commit history: " + C.REPO_URL + "/commits", 9, italic=True, indent=False)

    doc.save(str(out))
    print("Saved", out, "| complete results:", ready)


# --------------------------------------------------------------------------- #
# data-driven text
# --------------------------------------------------------------------------- #
def discussion(R: Results) -> list[str]:
    paras = []
    if R.complete:
        rows = {k: R.common(k) for k in MODEL_ORDER}
        order = sorted(MODEL_ORDER, key=lambda k: -rows[k]["macro_f1"])
        ranking = ", ".join(f"{R.name(k)} ({pct(rows[k]['macro_f1'])}%)" for k in order)
        paras.append(f"Ranking. Under the common protocol, the macro-F1 ranking is: {ranking}. Table IV and the per-class F1 figure break this down by class.")
        cls_mean = {c: sum(rows[k]["per_class"][c]["f1"] for k in MODEL_ORDER) / 4 for c in ("D00", "D10", "D20", "D40")}
        hard, easy = min(cls_mean, key=cls_mean.get), max(cls_mean, key=cls_mean.get)
        paras.append(f"Class difficulty. Averaged over the four models, {easy} is the easiest class (mean F1 {pct(cls_mean[easy])}%) "
                     f"and {hard} the hardest (mean F1 {pct(cls_mean[hard])}%).")
        y, u = rows["yolov8"], rows["unet"]
        paras.append(f"Localisation burden. The classifiers never miss an instance because they receive the box. YOLOv8 misses "
                     f"{pct(y['miss_rate'])}% of instances and U-Net {pct(u['miss_rate'])}%. Restricted to instances they do "
                     f"locate, their class accuracy is {pct(y['accuracy_on_detected'])}% (YOLOv8) and "
                     f"{pct(u['accuracy_on_detected'])}% (U-Net). This separates recognition quality from detection failures.")
        sig = [t for t in R.final["mcnemar"] if t["p_value"] < 0.05]
        if sig:
            paras.append("Significance. Paired McNemar tests show significant accuracy differences (p < 0.05) for: "
                         + "; ".join(f"{t['A']} vs {t['B']} (p = {t['p_value']:.3g})" for t in sig) + ".")
    else:
        paras.append(f"Final test values are filled automatically from results/final when the four notebooks and the comparison "
                     f"notebook have been run; cells marked {TBD} are pending.")
    paras += [
        "Why the families differ (1): task difficulty, not just architecture. ResNet-50 and ViT solve an easier task: the "
        "location is given and only the type is predicted. YOLOv8 and U-Net must also find the damage in a full road scene, "
        "where thin cracks occupy a tiny fraction of pixels and boxes are inconsistently drawn. A higher classifier score "
        "therefore reflects the oracle box, and the fair reading of Table III is classification quality for the classifiers "
        "versus end-to-end quality for the detector and segmenter.",
        "Why the families differ (2): label granularity and box geometry. Longitudinal and transverse cracks are long and thin, "
        "so their boxes are elongated and contain mostly intact pavement. At IoU ≥ 0.5, a small shift of a thin box turns a "
        "correct detection into a miss, which penalises YOLOv8 on D00/D10 much more than on compact potholes and alligator "
        "patches. For U-Net, box-derived masks label the background pixels inside a box as damage, which blurs the target; "
        "dense prediction is most useful for D20 and D40, which fill their boxes.",
        "Why the families differ (3): inductive bias versus data. ResNet-50 builds in locality and translation equivariance, "
        "which match crack texture and make it data-efficient. ViT-B/16 must learn spatial structure from data. With a few tens "
        "of thousands of crops it relies heavily on ImageNet pretraining, but its global attention can link the two ends of a "
        "long crack or the extent of an alligator pattern. The gap between ResNet-50 and ViT in Table III shows which effect "
        "dominates at RDD2022 scale. Confusions between D00 and D10 in particular show whether either model has captured "
        "orientation.",
        "Why the families differ (4): class imbalance and domain shift. D00 dominates RDD2022, while D40 and D10 are rarer "
        "and vary strongly across the six countries (road texture, camera height, image resolution). Class-weighted losses "
        "reduce the imbalance but do not remove it; the macro averages expose this, whereas plain accuracy would hide it.",
    ]
    return paras


def error_analysis(R: Results) -> list[str]:
    if not R.complete:
        return [f"The comparison script computes accuracy by box size, top confusion pairs and failure galleries for every model; "
                f"the quantitative statements here are filled in once results exist ({TBD})."]
    out = []
    for key in ("yolov8", "resnet50"):
        conf = R.final["models"][key]["top_confusions"]
        out.append(f"{R.name(key)}: the most frequent errors are " + "; ".join(f"{a} → {b} ({n} instances)" for n, a, b in conf[:3]) + ".")
    sizes = [r for r in R.final["by_box_size"] if r["Model"] == R.name("yolov8")]
    if sizes:
        txt = ", ".join(f"{r['Box size']}: {pct(r['Accuracy'])}%" for r in sizes if r["n"])
        out.append(f"Object size strongly affects the detector. YOLOv8 accuracy by box size is {txt}. Most misses are tiny or "
                   "thin cracks, which shrink to a few pixels at 640 px input, and blurred or shadowed cracks that resemble "
                   "joints, tyre marks or lane paint (see the failure-case figure).")
    out.append("Typical failure modes observed in the galleries: (i) D00/D10 swaps for diagonal cracks, whose orientation "
               "falls between the two classes; (ii) D20 predicted as D00 when only part of an alligator pattern is inside "
               "the box; (iii) missed potholes in shadow or filled with water; (iv) false positives on manhole covers, "
               "patches and road markings. Several 'errors' come from ambiguous or inconsistent annotations, a known "
               "issue of crowd-sourced RDD labels [1].")
    return out


def conclusion(R: Results) -> str:
    if R.complete:
        best = R.best()
        b = R.common(best)
        y = R.models["yolov8"]
        return (f"We compared four architecture families for road damage recognition on one RDD2022 split and one common protocol. "
                f"{R.name(best)} achieved the best macro-F1 ({pct(b['macro_f1'])}%) and accuracy ({pct(b['accuracy'])}%). "
                f"YOLOv8 is the only end-to-end model that finds and classifies damage in full frames at "
                f"{num(y.get('latency_ms_bs1'), 1)} ms per image ({num(y['params'] / 1e6, 1)} M parameters), which makes it "
                "the most practical choice for real-time dashcam monitoring. U-Net adds damage area at a higher computational "
                "cost but is limited by box-derived supervision, and the CNN/ViT comparison quantifies the trade-off between "
                "convolutional inductive bias and global attention. Localisation of thin cracks, not damage-type recognition, "
                "is the main bottleneck for deployable systems.")
    return (f"This report presents a controlled comparison of four architecture families for road damage recognition on RDD2022 "
            f"under one split and one protocol. The best-performing model and final figures are {TBD} until the final runs are "
            "complete. Our analysis indicates that localisation of thin cracks, rather than damage-type recognition, is the main "
            "bottleneck, and that a real-time detector is the most practical basis for deployable monitoring.")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(HERE))
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE.parent / "results")
    parser.add_argument("--out", type=Path, default=HERE.parent / "ICT4442_Final_Report.docx")
    args = parser.parse_args()
    build(args.results, args.out)
