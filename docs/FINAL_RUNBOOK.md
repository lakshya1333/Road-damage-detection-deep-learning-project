# Final submission runbook (read this first)

Presentations run 26–29 Oct 2026. Report and code are due **Sat 31 Oct 2026, 11:59 PM**. Code, notebooks, report and slides are all ready. The only missing piece is the **real final numbers**, and only you can produce them: they need the RDD2022 data and a GPU. Every number in the report and slides is either one of your real Phase-2 values or a highlighted `TBD` that fills in automatically.

## Step 0: push the code (Lakshya, 10 min)

Copy this folder into your cloned repo. Each member then commits **their own files** under their own Git identity. This is the honest history the guidelines check.

| Member | Files to commit (suggested message) |
|---|---|
| Lakshya | `src/common/`, `src/models/train_classifier.py`, `README.md`, `requirements.txt`, `.gitignore`, `notebooks/01_*` (`feat: shared data contract + CNN classifier`) |
| Dhruv | `src/models/train_yolo.py`, `src/inference/video_infer.py`, `notebooks/02_*` (`feat(det): YOLOv8 training + instance matching + video`) |
| Kumar | `src/data/build_manifest.py`, `src/models/train_unet.py`, `notebooks/03_*` (`feat(seg): manifest + U-Net with box-derived masks`) |
| Manas | `src/evaluation/compare_models.py`, `notebooks/04_*`, `notebooks/05_*`, `reporting/` (`feat(eval): common protocol comparison + report builders`) |

Afterwards, each member commits their own `results/<model>/` folder (metrics, CSV, PNG; no weights).

## Step 1: run the four notebooks in parallel (each member on their own Kaggle account)

1. Kaggle → New Notebook → File → Import → choose your notebook from `notebooks/`.
2. Settings: Accelerator **GPU T4 x2**, Internet **On**. Add Input: **aliabdelmenam/rdd-2022** (the dataset you used in Phase 2).
3. Run All. Check that the class mapping printed by the manifest cell is D00/D10/D20/D40.
4. Download `results_<model>.zip` from the Output panel.

| Notebook | Approx. T4 time | If short of time |
|---|---|---|
| 01 ResNet-50 | 35–50 min | `--epochs 8` |
| 02 YOLOv8s | 3–3.5 h | `--epochs 20`, or `--model yolov8n.pt` |
| 03 U-Net | 1.5–2 h | `--size 384 --epochs 10` |
| 04 ViT-B/16 | 45–60 min | `--epochs 8` |

## Step 2: comparison + report + slides (anyone, about 15 min)

Open `05_final_comparison.ipynb` on Kaggle and add the four zips plus the dataset as inputs, then Run All. It writes `results/final/`, then **rebuilds `ICT4442_Final_Report.docx` and `ICT4442_Final_Presentation.pptx` with the real numbers.**
On a laptop you can do the same with `python reporting/build_report.py` and `python reporting/build_slides.py` once `results/` is unzipped into the repo.

## Step 3: before submitting

- [ ] Search the report and slides for `TBD`: none should remain.
- [ ] Each member rewrites their own Methodology subsection and Section C.3 **in their own words**. The guideline requires ≤ 20 % AI content and ≤ 15 % plagiarism on Turnitin, and the C.3 page must be written by the members themselves.
- [ ] Keep or adjust the AI-tool declaration in the Acknowledgement. The guidelines require you to say where an LLM was used.
- [ ] Sign C.3. Submit report + GitHub link on MS Forms.
- [ ] Remove or annotate the fabricated "Proposed Dual-Branch" output in the Phase-2 notebook.

## If you present before the final runs finish

Present with the Phase-2 slide (real numbers) and the method slides. Say plainly that the final test numbers are being produced with the shared protocol. Never fill `TBD` by hand with guesses.
