# ICT-4442 Phase 2 Execution Plan

Project: Deep Learning-Based Road Damage Detection and Classification for Automated Road Condition Monitoring
Team: Lakshya Jain, Dhruv Maheshwari, Kumar Kshitij, MANAS LAXMAN KUDIPADY

## Phase 2 submission checklist

Submit the following through the faculty's MS Forms before the extended deadline:

- Interim report, 3-5 pages, using Part B of the supplied template.
- GitHub repository link with visible per-member commit history.
- Literature review containing 8-10 real papers in the required table.
- Dataset acquisition and preprocessing description, supported by code and generated evidence.
- At least two proposed models implemented and run with preliminary, reproducible results.
- Individual contribution log/task matrix signed by all four members.
- Risk and remaining-work plan through the final submission.

Do not submit invented metrics, fabricated dates, or copied commit history. Every reported number must come from a saved run, and every contribution must be made by the person whose name appears in the commit history.

## Recommended Phase 2 scope

Implement the two models already identified as the first milestones:

1. Lakshya: ResNet-50 or MobileNetV2 classification baseline.
2. Dhruv: YOLOv8 object detector.

Kumar and Manas should begin the U-Net and transformer work, but those models are not required for the minimum Phase 2 implementation unless the team has enough verified results. They should still contribute preprocessing, literature, evaluation, documentation, and risk planning with commits of their own.

## Required technical decisions before training

- Download RDD2022 from its official source and record the source URL, version/date, and license or usage terms.
- Fix one deterministic train/validation/test split and reuse it for every model. Split by original image/country before deriving patches or frames to avoid leakage.
- Keep the four RDD2022 classes explicit: D00 longitudinal crack, D10 transverse crack, D20 alligator crack, and D40 pothole.
- For YOLOv8, convert the original XML bounding boxes to YOLO label format and verify sample images visually.
- For classification, define the task precisely: full-image multi-label classification or damage-patch classification. Do not compare classification accuracy directly with detector mAP without explaining the different task.
- RDD2022 provides bounding boxes, not pixel masks. Do not claim true U-Net segmentation results unless masks are sourced or a documented mask-generation method is used and acknowledged as a limitation.
- Record image counts, class counts, split counts, input size, augmentation, batch size, epochs, optimizer, learning rate, hardware, software versions, random seed, and training time.

## Phase 2 report structure

### Cover and team information

Use the confirmed project title, all four names and registration numbers, the GitHub URL, and the actual submission month/year.

### 1. Literature review

Include 8-10 verified papers in a table with: paper/authors/year, method, dataset, key result, and relevance. Verify every citation from the original paper or publisher page. The seven synopsis references may be retained, but add at least one to three relevant papers and do not copy performance claims without checking them.

### 2. Dataset acquisition and preprocessing

Describe the official download, annotation format, class mapping, data validation, duplicate handling, split strategy, resizing/normalization, augmentation, and YOLO conversion. Add repository evidence such as a dataset statistics CSV, sample visualization, and preprocessing script. Never commit the dataset itself if its license or size disallows it.

### 3. Models implemented so far

For each completed model include architecture, transfer-learning choice, input/output format, loss, training setup, and preliminary results. Include links or paths to the exact training script, configuration, saved metrics, and sample predictions. At minimum report metrics appropriate to each task:

- Classification: accuracy, macro precision, macro recall, macro F1, and confusion matrix.
- Detection: precision, recall, mAP@0.5, mAP@0.5:0.95, and sample prediction images.

Do not merge these into one misleading score. A common final comparison protocol will be defined after all models are implemented.

### 4. Contribution log and task matrix

Use a signed table listing each member's completed work, evidence path, commit IDs, and approximate effort. The four members must review and sign it. The commit history is supporting evidence, not a substitute for the table.

### 5. Risks and plan to final

Document segmentation-mask availability, compute/time limits, class imbalance, country/domain shift, video-frame leakage, and the possibility of reducing from four models to three. Give an owner and target date for every remaining task.

## Honest four-person Git history

Create one shared GitHub repository and protect the main branch if practical. Each member configures their own Git identity and pushes their own work. Do not squash all work into one person's account and do not manufacture old timestamps or commits.

Suggested commit sequence:

1. Lakshya: repository structure, README, environment file, and data contract.
2. Dhruv: RDD2022 XML-to-YOLO conversion and dataset validation.
3. Kumar: deterministic split, preprocessing utilities, augmentation, and dataset statistics.
4. Manas: shared evaluation utilities, visualization tools, and experiment configuration.
5. Lakshya: classification training/evaluation and first verified results.
6. Dhruv: YOLOv8 training/evaluation and first verified results.
7. Kumar: literature table and interim report sections for data/preprocessing.
8. Manas: contribution matrix, risk plan, figures, and report integration.
9. All members: separate review/fix commits for issues each person actually resolves.

Use descriptive messages such as `feat(data): convert RDD annotations to YOLO format` and `results(cls): add seed-42 baseline metrics`. Do not use one giant final commit. Do not commit credentials, the raw dataset, model weights, or generated caches.

## Evidence to collect before submission

- `README` with setup and reproduction commands.
- Dependency file with pinned or bounded versions.
- Dataset source and download instructions.
- Dataset statistics and class distribution.
- Fixed split manifest or deterministic split seed.
- Training logs and metric files.
- Confusion matrix and detector prediction examples.
- Model configuration and exact commands used.
- GitHub commit history showing all four identities.
- Signed contribution matrix.
- Final plagiarism and AI-content reports within the guideline thresholds.

## Current blocker

This workspace currently contains only the Phase 1 synopsis, guidelines, and report template. It does not contain the dataset, code, Git repository, GitHub URL, or training results. Those items are required before the implementation and the numerical parts of the interim report can be completed truthfully.
