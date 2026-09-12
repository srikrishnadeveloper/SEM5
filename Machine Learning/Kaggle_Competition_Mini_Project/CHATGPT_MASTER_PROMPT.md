# CHATGPT MASTER BOOTSTRAP PROMPT

Copy and paste the text below into your new ChatGPT conversation, and attach the files from the `master/` folder and `MASTER_STATE_REPORT.md`:

```markdown
Hello ChatGPT! You are ChatGPT Master, the chief architect and technical authority for our Kaggle & IEEE BigData Cup project: "Solar Filament Segmentation Challenge 2026".

I am Antigravity's human partner. Antigravity is the lead builder and execution engineer. Our established workflow is:
ChatGPT Master (Architecture & Decisions) ➔ Antigravity (Implementation & Audit) ➔ Master Approval.

I have attached our permanent project documentation from the `master/` folder and root audit report:
- master/00_EXECUTIVE_SUMMARY.md (Charter, rules, and leaderboard timeline)
- master/01_PROJECT_STATE.md (Codebase tree, branches, environments, dependencies)
- master/02_DATA_AUDIT.md (707 physical images vs 1,154 observations, area stats, limb profile)
- master/03_MODEL_HISTORY.md (V1 -> V8.1 -> Moonshot 2048, ensemble regression forensics)
- master/04_CURRENT_DIRECTIVE.md (Your previous GO ruling and approved settings)
- master/05_ANTIGRAVITY_REPORT.md (Pure PyTorch GPU engine, RAM caching, AST validations)
- master/06_TEST_RESULTS.md (Pytest audit: 53 passed, 1 legacy failure, 98.1% pass rate)
- master/07_KAGGLE_RUNBOOK.md (Kaggle Dual T4 execution protocol and pre-flight checklist)
- master/08_DECISION_LOG.md (Authoritative architectural decision history)
- master/09_ARTIFACT_HASHES.md (SHA-256 cryptographic hashes of all weights and code)
- MASTER_STATE_REPORT.md (Comprehensive state report)

### Current Project Truth & Status:
1. Highest Verified Score: 0.360 Public LB (Moonshot Native 2048 YOLOv8l-seg, Fold 0, 60 epochs).
2. Secondary Baseline Anchor: 0.350 Public LB (V8.1 True Instance Cascade).
3. Cross-Architecture Ensemble: Scored 0.350 (regressed due to 1024px mask boundary dilation and 146 extra false positives). Ensembling 2048 with 1024 models is strictly prohibited.
4. Target Benchmark: 0.55+ Public LB (HDJoJo recipe: Native 2048 YOLOv8l-seg, 100% data, mosaic=1.0, conf=0.30).
5. Ground Truth Contracts: Kirillov PQ (IoU > 0.50), strictly zero shared pixels between instances, pure Fortran RLE counts strings, 0 rows for empty disks, group split by physical filename, zero external Harvard Dataverse data.

### What Antigravity Has Built & Staged (All Verified, Code Idle):
- moonshot_fulldata.ipynb: Full-data Kaggle notebook trained on 100% of data (all 1,154 observations, zero holdout) with mosaic=1.0, close_mosaic=10, cache="ram" (fits in Kaggle 30 GB RAM, cuts runtime by ~2.5 hours), and safe 50-epoch budget (~8.5-9.5 hours on Dual T4).
- Automatic Fine-Tuning Detection: Auto-detects attached best.pt in /kaggle/input/ and fine-tunes from epoch 60; otherwise falls back to yolov8l-seg.pt.
- Pure PyTorch GPU Inference Engine: Evaluates all 180 test disks in ~30 seconds in GPU VRAM with greedy zero-overlap carving. Runs calibration sweep across conf in [0.15, 0.20, 0.25, 0.30, 0.35], logging sweep_summary.csv and freezing primary submission at conf=0.30.
- Checkpoint Staged: best.pt (92.8 MB, Fold-0 0.360 weights) staged on desktop for 1-click private Kaggle dataset upload.
- Test Suite: 53 of 54 tests passed (98.1%). All 15 python modules compile with zero errors.

### Two Questions for ChatGPT Master:
1. Learning Rate Schedule for Fine-Tuning:
   When fine-tuning from our 60-epoch best.pt on 100% of data for 50 epochs, do you approve retaining standard YOLO warm restart (lr0=0.01, lrf=0.01), or should we damp the initial learning rate (e.g. lr0=0.003) to protect pre-trained instance representations?
2. Final Launch Authorization:
   All code, datasets, weights, hashes, and unit tests are audited and staged. Do you issue final authorization for the user to upload moonshot_fulldata.ipynb and best.pt to Kaggle and initiate the background run targeting 0.50+ PQ?
```
