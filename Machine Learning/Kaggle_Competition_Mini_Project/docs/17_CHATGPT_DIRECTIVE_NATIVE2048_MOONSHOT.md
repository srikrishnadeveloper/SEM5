Directive 17 — Native-2048 Moonshot Toward 0.60 PQ

Issued: 2026-09-06
Authority: ChatGPT Master
Executor: Antigravity
Status: GO — build the legitimate high-resolution branch
Target: First reach the current 0.55 cluster; then attempt to beat the current 0.56 public leader and move toward 0.60+

0. Read this first

This directive supersedes the incremental roadmap in CHATGPT_MASTER_CONTROL.md wherever the two differ. All host-safety, leakage, split, RLE, and zero-overlap rules remain frozen.

We are deliberately making a large architectural move. This is not a promise that 0.60 will be reached. As of 2026-09-06, the live public leaderboard leader is 0.56, so 0.60 is beyond the demonstrated current frontier.

1. Forensic findings that control this build

1.1 The Kaggle Code-page numbers are not current truth

The Code page still displays historical best scores from the broken pre-rescore evaluator. Examples include 1.00, 0.93, 0.90, 0.70, and 0.67. They must not be treated as current PQ.

The live leaderboard currently begins at 0.56 and then contains a large 0.55 tie cluster.

1.2 Empty-mask exploit is forbidden

The public Please fix the Leaderboard notebook proved that five empty masks could once score 1.00. The official competition page says the board was re-scored on 2026-08-12 and, on 2026-08-20, explicitly criticized attempts to hack the system. Do not reproduce or submit any missing-image/empty-mask exploit.

1.3 Several apparent champions are not reusable model evidence

(LB 1st) Solar Filament Segmentation 2026 trains on empty targets because it indexes the COCO dictionary incorrectly, then emits 180 PPP8 empty masks. Its historical 0.93 is evaluator failure, not segmentation.

Solar Filament Unet Segmentation | 0.55+ visibly trains a weak model with near-zero validation Dice, but its final submission comes from a compressed CHAMPION_PAYLOAD containing 1,342 precomputed test RLE masks. It is a static submission replay, not a 0.55 U-Net.

The historical LB 0.70 | Topology-Safe... is a genuine Mask R-CNN + crop U-Net inference artifact, but its 0.70 was measured before the current PQ re-score. It is useful research code, not current 0.70 evidence.

1.4 The strongest transferable signal is native-resolution large YOLO

The current public 0.55 inference recipe uses:

YOLOv8l-seg

inference at native imgsz=2048

conf=0.30

iou=0.00

max_det=100

1,342 positive instance masks

no dummy rows for missing detections

Its referenced checkpoint is private/unavailable from the public notebook input, so it cannot simply be copied as a reproducible model. Nevertheless, the design signal is strong: large instance segmentation at native 2048 resolution, not YOLO11s at 1024 followed by mandatory refinement.

1.5 Architecture decision

Build a new direct-mask anchor around a large native-resolution YOLO model. Keep V8.1 as a comparison and optional auxiliary branch, not as the controlling architecture.

2. Primary hypothesis

V8.1 loses too many faint/thin filaments and too much boundary detail in the YOLO11s @ 1024 -> crop refiner cascade. A large segmentation detector trained and inferred at 1536–2048, followed by OOF-calibrated instance selection and optional gated refinement, can produce a step-change in recognition quality without the topology damage caused by semantic splitting.

3. Mandatory project layout

Create a new isolated branch; do not overwrite V8.1:

moonshot_2048/
  config.py
  data.py
  train_yolov8l.py
  train_yolo11l.py
  predict_native.py
  match_and_calibrate.py
  ensemble_instances.py
  gated_refiner.py
  audit_submission.py
  run_experiment.py
notebooks/
  build_moonshot_2048_kaggle.py
  Moonshot_2048_Train_Fold0.ipynb
  Moonshot_2048_Inference.ipynb
tests/
  test_moonshot_group_split.py
  test_moonshot_matching.py
  test_moonshot_ensemble.py
  test_moonshot_submission.py
docs/
  18_MOONSHOT_BUILD_REPORT.md

Use current repository utilities where verified. Copying code is allowed only when the behavior remains covered by tests.

4. Phase A — forensic import and environment audit

Search the local project for the Kaggle .ipynb files the human collected.

Record filename, SHA256, number of cells, referenced datasets/models, and whether each notebook contains:

real training;

external/private weights;

compressed/base64/lzma test predictions;

dummy or zero-area masks;

full 180-image inference;

current host-safe zero-overlap enforcement.

Never execute unknown embedded payloads during the audit.

If the actual notebooks are absent, report them as missing but continue using the verified recipes in this directive.

Probe whether the public HDJoJo checkpoint path is actually attachable. Do not fail the build if it is private.

Output a PUBLIC_NOTEBOOK_AUDIT table in the build report.

5. Phase B — dataset and evaluation foundation

5.1 Training records

Use only official competition training images and JSON.

Map category IDs 1–4 to one filament class.

Preserve each annotator observation as its own target sample.

Give duplicate physical images annotator-specific training filenames/links so labels are never overwritten.

Group train/validation by physical filename/year. Assert zero physical filename leakage.

Never use public Dataverse test-overlap labels or hidden/test-derived labels.

5.2 Evaluator

Use the exact current Kirillov PQ behavior with IoU strictly greater than 0.50.

For each physical validation image, score predictions against every annotator observation separately.

Report pq_mean, pq_max, SQ, RQ, TP, FP, FN, count ratio, and metrics by area/radial-position/crowding bins.

Select all thresholds with pq_mean, not pixel Dice.

6. Phase C — direct native-resolution anchor

6.1 First major model

Train yolov8l-seg.pt using the official train split.

Initial configuration:

Parameter

Value

image size

2048

epochs

60 initially, extend to 100 only if validation is still improving

batch

largest stable value, expected 1–2 on a 16 GB T4

AMP

on

workers

2

device

one GPU; do not use notebook DDP/DataParallel

patience

15

max detections

100

classes

1

seed

42

If 2048 OOMs at batch 1, use 1792, then 1536. Do not silently fall back to 1024.

Keep augmentations conservative for solar morphology. Allow flips, small rotations, mild scale/translation and contrast changes. Avoid aggressive mosaic/copy-paste until the direct baseline exists.

6.2 Validation sweep

Cache raw validation predictions once, then sweep:

confidence: {0.15, 0.20, 0.25, 0.30, 0.35, 0.40}

NMS IoU: {0.00, 0.05, 0.10, 0.20, 0.30}

minimum native area: {50, 100, 200, 400}

inference size: trained size plus one lower safe size

Every candidate must pass strict positive-area and zero-overlap sanitization before PQ scoring.

6.3 Promotion gate

Promote this branch only if its legal grouped-validation pq_mean is at least 0.48 and exceeds the V8.1 legal reference 0.4389 by at least 0.04 absolute.

If it fails, diagnose small-filament recall, crowded-instance merges, and boundary SQ before authorizing more GPU training.

7. Phase D — second large model and disagreement learning

After the native YOLOv8l anchor passes:

Train YOLO11l-seg at the largest stable size in {2048, 1792, 1536} using the identical split.

Produce OOF masks from both models.

Match cross-model instances using box prefiltering plus mask IoU.

For every candidate instance, compute:

model confidence;

mask area, perimeter, elongation, solidity and component count;

radial position on the disk;

mean/quantile grayscale darkness and local contrast;

cross-model agreement IoU;

TTA stability;

optional V8.1-refiner agreement.

Match OOF candidates to annotator GT and train a small calibrated classifier/regressor to estimate:

probability of a valid IoU>0.50 match;

expected matched IoU.

Optimize the selection threshold directly for grouped-validation PQ.

Use logistic regression or histogram gradient boosting first. The selector is valuable only if it improves held-out PQ; complexity is not a goal.

8. Phase E — topology-safe instance ensemble

Cluster predictions across model/fold/TTA views. For each cluster:

do not blindly union masks;

compare highest-confidence mask, majority-vote mask, intersection-biased mask, and optional refined mask on OOF;

choose the variant with the best learned expected IoU;

keep unmatched proposals only when calibrated match probability clears the PQ-optimal threshold;

sort by calibrated match probability, then expected IoU, then area;

apply final greedy pixel carve;

drop post-carve fragments below the selected minimum area;

assert pairwise overlap equals zero pixels.

The ensemble must improve grouped pq_mean by at least 0.02 over the direct YOLOv8l anchor before promotion.

9. Phase F — optional gated crop refiner

Do not make refinement mandatory.

Train a native-detail refiner using proposal-jittered GT crops, hard negative crops, and train-fold-only data. Start with a 512 crop and a strong but T4-safe backbone such as ConvNeXt-T or EfficientNet-B3 U-Net++.

Candidate losses:

BCE/Dice base;

boundary or distance-transform auxiliary loss;

a small clDice term only if it improves PQ and fragmentation metrics.

For each OOF proposal, compare the direct detector mask against the refined mask. Learn a gate that selects refinement only when expected IoU improves and topology remains stable. Reject refinement when component count explodes, area ratio leaves calibrated bounds, or proposal/refiner Dice is too low.

Promotion gate: at least +0.01 grouped pq_mean with no material regression for small or crowded filaments.

10. Kaggle inference candidates

The production notebook may generate several audited local files, but ChatGPT will authorize submission of only the best OOF-selected candidate:

candidate_a_yolov8l_native.csv

candidate_b_dual_yolo_calibrated.csv

candidate_c_gated_refiner.csv

Each audit report must include:

180 images processed;

row count and represented-stem count;

zero-area count = 0;

invalid-RLE count = 0;

wrong-shape count = 0;

pairwise shared pixels = 0;

duplicate filament_id count = 0;

rows/image distribution;

mask-area distribution;

exact configuration and artifact SHA256 values.

Images with zero predictions emit zero rows. Do not force 180 represented stems.

11. Forbidden shortcuts

Do not submit PPP8, all-zero anchors, missing-image exploits, or deliberately malformed/incomplete files.

Do not embed or replay precomputed test RLE payloads as the production algorithm.

Do not train on public MAGFiLO data that overlaps the competition test set.

Do not select with pixel Dice alone.

Do not use random image-ID splits that leak physical observations.

Do not claim historical Code-page scores as current PQ.

Do not claim 0.60 before Kaggle verifies it.

Do not overwrite V8.1 or delete its verified weights/artifacts.

12. Required execution order

Implement Phase A/B and all tests.

Build the fold-0 native YOLOv8l notebook.

Run compile/unit/synthetic smoke tests locally.

Stop and return the dry-run/config/audit report to ChatGPT.

After ChatGPT GO, human runs fold-0 training on Kaggle.

Return grouped validation metrics and failure bins.

ChatGPT decides whether to train the second model/remaining folds.

This is one moonshot program with evidence gates, not a sequence of tiny unrelated tweaks.

13. Required first response from Antigravity

Do not begin a long GPU run yet. Implement the structure, audit available notebooks, verify the dataset conversion and grouped split, build the training notebook, and return:

MOONSHOT_STATUS
PUBLIC_NOTEBOOK_AUDIT
ARCHITECTURE_IMPLEMENTED
FILES_CREATED_OR_CHANGED
TESTS_AND_RESULTS
KAGGLE_NOTEBOOK_PATH_AND_SHA256
EXPECTED_GPU_MEMORY_AND_RUNTIME
BLOCKERS
QUESTIONS_FOR_CHATGPT

14. Source pages reviewed by ChatGPT

Official evaluation and announcements: https://www.kaggle.com/competitions/filament-segmentation-2026/overview/evaluation

Live leaderboard: https://www.kaggle.com/competitions/filament-segmentation-2026/leaderboard

Native-2048 YOLOv8l 0.55 inference: https://www.kaggle.com/code/hdjojo/solar-filament-seg-inference

Static 0.55 payload notebook: https://www.kaggle.com/code/lamhuy8904/solar-filament-unet-segmentation-0-55

Historical topology-safe 0.70 handoff: https://www.kaggle.com/code/phuongncn/lb-0-70-topology-safe-solar-filaments-handoff

Empty-mask evaluator disclosure: https://www.kaggle.com/code/artkomissar/please-fix-the-leaderboard

