# GROK RED-TEAM + MASTER TAKEOVER — 2026-09-05

Roles now: Grok = master. Antigravity = executor. ChatGPT = inactive.
This document is both the independent red-team (requested before the 0.350 run) and the R2 decision after it.

## OFFICIAL_FACTS_VERIFIED

- Competition: Solar Filament Segmentation Challenge 2026, IEEE BigData Cup 02. Prize up to $3,000. Deadline recovered earlier as 15 Nov 2026 (confirm on Kaggle).
- Task: **instance** masks of individual filaments on 2048×2048 GONG H-α. [Kaggle](https://www.kaggle.com/competitions/filament-segmentation-2026)
- Primary LB metric after 7 Aug 2026: **Panoptic Quality**. Rescore announced 12 Aug. Kirillov 2019: unique match iff **IoU > 0.5**. PQ = SQ × RQ.
- MAGFiLO v1.0: 10,244 filaments, 1,593 files, 958 unique disks. Categories on mlecofi: Left=1, Right=2, Unidentifiable=3, Ambiguous=4. [mlecofi MAGFiLO](https://www.mlecofi.net/magfilo)
- Kaggle train: 8,199 anns. Test: 180 disks, ~2,045 anns still sitting in the public Dataverse (leakage if you train full MAGFiLO).
- Ultralytics default `overlap_mask: True`, `mask_ratio: 4`. `Boxes.xyxy` is in **original image pixel space** (`orig_shape`), not network input space. [Ultralytics default.yaml](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/default.yaml), [docs Boxes.xyxy](https://docs.ultralytics.com/it/reference/engine/results/)
- Current PyPI ultralytics is **8.4.140** (2026-09-04). Pin 8.4.103 is a frozen known-good, not latest. [PyPI ultralytics](https://pypi.org/project/ultralytics/)
- Host Self Evaluation notebook: **not retrieved** in this crawl (Kaggle JS). Treat local `metrics/pq.py` as Kirillov-standard, not proven byte-identical to host.

## CURRENT_PUBLIC_SOTA_CHECK

Kaggle code/LB pages did not render here. Treat title-scores as **unverified**.

| Claim | Source type | Post-rescore? | Use? |
|---|---|---|---|
| Our v1–v4 **0.22 / 0.26 / 0.30** | This team’s submits | After metric change for 0.30 | Trust |
| Our V8.1 **0.350** | User report, 1231-row CSV audit | Yes | Trust the number; I did not see the Kaggle UI |
| sivapnlr80 ~0.60 V21 | ChatGPT | Unverified | Do **not** rewrite V8.1 because of a title |
| hdjojo inference ~0.55, ~3 min T4x2 | ChatGPT / notebook exists | Unverified | Worth **reading**, not cloning |
| lamhuy8904 title “0.55+” | Title only; earlier note Best Score 0.34 V11 | Unverified / likely stale | Do not copy |
| Topology-safe “0.70” | Pre-rescore marketing | **No** — post-rescore ~0.34 claimed | Ignore thresholds |
| Mask R-CNN + U-Net ~0.33 | ChatGPT | Plausible | Same class as us, not ahead |
| Ghosh YOLO11m two-stage PQ **0.4145** | ResearchGate paper | Off-Kaggle MAGFiLO | Architecture confirmation |

**Do not alter Baseline-1 family because a public kernel title says 0.60.** Information gain from a cheap conf/min_area/mask-fallback sweep on **our** 0.350 weights is higher than a rewrite.

## ANTIGRAVITY_CLAIMS_VALIDATED

Evaluated against files in this chat (often **pre-R1**) vs ChatGPT’s production report (SHA `331ce02d…`).

| Claim | Verdict | Note |
|---|---|---|
| `$data_dir` shell bug fixed via subprocess | CONFIRMED | `build_v8_1_kaggle_nb.py` uses `subprocess.check_call` |
| YOLO `boxes.xyxy` not scaled 2048/1024 | CONFIRMED in attached infer | Matches Ultralytics docs |
| Unreadable image emits a row | PARTIAL | Attached infer emits zero-mask `{stem}_1`; production report says 0 rows for empties |
| No hardcoded PPP2/PPPP4 | CONFIRMED in attached encoder | |
| YOLO batch=2, device=0 | PARTIAL | Attached `config.py` still BATCH=4; notebook cell used `--batch 2`. Production may differ |
| Converter drops cat-4 | CONFIRMED in **attached** train/refiner (`if category_id == 4: continue`) | ChatGPT later said INCLUDE. Production 0.350 run may have included. **Must check the SHA 331ce tree** |
| One class filament | CONFIRMED | |
| Annotator observations separate | CONFIRMED | `stem_ann_{image_id}` |
| Physical filename leakage prevented | CONFIRMED (GroupKFold by year on samples) | 579/128 unique JPEGs claimed |
| 951/203 annotator samples | UNPROVEN here | Consistent with 1154 JSON images |
| Cross-FS materialize fallback | CONFIRMED in converter | symlink → hardlink → copy2 |
| Streamed val GT masks | UNPROVEN in attached `1_train_yolo.py` (file truncated) | Report says fixed |
| Notebook embeds match disk | UNPROVEN here | Hashes reported, not recomputed |
| Compile / PQ tests / RLE / smoke | UNPROVEN here | Report says pass; 0.350 submit implies plumbing worked |
| 0.350 public LB | UNPROVEN by me | User-reported. Treat as true unless Kaggle UI contradicts |

## MASTER_FINDINGS_VALIDATED

1. **Crop refiner runtime** — **SAFE_FOR_BASELINE** (after the fact). Full-2048 CLAHE per crop is wasteful (~8k crops × 20 epochs × decode+CLAHE+blur). Kernel still finished in ~43 min, so it is not a gate. **EXPERIMENT_LATER**: cache 3ch per source JPEG or crop-then-CLAHE. Crop-then-CLAHE **does** change distribution vs full-disk CLAHE; do not silently switch.

2. **Fake letterbox** — **MUST_FIX** was correct **before** the run; ChatGPT added `geometry.py`. Attached infer **still** does `cv2.resize` on a possibly non-square clipped window with no pad. If production `geometry.py` is in the 331ce notebook, this is **SAFE_FOR_BASELINE** now. Re-verify in R2 by asserting crop_h == crop_w before resize on 20 edge boxes. Limb filaments can stretch; not fatal for a first 0.350.

3. **No refiner val checkpoint** — **SAFE_FOR_BASELINE**. Keep final epoch for R2. **EXPERIMENT_LATER**: physical-file grouped crop val + best Dice/PQ ckpt. Do not leak val JPEGs into refiner train.

4. **GT-box train vs YOLO-box infer** — **SAFE_FOR_BASELINE** (did not block 0.350). **Next experiment after R2 sweep**: box jitter (±10–20%) and/or OOF YOLO boxes as refiner train. This is the most likely SQ leak.

5. **Local PQ vs host** — **UNPROVEN** vs Self Evaluation notebook (not fetched). Algorithmically: IoU>0.5 greedy unique match is Kirillov-correct. Empty-empty = PQ 1 is reasonable. **MUST_FIX** only if val-selected conf disagrees with a host-notebook clone. Until then do not retune the scorer.

6. **conf 0.25 hardcoded** — **MUST_FIX for R2** (inference only). Val already sweeps 0.15/0.25/0.35; test used 0.25. Pass **best val pq_mean conf** into Stage 3. That is the entire R2.

7. **Unpinned pip** — **SAFE_FOR_BASELINE** if production pinned 8.4.103. Keep the pin. Do not bump to 8.4.140 in R2. `device=0` is a single GPU. DDP `device=[0,1]` stays forbidden.

8. **overlap_mask** — **EXPERIMENT_LATER**. Default True merges overlapping GT into one training mask (integer overlap encoding). Thin crossing filaments are rare; MAGFiLO `iscrowd=0`. Ablate True vs False on val PQ later. Not a correctness blocker.

9. **YOLO masks thrown away** — **SAFE_FOR_BASELINE**, **highest-ROI R2 knob**. Enable fallback: if refiner empties, keep YOLO-seg mask. Zero extra train cost.

10. **Greedy non-overlap trim** — **SAFE_FOR_BASELINE**. MAGFiLO instances can touch. Trimming hurts SQ; NMS-at-0.5 (allow overlap pixels) may help. Ablate as `overlap_mode`. Do not guess.

11. **min_area=100** — **SAFE_FOR_BASELINE**, sweep in R2. GT 10th pctile ~410 px; 100 is permissive (FP risk) not recall-killing. 750 was the old RQ killer. Do not raise blindly.

12. **Doc drift (“True Union”, “DDP”, “Master: Grok”)** — **EXPERIMENT_LATER** / qualitative 30% rubric. Clean before IEEE report, not before R2 GPU.

13. **Public kernels** — **CONFIRMED as a class of bugs**, not line-by-line:
    - Filename-keyed lookup on raw COCO dict → empty masks / 180 dummy rows: do not copy.
    - Random split on image_id with duplicate physical JPEGs: leakage. Ours (year GroupKFold + unique files) is better.
    - Training Ambiguous as its **own class**: wrong for single-class PQ. Including it as filament class 0 is right.
    - DDP T4x2: useful runtime evidence, bad in notebooks.

## HIDDEN_BUGS_CHATGPT_MISSED

Evidence-backed, from attached (pre-R1) sources. Confirm against SHA 331ce before patching twice.

1. **`argparse --tta` with `store_true` and `default=True`** (`3_infer_cascade.py`). You cannot turn TTA off. Harmless if you want TTA on; not a gate.

2. **`--residual 1` is a no-op.** Print path claims enable; `run_cascade_inference` never implements residual. Fine (we want it off). Do not advertise it as a switch.

3. **Refiner `__getitem__` silent zero image** if `cv2.imread` fails (attached). ChatGPT said R1.1 raised RuntimeError. If production still zeros, it poisons crops. **MUST_FIX** if still present.

4. **Count mismatch vs hidden GT.** 1231 preds vs ~2045 test anns. If host uses all annotator copies, unique matching caps RQ (one pred cannot match two overlapping GT copies). If host uses one annotator, we are missing ~40% of filaments. R2 val histogram decides which. ChatGPT called 6.92 “exact match with MAGFiLO ~7.1” — that 7.1 is **train files including copies averaged down**. Test annotation density is higher. **Do not treat 6.92 as calibrated recall.**

5. **Notebook Cell 8 interpolates `{best_yolo}` inside a `!python` string.** Works in Jupyter. If they convert to a script, it breaks. SAFE in current notebook.

6. **Solar disk R=0.93** can clip off-limb prominences that MAGFiLO sometimes labels. **EXPERIMENT_LATER** (try 0.98 / no mask).

7. **Converter `PROJECT_ROOT / args.data_dir`:** if someone passes a relative path from a different cwd, Kaggle notebook is OK because it passes `str(data_dir)` absolute.

## HOST_PQ_COMPATIBILITY

| Behavior | Our `metrics/pq.py` (embedded) | Kirillov 2019 | Host Self Eval |
|---|---|---|---|
| Match threshold | IoU **>** 0.5 | IoU **>** 0.5 | **UNFETCHED** |
| Matching | greedy sort by IoU, unique | unique (trivial if >0.5) | unknown |
| empty ∩ empty | PQ=1 | undefined / 1 in practice | unknown |
| empty vs blob | PQ=0, FN or FP | yes | unknown |
| Aggregation | unweighted mean over images in `pq_score_multi` | dataset-level sum TP/FP/FN also used in literature | **critical unknown** |
| RLE | pycocotools Fortran 3D | n/a | likely same |
| Multi-annotator | not in scorer; eval wrapper can mean/max | n/a | unknown |

**MUST_FIX** only if we later clone the host notebook and find dataset-level PQ (sum TP across images) vs per-image mean. Those two numbers diverge when image instance counts vary. Until then, report **both** in R2 val table.

## BASELINE_1_RECOMMENDATION

Keep **exactly** this family:

`YOLO11s-seg @ 1024 → adaptive square crop → ResNet-34 U-Net @ 384 → 4-flip TTA → score assembly → COCO RLE`

Do **not** switch the first trained experiment to a public 0.60 U-Net. V8.1 already cleared 0.30 and produced a valid instance CSV. Next compute should buy **diagnostics + inference knobs**, not a new proposer.

## MINIMAL_PRE_KAGGLE_PATCH_LIST

The 0.350 run already happened. Remaining **must** items before the **next** GPU submit:

1. Load **existing** 0.350 weights. Do not retrain unless missing.
2. Val table of conf × min_area × yolo_mask_fallback × overlap_mode with PQ/SQ/RQ.
3. Stage 3 uses the **argmax val pq_mean** conf, not hardcoded 0.25.
4. Keep 0-row empty-disk policy.
5. Confirm production converter **includes category_id=4 as class 0**. If the 0.350 run still dropped them, turn include **on** only as a labeled ablation (second submit), not mixed with the conf sweep.

No geometry rewrite, no YOLO11m, no residual, no pin bump.

## EXPERIMENT_QUEUE_AFTER_BASELINE

Ordered by expected PQ / GPU-hour:

1. **R2 inference sweep** (this directive) — minutes if weights exist.
2. YOLO-mask fallback if table says so (may already be inside R2).
3. Refiner train on **jittered / OOF-YOLO boxes** (Stage 2 only, ~20 min).
4. Longer YOLO11s (40 ep) or imgsz 1280 on s, still not m.
5. `overlap_mask=False` YOLO retrain ablation.
6. Residual proposals, gated by val PQ on vs off.
7. YOLO11m-seg @ 1280 only after s saturates.
8. Read hdjojo / sivapnlr80 **code** (not titles) and steal **one** postprocess idea.

## VERDICT

**VERDICT: GO** for Directive R2 as specified in `11_R2_DIRECTIVE.md`.

Not GO for YOLO11m, residual fusion, or a public-0.60 rewrite.

### Human monitoring lines (when Antigravity’s notebook runs)

- Weights resolved: `best.pt` and `crop_refiner_r34.pth` exist (or a clear retrain).
- Val table printed with PQ/SQ/RQ; chosen conf ≠ 0.25 is allowed.
- Test CSV: 0 invalid RLEs, 0 zero-area masks, mean rows/image printed.
- n_rows likely 1,200–2,000. If n_rows stays ~1231 at conf=0.10, recall is not a conf problem.
- Wall time should be ≪ 43 min if no retrain; ~45–90 min if retrain required.
- Paste the footer to Grok **before** treating any new LB number as success.

Do not start a Kaggle run until Antigravity returns the val table (or proves weights are missing and must retrain in-kernel).
