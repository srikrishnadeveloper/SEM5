# ChatGPT Master Control — Solar Filament Segmentation 2026

**Effective date:** 2026-09-06  
**Master / final technical authority:** ChatGPT  
**Executor / builder:** Antigravity  
**Human:** Cloud operator and copy/paste bridge  
**Grok:** Inactive; historical advice only unless the human explicitly says `Grok back`

## 1. Authority order

When two files disagree, use this order:

1. `16_CHATGPT_MASTER_BRIEFING_AND_WORKFLOW.md`
2. `00_MASTER_CONTEXT_CHATGPT.md`
3. current ChatGPT directive issued after this control sheet
4. current executable V8.1 source and its passing tests
5. current official Kaggle rules and host behavior
6. R3 directive/GO records
7. Grok verdicts, old worklogs, research notes, and retired V1–V8 plans

Never let an older document silently override a newer host-verified rule.

## 2. Mission and success criteria

Predict one pixel-precise instance mask per solar filament in 2048×2048 GONG H-alpha images and submit COCO RLE rows to Kaggle.

- Immediate target: exceed **0.40 public PQ**.
- Verified production anchor: **0.350 public PQ** from V8.1 Baseline 1.
- Do not claim any score as verified unless the user provides the Kaggle result or it is independently verified from current official evidence.

## 3. Competition truths

- Primary metric: Kirillov Panoptic Quality, with one-to-one matches at **IoU > 0.50**.
- Dataset: 707 physical train JPEGs, 1,154 annotator observations, 8,199 polygons, and 180 test JPEGs.
- All category IDs 1–4 map to the single class `filament`; chirality is not scored.
- Split by physical filename/year group, never COCO `image_id` alone.
- Keep annotator observations separate; never OR-merge their masks.
- Submission columns are exactly `filament_id,segmentation_rle`.
- Emit one row per positive-area predicted instance.
- If an image has no predictions, emit **zero rows** for it. Dummy zero masks are retired and forbidden.
- Masks from the same disk must share **zero pixels**.
- Encode positive masks with `pycocotools` in Fortran order and verify decode shape 2048×2048.
- Never train on the full public Harvard Dataverse MAGFiLO release because it contains competition test imagery.
- Never expose or commit the Kaggle API token.

## 4. Current production stack

```text
2048×2048 GONG image
  -> YOLO11s-seg proposer @ 1024
  -> native-coordinate proposal boxes
  -> boundary-safe square crop, side 256–512
  -> 3 channels: raw + CLAHE + unsharp/high-pass
  -> ResNet-34 U-Net crop refiner @ 384
  -> 4-flip crop TTA
  -> paste to native 2048 canvas + solar-limb clip
  -> confidence/area ordered greedy pixel carve
  -> drop carved masks below min_area
  -> assert strict pairwise zero overlap
  -> COCO Fortran RLE
  -> submission.csv
```

Baseline 1 uses residual discovery **off**. Training uses one GPU to avoid notebook DDP/DataParallel instability; dual GPUs may be used for model-parallel inference.

## 5. Verified state versus live state

### Verified

- V8.1 Baseline 1 public score: **0.350**.
- Baseline artifact produced 1,231 rows across 178 disks; two disks emitted zero rows.
- Historical semantic pipelines plateaued around 0.22–0.30 because merged/split topology and false positives dominate PQ.
- The legal validation sweep winner was trim mode at approximately **0.4389 local PQ**, with `conf=0.20`, `min_area=400`, fallback off.
- Allow-overlap scored higher locally but was rejected by the host and is illegal for submission.

### Live / not yet verified

- Directive R3 was launched on Kaggle using the host-safe settings below.
- The R3 audit output and official public leaderboard score are still required.
- Expected score ranges, public notebook titles, and alleged 0.55/0.60 methods are hypotheses until current post-rescore evidence is verified.

## 6. Frozen R3 configuration

| Setting | Value |
|---|---:|
| confidence | 0.20 |
| minimum post-carve area | 400 px |
| YOLO fallback | off / 0 |
| overlap mode | trim |
| crop TTA | on |
| residual discovery | off |
| current R3 notebook SHA256 | `29a7120d69e83bd2b7bf678e3236c23f6aa1725bc0e64575804d152a15ea61ed` |

The earlier SHA `331ce02d...` identifies the 0.350 baseline lineage, not the current R3 notebook.

## 7. Immediate workflow

### Gate A — collect the active R3 evidence

The human returns only:

- conversion train/validation counts;
- last YOLO epochs and `pq_mean` / `pq_max` table;
- first-image native `xyxy` debug range;
- submission audit: rows, unique stems, empty count, rows/image distribution;
- zero-overlap audit result;
- official public leaderboard score.

### Gate B — ChatGPT verdict

ChatGPT checks:

1. source/artifact identity and frozen configuration;
2. 180 test images processed (not necessarily 180 represented stems because empty images emit no rows);
3. every RLE is positive-area and decodes to 2048×2048;
4. zero shared pixels between instances on each disk;
5. plausible instance-count distribution;
6. score compared with the verified 0.350 anchor.

Then ChatGPT issues exactly one decision: `ACCEPT`, `FIX INFERENCE`, or `NEXT EXPERIMENT`.

### Gate C — next experiment, chosen from evidence

- If R3 is below 0.30: treat as artifact/weights/coordinate/contract failure; do not train a larger model.
- If R3 is 0.30–0.40: diagnose proposer recall and legal trim damage before expanding architecture.
- If R3 exceeds 0.40: preserve the stack and run one controlled ablation at a time.

Priority order after a healthy R3 result:

1. Measure proposer false negatives and recall by filament size/location.
2. Test YOLO11m-seg or 1280 input as a single proposer-recall ablation.
3. Test radial limb normalization as a controlled preprocessing ablation.
4. Test one stronger crop-refiner backbone.
5. Test proposer/refiner ensembling.
6. Test residual discovery last, only with local PQ evidence and zero-overlap enforcement.

## 8. Execution contract for Antigravity

Every implementation request must contain:

- objective and one measurable hypothesis;
- exact files allowed to change;
- frozen behavior that must remain unchanged;
- tests and smoke commands;
- expected output artifacts;
- stop condition;
- report footer: `SUMMARY`, `FILES_TOUCHED`, `VERIFICATION`, `RESULTS`, `RISKS`, `QUESTIONS_FOR_CHATGPT`.

Antigravity implements and reports evidence. It does not redefine competition truth, approve its own experiment, or promote an unverified score.

## 9. Resource map

| Resource | Purpose | Status |
|---|---|---|
| `16_CHATGPT_MASTER_BRIEFING_AND_WORKFLOW.md` | Main handoff, current architecture, R3 state | Authoritative |
| `00_MASTER_CONTEXT_CHATGPT.md` | Compact truths, roles, constraints | Authoritative |
| `AGENTS(1).md` | Current authority header plus repo operating guide | Current, but use top authority header first |
| `AGENTS.md` in project sources | Older Grok-era repo guide | Historical where conflicting |
| `WORKLOG.md` | Chronological evidence and score history | Evidence log |
| `LESSONS_LEARNED.md` | Failure prevention | Retain, but newer submission rules override old empty-mask rule |
| `POST_MORTEM_AND_GAINS.md` | Why V1–V8 were retired/evolved | Historical design rationale |
| `PLAN.md`, `BUILD_REPORT.md` | Phase A/B construction record | Completed historical gates |
| Grok Phase B/C verdicts and `07_GO_KAGGLE.md` | Prior approvals and launch checklist | Historical; superseded where host behavior changed |
| `RESEARCH_DOSSIER.md` | Candidate ideas and literature | Idea bank, not production truth |
| `Pasted markdown (2).md` | Full conversation/research lineage | Raw archive; never treat isolated claims as verified |
| `.env` | Local credentials/config | Secret; never paste, log, or commit |
| `requirements.txt`, `.gitignore`, `README.md`, `CONTINUATION.md` | Environment and legacy runbook | Supporting resources |
| `yolo11n-seg.pt` | Ultralytics seed weight | Available resource, but production baseline is YOLO11s-seg |

## 10. Non-negotiable lessons

- Optimize and select with PQ, not Dice alone.
- Never return to a large semantic ensemble without direct PQ evidence.
- Never use overlapping submission masks.
- Never manufacture an empty-image row.
- Never resize native YOLO coordinates a second time.
- Never use `nn.DataParallel` in the Kaggle notebook.
- Run compile, unit, contract, real-image smoke, RLE round-trip, and notebook audit before spending GPU time.
- Change one causal lever per experiment and preserve a reproducible artifact hash.

## 11. Start-here message for the next session

> Use `CHATGPT_MASTER_CONTROL.md` as the control plane. ChatGPT is master; Antigravity is executor; Grok is inactive. The verified anchor is V8.1 at 0.350 public PQ. R3 uses conf 0.20, min_area 400, fallback 0, trim, TTA on, strict zero overlap, and no empty rows. First request the R3 Kaggle audit and score. Do not authorize a new training run until that evidence is reviewed.
