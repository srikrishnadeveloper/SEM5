# Work Log

Appended by the agent after every working session. Newest entries at top.

---

## 2026-09-10 — Create college-workflow OpenCode skill, reconcile lab-report-pdf, and add model/subagent guide

- **Goal:** Document the repeatable college project workflows as a reusable OpenCode skill, reconcile `lab-report-pdf` with current `AGENTS.md` rules, and add a parent-model / subagent selection guide.
- **Deliverable:** `C:\Users\srik2\Desktop\College\.opencode\skills\college-workflow\SKILL.md`, `C:\Users\srik2\Desktop\College\.opencode\skills\lab-report-pdf\SKILL.md`, `C:\Users\srik2\Desktop\College\.opencode\skills\lab-report-pdf\references\spec_schema.md`
- **What was done:**
  - Wrote `college-workflow/SKILL.md` (runbook: identity, core principles, 8 workflows, tool guide, verification checklist, common pitfalls, WORKLOG convention).
  - Wrote `college-workflow/references/workflow_cheatsheet.md` with one-line lookups for each workflow.
  - Wrote `college-workflow/scripts/strip_comments.py` (Java `//` and `/* */` removal; Python whole-line `#` removal; quote-aware; preserves package/imports).
  - Wrote `college-workflow/scripts/verify_pdf.py` (pymupdf page count, character count, image count, overflow scan; non-zero exit on overflow).
  - Reconciled `lab-report-pdf/SKILL.md` and `references/spec_schema.md` with `AGENTS.md`: removed the System-Design `plain: true` exception, changed `paragraph` type to single string, added `_nocomments` stripping, no `None` in JSON, and full absolute path rules.
  - Added a `Subagent & model selection` section to `college-workflow/SKILL.md` listing the actual working parent models (`windsurf/swe-1.7`, `windsurf/swe-1.7-medium`, `windsurf/swe-1.6`, `windsurf/glm-5.2`, `opencode/*`) and subagent profiles (`build`, `coder`, `tester`, `explorer`, `general`).
  - Fixed the `plain` mode contradiction in `college-workflow/SKILL.md` to match the reconciled rule: all subjects polished, never `plain: true`.
  - Added a runbook note to `AGENTS.md` pointing to the new skill and helper scripts.
  - Tested `strip_comments.py` on `CrawlerService.java`; it removed 33 line and 19 block comments while preserving package/imports and `://` in regex.
  - Tested `verify_pdf.py` on `Machine Learning\exp1\Assignment_1_Srikrishna_O_S.pdf`; it correctly reported 11 overflow spans (`bbox.x1 > 545 pt`) on pages 13–14.
- **Lessons learned:** Keep the skill concise and reference the existing `lab-report-pdf` skill rather than duplicating it; `pymupdf` now recommends `import pymupdf as fitz` to avoid deprecation warnings; reconcile `SKILL.md` and `spec_schema.md` with `AGENTS.md` as the source of truth to avoid contradictions.

---

## 2026-09-10 — System Design webcrawler: build fix, regenerated _nocomments.java files, and PDF coverage verification

- **Goal:** Fix the original webcrawler Spring Boot project so it compiles, regenerate the stripped `for-pdf\*_nocomments.java` files, and verify lab-sheet coverage.
- **Deliverable:** `C:\Users\srik2\Desktop\College\System Design\webcrawler\for-pdf\*_nocomments.java`
- **What was done:**
  - Ran `mvn clean package -DskipTests` in `System Design\webcrawler\`; build failed with 13 missing-import errors in `CrawlRequest.java` (jakarta.validation.constraints.*), `CrawlJobStatusResponse.java` (java.time.Instant), and `UrlValidator.java` (java.net.* / java.util.Set).
  - Added only the missing imports to the three original files under `src\main\java\com\sscse\webcrawler`; no logic or comments were changed.
  - Re-ran `mvn clean package -DskipTests`; build succeeded and produced `target\webcrawler.jar`.
  - Regenerated all 13 `for-pdf\*_nocomments.java` files from the now-correct originals using a Python stripper that removes Javadoc, block, and whole-line `//` comments while preserving package declarations, imports, and all logic.
  - Verified the regenerated stripped files contain the new imports and are free of stray `//` or `/*` comments (only the `://` inside a regex string remains).
  - Read `C:\Users\srik2\Desktop\extra\webcrawler_pdf.txt` (lab sheet) and checked all 11 required areas: Spring Boot entry, seed URL/dataset, URL queue, page retrieval, hyperlink extraction, visited-URL set, URL validation, Redis integration, REST API, Docker, and JMeter. All 11 are implemented in the project.
- **Lessons learned:** Always compile the original source before regenerating stripped PDF copies; missing imports propagate into the stripped files and would break a report that uses them as code blocks. Keeping `for-pdf\` flat and outside `src\main\java` avoids duplicate-class Maven errors.

---

## 2026-09-10 — System Design webcrawler: stripped _nocomments.java files for report PDF

- **Goal:** Create stripped `_nocomments.java` versions of the 13 key webcrawler Java files for a future lab report PDF.
- **Deliverable:** `C:\Users\srik2\Desktop\College\System Design\webcrawler\for-pdf\*_nocomments.java`
- **What was done:**
  - Read the 13 original Java files under `src\main\java\com\sscse\webcrawler` without modifying them.
  - Created `for-pdf\` and wrote a `_nocomments.java` counterpart for each, keeping package declarations, imports, and all logic identical.
  - Removed Javadoc blocks and most inline `//` comments; left no polished/AI-sounding rewrites.
  - Verified no stray `/*` or `//` comments remain in the stripped files except `://` inside a regex string.
  - Total line-count reduction: 1,361 original lines → 798 stripped lines (563 lines saved).
- **Lessons learned:** Keep stripped files out of `src\main\java` so Maven does not compile duplicate classes; a flat `for-pdf\` folder keeps the PDF source separate.

---

## 2026-09-01 — Solar Filament Segmentation 2026: V7 Mega Master Grandmaster Pipeline Audit

- **Goal:** Conduct a brutal, mathematically rigorous, line-by-line forensic audit of the V7 Mega Master pipeline (`v7/v7_mega_master_pipeline.py`) and cross-check against `v6/train_yolo_detector.py` and `v6/infer_yolo_plus_14models_ensemble.py` across 7 mandatory vectors: coordinate/bbox scaling, preprocessing channel math, TTA symmetry, confidence fusion arbitration, fallback instance separation, RLE compliance, and hidden pipeline issues.
- **Deliverable:** `C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\v7\V7_MEGA_MASTER_AUDIT.md`
- **What was done:**
  - Verified coordinate scaling 1024→2048 is exact up to integer flooring; padding and bounds checks are correct.
  - Identified the unsharp mask is computed in saturated uint8, throwing away dynamic range; provided float32 fix.
  - Called out the false "8-Way D4 TTA" claim — the code only does 4 flips; provided full D4 (8-way) tensor implementation.
  - Analyzed the fused-score priority function; concluded it is sound but untuned and suggested validation-based tuning with optional area term.
  - Showed that fallback connected-components + 3×3 closing can merge distinct filaments at saddle points; provided distance-transform watershed splitting for large components.
  - Confirmed COCO RLE encoding and empty-mask fallback are correct.
  - Exposed dead code / false marketing: Gaussian apodization window is never used, EdgeAttNet/boundary loss are not implemented.
  - Reiterated the V6 DeepLab encoder-guessing bug and the harmful Mosaic/CopyPaste/Mixup YOLO augmentations.
  - Wrote corrected production code blocks for features, TTA, watershed fallback, and tunable fusion.
- **Lessons learned:** V7 is mostly V6 with misleading claims and unimplemented features. The submission is valid, but the score will not improve until the unsharp saturation, true 8-way TTA, watershed fallback splitting, and validation-based threshold tuning are implemented. Brutal honesty: rename the file or deliver the features.

## 2026-09-01 — Solar Filament Segmentation 2026: V6 YOLOv8x + 14-Model UNet++ Ensemble Audit

- **Goal:** Audit the V6 production engine: YOLOv8x-seg detector training pipeline (`v6/train_yolo_detector.py`) and the fused YOLO + 14-model UNet++/DeepLabV3+ ensemble inference pipeline (`v6/infer_yolo_plus_14models_ensemble.py`).
- **Deliverable:** `C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\v6\V6_YOLO_UNET_ENSEMBLE_AUDIT.md`
- **What was done:**
  - Verified COCO polygon normalization to YOLO format, GroupKFold by year, preprocessing, YOLO training hyperparameters/augmentations, checkpoint discovery, per-model dynamic input-channel inference, box scaling, two-stage fusion, confidence-score arbitration, and COCO RLE encoding.
  - Found no critical correctness bugs (RLE, scaling, normalization, and non-overlap are correct).
  - Identified one runtime-blocking bug: `cv2.CISTAT_AREA` typo in the fallback connected-components path.
  - Identified high-impact design issues: only the first GroupKFold fold is used; Mosaic/CopyPaste/Mixup augmentations are likely harmful for thin solar filaments; thresholds are hard-coded and untuned.
  - Provided corrected fallback code, a better augmentation recipe, and tuning recommendations.
- **Lessons learned:** V6 is the strongest pipeline audited so far. The biggest scoring improvements come from training on all 5 folds, disabling Mosaic/CopyPaste/Mixup for filament-shaped objects, fixing the `CC_STAT_AREA` typo, and tuning thresholds on validation PQ.

## 2026-09-01 — Solar Filament Segmentation 2026: V5 I079 Detector+Refiner Pipeline Audit

- **Goal:** Audit the two-stage Mask R-CNN + crop U-Net refiner inference pipeline (`v5/infer_i079_detector_refiner.py`) based on the public `phuongncn/solar-filament-i079-weights` model that reports LB 0.70.
- **Deliverable:** `C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\v5\V5_I079_AUDIT.md`
- **What was done:**
  - Read and audited every block: weight loading, detector/refiner instantiation, preprocessing, detector inference, box scaling, crop extraction, refiner TTA, mask repasting, non-overlap logic, RLE encoding, and submission generation.
  - Verified mathematical correctness of box scaling, crop padding, flip TTA, greedy non-overlap removal, and pycocotools RLE encoding.
  - Found no critical bugs (the submission is valid). Identified high-impact improvements: missing solar limb mask, inconsistent detector/refiner normalization, and untuned confidence/mask thresholds.
  - Also noted medium issues: fragile checkpoint discovery by glob, `strict=False` weight loading, small PADDING=16, and unused detector masks.
  - Provided a corrected/improved crop+repaste block and tuning recommendations.
- **Lessons learned:** The V5 pipeline is significantly more correct than V4 and should reproduce the ~0.70 public LB. The biggest scoring levers are adding a solar disk mask, tuning thresholds/area on validation, and ensembling with the existing U-Net models.

## 2026-09-01 — Solar Filament Segmentation 2026: V4 Hysteresis Pipeline Audit

- **Goal:** Perform a rigorous line-by-line forensic audit of the V4 grandmaster hysteresis and non-overlap ensemble inference pipeline (`v4/infer_hysteresis_standalone.py` and `v4/v4_grandmaster_hysteresis_ensemble.ipynb`).
- **Deliverable:** `C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\v4\V4_HYSTERESIS_AUDIT.md`
- **What was done:**
  - Read and compared the standalone script and the notebook code cell (they are identical).
  - Audited imports, checkpoint discovery, model loading, preprocessing, TTA, probability upsampling, hysteresis instance extraction, RLE encoding, and submission generation.
  - Identified 3 critical bugs (invalid 2D RLE encoding, invalid `"PPP2"` dummy RLE, fragile architecture/encoder guessing), 4 high-impact design issues (untrue hysteresis merging adjacent filaments, hard-coded thresholds, incomplete TTA, area-based overlap priority), and several medium/low issues.
  - Provided mathematical validation of the hysteresis algorithm versus true Canny-style hysteresis and explained PQ/mIoU scoring implications.
  - Wrote a corrected `extract_hysteresis_instances` function with true hysteresis, watershed splitting, and score-based non-overlap priority, plus a corrected RLE helper.
- **Lessons learned:** The pipeline is conceptually strong but unsafe to run as-is. The RLE bug alone can invalidate the submission. The hysteresis function merges distinct filaments via weak bridges, which will hurt PQ. Hyperparameters must be tuned on OOF validation rather than hard-coded.

## 2026-09-01 — Solar Filament Segmentation 2026: Research Dossier

- **Goal:** Perform an exhaustive deep-research and codebase-mining pass across the 100+ requested vectors for the Solar Filament Segmentation 2026 Kaggle challenge and compile a comprehensive `RESEARCH_DOSSIER.md`.
- **Deliverable:** `C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\RESEARCH_DOSSIER.md`
- **What was done:**
  - Conducted 10+ targeted web searches covering EdgeAttNet, Flat U-Net, Compound U-Net, clDice, Lovász loss, Focal Tversky, Boundary loss, Sartorius/HuBMAP gold solutions, watershed post-processing, pycocotools RLE, and Panoptic Quality.
  - Fetched arXiv abstracts for EdgeAttNet (2509.02964), Flat U-Net (2502.07259), and the EdgeAttNet GitHub for architecture details.
  - Extracted copy-paste-ready PyTorch code for: EG-MHSA attention, SCA/CSA blocks, clDice, Focal Tversky, Boundary loss, Lovász-Hinge, watershed splitting, pycocotools RLE encoding, and a self-contained PQ evaluator.
  - Synthesized a hyperparameter/loss matrix and a 4-phase Top-10/medal roadmap.
  - Wrote a 32,585-byte markdown dossier with sections on architecture, losses, post-processing, metrics, Kaggle gold case studies, and next steps.
- **Lessons learned:** The winning playbook is a combination of barb-aware attention, topology-preserving clDice + boundary losses, hysteresis/watershed post-processing tuned with PQ (not Dice), D4 TTA, and 5-fold ensembling. Post-processing and metric-aware threshold/area tuning is at least as important as the model itself.

## 2026-08-30 — Computer Networks Assignment 1, Question 3: DNS geographic distribution

- **Goal:** Build the report for Q3 — "Design and Develop a Comparative Framework to Evaluate the Impact of Geographic Distribution of DNS Servers on Network Performance" (UCS3501).
- **Deliverable:**
  - `C:\Users\srik2\Desktop\College\Computer Networks\Assignment 1\Assignment_1_Q3_DNS_Geographic_Distribution_Srikrishna_O_S.pdf`
  - `C:\Users\srik2\Desktop\College\Computer Networks\Assignment 1\spec_q3.json`
  - `C:\Users\srik2\Desktop\College\Computer Networks\Assignment 1\dns_data.json`
  - `C:\Users\srik2\Desktop\College\Computer Networks\Assignment 1\q3_plots\`
- **What was done:**
  - Extracted the full question paper from `C:\Users\srik2\Downloads\Theory Assignment 1.pdf`.
  - Wrote a Python collector (`dnspython` + `ip-api.com`) to gather A-record resolution times, authoritative NS records, and IP geolocation for `www.nus.edu.sg`, `www.ntu.edu.sg`, `www.iitm.ac.in`, and `www.nic.in`.
  - Generated 4 plots: resolution time bar chart, authoritative NS count, geographic distribution, and a world-map scatter.
  - Wrote `spec_q3.json` and built the PDF with the lab-report-pdf skill.
  - Verified: 6 pages, 8,160 chars, 4 images, no overflow.
  - Copied deliverables into the subject folder `Computer Networks\Assignment 1\`.
- **Lessons learned:** Use apex domains (not `www`) to query `NS` records; `www.nic.in` uses Akamai (Frankfurt) for A-record and NIC New Delhi for NS, while `www.nus.edu.sg` uses Incapsula London — CDNs make geographic analysis important.

---

## 2026-08-25 — Solar Filament Segmentation 2026: generate three Colab notebook payloads

- **Goal:** Build three self-contained Google Colab notebook payloads for the Kaggle Solar Filament Segmentation project, one per Colab account, with complementary model/duration/fold configurations.
- **Deliverable:**
  - `C:\Users\srik2\AppData\Local\Temp\opencode\colabs\account_1_payload.json`
  - `C:\Users\srik2\AppData\Local\Temp\opencode\colabs\account_2_payload.json`
  - `C:\Users\srik2\AppData\Local\Temp\opencode\colabs\account_3_payload.json`
  - `C:\Users\srik2\AppData\Local\Temp\opencode\colabs\variations_summary.md`
- **What was done:**
  - Read the local `notebooks/build_colab_nb.py` and all `code/*.py` modules to understand the Colab flow and environment-variable-driven config.
  - Designed three complementary variations: fast 512×512 `tu-efficientnet_b0` 10 epochs fold 0; medium 768×768 `tu-efficientnet_b1` 15 epochs fold 1 with different dice/focal/BCE weights; strong 1024×1024 `tu-efficientnet_b3` 30 epochs fold 2 with batch 1 / accumulation 2 and TTA.
  - Wrote `generate_payloads.py` to embed each code module in `%%writefile` cells, set `KAGGLE_API_TOKEN` from `userdata.get` with a fallback, download data via `kagglehub.competition_download`, set per-account environment overrides, then run `train.train_fold()` and `infer.run_test_submission()`.
  - Validated all payloads as valid JSON and ran `py_compile` on every cell source (both executable cells and the file bodies behind `%%writefile`); confirmed no syntax errors.
  - Generated `variations_summary.md` with a table of the three configurations.
- **Lessons learned:** Colab payload generation should keep env-var overrides in a dedicated cell before `import config`; `%%writefile` cells must start with the magic; and kagglehub needs either `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`, so the token cell parses a JSON kaggle.json when possible.

## 2026-08-25 — Solar Filament Segmentation 2026: verify brief/baseline and produce alternative plan

- **Goal:** Verify facts, URLs, and formulas in `01_competition_brief.md` and `public_baseline_resnet_unet_summary.md`, then propose a concrete alternative high-level approach for a top-100 finish.
- **Deliverable:** `C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\research\verification_and_recommendations.md`
- **What was done:**
  - Web-searched and fetched the IEEE BigData Cup 2026 page, MAGFiLO dataset page, MAGFiLO Scientific Data paper, EdgeAttNet arXiv/paper, Flat U-Net paper, and Panoptic Segmentation paper to cross-check URLs and claims.
  - Downloaded the public Kaggle baseline notebook from its GitHub mirror (`avikds/Kaggle-Notebooks-Avik`) and confirmed source cells: dataset counts (1,154 / 8,199 / 4), ResNet34UNet architecture, ScientificDiceLoss, BCE+Dice training, 5-epoch log, and 2048 post-processing/RLE.
  - Verified the actual competition evaluation is based on `mIoU_pairwise` and `mIoU_multiscale` (IEEE proposal + EdgeAttNet paper), contradicting the brief's 70/30 Dice/Panoptic-Quality claim (lines 89-123).
  - Wrote a 17-section verification report with line-by-line verdict tables and a concrete alternative: EdgeAttNet-style edge-guided U-Net, solar-specific preprocessing (Hough disk mask, radial flattening, CLAHE, Gaussian blur), 1024×1024 tiled training, boundary-aware loss, 5-fold stratified CV, hysteresis/CC/watershed post-processing, and a 5-day Colab first-mover plan.
- **Lessons learned:** The public baseline is accurate as a digest but the competition brief is wrong about scoring; any top-100 attempt must train and validate against pairwise/multiscale IoU, not Dice. The EdgeAttNet repo has circular/incomplete model files, so the architecture should be re-implemented in `segmentation-models-pytorch` rather than cloned as-is.

## 2026-08-25 — Solar Filament Segmentation 2026: deep literature & engineering dive

- **Goal:** Gather and synthesize at least 50 distinct sources across 8 requested topics for the Solar Filament Segmentation 2026 Kaggle competition and write a comprehensive markdown report.
- **Deliverable:** `C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\research\literature_deep_dive.md`
- **What was done:**
  - Listed the local project; read `01_competition_brief.md` and `public_baseline_resnet_unet_summary.md`.
  - Conducted 35+ targeted web searches covering public Kaggle notebooks, solar-filament architectures (EdgeAttNet, Flat U-Net, Compound U-Net, Mask R-CNN, YOLO, SAM), thin/astronomical segmentation, top Kaggle/CVPR solutions, loss functions (Dice, Focal, Lovász, Boundary, clDice), PQ post-processing, data augmentation, TTA, ensembling, and self-supervised pretraining.
  - Fetched key pages (EdgeAttNet arXiv, Flat U-Net arXiv, MAGFiLO Scientific Data, public baseline GitHub) for details.
  - Wrote a 44,687-byte markdown report with 107 distinct cited sources, each containing title, URL, key takeaway, and application; included a 9-step suggested pipeline.
- **Lessons learned:** The winning playbook is an ensemble of metric-aware components: barb-aware attention, topology/clDice + boundary losses, D4 TTA, watershed/center-NMS post-processing, and weighted multi-fold/model ensembling. The public baseline is far below these practices and should be rebuilt.

---

## 2026-08-24 — Solar Filament Segmentation 2026: critical approach review

- **Goal:** Review all files in `Machine Learning\Kaggle_Competition_Mini_Project\research` and `\data` and produce a gap-analysis report for finishing above the 100th position.
- **Deliverable:** `C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\research\approach_check.md`
- **What was done:**
  - Inventoried the project: only `research\` has files; `code\`, `notebooks\`, `plots\`, `submissions\` are empty; no `requirements.txt`, `README.md`, or `.gitignore`.
  - Read `research\01_competition_brief.md`, `research\public_baseline_resnet_unet_summary.md`, and extracted all 35 cells of `research\public_baseline_resnet_unet.ipynb`.
  - Inspected `data\MAGFiLO_1.0_Kaggle_2026\train\MAGFiLO_1.0_Annotations_kaggle2026_train.json` (1,154 image entries, 707 unique JPEGs, 8,199 annotations, 296 files with multiple image_ids, `Ambiguous` class empty).
  - Verified image sizes are 2048 × 2048 and plotted no actual plots; recorded distributions from JSON.
- **Key gaps found:** data leakage through image_id-level split, 512×512 training collapsing small filaments, ResNet34 pseudo-RGB, BCE instead of promised Focal loss, 5 epochs, no PQ metric, hand-picked threshold/area filter, no TTA/ensemble, no modular code, no Colab notebook, no lab-report `spec.json` or `WORKLOG` entry.
- **Lessons learned:** A public-baseline notebook is not a competitive pipeline; a top-100 attempt needs group-stratified cross-validation, higher resolution, a stronger encoder/decoder, metric-aware post-processing, and a real lab-report workflow.

---

## 2026-08-24 — Q4 (Srikrishna) and Q16 (Arjun M) single-question PDFs

- Rebuilt UEC3942 Assignment as **Q4 only** (user's allotted question): `Assignment_1_Q4_Optical_Fire_Detection_Srikrishna_O_S.pdf` — 5 pages, 4 figures (block diagram, 90-deg scatter chamber, waveform w/ hysteresis, state machine) + IR LED photo.
- Built **Q16 for friend Arjun M (IT, 3122245002302)**: `Assignment_1_Q16_Bridge_SHM_Arjun_M.pdf` — 6 pages, 3 new figures (SHM chain, bridge gauge-placement elevation, FFT frequency-drift damage plot) + foil strain gauge photo (Wikimedia).
- Both verified with pymupdf before reporting (pages/chars/images/overflow).
- Reusable fig scripts in `Temp\opencode\q4_figs.py` and `q16_figs.py`; specs `spec_q4.json`, `spec_q16_arjun.json` in UEC3942 folder.
- Note: allotment XLSX maps S.No 16 -> Arjun M, IT, 3122245002302, Q16 (bridge SHM strain gauges).

---

## 2026-08-24 — Semester submission audit (all current-sem subjects)

**Goal:** Check everything needed for submission this sem (arrears Discrete Math + Probability excluded) using local folders + LMS via MCP.

**Findings (full details in `SUBMISSIONS.md`):**
- Local labs all complete: Networks exp2-7, ML Lab exp1-5, System Design exp2/3/5/6, UEC3942 A1 — every folder has its report PDF.
- **PENDING FOUND: UCS3502 Principles of Machine Learning (Theory) Assignment 1 & 2, 40 marks.** Question paper at `Downloads\ASSIGNMENT-2 - Theory (2).pdf`. Team of 3, IEEE Big Data Cup 2026 challenge (Challenge 01 Suicide Risk Detection chosen), report in LaTeX + code + screenshots as single PDF on LMS (course UCS3502-POML-A-26).
- Unverified LMS courses: DAA (id=4757), CP (id=4773), IP (id=4766) — browser was in active use, could not sweep them. Re-check later.

**How it was done:** explorer subagent inventoried local folders + LMS manifest; I drove Edge via computer-use (set_value on address bar + Enter). Lesson: UI element indexes shift between get_app_state calls when tabs change — use keyboard (ctrl+l, type, Return) instead of clicking tree elements. browsermcp extension still not connected (needs manual Connect click).

---

## 2026-08-24 — UEC3942 Assignment 1 PDF (Sensors and Actuators)

**Goal:** Build the Assignment-1 submission PDF for UEC3942 from the question DOCX + allotment XLSX in Downloads. Srikrishna O S is allotted Q4.

**Final deliverable:**
- `C:\Users\srik2\Desktop\College\UEC3942\Assignment_1_Srikrishna_O_S.pdf` — 11 pages, 11 images, no overflow.
- Per question: 1 short principle paragraph + labeled bullets (Sensing / Conditioning / Control / Output) + figures where useful.

**What was done:**
1. Extracted DOCX (python-docx) and XLSX (openpyxl); student gets Q4.
2. Built spec.json with all 20 Q and A, polished style (`plain: false`).
3. Generated 5 matplotlib block diagrams in `UEC3942\diagrams\` (Q1 loop, Q4 fire chamber, Q9 PPG, Q16 bridge, Q18 TPMS) themed to match the report (#DDE7F5 boxes, #2F5597 borders).
4. Downloaded 6 component photos from Wikimedia Commons (API search srnamespace=6 + Special:FilePath, width=800) into `UEC3942\images\`: PT100, thermocouple, LDR, IR LED, IR thermometer, TPMS. Wikimedia rate-limits aggressively (WinError 10054) — sleep 3-10 s between requests.
5. Rebuilt, verified with pymupdf: page count, embedded image count, no span past 545 pt.

**Lessons learned this session (also added to AGENTS.md):**
- `build_pdf.py` silently SKIPS sections using `paragraphs` (plural). Only `paragraph` and `bullets` render. This caused several "empty PDF" rounds.
- JSON spec must never contain Python `None` — use a string or omit the field.
- PowerShell here: never use bash heredocs (`<< 'EOF'`) with python -c; write a script into Temp\opencode and run it.
- After every build, verify content actually landed: extract text with pymupdf and check char count before telling the user it is done.
- Wikimedia Commons is the reliable free image source; use the API, throttle requests, keep a browser-independent User-Agent.

---

## 2026-08-24 — Earlier session fixes (recap)

- System Design exp6 spec had `"plain": true` producing a plain PDF; user wants polished everywhere. Changed to `false`; AGENTS.md rule updated: ALL subjects polished, plain mode removed.
- Networks Lab exp6 PDF rebuilt with polished style.
- System Design exp5 (Consistent Hashing) PDF built from existing Spring Boot code; fixed spec path typo (`hashhing` -> `hashing\hashing`).
- ML exp4 (Perceptron vs MLP) and ML exp5 (Naive Bayes + KNN) code refactored with natural comments, run for real, plots captured, PDFs built. Spambase dataset downloaded from UCI; `kd_tree` (not `kdtree`) is the sklearn algorithm name.
- AGENTS.md updated: full absolute paths required for every PDF reference.
