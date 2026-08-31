# Lessons Learned — Hard-Won Rules for This Pipeline

> **Purpose:** Every bug, crash, and wasted Colab session is documented here so the same
> mistake is NEVER repeated. Read this file before every code change.

---

## 1. FOLD INDEX IS 0-BASED (0..n_folds-1)

**What went wrong:** User entered `FOLD = 5` with `n_folds = 5`. Valid folds are `0, 1, 2, 3, 4`.
Fold 5 doesn't exist in the fold assignment, so the split became `701 train / 6 val` — essentially
training on the entire dataset with a garbage validation set. Checkpoint selection becomes meaningless.

**Rule:** Always validate `0 <= cfg.fold < cfg.n_folds` and raise a clear error if not.
Display valid range in the Colab form comment.

---

## 2. NEVER DELETE A FUNCTION WITHOUT GREP-CHECKING ALL CALL SITES

**What went wrong:** During cache optimization, `_downsample_mask_max()` was accidentally deleted
even though `search_threshold()` still calls it. Python doesn't catch this at import time — it
only crashes at runtime, deep into a 2-hour Colab training session.

**Rule:** After every refactor, run the AST audit:
```python
python -c "
import ast
with open('notebooks/filament_top50_single.py', encoding='utf-8') as f:
    tree = ast.parse(f.read())
defs = set()
for n in ast.walk(tree):
    if isinstance(n, (ast.FunctionDef, ast.ClassDef)): defs.add(n.name)
    if isinstance(n, ast.Assign):
        for t in n.targets:
            if isinstance(t, ast.Name): defs.add(t.id)
for n in ast.walk(tree):
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
        if n.func.id.startswith('_') and n.func.id not in defs:
            print(f'BUG: {n.func.id} called but not defined')
print('Done')
"
```

---

## 3. PREPROCESSING MUST MATCH BETWEEN CACHE AND INFERENCE

**What went wrong:** `build_cache()` ran `preprocess_solar()` (Hough + radial flatten + CLAHE)
at full 2048×2048 resolution — taking ~60 seconds per image. With 707 images, the cache build
took 12+ hours. Meanwhile inference also runs `preprocess_solar()` on full-res images.

**Rule:** Preprocessing functions that touch every pixel (Hough circles, radial flatten) must
use fast thumbnails internally. `solar_disk_hough()` now runs on a 256×256 thumbnail.
`radial_flatten()` uses vectorized `np.bincount` instead of a Python for-loop.

---

## 4. lr_scheduler.step() MUST COME AFTER optimizer.step()

**What went wrong:** PyTorch warning: "Detected call of `lr_scheduler.step()` before
`optimizer.step()`". This causes the first LR value to be skipped, so warmup is off by one.

**Rule:** In the training loop, the scheduler must step AFTER the optimizer has stepped
at least once in that epoch. With gradient accumulation, `sched.step()` must be called
only after the accumulated `optimizer.step()` has executed.

---

## 5. VRAM REPORTING: USE memory_reserved(), NOT memory_allocated()

**What went wrong:** Logs showed `VRAM: 0.27 GB` for a UNet++ B4 at 1024×1024 on T4.
The real usage is 8-12 GB. `torch.cuda.memory_allocated()` only shows the current
tensor allocation, not the total GPU memory reserved by PyTorch's caching allocator.

**Rule:** Use `torch.cuda.memory_reserved()` for the true VRAM footprint, or
`torch.cuda.max_memory_allocated()` for peak usage.

---

## 6. DEFAULT PRESET MUST BE "fast" FOR FIRST RUNS

**What went wrong:** `CELL_CONFIG` defaulted to `PRESET = "balanced"` (1024px, ~70min).
Even though the Colab `#@param` dropdown showed "fast" selected, the Python variable
was still `"balanced"` because the user didn't re-execute the cell after changing the dropdown.

**Rule:** Default to `"fast"` so first runs complete quickly. Users upgrade to
`"balanced"` / `"max"` only after confirming the pipeline works.

---

## 7. ALWAYS RUN BOTH TEST SUITES BEFORE UPLOADING TO COLAB

**What went wrong:** Multiple times, code was uploaded to Colab without running the
local test suites first. Bugs that take 1 second to catch locally wasted hours of
Colab GPU time.

**Rule:** Before every `build_single_notebook.py` run:
```powershell
python -m py_compile notebooks\filament_top50_single.py
python tests\test_unit_fixes.py
python tests\smoke_synthetic.py
```
All three must pass. No exceptions.

---

## 8. COLAB #@param FORMS DON'T AUTO-UPDATE VARIABLES

**What went wrong:** User selected "fast" in the dropdown but the cell wasn't
re-executed, so the Python variable still held the old default "balanced".

**Rule:** Print the actual configuration values AFTER apply_preset() so the user
can visually confirm what's actually running. Add an assertion that catches
obviously wrong values (like fold >= n_folds).

---

## 9. POWERSHELL 2>&1 GIVES FALSE EXIT CODE 1

**What went wrong:** `python script.py 2>&1` in PowerShell treats any stderr output
(including tqdm progress bars) as a NativeCommandError, reporting exit code 1 even
when Python exited cleanly.

**Rule:** Don't rely on exit code when using `2>&1` in PowerShell. Check the actual
output text (e.g. "ALL TESTS PASSED") instead.

---

## 10. CACHE BUILDING: RESIZE FIRST, THEN PREPROCESS

**What went wrong:** Original code loaded 2048×2048 JPEG, ran expensive preprocessing
(Hough + radial flatten + CLAHE) at full res, THEN resized to 1024×1024 cache res.
75% of the computation was thrown away.

**Rule:** Resize to cache_res FIRST, then run preprocessing on the smaller image.
The quality difference is negligible but the speed difference is 10-50×.

---

## 11. V1 OVER-ENGINEERING VS V2 PROVEN SIMPLICITY

**What went wrong:** V1 piled on 5 loss terms (BCE + Dice + Tversky + Boundary + clDice) with pos_weight=50, complex Hough-circle / radial-flattening preprocessing at different resolutions, custom architectures (EdgeAttNet), and 1,690 lines of single-file code. This caused conflicting loss gradients (stuck at loss 9.0), distribution mismatch between train and test, and hours of wasted GPU credits.

**Rule:** V2 is built around proven Kaggle fundamentals:
1. `smp.UnetPlusPlus` + ImageNet-pretrained `efficientnet-b4` + 3-channel RGB.
2. Direct image resizing and Albumentations augmentations (no complex solar-specific distortion).
3. Balanced `0.30*BCE + 0.40*Dice + 0.30*Focal` loss (no pos_weight, no conflicting boundary/clDice terms).
4. Fast 512×512 training with 4-flip TTA, upsampling float probability map to 2048×2048 for smooth RLE.
5. All pipeline code in clean ~450 lines in `v2/pipeline.py`.

---

## 12. NEVER USE `nn.DataParallel` IN KAGGLE / JUPYTER NOTEBOOKS

**What went wrong:** `nn.DataParallel` was added to `v4/pipeline.py` to utilize dual Tesla T4 GPUs on Kaggle. In Kaggle's shared-memory Jupyter Linux container, `nn.DataParallel` multi-threading duplicates tensors across host CPU memory on every forward/backward pass. Over 255 steps at 1024px, host RAM accumulated up to Kaggle's 13GB ceiling, causing the kernel to freeze and the Linux OOM watchdog to terminate the process (`Kernel died while waiting for execute reply. Your notebook tried to allocate more memory than is available`).



---

## 14. NEVER RUN SEGFORMER (MIT-B3) AT 1024px ON 16GB T4 GPU

**What went wrong:** In `grandmaster`, `Segformer (mit_b3)` crashed with `OutOfMemoryError: CUDA out of memory` during Epoch 1 on T4. Vision Transformer self-attention maps at 1024×1024 resolution exceed 16GB VRAM during feature concatenation (`self.fuse_stage`).

**Rule:** 
1. Use CNN-based backbones for 1024px training on T4 GPUs: `UnetPlusPlus (efficientnet-b4)` and `DeepLabV3Plus (resnet50)`. Both stay comfortably under 5GB VRAM.
2. If using `Segformer (mit_b3)`, cap resolution at 512×512 or use batch size 1 with gradient accumulation.


**What went wrong:** In dual-GPU sessions (`GPU T4 x2`), sending all models to `cuda:0` leaves GPU 1 at 0.00% utilization, wasting half the available compute.

**Rule:** During multi-model ensemble inference:
1. Distribute models alternating across `cuda:0` and `cuda:1` (`dev = f'cuda:{i % n_gpus}'`).
2. Run forward passes on each model's designated GPU.
3. This lights up BOTH GPUs to 100%, doubles available VRAM, and cuts total test inference time in half with zero memory locking!



