# infer.py - D4/multiscale inference and probability ensembling
import glob, os
import cv2, numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import config, model, postprocess


def load_image_for_inference(path, image_size=config.TRAIN_RES):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None or not img.size: raise FileNotFoundError(path)
    x = cv2.resize(img.astype(np.float32) / 255., (image_size, image_size), interpolation=cv2.INTER_AREA)
    if config.IN_CHANNELS == 3: x = np.repeat(x[None], 3, axis=0)
    else: x = x[None]
    return torch.from_numpy((x - .5) / .5).unsqueeze(0).float(), img.shape


def _d4_pairs(x, enabled):
    if not enabled: return [(x, lambda y: y)]
    pairs = []
    for k in range(4):
        r = torch.rot90(x, k, (-2, -1)); pairs.append((r, lambda y, k=k: torch.rot90(y, -k, (-2, -1))))
        rf = torch.flip(r, [-1]); pairs.append((rf, lambda y, k=k: torch.rot90(torch.flip(y, [-1]), -k, (-2, -1))))
    return pairs


def tta_inference(net, tensor, image_size=None, device=config.device, flips=None,
                  d4=config.TTA_D4, scales=None):
    net.eval(); scales = scales or config.TTA_SCALES; out = []
    with torch.no_grad():
        for scale in scales:
            size = [max(32, int(v * scale) // 32 * 32) for v in tensor.shape[-2:]]
            scaled = F.interpolate(tensor, size, mode="bilinear", align_corners=False)
            for aug, undo in _d4_pairs(scaled, d4):
                with torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP and device.type == "cuda"):
                    p = torch.sigmoid(net(aug.to(device)))
                out.append(F.interpolate(undo(p), tensor.shape[-2:], mode="bilinear", align_corners=False))
    return torch.stack(out).mean(0)


def resize_prob_to_full(prob, full_shape=(2048, 2048)):
    if torch.is_tensor(prob): prob = prob.detach().float().cpu().numpy()
    prob = np.asarray(prob).squeeze().astype(np.float32)
    if prob.ndim != 2 or not prob.size: return np.zeros(full_shape, np.float32)
    return cv2.resize(np.ascontiguousarray(prob), (full_shape[1], full_shape[0]), interpolation=cv2.INTER_LINEAR)


def predict_image(net, path, device=config.device, image_size=config.TRAIN_RES, use_tta=False):
    tensor, shape = load_image_for_inference(path, image_size)
    if use_tta: prob = tta_inference(net, tensor, device=device, d4=config.TTA_D4, scales=config.TTA_SCALES)
    else:
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP and device.type == "cuda"):
            prob = torch.sigmoid(net(tensor.to(device)))
    return resize_prob_to_full(prob, shape)


def run_ensemble_averaging(prob_dirs, output_dir=None):
    """Average matching .npy maps across fold/model directories."""
    ids = sorted({os.path.basename(p) for d in prob_dirs for p in glob.glob(os.path.join(d, "*.npy"))})
    result = {}; output_dir and os.makedirs(output_dir, exist_ok=True)
    for image_id in ids:
        maps = [np.load(os.path.join(d, image_id)).astype(np.float32) for d in prob_dirs if os.path.exists(os.path.join(d, image_id))]
        if maps:
            avg = np.mean(maps, axis=0); result[os.path.splitext(image_id)[0]] = avg
            if output_dir: np.save(os.path.join(output_dir, image_id), avg.astype(np.float16))
    return result


def _rows_from_probs(probs, paths, threshold, min_area):
    rows = []
    for path in paths:
        base = os.path.splitext(os.path.basename(path))[0]; prob = probs.get(base)
        if prob is None or postprocess.is_empty_probability(prob): continue
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if config.CRF: prob = postprocess.apply_dense_crf(image, prob, config.CRF_ITER)
        rles, _ = postprocess.full_pipeline(prob, threshold=threshold, min_area=min_area, disk=postprocess.solar_disk_mask(image))
        rows += [{"filament_id": f"{base}_{i:04d}", "segmentation_rle": r} for i, r in enumerate(rles)]
    return rows


def run_on_directory(net, image_dir, output_csv, device=config.device, image_size=config.TRAIN_RES,
                     threshold=config.PROB_THRESHOLD, min_area=config.MIN_AREA, use_tta=False,
                     prob_dir=None, filenames=None, ensemble_dirs=None):
    paths = [os.path.join(image_dir, f) for f in filenames] if filenames else sorted(p for p in glob.glob(os.path.join(image_dir, "*")) if p.lower().endswith((".jpg", ".jpeg")))
    probs = run_ensemble_averaging(ensemble_dirs) if ensemble_dirs else {}
    if not ensemble_dirs:
        if prob_dir: os.makedirs(prob_dir, exist_ok=True)
        for path in tqdm(paths, desc="infer"):
            base = os.path.splitext(os.path.basename(path))[0]
            try: probs[base] = predict_image(net, path, device, image_size, use_tta)
            except Exception as e: print(f"warning: inference failed for {path}: {e}"); continue
            if prob_dir: np.save(os.path.join(prob_dir, f"{base}.npy"), probs[base].astype(np.float16))
    df = pd.DataFrame(_rows_from_probs(probs, paths, threshold, min_area), columns=["filament_id", "segmentation_rle"])
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True); df.to_csv(output_csv, index=False)
    print(f"saved {output_csv} with {len(df)} filaments"); return df


def run_validation_oof(fold=config.VAL_FOLD, image_size=config.TRAIN_RES, threshold=config.PROB_THRESHOLD, use_tta=False, ensemble_dirs=None):
    from dataset import load_coco_annotations, group_annotations_by_filename, get_train_val_files
    _, anns = group_annotations_by_filename(load_coco_annotations(config.TRAIN_JSON)); _, files = get_train_val_files(dict.fromkeys(anns), config.N_FOLDS, fold)
    net = model.get_model().to(config.device); path = os.path.join(config.MODELS_DIR, f"best_fold_{fold}.pth")
    if not ensemble_dirs: model.load_checkpoint(net, path)
    out = os.path.join(config.SUBMISSIONS_DIR, f"oof_fold_{fold}.csv")
    run_on_directory(net, config.TRAIN_IMAGES, out, image_size=image_size, threshold=threshold, use_tta=use_tta,
                     filenames=files, prob_dir=os.path.join(config.SUBMISSIONS_DIR, f"oof_probs_fold_{fold}"), ensemble_dirs=ensemble_dirs)
    return out


def run_test_submission(checkpoint_path=None, output_csv=None, image_size=config.TRAIN_RES,
                        threshold=config.PROB_THRESHOLD, use_tta=False, ensemble_dirs=None):
    candidates = sorted(glob.glob(os.path.join(config.MODELS_DIR, "best_fold_*.pth")))
    checkpoint_path = checkpoint_path or (candidates[-1] if candidates else None)
    if not checkpoint_path and not ensemble_dirs: raise FileNotFoundError("no checkpoint or ensemble probabilities found")
    output_csv = output_csv or os.path.join(config.SUBMISSIONS_DIR, "submission.csv")
    net = model.get_model().to(config.device)
    if not ensemble_dirs: model.load_checkpoint(net, checkpoint_path)
    return run_on_directory(net, config.TEST_IMAGES, output_csv, image_size=image_size, threshold=threshold,
        use_tta=use_tta, prob_dir=os.path.join(config.SUBMISSIONS_DIR, "test_probs"), ensemble_dirs=ensemble_dirs)


if __name__ == "__main__": print("infer module ready")
