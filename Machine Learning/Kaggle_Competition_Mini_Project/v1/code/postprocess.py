# postprocess.py
# turn a probability map into a list of filament instance masks and encode them.

from typing import List, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import pycocotools.mask as mask_util
from skimage.morphology import remove_small_holes, remove_small_objects


def is_empty_probability(prob, threshold=0.10, min_pixels=50):
    return prob.size == 0 or float(prob.max()) < threshold or int((prob > threshold).sum()) < min_pixels


def apply_dense_crf(image, prob, n_iter=5):
    """Optional dense CRF; safely returns the input when pydensecrf is absent."""
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        return prob
    if image.ndim == 2: image = np.repeat(image[..., None], 3, axis=-1)
    h, w = prob.shape; dense = dcrf.DenseCRF2D(w, h, 2)
    dense.setUnaryEnergy(unary_from_softmax(np.stack([1 - prob, prob]).astype(np.float32)))
    dense.addPairwiseGaussian(sxy=3, compat=3)
    dense.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image.astype(np.uint8), compat=10)
    return np.asarray(dense.inference(n_iter), dtype=np.float32).reshape(2, h, w)[1]


def optimize_threshold_area(oof_probs, gt_instances, thresholds, min_areas):
    """Grid-search OOF probabilities for competition multiscale IoU."""
    import metrics
    best = {"score": -1.0, "threshold": 0.5, "min_area": 200}
    for threshold in thresholds:
        for min_area in min_areas:
            preds = [mask_to_instances(clean_mask(p > threshold, min_obj=min_area), min_area=min_area,
                                       watershed_split=False) for p in oof_probs]
            pred_arrays = [np.stack(x) if x else np.zeros((0, *p.shape), np.uint8) for x, p in zip(preds, oof_probs)]
            score = metrics.mean_mIoU_multiscale(pred_arrays, gt_instances)["mIoU_multiscale_micro"]
            if score > best["score"]: best = {"score": score, "threshold": float(threshold), "min_area": int(min_area)}
    return best


def clean_mask(mask, open_r=1, close_r=2, min_hole=100, min_obj=100):
    mask = mask.astype(bool)
    if open_r: mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2*open_r+1,)*2, np.uint8)).astype(bool)
    if close_r: mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2*close_r+1,)*2, np.uint8)).astype(bool)
    return remove_small_objects(remove_small_holes(mask, area_threshold=min_hole), min_size=min_obj).astype(np.uint8)


def mask_nms(instances, iou_threshold=0.7):
    kept = []
    for mask in sorted(instances, key=lambda x: x.sum(), reverse=True):
        if all((np.logical_and(mask, k).sum() / max(np.logical_or(mask, k).sum(), 1)) < iou_threshold for k in kept): kept.append(mask)
    return kept


def solar_disk_mask(image: np.ndarray) -> np.ndarray:
    """
    estimate a binary solar disk mask for a 2048x2048 H-alpha image.
    uses a simple threshold + largest connected component.
    """
    if image.max() <= 1.0:
        img8 = (image * 255).astype(np.uint8)
    else:
        img8 = image.astype(np.uint8)

    # threshold well above the black padding
    _, binary = cv2.threshold(img8, 10, 255, cv2.THRESH_BINARY)
    # find largest component
    num, labels = cv2.connectedComponents(binary, connectivity=8)
    if num <= 1:
        return binary
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # background
    largest = sizes.argmax()
    disk = (labels == largest).astype(np.uint8)
    return disk


def morphological_cleanup(mask: np.ndarray, close_ksize: int = 3) -> np.ndarray:
    """small closing to fill holes, then small opening to remove noise."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def mask_to_instances(
    mask: np.ndarray,
    min_area: int = 200,
    max_area: int = 500000,
    watershed_split: bool = True,
) -> List[np.ndarray]:
    """
    split a binary mask into separate instance masks.

    - remove small components
    - optionally split merged components with watershed on distance transform
    - return a list of binary masks, one per filament
    """
    labeled, num = ndimage.label(mask)
    if num == 0:
        return []

    instances = []
    for i in range(1, num + 1):
        comp = (labeled == i).astype(np.uint8)
        area = comp.sum()
        if area < min_area or area > max_area:
            continue

        if watershed_split:
            # distance transform
            dist = ndimage.distance_transform_edt(comp)
            # find peaks that are not too close
            coords = peak_local_max(
                dist,
                min_distance=20,
                exclude_border=False,
                footprint=np.ones((5, 5)),
                labels=comp,
            )
            if len(coords) > 1:
                mask_internal = np.zeros(comp.shape, dtype=bool)
                mask_internal[tuple(coords.T)] = True
                markers, _ = ndimage.label(mask_internal)
                ws = watershed(-dist, markers, mask=comp.astype(bool), watershed_line=True)
                # extract the split components
                unique_markers = np.unique(markers)
                for m in unique_markers:
                    if m == 0:
                        continue
                    sub = (ws == m).astype(np.uint8)
                    if min_area <= sub.sum() <= max_area:
                        instances.append(sub)
                continue
        instances.append(comp)

    return instances


def rle_encode(mask: np.ndarray) -> str:
    """encode a single binary mask to COCO RLE with pycocotools."""
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("utf-8")


def rle_decode(rle: str, height: int = 2048, width: int = 2048) -> np.ndarray:
    """decode a COCO RLE string to binary mask."""
    return mask_util.decode({"size": [height, width], "counts": rle})


def threshold_search(
    model,
    loader,
    device,
    thresholds: np.ndarray = np.arange(0.3, 0.7, 0.05),
) -> float:
    """
    find the best probability threshold on a validation loader.
    we look for the threshold that maximizes PQ (or dice, if PQ is unavailable).
    """
    import metrics

    best_score = -1.0
    best_thr = 0.5
    for thr in thresholds:
        scores = []
        for imgs, masks, _ in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            with torch.no_grad():
                logits = model(imgs)
                if logits.shape != masks.shape:
                    logits = torch.nn.functional.interpolate(
                        logits, size=masks.shape[2:], mode="bilinear", align_corners=False
                    )
                probs = torch.sigmoid(logits)
                preds = (probs > thr).float()

            for p, m in zip(preds.detach().cpu().numpy(), masks.detach().cpu().numpy()):
                pred_labeled, pred_n = ndimage.label(p[0])
                true_labeled, true_n = ndimage.label(m[0])
                pred_layers = np.stack([
                    (pred_labeled == i).astype(np.float32)
                    for i in range(1, pred_n + 1)
                ]) if pred_n > 0 else np.zeros((0, p.shape[1], p.shape[2]), dtype=np.float32)
                true_layers = np.stack([
                    (true_labeled == i).astype(np.float32)
                    for i in range(1, true_n + 1)
                ]) if true_n > 0 else np.zeros((0, m.shape[1], m.shape[2]), dtype=np.float32)

                pq, _, _, _ = metrics.panoptic_quality(
                    torch.from_numpy(true_layers), torch.from_numpy(pred_layers)
                )
                scores.append(pq)

        score = float(np.mean(scores))
        if score > best_score:
            best_score = score
            best_thr = thr
    print(f"best threshold {best_thr:.2f} with PQ {best_score:.4f}")
    return best_thr


def full_pipeline(
    prob: np.ndarray,
    threshold: float = 0.45,
    close_ksize: int = 3,
    min_area: int = 200,
    max_area: int = 500000,
    apply_watershed: bool = True,
    disk: np.ndarray = None,
) -> Tuple[List[str], List[np.ndarray]]:
    """
    take a (H, W) probability map and return:
        - list of RLE strings (one per filament)
        - list of binary masks
    """
    mask = (prob > threshold).astype(np.uint8)
    mask = morphological_cleanup(mask, close_ksize)
    mask = clean_mask(mask, open_r=0, close_r=0, min_hole=100, min_obj=min_area)

    if disk is not None:
        mask = mask * disk

    instances = mask_nms(mask_to_instances(mask, min_area, max_area, apply_watershed))
    rles = [rle_encode(inst) for inst in instances]
    return rles, instances


if __name__ == "__main__":
    # quick test
    fake = np.zeros((2048, 2048), dtype=np.float32)
    fake[500:600, 500:700] = 0.9
    fake[800:820, 800:1200] = 0.9
    rles, insts = full_pipeline(fake, threshold=0.5)
    print("encoded filaments:", len(rles))
