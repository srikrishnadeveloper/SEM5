# metrics.py
# competition-style metrics for the Solar Filament Segmentation Challenge 2026.
#
# Important note on the evaluation:
#   The older 01_competition_brief.md describes a 70 % quantitative / 30 %
#   qualitative split based on Mean Dice + Panoptic Quality.  The research
#   verification (verification_and_recommendations.md §4) and the EdgeAttNet
#   paper (arXiv:2509.02964, §IV) show that the actual Kaggle/IEEE BigData Cup
#   2026 scoring uses mIoU_pairwise and mIoU_multiscale.  We therefore keep
#   the Dice/PQ helpers (train.py and postprocess.py still call dice_score and
#   panoptic_quality) and add the two competition metrics as first-class
#   functions.
#
# References:
#   - EdgeAttNet, arXiv:2509.02964, §IV: pairwise and multiscale IoU.
#   - Multiscale IoU, arXiv:2105.14572 / ICIP 2021: edge-based, multi-scale
#     intersection ratio.  The official reference implementation (Bitbucket
#     gsudmlab/multiscale_iou) is mirrored closely in PySODMetrics; we follow
#     the same defaults (Sobel edges, cell sizes 2^0..2^9, trapezoidal integral,
#     +1 smoothing to avoid divide-by-zero).
#   - Self-evaluation notebook: relaxed Panoptic Quality used as a proxy.

from typing import List, Tuple, Optional, Union

import numpy as np
import torch
from scipy import ndimage


# default 10 cell sizes used by the Multiscale IoU / EdgeAttNet reference.
MIoU_CELL_SIZES: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def _trapezoid(y, dx, axis):
    """
    NumPy 2.0 removed np.trapz in favor of np.trapezoid; older code uses
    np.trapz, and older SciPy uses scipy.integrate.trapz.  Try all common
    names so the code works across dependency versions.
    """
    for mod in (np,):
        for name in ("trapezoid", "trapz"):
            fn = getattr(mod, name, None)
            if fn is not None:
                return fn(y, dx=dx, axis=axis)
    from scipy import integrate
    for name in ("trapezoid", "trapz"):
        fn = getattr(integrate, name, None)
        if fn is not None:
            return fn(y, dx=dx, axis=axis)
    raise ImportError("no trapezoid implementation found in numpy or scipy")


def dice_score(pred: np.ndarray, true: np.ndarray, eps: float = 1e-7) -> float:
    """pixel-level dice for one binary mask."""
    pred = pred.astype(np.float32).ravel()
    true = true.astype(np.float32).ravel()
    inter = (pred * true).sum()
    return (2.0 * inter + eps) / (pred.sum() + true.sum() + eps)


def iou_score(pred: np.ndarray, true: np.ndarray, eps: float = 1e-7) -> float:
    """pixel-level intersection over union for one binary mask."""
    pred = pred.astype(np.float32).ravel()
    true = true.astype(np.float32).ravel()
    inter = (pred * true).sum()
    union = pred.sum() + true.sum() - inter
    return (inter + eps) / (union + eps)


def rles_to_layers(rles: List[str], height: int = 2048, width: int = 2048) -> np.ndarray:
    """
    decode a list of compressed COCO RLE strings into (n_masks, H, W) array.
    pycocotools is imported lazily so the rest of the module can be used
    without it being installed.
    """
    import pycocotools.mask as mask_util

    if not rles:
        return np.zeros((0, height, width), dtype=np.float32)
    rle_dicts = [{"size": [height, width], "counts": rle} for rle in rles]
    masks = mask_util.decode(rle_dicts)
    return masks.transpose(2, 0, 1).astype(np.float32)


def _to_float_layers(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """make sure we have a (n, H, W) float32 torch tensor."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.detach().cpu().float()
    if x.dim() == 2:
        x = x.unsqueeze(0)
    while x.dim() > 3:
        if x.shape[0] == 1:
            x = x.squeeze(0)
        elif x.shape[1] == 1:
            x = x.squeeze(1)
        else:
            break
    return x


def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """detach and move torch tensors or pass numpy through."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_overlap_matrices(
    gt_layers: torch.Tensor, pred_layers: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    pairwise iou and dice between GT and predicted instance masks.
    shapes: (n_gt, H, W) and (n_pred, H, W).
    """
    n_gt, h, w = gt_layers.shape
    n_pred = pred_layers.shape[0]

    gt_flat = gt_layers.reshape(n_gt, h * w)
    pred_flat = pred_layers.reshape(n_pred, h * w)

    intersection = torch.matmul(gt_flat, pred_flat.t())
    gt_areas = gt_flat.sum(dim=1, keepdim=True)
    pred_areas = pred_flat.sum(dim=1, keepdim=True).t()

    union = gt_areas + pred_areas - intersection
    iou = torch.where(
        union == 0, torch.tensor(0.0, device=gt_layers.device), intersection / union
    )
    dice = torch.where(
        union == 0,
        torch.tensor(0.0, device=gt_layers.device),
        2 * intersection / (gt_areas + pred_areas),
    )
    return iou, dice


def fp_count_hit(hit_matrix: torch.Tensor) -> int:
    """predicted masks that match no gt mask."""
    return (hit_matrix.sum(dim=0) == 0).sum().item()


def fn_count_hit(hit_matrix: torch.Tensor) -> int:
    """gt masks that match no predicted mask."""
    return (hit_matrix.sum(dim=1) == 0).sum().item()


def panoptic_quality(
    gt_layers: torch.Tensor,
    pred_layers: torch.Tensor,
    iou_threshold: float = 0.5,
) -> Tuple[float, int, int, int]:
    """
    compute the relaxed Panoptic Quality used by the public self-evaluation
    notebook: every (gt_i, pred_j) pair with IoU > iou_threshold counts as a
    true positive and contributes its IoU.  This is NOT the standard one-to-one
    PQ from Kirillov et al., but it is the formula currently wired into the
    public baseline / self-eval notebook and is kept for backwards compat.

    returns (pq_score, n_tp, n_fp, n_fn).
    """
    n_gt, n_pred = gt_layers.shape[0], pred_layers.shape[0]
    if n_gt == 0:
        return 0.0, 0, n_pred, 0
    if n_pred == 0:
        return 0.0, 0, 0, n_gt

    iou_matrix, _ = get_overlap_matrices(gt_layers, pred_layers)
    hit_matrix = iou_matrix > iou_threshold

    # each pair with iou > 0.5 counts as a true positive, summing its iou
    tp_iou_scores = iou_matrix[hit_matrix].tolist()
    tp = len(tp_iou_scores)
    fp = fp_count_hit(hit_matrix)
    fn = fn_count_hit(hit_matrix)

    denom = tp + 0.5 * fp + 0.5 * fn
    if denom == 0:
        return 0.0, tp, fp, fn
    pq = sum(tp_iou_scores) / denom
    return pq, tp, fp, fn


def mean_pq(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> dict:
    """
    average PQ over a list of images.
    pred_masks and gt_masks are lists of (n_instances, H, W) arrays.
    """
    pq_scores = []
    total_tp = total_fp = total_fn = 0
    for p, g in zip(pred_masks, gt_masks):
        p = torch.from_numpy(p).float()
        g = torch.from_numpy(g).float()
        pq, tp, fp, fn = panoptic_quality(g, p)
        pq_scores.append(pq)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    return {
        "mean_pq": float(np.mean(pq_scores)),
        "pq_scores": pq_scores,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


def _get_edge(mask: np.ndarray) -> np.ndarray:
    """sobel edge map for a binary mask.  returns {0,1} uint8."""
    sx = ndimage.sobel(mask, axis=0, mode="constant")
    sy = ndimage.sobel(mask, axis=1, mode="constant")
    return (np.hypot(sx, sy) > 0).astype(np.uint8)


def _shrink_by_grid(mask: np.ndarray, cell_size: int) -> np.ndarray:
    """
    box-counting downsample used by the Multiscale IoU reference.  one output
    cell is 1 if any input pixel in that cell is 1.  zero-padding is added to
    the top/left to make the spatial dimensions divisible by cell_size.
    """
    if cell_size <= 0:
        raise ValueError("cell_size must be a positive integer")
    if cell_size == 1:
        return (mask > 0).astype(mask.dtype)

    h, w = mask.shape[:2]
    pad_h = (cell_size - h % cell_size) % cell_size
    pad_w = (cell_size - w % cell_size) % cell_size

    if pad_h > 0 or pad_w > 0:
        mask = np.pad(mask, ((pad_h, 0), (pad_w, 0)), mode="constant", constant_values=0)

    h, w = mask.shape[:2]
    mask = mask.reshape(h // cell_size, cell_size, w // cell_size, cell_size)
    return mask.any(axis=(1, 3)).astype(mask.dtype)


def mIoU_pairwise(
    gt_layers: Union[np.ndarray, torch.Tensor],
    pred_layers: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-7,
) -> Tuple[float, int]:
    """
    Mean pairwise IoU for one image (EdgeAttNet, arXiv:2509.02964, §IV-A).

    For every (gt_i, pred_j) with non-zero spatial intersection we compute
    IoU = |gt_i ∩ pred_j| / |gt_i ∪ pred_j| and average over those pairs.
    Pairs with no overlap are not included in the mean.

    returns (mIoU_pairwise, n_pairs_with_overlap).
    """
    gt = _to_float_layers(gt_layers)
    pred = _to_float_layers(pred_layers)
    iou_matrix, _ = get_overlap_matrices(gt, pred)
    hit = iou_matrix > eps
    n_pairs = int(hit.sum().item())
    if n_pairs == 0:
        return 0.0, 0
    return float(iou_matrix[hit].mean().item()), n_pairs


def mIoU_multiscale(
    gt_layers: Union[np.ndarray, torch.Tensor],
    pred_layers: Union[np.ndarray, torch.Tensor],
    cell_sizes: Optional[List[int]] = None,
    use_edges: bool = True,
    smooth: float = 1.0,
    use_trapz: bool = True,
    clip_large: bool = False,
    eps: float = 1e-7,
) -> Tuple[float, int]:
    """
    Mean multiscale IoU for one image (EdgeAttNet / Multiscale IoU).

    Per pair (gt_i, pred_j) with non-zero intersection, the MIoU is computed
    by downsampling the (edge) masks at each cell size, taking the ratio
    r = n(s(gt) ∩ s(pred)) / n(s(gt)) with optional +1 smoothing to avoid
    divide-by-zero, and integrating r over the scale range [0, 1] with the
    trapezoidal rule.  We then average over all overlapping pairs.

    Args:
        gt_layers: (n_gt, H, W) tensor/array of binary instance masks.
        pred_layers: (n_pred, H, W) tensor/array of binary instance masks.
        cell_sizes: list of cell sizes for box counting.  Default is
            [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] from the reference.
        use_edges: if True (default), extract Sobel edges before box counting
            as in the official MIoU implementation.  If False, downsample the
            filled region masks instead.
        smooth: additive smoothing for the intersection-ratio denominator.
            1.0 matches the PySODMetrics / reference implementation.  0.0 uses
            the unsmoothed paper formula with r=0 when n(s(gt)) == 0.
        use_trapz: if True (default), integrate with trapezoidal rule as in
            the reference; otherwise use a plain mean over scales.
        clip_large: if True, drop cell sizes larger than min(H, W) so every
            scale has at least a 2x2 downsampled grid.  Default False to match
            the reference (which pads the image up to the cell size).
        eps: threshold for deciding whether two masks overlap.

    returns (mIoU_multiscale, n_pairs_with_overlap).
    """
    if cell_sizes is None:
        cell_sizes = list(MIoU_CELL_SIZES)

    gt = _to_float_layers(gt_layers)
    pred = _to_float_layers(pred_layers)
    n_gt, H, W = gt.shape
    n_pred = pred.shape[0]

    # find overlapping pairs with the original (scale=1) IoU.
    iou_matrix, _ = get_overlap_matrices(gt, pred)
    pair_mask = iou_matrix > eps
    n_pairs = int(pair_mask.sum().item())
    if n_pairs == 0:
        return 0.0, 0

    gt_np = _to_numpy(gt)
    pred_np = _to_numpy(pred)

    if use_edges:
        gt_np = np.stack([_get_edge(m) for m in gt_np])
        pred_np = np.stack([_get_edge(m) for m in pred_np])
    else:
        gt_np = (gt_np > 0).astype(np.uint8)
        pred_np = (pred_np > 0).astype(np.uint8)

    if clip_large:
        max_cell = min(H, W)
        cell_sizes = [c for c in cell_sizes if c <= max_cell]
    if not cell_sizes:
        return 0.0, n_pairs

    ratios = []
    for cell_size in cell_sizes:
        gt_shrunk = np.stack([_shrink_by_grid(m, cell_size) for m in gt_np])
        pred_shrunk = np.stack([_shrink_by_grid(m, cell_size) for m in pred_np])

        h_new, w_new = gt_shrunk.shape[1:]
        gt_flat = gt_shrunk.reshape(n_gt, -1).astype(np.float64)
        pred_flat = pred_shrunk.reshape(n_pred, -1).astype(np.float64)

        intersection = gt_flat @ pred_flat.T
        gt_count = gt_flat.sum(axis=1, keepdims=True)

        if smooth > 0.0:
            ratio = (intersection + smooth) / (gt_count + smooth)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(gt_count > 0, intersection / gt_count, 0.0)
        ratios.append(ratio)

    ratios = np.stack(ratios, axis=0)  # (n_scales, n_gt, n_pred)

    if use_trapz and len(cell_sizes) > 1:
        mIoU_per_pair = _trapezoid(ratios, dx=1.0 / (len(cell_sizes) - 1), axis=0)
    else:
        mIoU_per_pair = ratios.mean(axis=0)

    pair_mask_np = pair_mask.detach().cpu().numpy()
    values = mIoU_per_pair[pair_mask_np]
    if values.size == 0:
        return 0.0, n_pairs
    return float(values.mean()), n_pairs


def mean_mIoU_pairwise(
    pred_masks: List[Union[np.ndarray, torch.Tensor]],
    gt_masks: List[Union[np.ndarray, torch.Tensor]],
    eps: float = 1e-7,
) -> dict:
    """
    macro and micro mIoU_pairwise over a list of images.
    pred_masks / gt_masks are lists of (n_instances, H, W) arrays.
    """
    scores = []
    n_pairs = []
    weighted_sum = 0.0
    total_pairs = 0
    for p, g in zip(pred_masks, gt_masks):
        s, n = mIoU_pairwise(g, p, eps=eps)
        scores.append(s)
        n_pairs.append(n)
        weighted_sum += s * n
        total_pairs += n

    macro = float(np.mean(scores)) if scores else 0.0
    micro = weighted_sum / total_pairs if total_pairs > 0 else 0.0
    return {
        "mIoU_pairwise_macro": macro,
        "mIoU_pairwise_micro": micro,
        "mIoU_pairwise_scores": scores,
        "n_pairs": n_pairs,
        "total_pairs": total_pairs,
    }


def mean_mIoU_multiscale(
    pred_masks: List[Union[np.ndarray, torch.Tensor]],
    gt_masks: List[Union[np.ndarray, torch.Tensor]],
    cell_sizes: Optional[List[int]] = None,
    use_edges: bool = True,
    smooth: float = 1.0,
    use_trapz: bool = True,
    clip_large: bool = False,
    eps: float = 1e-7,
) -> dict:
    """
    macro and micro mIoU_multiscale over a list of images.
    pred_masks / gt_masks are lists of (n_instances, H, W) arrays.
    """
    scores = []
    n_pairs = []
    weighted_sum = 0.0
    total_pairs = 0
    for p, g in zip(pred_masks, gt_masks):
        s, n = mIoU_multiscale(
            g,
            p,
            cell_sizes=cell_sizes,
            use_edges=use_edges,
            smooth=smooth,
            use_trapz=use_trapz,
            clip_large=clip_large,
            eps=eps,
        )
        scores.append(s)
        n_pairs.append(n)
        weighted_sum += s * n
        total_pairs += n

    macro = float(np.mean(scores)) if scores else 0.0
    micro = weighted_sum / total_pairs if total_pairs > 0 else 0.0
    return {
        "mIoU_multiscale_macro": macro,
        "mIoU_multiscale_micro": micro,
        "mIoU_multiscale_scores": scores,
        "n_pairs": n_pairs,
        "total_pairs": total_pairs,
    }


def ap_at_iou50(gt_layers, pred_layers) -> float:
    """Greedy one-to-one instance AP/precision at IoU 0.50."""
    gt, pred = _to_float_layers(gt_layers), _to_float_layers(pred_layers)
    if pred.shape[0] == 0:
        return 1.0 if gt.shape[0] == 0 else 0.0
    if gt.shape[0] == 0:
        return 0.0
    ious, _ = get_overlap_matrices(gt, pred)
    matches = 0
    used_gt = set()
    for j in range(pred.shape[0]):
        value, i = ious[:, j].max(dim=0)
        if value >= 0.5 and int(i) not in used_gt:
            used_gt.add(int(i)); matches += 1
    return matches / max(pred.shape[0] + gt.shape[0] - matches, 1)


def calculate_batch_dice(pred: torch.Tensor, true: torch.Tensor) -> float:
    """simple batch dice (same as the public baseline name)."""
    pred = (torch.sigmoid(pred) > 0.5).float()
    pred = pred.detach().cpu().numpy()
    true = true.detach().cpu().numpy()
    return dice_score(pred, true)


if __name__ == "__main__":
    # tiny sanity test: one gt filament and one perfect + one shifted pred.
    g = np.zeros((256, 256), dtype=np.float32)
    g[50:80, 50:150] = 1.0

    p1 = g.copy()  # perfect match
    p2 = g.copy()
    p2[:, 2:] = p2[:, :-2]  # shifted right by 2 pixels

    gt_layers = torch.from_numpy(g[np.newaxis, :, :]).float()
    pred_layers = torch.from_numpy(np.stack([p1, p2])).float()

    print("dice perfect:", dice_score(p1, g))
    print("iou perfect:", iou_score(p1, g))
    print("mIoU_pairwise:", mIoU_pairwise(gt_layers, pred_layers))
    print("mIoU_multiscale:", mIoU_multiscale(gt_layers, pred_layers))
    print("PQ:", panoptic_quality(gt_layers, pred_layers))
