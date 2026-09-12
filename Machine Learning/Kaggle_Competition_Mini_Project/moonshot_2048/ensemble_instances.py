"""
moonshot_2048/ensemble_instances.py — Topology-Safe Cross-Model Instance Ensemble.

Authority: ChatGPT Master
Directive: 17 — Native-2048 Moonshot Toward 0.60 PQ
Executor: Antigravity

Frozen Ensemble Rules:
1. NEVER blindly union instance masks (destroys fine filament topology and inflates false positives).
2. Cluster predictions across models (e.g. YOLOv8l + YOLO11l or TTA views) based on bounding box / mask IoU.
3. For each cluster, evaluate candidate representations:
   - Highest-confidence proposal
   - Majority-vote consensus (threshold >= 0.5)
   - Intersection-biased consensus (strict precision)
4. Sort consensus instances by calibrated match probability / confidence, then area.
5. Apply strict greedy pixel-carve zero-overlap sanitizer.
6. Assert 0 shared pixels between any two emitted masks per disk.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from metrics.pq import _iou
from moonshot_2048.config import MoonshotConfig
from moonshot_2048.predict_native import sanitize_instances_zero_overlap


def cluster_instances(
    proposals_list: List[Dict[str, Any]],
    cluster_iou_thresh: float = 0.3,
) -> List[List[Dict[str, Any]]]:
    """Group overlapping instance proposals into consensus clusters."""
    if not proposals_list:
        return []

    n = len(proposals_list)
    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue
        cluster = [proposals_list[i]]
        visited[i] = True

        for j in range(i + 1, n):
            if visited[j]:
                continue
            # Check mask IoU
            iou_val = _iou(proposals_list[i]["mask"], proposals_list[j]["mask"])
            if iou_val >= cluster_iou_thresh:
                cluster.append(proposals_list[j])
                visited[j] = True

        clusters.append(cluster)
    return clusters


def resolve_cluster_consensus(
    cluster: List[Dict[str, Any]],
    method: str = "best_conf",
    vote_threshold: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """Resolve a consensus mask for a cluster of candidate proposals.
    
    Methods:
      - 'best_conf': select the proposal with the highest model confidence.
      - 'majority_vote': pixel-wise voting across all proposals in cluster.
      - 'intersection': only keep pixels shared by all proposals in cluster.
    """
    if len(cluster) == 1:
        return cluster[0]["mask"], float(cluster[0]["conf"])

    if method == "best_conf":
        best_item = max(cluster, key=lambda x: x["conf"])
        return best_item["mask"], float(best_item["conf"])

    # Voting methods
    stacked = np.stack([item["mask"].astype(np.float32) for item in cluster], axis=0)
    mean_prob = np.mean(stacked, axis=0)
    avg_conf = float(np.mean([item["conf"] for item in cluster]))

    if method == "majority_vote":
        consensus_mask = (mean_prob >= vote_threshold).astype(np.uint8)
    elif method == "intersection":
        consensus_mask = (mean_prob == 1.0).astype(np.uint8)
    else:
        best_item = max(cluster, key=lambda x: x["conf"])
        return best_item["mask"], float(best_item["conf"])

    if consensus_mask.sum() == 0:
        # Fallback to highest confidence if voting vanished
        best_item = max(cluster, key=lambda x: x["conf"])
        return best_item["mask"], float(best_item["conf"])

    return consensus_mask, avg_conf


def ensemble_disk_predictions(
    model_predictions: List[List[Dict[str, Any]]],
    method: str = "best_conf",
    cluster_iou_thresh: float = 0.3,
    min_area: int = MoonshotConfig.MIN_AREA,
) -> Tuple[List[np.ndarray], List[float]]:
    """Ensemble instance predictions from multiple models/views for a single disk."""
    # Flatten all proposals
    all_proposals = []
    for model_idx, preds in enumerate(model_predictions):
        for p in preds:
            p_copy = dict(p)
            p_copy["model_idx"] = model_idx
            all_proposals.append(p_copy)

    if not all_proposals:
        return [], []

    # Form clusters
    clusters = cluster_instances(all_proposals, cluster_iou_thresh=cluster_iou_thresh)

    candidate_masks = []
    candidate_confs = []

    for cl in clusters:
        c_mask, c_conf = resolve_cluster_consensus(cl, method=method)
        if c_mask.sum() >= min_area:
            candidate_masks.append(c_mask)
            candidate_confs.append(c_conf)

    # Strict greedy pixel-carve zero-overlap sanitizer
    sanitized_masks, sanitized_confs = sanitize_instances_zero_overlap(
        candidate_masks, candidate_confs, min_area=min_area
    )
    return sanitized_masks, sanitized_confs
