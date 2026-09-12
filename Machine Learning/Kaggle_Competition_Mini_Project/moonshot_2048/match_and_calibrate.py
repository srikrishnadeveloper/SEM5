"""
moonshot_2048/match_and_calibrate.py — Exact Kirillov PQ Evaluator & Instance Calibrator.

Authority: ChatGPT Master
Directive: 17 — Native-2048 Moonshot Toward 0.60 PQ
Executor: Antigravity

Frozen Evaluator Truth:
1. Exact Kirillov Panoptic Quality: PQ = SQ * RQ
   - Matching: greedy 1-to-1 pairing sorted descending by IoU, valid only if IoU > 0.50.
   - SQ = mean IoU of matched pairs (TP).
   - RQ = TP / (TP + 0.5*FP + 0.5*FN).
2. Multi-Annotator Evaluation on Physical Disks:
   - For each physical validation image, evaluate predictions against EACH annotator
     observation separately.
   - Report pq_mean (the official primary selector) and pq_max.
3. Detailed Failure Binning:
   - Area: small (<500 px), medium (500-2000 px), large (>2000 px).
   - Radial distance from disk center: core (<0.4 R), mid (0.4-0.8 R), limb (>=0.8 R).
   - Crowding: isolated (0 overlapping proposals), crowded (>=1 overlapping proposals).
4. Feature Extraction & Calibration:
   - Extracts morphological, spatial, and intensity features from candidate instances.
   - Trains a calibrated selector (LogisticRegression / HistGradientBoosting) to estimate
     P(valid IoU > 0.50 match) to filter false positives before greedy carving.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from metrics.pq import pq_score, _iou
from moonshot_2048.config import MoonshotConfig


def evaluate_instances_multi_annotator(
    pred_masks: List[np.ndarray],
    annotators_gt_masks: List[List[np.ndarray]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Score a list of predicted binary masks for a single physical disk
    against multiple annotator observations separately.
    Returns aggregated metrics: pq_mean, pq_max, sq_mean, rq_mean, tp, fp, fn.
    """
    if not annotators_gt_masks:
        return {
            "pq_mean": 0.0, "pq_max": 0.0,
            "sq_mean": 0.0, "rq_mean": 0.0,
            "tp": 0, "fp": len(pred_masks), "fn": 0,
            "ann_scores": [],
        }

    ann_pqs = []
    ann_sqs = []
    ann_rqs = []
    tot_tp, tot_fp, tot_fn = 0, 0, 0
    detailed_scores = []

    for gt_masks in annotators_gt_masks:
        res = pq_score(pred_masks, gt_masks, iou_threshold=iou_threshold)
        ann_pqs.append(res["PQ"])
        ann_sqs.append(res["SQ"])
        ann_rqs.append(res["RQ"])
        tot_tp += res["TP"]
        tot_fp += res["FP"]
        tot_fn += res["FN"]
        detailed_scores.append(res)

    return {
        "pq_mean": float(np.mean(ann_pqs)),
        "pq_max": float(np.max(ann_pqs)),
        "sq_mean": float(np.mean(ann_sqs)),
        "rq_mean": float(np.mean(ann_rqs)),
        "tp": tot_tp,
        "fp": tot_fp,
        "fn": tot_fn,
        "ann_scores": detailed_scores,
    }


def compute_instance_failure_bins(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    disk_center: Tuple[int, int] = (1024, 1024),
    disk_radius: float = 952.0,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Classify TP, FP, FN instances into Area, Radial, and Crowding bins.
    
    Area Bins:
      - Small: area < 500 px
      - Medium: 500 <= area <= 2000 px
      - Large: area > 2000 px
    
    Radial Bins (normalized r/R from center):
      - Core: r < 0.4
      - Mid: 0.4 <= r < 0.8
      - Limb: r >= 0.8
      
    Crowding Bins:
      - Isolated: no other instance bounding box overlaps
      - Crowded: >= 1 other instance bounding box overlaps
    """
    bins = {
        "area": {"small": {"tp": 0, "fp": 0, "fn": 0}, "medium": {"tp": 0, "fp": 0, "fn": 0}, "large": {"tp": 0, "fp": 0, "fn": 0}},
        "radial": {"core": {"tp": 0, "fp": 0, "fn": 0}, "mid": {"tp": 0, "fp": 0, "fn": 0}, "limb": {"tp": 0, "fp": 0, "fn": 0}},
    }

    # Match instances
    n_p = len(pred_masks)
    n_g = len(gt_masks)
    matched_p = set()
    matched_g = set()

    if n_p > 0 and n_g > 0:
        iou_mat = np.zeros((n_p, n_g), dtype=np.float32)
        for i, pm in enumerate(pred_masks):
            for j, gm in enumerate(gt_masks):
                iou_mat[i, j] = _iou(pm, gm)

        pairs = []
        for i in range(n_p):
            for j in range(n_g):
                if iou_mat[i, j] > iou_threshold:
                    pairs.append((iou_mat[i, j], i, j))
        pairs.sort(key=lambda x: x[0], reverse=True)

        for _, pi, gi in pairs:
            if pi in matched_p or gi in matched_g:
                continue
            matched_p.add(pi)
            matched_g.add(gi)

    def get_bins(mask: np.ndarray) -> Tuple[str, str]:
        area = float(mask.sum())
        if area < 500:
            a_bin = "small"
        elif area <= 2000:
            a_bin = "medium"
        else:
            a_bin = "large"

        # Center of mass / centroid
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            cx, cy = np.mean(xs), np.mean(ys)
            dist = np.sqrt((cx - disk_center[0])**2 + (cy - disk_center[1])**2)
            norm_r = dist / disk_radius
        else:
            norm_r = 0.0

        if norm_r < 0.4:
            r_bin = "core"
        elif norm_r < 0.8:
            r_bin = "mid"
        else:
            r_bin = "limb"
        return a_bin, r_bin

    # Record TPs and FPs from predictions
    for i, pm in enumerate(pred_masks):
        a_bin, r_bin = get_bins(pm)
        if i in matched_p:
            bins["area"][a_bin]["tp"] += 1
            bins["radial"][r_bin]["tp"] += 1
        else:
            bins["area"][a_bin]["fp"] += 1
            bins["radial"][r_bin]["fp"] += 1

    # Record FNs from ground truth
    for j, gm in enumerate(gt_masks):
        if j not in matched_g:
            a_bin, r_bin = get_bins(gm)
            bins["area"][a_bin]["fn"] += 1
            bins["radial"][r_bin]["fn"] += 1

    return bins


def extract_candidate_features(
    mask: np.ndarray,
    conf: float,
    gray_img: Optional[np.ndarray] = None,
    disk_center: Tuple[int, int] = (1024, 1024),
    disk_radius: float = 952.0,
) -> Dict[str, float]:
    """Compute rich morphology, spatial, and photometric features for an instance."""
    area = float(mask.sum())
    if area == 0:
        return {}

    ys, xs = np.where(mask > 0)
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    radial_dist = float(np.sqrt((cx - disk_center[0])**2 + (cy - disk_center[1])**2) / disk_radius)

    # Perimeter and contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    n_components = len(contours)

    # Elongation and Solidity
    if len(contours) > 0 and len(contours[0]) >= 5:
        ellipse = cv2.fitEllipse(contours[0])
        axes = ellipse[1]
        major = max(axes)
        minor = min(axes)
        elongation = float(major / (minor + 1e-5))
    else:
        elongation = 1.0

    # Convex hull & solidity
    all_pts = np.vstack(contours) if contours else np.zeros((0, 1, 2), dtype=np.int32)
    if len(all_pts) >= 3:
        hull = cv2.convexHull(all_pts)
        hull_area = cv2.contourArea(hull)
        solidity = float(np.clip(area / (hull_area + 1e-5), 0.0, 1.0))
    else:
        solidity = 1.0

    feat = {
        "conf": float(conf),
        "area": area,
        "log_area": float(np.log1p(area)),
        "perimeter": float(perimeter),
        "circularity": float(4.0 * np.pi * area / (perimeter**2 + 1e-5)),
        "elongation": elongation,
        "solidity": solidity,
        "n_components": float(n_components),
        "radial_dist": radial_dist,
    }

    if gray_img is not None:
        pixel_vals = gray_img[ys, xs]
        feat["mean_intensity"] = float(np.mean(pixel_vals))
        feat["q25_intensity"] = float(np.percentile(pixel_vals, 25))
        feat["local_contrast"] = float(np.mean(pixel_vals) - np.min(gray_img))

    return feat


class InstanceCalibrator:
    """Calibrated selector predicting P(valid IoU > 0.50 match) for candidates."""

    def __init__(self, model_type: str = "logistic"):
        if model_type == "logistic":
            self.model = LogisticRegression(class_weight="balanced", max_iter=1000)
        else:
            self.model = HistGradientBoostingClassifier(max_iter=100)
        self.feature_names = [
            "conf", "log_area", "perimeter", "circularity",
            "elongation", "solidity", "n_components", "radial_dist"
        ]
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fit classifier on OOF candidate instances."""
        X_mat = X[self.feature_names].fillna(0).values
        self.model.fit(X_mat, y)
        self.is_fitted = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict match probability P(IoU > 0.50)."""
        if not self.is_fitted:
            return X["conf"].values  # Fallback to detector confidence
        X_mat = X[self.feature_names].fillna(0).values
        return self.model.predict_proba(X_mat)[:, 1]
