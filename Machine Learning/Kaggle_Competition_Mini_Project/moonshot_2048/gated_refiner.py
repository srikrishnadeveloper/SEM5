"""
moonshot_2048/gated_refiner.py — Optional Gated Crop Refiner & Topology Guard.

Authority: ChatGPT Master
Directive: 17 — Native-2048 Moonshot Toward 0.60 PQ
Executor: Antigravity

Frozen Refiner Specification:
1. Optional, NOT mandatory: Refinement is gated per-instance.
2. Rejection criteria:
   - Reject if component count explodes (fragmentation).
   - Reject if refined area / proposal area ratio leaves bounds [0.5, 2.0].
   - Reject if proposal-refiner Dice < 0.60 (unstable divergence).
3. If rejected, preserve the original direct detector proposal.
"""

from typing import Dict, Tuple, Optional
import cv2
import numpy as np


class GatedRefinerGuard:
    """Topology and divergence gate controlling instance refinement."""

    def __init__(
        self,
        min_dice_stability: float = 0.60,
        min_area_ratio: float = 0.50,
        max_area_ratio: float = 2.00,
        max_component_increase: int = 1,
    ):
        self.min_dice = min_dice_stability
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.max_component_increase = max_component_increase

    def should_accept_refined_mask(
        self,
        original_mask: np.ndarray,
        refined_mask: np.ndarray,
    ) -> Tuple[bool, str]:
        """Decide whether to accept or reject the refined mask."""
        orig_area = float(original_mask.sum())
        ref_area = float(refined_mask.sum())

        if orig_area == 0:
            return False, "original_empty"
        if ref_area == 0:
            return False, "refined_empty"

        # 1. Area ratio check
        ratio = ref_area / orig_area
        if ratio < self.min_area_ratio or ratio > self.max_area_ratio:
            return False, f"area_ratio_out_of_bounds_{ratio:.2f}"

        # 2. Dice stability check
        intersection = float(np.logical_and(original_mask, refined_mask).sum())
        dice = (2.0 * intersection) / (orig_area + ref_area + 1e-5)
        if dice < self.min_dice:
            return False, f"dice_instability_{dice:.2f}"

        # 3. Topology component explosion check
        _, orig_labels = cv2.connectedComponents(original_mask.astype(np.uint8))
        _, ref_labels = cv2.connectedComponents(refined_mask.astype(np.uint8))
        orig_comp = int(np.max(orig_labels))
        ref_comp = int(np.max(ref_labels))

        if ref_comp > (orig_comp + self.max_component_increase):
            return False, f"component_explosion_{orig_comp}->{ref_comp}"

        return True, "accepted"
