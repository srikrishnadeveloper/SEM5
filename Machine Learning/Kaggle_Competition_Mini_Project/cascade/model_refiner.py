"""Dedicated high-resolution patch segmenter for Cascade 2.0.

Architecture:
- SMP U-Net++ or U-Net with high-resolution skip connections.
- 3-channel input: [Raw Grayscale, CLAHE Contrast Channel, Coarse YOLO Mask Prior].
- Composite Loss: BCE + Soft-Dice + Focal Loss for sharp thin boundary delineation.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class CompositeRefinerLoss(nn.Module):
    """Composite loss balancing pixel BCE, soft Dice overlap, and Focal hard-pixel mining."""

    def __init__(
        self,
        bce_weight: float = 0.4,
        dice_weight: float = 0.4,
        focal_weight: float = 0.2,
        smooth: float = 1.0,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth
        self.gamma = gamma
        self.bce_fn = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw unnormalized logits
            targets: (B, 1, H, W) binary ground truth {0, 1}
        """
        # 1. Binary Cross-Entropy
        loss_bce = self.bce_fn(logits, targets)

        # 2. Soft Dice Loss
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        cardinality = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss_dice = 1.0 - dice.mean()

        # 3. Focal Loss
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_factor = (1.0 - p_t) ** self.gamma
        bce_pointwise = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss_focal = (focal_factor * bce_pointwise).mean()

        total_loss = (
            self.bce_weight * loss_bce
            + self.dice_weight * loss_dice
            + self.focal_weight * loss_focal
        )
        return total_loss


def build_crop_refiner(
    arch: str = "unetplusplus",
    encoder_name: str = "tu-efficientnet_b2",
    in_channels: int = 3,
    classes: int = 1,
    encoder_weights: Optional[str] = "imagenet",
) -> nn.Module:
    """Build and initialize the high-resolution patch refiner network."""
    arch_lower = arch.lower()
    if arch_lower in ("unetplusplus", "unet++"):
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    elif arch_lower == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    else:
        raise ValueError(f"Unsupported refiner architecture: {arch}. Choose 'unetplusplus' or 'unet'.")

    return model
