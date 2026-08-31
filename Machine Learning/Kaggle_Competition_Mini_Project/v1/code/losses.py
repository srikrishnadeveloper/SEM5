# losses.py - imbalance and boundary aware segmentation losses
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch.losses as smp_losses
import config


class DiceLoss(nn.Module):
    def forward(self, logits, target):
        prob = torch.sigmoid(logits).flatten()
        target = target.flatten()
        return 1 - (2 * (prob * target).sum() + 1e-6) / (prob.sum() + target.sum() + 1e-6)


class FocalLoss(nn.Module):
    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.where(target > 0.5, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        return (config.FOCAL_ALPHA * (1 - pt).pow(config.FOCAL_GAMMA) * bce).mean()


class BoundaryLoss(nn.Module):
    """Distance-map loss. Returns zero when a distance map is not supplied."""
    def forward(self, prob, dist_map=None):
        if dist_map is None:
            return prob.sum() * 0.0
        if dist_map.ndim == 3:
            dist_map = dist_map.unsqueeze(1)
        return torch.einsum("bchw,bchw->", prob, dist_map.to(prob)) / prob.numel()


class CombinedTop50Loss(nn.Module):
    def __init__(self, pos_weight=1.0, bce_w=config.LOSS_BCE_W,
                 dice_w=config.LOSS_DICE_W, tversky_w=config.LOSS_TVERSKY_W,
                 boundary_w=config.LOSS_BOUNDARY_W):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(max(pos_weight, 1.0))))
        self.dice = DiceLoss()
        self.tversky = smp_losses.TverskyLoss(mode="binary", alpha=0.3, beta=0.7,
                                               gamma=4 / 3, from_logits=True)
        self.boundary = BoundaryLoss()
        self.weights = bce_w, dice_w, tversky_w, boundary_w

    def forward(self, logits, target, dist_map=None):
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        b, d, t, bd = self.weights
        return (b * bce + d * self.dice(logits, target) + t * self.tversky(logits, target)
                + bd * self.boundary(torch.sigmoid(logits), dist_map))


class CombinedLoss(CombinedTop50Loss):
    """Backwards-compatible name."""


def estimate_pos_weight(loader, max_batches=32):
    pos = total = 0
    for i, (_, masks, _) in enumerate(loader):
        pos += masks.sum().item(); total += masks.numel()
        if i + 1 >= max_batches: break
    return min(max((total - pos) / max(pos, 1), 1.0), 1000.0)


if __name__ == "__main__":
    criterion = CombinedTop50Loss(pos_weight=10)
    print("combined loss:", criterion(torch.randn(2, 1, 64, 64),
                                      torch.randint(0, 2, (2, 1, 64, 64)).float()).item())
