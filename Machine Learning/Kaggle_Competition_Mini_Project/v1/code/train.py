# train.py - one-fold top-50 training loop
import csv, os, time
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import config, dataset, losses, metrics, model


def _instances(mask):
    lab, n = ndimage.label(mask > 0)
    return np.stack([(lab == i).astype(np.float32) for i in range(1, n + 1)]) if n else np.zeros((0, *mask.shape), np.float32)


def train_one_epoch(net, loader, criterion, optimizer, scheduler=None, scaler=None,
                    accumulation_steps=config.ACCUMULATION_STEPS, device=config.device, ema=None):
    net.train(); optimizer.zero_grad(set_to_none=True)
    total_loss = total_dice = 0.0
    scaler = scaler or GradScaler(enabled=False)
    for step, (imgs, masks, _) in enumerate(tqdm(loader, desc="train")):
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=config.USE_AMP and device.type == "cuda"):
            logits = net(imgs)
            if isinstance(logits, (list, tuple)): logits = logits[0]
            if logits.shape[-2:] != masks.shape[-2:]: logits = nn.functional.interpolate(logits, masks.shape[-2:], mode="bilinear", align_corners=False)
            raw_loss = criterion(logits, masks)
        scaler.scale(raw_loss / accumulation_steps).backward()
        if (step + 1) % accumulation_steps == 0 or step + 1 == len(loader):
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if ema is not None: ema.update_parameters(net)
        total_loss += raw_loss.item()
        total_dice += metrics.dice_score((torch.sigmoid(logits) > .5).cpu().numpy(), masks.cpu().numpy())
    return total_loss / max(len(loader), 1), total_dice / max(len(loader), 1)


@torch.no_grad()
def validate(net, loader, criterion, device=config.device, threshold=config.PROB_THRESHOLD, prob_dir=None):
    net.eval(); loss_sum = dice_sum = 0.; pred_all, gt_all, pqs = [], [], []
    if prob_dir: os.makedirs(prob_dir, exist_ok=True)
    for imgs, masks, ids in tqdm(loader, desc="val"):
        imgs, masks = imgs.to(device), masks.to(device)
        with autocast(device_type=device.type, enabled=config.USE_AMP and device.type == "cuda"):
            logits = net(imgs)
            if logits.shape[-2:] != masks.shape[-2:]: logits = nn.functional.interpolate(logits, masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, masks)
        probs = torch.sigmoid(logits).float().cpu().numpy(); true = masks.cpu().numpy()
        loss_sum += loss.item()
        for p, g, image_id in zip(probs, true, ids):
            if prob_dir: np.save(os.path.join(prob_dir, f"{image_id}.npy"), p[0].astype(np.float16))
            pi, gi = _instances(p[0] > threshold), _instances(g[0] > .5)
            pred_all.append(pi); gt_all.append(gi)
            dice_sum += metrics.dice_score(p[0] > threshold, g[0] > .5)
            pqs.append(metrics.panoptic_quality(torch.from_numpy(gi), torch.from_numpy(pi))[0])
    pair = metrics.mean_mIoU_pairwise(pred_all, gt_all)["mIoU_pairwise_micro"]
    multi = metrics.mean_mIoU_multiscale(pred_all, gt_all)["mIoU_multiscale_micro"]
    n = max(len(pred_all), 1)
    return {"loss": loss_sum / max(len(loader), 1), "dice": dice_sum / n,
            "pq": float(np.mean(pqs)) if pqs else 0., "mIoU_pairwise": pair, "mIoU_multiscale": multi}


def train_fold(fold=config.VAL_FOLD, epochs=config.EPOCHS, save_dir=config.MODELS_DIR, log_dir=None, resume=None):
    os.makedirs(save_dir, exist_ok=True); log_dir = log_dir or save_dir
    dataset.set_seed(config.SEED + fold)
    train_loader, val_loader, _, _ = dataset.make_dataloaders(val_fold=fold, image_size=config.TRAIN_RES,
        batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    net = model.get_model().to(config.device)
    pos_weight = config.POS_WEIGHT or losses.estimate_pos_weight(train_loader)
    criterion = losses.CombinedTop50Loss(pos_weight).to(config.device)
    optimizer = AdamW(net.parameters(), lr=config.LR, weight_decay=config.WD)
    warm = min(config.WARMUP_EPOCHS, max(epochs - 1, 0)) if config.USE_WARMUP else 0
    if warm:
        scheduler = SequentialLR(optimizer, [LinearLR(optimizer, .01, 1., warm),
            CosineAnnealingLR(optimizer, max(epochs - warm, 1), eta_min=config.MIN_LR)], [warm])
    else: scheduler = CosineAnnealingLR(optimizer, max(epochs, 1), eta_min=config.MIN_LR)
    scaler = GradScaler(enabled=config.USE_AMP and config.device.type == "cuda")
    ema = model.create_ema_model(net) if config.USE_EMA else None
    start, best, stale = 1, -1., 0
    resume = resume or config.RESUME
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location=config.device, weights_only=False); net.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"]); scheduler.load_state_dict(ck["scheduler"])
        start, best = ck.get("epoch", 0) + 1, ck.get("best", -1.)
        if ema and "ema_model" in ck: ema.load_state_dict(ck["ema_model"])
    csv_path, best_path = os.path.join(log_dir, f"fold_{fold}_metrics.csv"), os.path.join(save_dir, f"best_fold_{fold}.pth")
    if start == 1:
        with open(csv_path, "w", newline="") as f: csv.writer(f).writerow(["epoch","train_loss","train_dice","val_loss","val_dice","val_pq","mIoU_pairwise","mIoU_multiscale","best"])
    for epoch in range(start, epochs + 1):
        t = time.time(); tr_loss, tr_dice = train_one_epoch(net, train_loader, criterion, optimizer, scaler=scaler, ema=ema)
        eval_net = ema or net
        vals = validate(eval_net, val_loader, criterion, prob_dir=os.path.join(config.SUBMISSIONS_DIR, f"oof_probs_fold_{fold}") if epoch == epochs else None)
        scheduler.step(); score = vals.get(config.VAL_METRIC, vals["mIoU_multiscale"]); improved = score > best
        state = {"model": net.state_dict(), "ema_model": ema.state_dict() if ema else None, "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "epoch": epoch, "best": max(best, score), "metric": config.VAL_METRIC}
        if improved: best, stale = score, 0; torch.save(state, best_path)
        else: stale += 1
        if epoch % config.CHECKPOINT_EVERY == 0: torch.save(state, os.path.join(save_dir, f"fold_{fold}_epoch_{epoch}.pth"))
        print(f"epoch {epoch}/{epochs} train dice {tr_dice:.4f} | val Dice {vals['dice']:.4f} PQ {vals['pq']:.4f} pair {vals['mIoU_pairwise']:.4f} multiscale {vals['mIoU_multiscale']:.4f} | best {best:.4f} ({time.time()-t:.0f}s)")
        with open(csv_path, "a", newline="") as f: csv.writer(f).writerow([epoch,tr_loss,tr_dice,vals["loss"],vals["dice"],vals["pq"],vals["mIoU_pairwise"],vals["mIoU_multiscale"],int(improved)])
        if stale >= config.PATIENCE: print(f"early stopping at epoch {epoch}"); break
    # Always save OOF probabilities from the selected model.
    model.load_checkpoint(net, best_path); validate(net, val_loader, criterion, prob_dir=os.path.join(config.SUBMISSIONS_DIR, f"oof_probs_fold_{fold}"))
    return best_path


if __name__ == "__main__": train_fold()
