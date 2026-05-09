"""
train.py -- BrainMRI Model Training & Validation (Optimized)
=============================================================
Complete training pipeline:
    - Supports nnU-Net / Attention U-Net / TransUNet
    - 3D random patch cropping (suited for large volumes)
    - Loss functions: Dice+CE / Dice / CE / Focal (with Label Smoothing)
    - Metrics: Dice / IoU / Sensitivity (per-class + mean)
    - Per-epoch training loss and validation metrics
    - AMP mixed-precision support

Optimizations applied:
    1. Gradient accumulation (virtual batch_size=4) -- reduces gradient noise
    2. Gradient clipping (max_norm=1.0) -- prevents loss spikes
    3. EMA weight exponential moving average (decay=0.999) -- more stable training
    4. Label Smoothing (smoothing=0.1) -- improves generalization
    5. Lower learning rate (1e-4->5e-5) -- reduces training oscillation
"""

from __future__ import annotations

import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Project imports
from data_load import BrainMRIDataset
from model import build_model
from param_set import parse_args


# ============================================================================
#  Loss Functions
# ============================================================================


class DiceLoss(nn.Module):
    """Multi-class Soft Dice Loss."""

    def __init__(self, num_classes: int, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred: (B, C, D, H, W) logits; target: (B, D, H, W) int64."""
        pred_prob = torch.softmax(pred, dim=1)
        target_onehot = nn.functional.one_hot(target, self.num_classes)  # (B,D,H,W,C)
        target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()    # (B,C,D,H,W)

        dims = (0, 2, 3, 4)  # batch + spatial
        intersection = (pred_prob * target_onehot).sum(dim=dims)
        cardinality = pred_prob.sum(dim=dims) + target_onehot.sum(dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class FocalLoss(nn.Module):
    """Multi-class Focal Loss."""

    def __init__(self, num_classes: int, gamma: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


class _DiceCELoss(nn.Module):
    """Dice + CE combined loss (with Label Smoothing)."""

    def __init__(self, num_classes: int, smooth: float = 1.0):
        super().__init__()
        self.dice = DiceLoss(num_classes, smooth)
        # CE with Label Smoothing to prevent overfitting
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, pred, target):
        return self.dice(pred, target) + self.ce(pred, target)


def build_loss_fn(args) -> nn.Module:
    """Build loss function from args."""
    nc = args.num_classes
    if args.loss_fn == "dice_ce":
        return _DiceCELoss(nc, args.dice_smooth)
    elif args.loss_fn == "dice":
        return DiceLoss(nc, args.dice_smooth)
    elif args.loss_fn == "ce":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    elif args.loss_fn == "focal":
        return FocalLoss(nc, args.focal_gamma)
    else:
        raise ValueError(f"Unknown loss function: {args.loss_fn}")


# ============================================================================
#  EMA (Exponential Moving Average)
# ============================================================================


class ModelEMA:
    """
    Exponential moving average of model weights.

    Maintains a shadow model during training; verification / saving use
    shadow weights, which is equivalent to low-pass filtering the training
    process and significantly improves stability and generalization.

    Usage:
        ema = ModelEMA(model, decay=0.999)
        # After each forward pass:
        ema.update()
        # During evaluation, switch to shadow weights:
        ema.apply_shadow()
        evaluate(model)
        ema.restore()
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, device=None):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register(model)

    def _register(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"EMA: parameter '{name}' not found"
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
        self.backup.clear()


# ============================================================================
#  Evaluation Metrics
# ============================================================================


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """
    Compute per-class and mean Dice / IoU / Sensitivity.

    Parameters
    ----------
    pred : (B, C, D, H, W) logits
    target : (B, D, H, W) int64
    num_classes : Number of classes (including background)
    eps : Epsilon to prevent division by zero

    Returns
    -------
    dict with keys:
        dice_c{i}, iou_c{i}, sens_c{i} for each class
        dice_mean, iou_mean, sens_mean
    """
    pred_labels = pred.argmax(dim=1)  # (B, D, H, W)
    results: Dict[str, float] = {}

    dices, ious, senss = [], [], []

    for c in range(num_classes):
        pred_c = (pred_labels == c)
        tgt_c = (target == c)

        tp = (pred_c & tgt_c).sum().float().item()
        fp = (pred_c & ~tgt_c).sum().float().item()
        fn = (~pred_c & tgt_c).sum().float().item()

        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        sens = (tp + eps) / (tp + fn + eps)

        results[f"dice_c{c}"] = dice
        results[f"iou_c{c}"] = iou
        results[f"sens_c{c}"] = sens

        dices.append(dice)
        ious.append(iou)
        senss.append(sens)

    # Skip background (c=0) for mean
    results["dice_mean"] = float(np.mean(dices[1:])) if num_classes > 1 else 0.0
    results["iou_mean"] = float(np.mean(ious[1:])) if num_classes > 1 else 0.0
    results["sens_mean"] = float(np.mean(senss[1:])) if num_classes > 1 else 0.0

    return results


# ============================================================================
#  3D Patch Crop Utilities
# ============================================================================


def random_patch_crop(
    volume: torch.Tensor,
    seg: torch.Tensor,
    patch_size: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly crop a patch from a full 3D volume.
    volume: (C, D, H, W), seg: (D, H, W)
    """
    _, d, h, w = volume.shape
    pd, ph, pw = patch_size

    if d < pd or h < ph or w < pw:
        # Pad if volume is smaller than patch
        volume = nn.functional.pad(
            volume, [0, max(0, pw - w), 0, max(0, ph - h), 0, max(0, pd - d)]
        )
        seg = nn.functional.pad(
            seg.unsqueeze(0),
            [0, max(0, pw - w), 0, max(0, ph - h), 0, max(0, pd - d)],
        ).squeeze(0)
        _, d, h, w = volume.shape

    sd = torch.randint(0, d - pd + 1, (1,)).item()
    sh = torch.randint(0, h - ph + 1, (1,)).item()
    sw = torch.randint(0, w - pw + 1, (1,)).item()

    vol_patch = volume[:, sd:sd + pd, sh:sh + ph, sw:sw + pw]
    seg_patch = seg[sd:sd + pd, sh:sh + ph, sw:sw + pw]
    return vol_patch, seg_patch


def center_crop_or_pad(
    volume: torch.Tensor,
    target_size: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Center-crop or pad a volume to the target size.
    Used by TransUNet and other models that require a fixed input size.
    """
    _, d, h, w = volume.shape
    td, th, tw = target_size

    # Pad if needed
    pad_d = max(0, td - d)
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        volume = nn.functional.pad(volume, [
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2,
            pad_d // 2, pad_d - pad_d // 2,
        ])
        _, d, h, w = volume.shape

    sd = (d - td) // 2
    sh = (h - th) // 2
    sw = (w - tw) // 2
    return volume[:, sd:sd + td, sh:sh + th, sw:sw + tw]


# ============================================================================
#  Training One Epoch
# ============================================================================


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    args,
    scaler: GradScaler = None,
    ema: ModelEMA = None,
) -> float:
    """
    Train one epoch and return the average loss.

    Optimizations applied:
        - Gradient accumulation (accumulation_steps=4): effective batch_size=4
        - Gradient clipping (max_norm=1.0): prevents gradient explosion
        - EMA weight update: keeps shadow weights in sync
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    patch_size = tuple(args.patch_size)
    use_amp = args.amp == 1
    accum_steps = args.grad_accum_steps
    grad_scale = 1.0 / accum_steps  # Scale loss to compensate for accumulation

    for batch_idx, (volume, seg, _) in enumerate(dataloader):
        # volume: (B, 4, D, H, W), seg: (B, D, H, W)
        volume = volume.to(device, non_blocking=True)
        seg = seg.to(device, non_blocking=True)

        # Random patch cropping (reduces GPU memory)
        if args.random_patch == 1 and args.model_name != "transunet":
            patches_v, patches_s = [], []
            for i in range(volume.size(0)):
                pv, ps = random_patch_crop(volume[i], seg[i], patch_size)
                patches_v.append(pv)
                patches_s.append(ps)
            volume = torch.stack(patches_v)
            seg = torch.stack(patches_s)
        elif args.model_name == "transunet":
            # TransUNet requires fixed input size
            fixed_size = (args.vit_img_size,) * 3
            patches_v, patches_s = [], []
            for i in range(volume.size(0)):
                pv = center_crop_or_pad(volume[i], fixed_size)
                ps_vol = center_crop_or_pad(seg[i].unsqueeze(0), fixed_size)
                patches_v.append(pv)
                patches_s.append(ps_vol.squeeze(0))
            volume = torch.stack(patches_v)
            seg = torch.stack(patches_s)

        # Zero gradients at the start of each accumulation group
        if batch_idx % accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        # Forward pass
        if use_amp and scaler is not None:
            with autocast('cuda'):
                pred = model(volume)
                loss = loss_fn(pred, seg) * grad_scale
            scaler.scale(loss).backward()
        else:
            pred = model(volume)
            loss = loss_fn(pred, seg) * grad_scale
            loss.backward()

        # Optimizer step every accum_steps batches
        if (batch_idx + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update(model)

        total_loss += loss.item() * accum_steps  # Restore original scale
        n_batches += 1

        if (batch_idx + 1) % args.log_every == 0:
            print(f"  [Epoch {epoch}] Batch {batch_idx+1}/{len(dataloader)}  "
                  f"Loss: {loss.item()*accum_steps:.5f}")

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


# ============================================================================
#  Validation
# ============================================================================


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    args,
    ema: ModelEMA = None,
    use_ema: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate on the validation set. Returns (avg_loss, metrics_dict).

    By default, uses EMA shadow weights for inference, which gives
    more stable validation metrics.
    """
    model.eval()

    if ema is not None and use_ema:
        ema.apply_shadow(model)

    total_loss = 0.0
    n_batches = 0
    all_metrics = {}

    patch_size = tuple(args.patch_size)
    overlap = args.patch_overlap
    use_amp = args.amp == 1

    for volume, seg, _ in dataloader:
        volume = volume.to(device, non_blocking=True)
        seg = seg.to(device, non_blocking=True)
        B = volume.size(0)

        for i in range(B):
            vol_i = volume[i]      # (4, D, H, W)
            seg_i = seg[i]         # (D, H, W)

            # Sliding window inference on the full volume
            pred_full = _sliding_window_inference(
                model, vol_i, patch_size, overlap, device, use_amp
            )  # (C, D, H, W)

            pred_batch = pred_full.unsqueeze(0)   # (1, C, D, H, W)
            seg_batch = seg_i.unsqueeze(0)        # (1, D, H, W)

            loss = loss_fn(pred_batch, seg_batch)
            total_loss += loss.item()
            n_batches += 1

            metrics = compute_metrics(pred_batch, seg_batch, args.num_classes)
            for k, v in metrics.items():
                all_metrics[k] = all_metrics.get(k, 0.0) + v

    avg_loss = total_loss / max(n_batches, 1)
    n_samples = n_batches
    avg_metrics = {k: v / n_samples for k, v in all_metrics.items()}

    if ema is not None and use_ema:
        ema.restore(model)

    return avg_loss, avg_metrics


def _sliding_window_inference(
    model: nn.Module,
    volume: torch.Tensor,
    patch_size: Tuple[int, int, int],
    overlap: float,
    device: str,
    use_amp: bool,
) -> torch.Tensor:
    """
    Sliding window inference on a single 3D volume.

    Returns a probability map of the same spatial size as the input.

    volume: (C, D, H, W)
    """
    model.eval()
    C_in, D, H, W = volume.shape

    stride = tuple(int(p * (1 - overlap)) for p in patch_size)
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    # Output accumulators; num_classes is determined from the first forward pass
    output_sum = None
    count_map = torch.zeros(D, H, W, device=device)

    # Iterate over all patch positions
    d_starts = list(range(0, max(D - pd + 1, 1), max(sd, 1)))
    h_starts = list(range(0, max(H - ph + 1, 1), max(sh, 1)))
    w_starts = list(range(0, max(W - pw + 1, 1), max(sw, 1)))

    # Ensure the last patch covers the edge
    if d_starts[-1] + pd < D:
        d_starts.append(max(D - pd, 0))
    if h_starts[-1] + ph < H:
        h_starts.append(max(H - ph, 0))
    if w_starts[-1] + pw < W:
        w_starts.append(max(W - pw, 0))

    for d_s in d_starts:
        for h_s in h_starts:
            for w_s in w_starts:
                d_e = min(d_s + pd, D)
                h_e = min(h_s + ph, H)
                w_e = min(w_s + pw, W)

                patch = volume[:, d_s:d_e, h_s:h_e, w_s:w_e].unsqueeze(0)  # (1, C, d, h, w)

                # Pad to patch_size if needed
                if patch.shape[2:] != patch_size:
                    patch = nn.functional.pad(
                        patch,
                        [0, pw - patch.shape[4], 0, ph - patch.shape[3], 0, pd - patch.shape[2]],
                    )

                if use_amp:
                    with autocast('cuda'):
                        pred_patch = model(patch)   # (1, num_classes, pd, ph, pw)
                else:
                    pred_patch = model(patch)

                pred_prob = torch.softmax(pred_patch, dim=1)  # (1, C, pd, ph, pw)

                if output_sum is None:
                    num_classes = pred_prob.size(1)
                    output_sum = torch.zeros(num_classes, D, H, W, device=device)

                # Remove padding
                actual_d = d_e - d_s
                actual_h = h_e - h_s
                actual_w = w_e - w_s
                pred_prob = pred_prob[0, :, :actual_d, :actual_h, :actual_w]

                output_sum[:, d_s:d_e, h_s:h_e, w_s:w_e] += pred_prob
                count_map[d_s:d_e, h_s:h_e, w_s:w_e] += 1.0

    # Average overlapping regions
    count_map = count_map.clamp(min=1.0)
    output_sum = output_sum / count_map.unsqueeze(0)  # type: ignore[assignment]

    return output_sum  # (C, D, H, W) probabilities


# ============================================================================
#  Full Training Pipeline
# ============================================================================


def train(args=None) -> nn.Module:
    """
    Complete training pipeline: data loading -> model build -> training loop -> save best weights.

    Parameters
    ----------
    args : argparse.Namespace | None
        If None, parses from command line automatically.

    Returns
    -------
    model : Trained model
    """
    if args is None:
        args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    # -- Create output directories --
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # -- Dataset --
    print("=" * 70)
    print("  Loading datasets...")
    print("=" * 70)

    train_ds = BrainMRIDataset(
        split="train",
        target_spacing=args.target_spacing,
        use_cache=bool(args.use_cache),
    )
    val_ds = BrainMRIDataset(
        split="val",
        target_spacing=args.target_spacing,
        use_cache=bool(args.use_cache),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # Per-sample sliding window inference
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
    )

    print(f"  Training set: {len(train_ds)} samples")
    print(f"  Validation set: {len(val_ds)} samples")

    # -- Model --
    print(f"\n  Building model: {args.model_name}")
    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # -- Loss function --
    loss_fn = build_loss_fn(args)
    print(f"  Loss function: {args.loss_fn}")

    # -- Optimizer & Scheduler --
    print(f"  Learning rate: {args.effective_lr}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.effective_lr,
        weight_decay=args.weight_decay,
    )

    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_lr_step_size, gamma=args.step_lr_gamma
        )
    else:
        scheduler = None

    # -- AMP Scaler --
    scaler = GradScaler('cuda') if args.amp == 1 else None

    # -- EMA --
    ema = ModelEMA(model, decay=0.999, device=device)
    print(f"  EMA weight moving average: decay=0.999")

    # -- Gradient accumulation --
    print(f"  Gradient accumulation: accum_steps={args.grad_accum_steps}  "
          f"(effective batch_size={args.batch_size * args.grad_accum_steps})")

    # -- Resume from checkpoint --
    start_epoch = 1
    best_dice = 0.0
    if args.resume_ckpt and os.path.isfile(args.resume_ckpt):
        print(f"  Resuming from checkpoint: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dice = ckpt.get("best_dice", 0.0)
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    # -- Training loop --
    print("\n" + "=" * 70)
    print("  Starting training")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch, args, scaler, ema
        )

        # Validate (using EMA weights for more stability)
        val_loss, val_metrics = validate(
            model, val_loader, loss_fn, device, args, ema, use_ema=True
        )

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0

        # Print epoch summary
        print(
            f"\n[Epoch {epoch}/{args.epochs}]  "
            f"Train Loss: {train_loss:.5f}  |  "
            f"Val Loss: {val_loss:.5f}  |  "
            f"Dice(Mean): {val_metrics['dice_mean']:.4f}  |  "
            f"IoU(Mean): {val_metrics['iou_mean']:.4f}  |  "
            f"Sens(Mean): {val_metrics['sens_mean']:.4f}  |  "
            f"Time: {elapsed/60:.1f}min"
        )

        # Per-class details
        for c in range(args.num_classes):
            print(
                f"    Class {c}: "
                f"Dice={val_metrics[f'dice_c{c}']:.4f}  "
                f"IoU={val_metrics[f'iou_c{c}']:.4f}  "
                f"Sens={val_metrics[f'sens_c{c}']:.4f}"
            )

        # Save best model
        current_dice = val_metrics["dice_mean"]
        is_best = current_dice > best_dice
        if is_best:
            best_dice = current_dice

        if is_best or (epoch % args.save_every == 0):
            ckpt_path = os.path.join(
                ckpt_dir,
                "best_model.pth" if is_best else f"epoch_{epoch}.pth",
            )
            # Save with EMA weights (more stable model)
            if ema is not None:
                ema.apply_shadow(model)

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "val_metrics": val_metrics,
                "args": vars(args),
            }, ckpt_path)

            if ema is not None:
                ema.restore(model)

            tag = " (BEST)" if is_best else ""
            print(f"    -> Saved checkpoint: {ckpt_path}{tag}")

    print("\n" + "=" * 70)
    print(f"  Training complete!  Best Dice(Mean): {best_dice:.4f}")
    print(f"  Checkpoint directory: {ckpt_dir}")
    print("=" * 70)

    return model
