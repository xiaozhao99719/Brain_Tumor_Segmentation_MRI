"""
train.py — BrainMRI 模型训练与验证
====================================
完整训练流程:
  - 支持 nnU-Net / Attention U-Net / TransUNet
  - 3D 随机 patch 裁剪训练 (适配大体积)
  - 损失函数: Dice+CE / Dice / CE / Focal
  - 评估指标: Dice / IoU / Sensitivity (per-class + mean)
  - 每 epoch 打印训练 loss 和验证指标
  - AMP 混合精度支持
"""

from __future__ import annotations

import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# 从项目内导入
from data_load import BrainMRIDataset
from model import build_model
from param_set import parse_args


# ═══════════════════════════════════════════════════════════════
#  损失函数
# ═══════════════════════════════════════════════════════════════


class DiceLoss(nn.Module):
    """多类别 Soft Dice Loss。"""

    def __init__(self, num_classes: int, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred: (B, C, D, H, W) logits; target: (B, D, H, W) int64。"""
        pred_prob = torch.softmax(pred, dim=1)
        target_onehot = nn.functional.one_hot(target, self.num_classes)  # (B,D,H,W,C)
        target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()    # (B,C,D,H,W)

        dims = (0, 2, 3, 4)  # batch + spatial
        intersection = (pred_prob * target_onehot).sum(dim=dims)
        cardinality = pred_prob.sum(dim=dims) + target_onehot.sum(dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class FocalLoss(nn.Module):
    """多类别 Focal Loss。"""

    def __init__(self, num_classes: int, gamma: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


def build_loss_fn(args) -> nn.Module:
    """根据参数构建损失函数。"""
    nc = args.num_classes
    if args.loss_fn == "dice_ce":
        return _DiceCELoss(nc, args.dice_smooth)
    elif args.loss_fn == "dice":
        return DiceLoss(nc, args.dice_smooth)
    elif args.loss_fn == "ce":
        return nn.CrossEntropyLoss()
    elif args.loss_fn == "focal":
        return FocalLoss(nc, args.focal_gamma)
    else:
        raise ValueError(f"未知损失函数: {args.loss_fn}")


class _DiceCELoss(nn.Module):
    """Dice + CE 联合损失。"""

    def __init__(self, num_classes: int, smooth: float = 1.0):
        super().__init__()
        self.dice = DiceLoss(num_classes, smooth)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        return self.dice(pred, target) + self.ce(pred, target)


# ═══════════════════════════════════════════════════════════════
#  评估指标
# ═══════════════════════════════════════════════════════════════


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """
    计算 per-class 与 mean 的 Dice / IoU / Sensitivity。

    Parameters
    ----------
    pred : (B, C, D, H, W) logits
    target : (B, D, H, W) int64
    num_classes : 类别数 (含背景)
    eps : 防除零

    Returns
    -------
    dict with keys:
        dice_c{i}, iou_c{i}, sens_c{i}  for each class
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

    # 跳过背景 (c=0) 计算 mean
    results["dice_mean"] = float(np.mean(dices[1:])) if num_classes > 1 else 0.0
    results["iou_mean"] = float(np.mean(ious[1:])) if num_classes > 1 else 0.0
    results["sens_mean"] = float(np.mean(senss[1:])) if num_classes > 1 else 0.0

    return results


# ═══════════════════════════════════════════════════════════════
#  3D Patch 裁剪工具
# ═══════════════════════════════════════════════════════════════


def random_patch_crop(
    volume: torch.Tensor,
    seg: torch.Tensor,
    patch_size: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从完整 3D 体积中随机裁剪一个 patch。
    volume: (C, D, H, W), seg: (D, H, W)
    """
    _, d, h, w = volume.shape
    pd, ph, pw = patch_size

    if d < pd or h < ph or w < pw:
        # pad if volume smaller than patch
        volume = nn.functional.pad(volume, [0, max(0, pw - w), 0, max(0, ph - h), 0, max(0, pd - d)])
        seg = nn.functional.pad(seg.unsqueeze(0), [0, max(0, pw - w), 0, max(0, ph - h), 0, max(0, pd - d)])
        seg = seg.squeeze(0)
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
    """将体积中心裁剪/填充到目标尺寸。用于 TransUNet 等需要固定输入尺寸的模型。"""
    _, d, h, w = volume.shape
    td, th, tw = target_size

    # pad if needed
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


# ═══════════════════════════════════════════════════════════════
#  训练单个 Epoch
# ═══════════════════════════════════════════════════════════════


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    args,
    scaler: GradScaler = None,
) -> float:
    """训练一个 epoch, 返回平均 loss。"""
    model.train()
    total_loss = 0.0
    n_batches = 0

    patch_size = tuple(args.patch_size)
    use_amp = args.amp == 1

    for batch_idx, (volume, seg, info) in enumerate(dataloader):
        # volume: (B, 4, D, H, W), seg: (B, D, H, W)
        volume = volume.to(device, non_blocking=True)
        seg = seg.to(device, non_blocking=True)

        # 随机 patch 裁剪 (减少显存)
        if args.random_patch == 1 and args.model_name != "transunet":
            # 对 batch 中每个样本单独裁剪
            patches_v, patches_s = [], []
            for i in range(volume.size(0)):
                pv, ps = random_patch_crop(volume[i], seg[i], patch_size)
                patches_v.append(pv)
                patches_s.append(ps)
            volume = torch.stack(patches_v)
            seg = torch.stack(patches_s)
        elif args.model_name == "transunet":
            # TransUNet 需要固定输入尺寸
            fixed_size = (args.vit_img_size,) * 3
            patches_v, patches_s = [], []
            for i in range(volume.size(0)):
                pv = center_crop_or_pad(volume[i], fixed_size)
                ps_vol = seg[i].unsqueeze(0)
                ps_vol = center_crop_or_pad(ps_vol, fixed_size)
                patches_v.append(pv)
                patches_s.append(ps_vol.squeeze(0))
            volume = torch.stack(patches_v)
            seg = torch.stack(patches_s)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with autocast('cuda'):
                pred = model(volume)
                loss = loss_fn(pred, seg)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(volume)
            loss = loss_fn(pred, seg)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % args.log_every == 0:
            print(f"  [Epoch {epoch}] Batch {batch_idx+1}/{len(dataloader)}  "
                  f"Loss: {loss.item():.5f}")

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


# ═══════════════════════════════════════════════════════════════
#  验证
# ═══════════════════════════════════════════════════════════════


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    args,
) -> Tuple[float, Dict[str, float]]:
    """
    在验证集上评估, 返回 (avg_loss, metrics_dict)。
    采用滑窗推理 + 重叠融合以处理大体积。
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    # 累计指标
    all_metrics = {}

    patch_size = tuple(args.patch_size)
    overlap = args.patch_overlap
    use_amp = args.amp == 1

    for volume, seg, info in dataloader:
        volume = volume.to(device, non_blocking=True)
        seg = seg.to(device, non_blocking=True)
        B = volume.size(0)

        for i in range(B):
            vol_i = volume[i]      # (4, D, H, W)
            seg_i = seg[i]         # (D, H, W)

            # 对整个体积做滑窗推理
            pred_full = _sliding_window_inference(
                model, vol_i, patch_size, overlap, device, use_amp
            )  # (C, D, H, W)

            pred_batch = pred_full.unsqueeze(0)   # (1, C, D, H, W)
            seg_batch = seg_i.unsqueeze(0)         # (1, D, H, W)

            loss = loss_fn(pred_batch, seg_batch)
            total_loss += loss.item()
            n_batches += 1

            metrics = compute_metrics(pred_batch, seg_batch, args.num_classes)
            for k, v in metrics.items():
                all_metrics[k] = all_metrics.get(k, 0.0) + v

    avg_loss = total_loss / max(n_batches, 1)

    # 平均指标
    n_samples = n_batches
    avg_metrics = {k: v / n_samples for k, v in all_metrics.items()}

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
    对单个 3D 体积做滑窗推理, 返回与输入同尺寸的概率图。

    volume: (C, D, H, W)
    """
    model.eval()
    C_in, D, H, W = volume.shape
    num_classes = None  # 推断得到

    stride = tuple(int(p * (1 - overlap)) for p in patch_size)
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    # 输出累积器
    output_sum = torch.zeros(1, D, H, W, device=device)  # placeholder
    count_map = torch.zeros(D, H, W, device=device)

    # 遍历所有 patch 位置
    d_starts = list(range(0, max(D - pd + 1, 1), max(sd, 1)))
    h_starts = list(range(0, max(H - ph + 1, 1), max(sh, 1)))
    w_starts = list(range(0, max(W - pw + 1, 1), max(sw, 1)))

    # 确保最后一个 patch 能覆盖边缘
    if d_starts[-1] + pd < D:
        d_starts.append(max(D - pd, 0))
    if h_starts[-1] + ph < H:
        h_starts.append(max(H - ph, 0))
    if w_starts[-1] + pw < W:
        w_starts.append(max(W - pw, 0))

    first = True
    for d_s in d_starts:
        for h_s in h_starts:
            for w_s in w_starts:
                d_e = min(d_s + pd, D)
                h_e = min(h_s + ph, H)
                w_e = min(w_s + pw, W)

                patch = volume[:, d_s:d_e, h_s:h_e, w_s:w_e].unsqueeze(0)  # (1, C, d, h, w)

                # pad to patch_size if needed
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

                if first:
                    num_classes = pred_prob.size(1)
                    output_sum = torch.zeros(num_classes, D, H, W, device=device)
                    first = False

                # 去掉 padding
                actual_d = d_e - d_s
                actual_h = h_e - h_s
                actual_w = w_e - w_s
                pred_prob = pred_prob[0, :, :actual_d, :actual_h, :actual_w]

                output_sum[:, d_s:d_e, h_s:h_e, w_s:w_e] += pred_prob
                count_map[d_s:d_e, h_s:h_e, w_s:w_e] += 1.0

    # 平均重叠区域
    count_map = count_map.clamp(min=1.0)
    for c in range(num_classes):
        output_sum[c] /= count_map

    return output_sum  # (C, D, H, W) logits-like (实际是概率)


# ═══════════════════════════════════════════════════════════════
#  完整训练流程
# ═══════════════════════════════════════════════════════════════


def train(args=None) -> nn.Module:
    """
    完整训练流程: 数据加载 → 模型构建 → 训练循环 → 保存最优权重。

    Parameters
    ----------
    args : argparse.Namespace | None
        若为 None 则自动解析命令行参数。

    Returns
    -------
    model : 训练完成的模型
    """
    if args is None:
        args = parse_args()

    # 固定随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    # ── 创建输出目录 ──
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── 数据集 ──
    print("=" * 70)
    print("  加载数据集...")
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
        batch_size=1,  # 验证时逐样本推理
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
    )

    print(f"  训练集: {len(train_ds)} 样本")
    print(f"  验证集: {len(val_ds)} 样本")

    # ── 模型 ──
    print(f"\n  构建模型: {args.model_name}")
    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数量: {n_params:,}")

    # ── 损失函数 ──
    loss_fn = build_loss_fn(args)
    print(f"  损失函数: {args.loss_fn}")

    # ── 优化器 & 调度器 ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
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

    # ── AMP Scaler ──
    scaler = GradScaler('cuda') if args.amp == 1 else None

    # ── 恢复训练 ──
    start_epoch = 1
    best_dice = 0.0
    if args.resume_ckpt and os.path.isfile(args.resume_ckpt):
        print(f"  恢复检查点: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dice = ckpt.get("best_dice", 0.0)
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    # ── 训练循环 ──
    print("\n" + "=" * 70)
    print("  开始训练")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # 训练
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch, args, scaler
        )

        # 验证
        val_loss, val_metrics = validate(
            model, val_loader, loss_fn, device, args
        )

        # 学习率调度
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0

        # 打印 epoch 结果
        print(
            f"\n[Epoch {epoch}/{args.epochs}]  "
            f"Train Loss: {train_loss:.5f}  |  "
            f"Val Loss: {val_loss:.5f}  |  "
            f"Dice(Mean): {val_metrics['dice_mean']:.4f}  |  "
            f"IoU(Mean): {val_metrics['iou_mean']:.4f}  |  "
            f"Sens(Mean): {val_metrics['sens_mean']:.4f}  |  "
            f"Time: {elapsed/60:.1f}min"
        )

        # per-class 详细指标
        for c in range(args.num_classes):
            print(
                f"    Class {c}: "
                f"Dice={val_metrics[f'dice_c{c}']:.4f}  "
                f"IoU={val_metrics[f'iou_c{c}']:.4f}  "
                f"Sens={val_metrics[f'sens_c{c}']:.4f}"
            )

        # 保存最优模型
        current_dice = val_metrics["dice_mean"]
        is_best = current_dice > best_dice
        if is_best:
            best_dice = current_dice

        if is_best or (epoch % args.save_every == 0):
            ckpt_path = os.path.join(
                ckpt_dir,
                "best_model.pth" if is_best else f"epoch_{epoch}.pth",
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "val_metrics": val_metrics,
                "args": vars(args),
            }, ckpt_path)
            tag = " (BEST)" if is_best else ""
            print(f"    -> 保存检查点: {ckpt_path}{tag}")

    print("\n" + "=" * 70)
    print(f"  训练完成!  最优 Dice(Mean): {best_dice:.4f}")
    print(f"  检查点目录: {ckpt_dir}")
    print("=" * 70)

    return model
