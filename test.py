"""
test.py — BrainMRI 模型测试与评估
====================================
在测试集上加载训练好的模型权重, 做完整评估:
  - 滑窗推理 (与训练验证一致)
  - 计算 Dice / IoU / Sensitivity (per-class + mean)
  - 可选 TTA (4 翻转增强)
  - 可选保存预测结果为 NIfTI
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_load import BrainMRIDataset
from model import build_model
from param_set import parse_args
from train import compute_metrics, _sliding_window_inference


# ═══════════════════════════════════════════════════════════════
#  TTA 辅助
# ═══════════════════════════════════════════════════════════════


def _tta_transforms(volume: torch.Tensor):
    """
    生成 4 种翻转变体 + 逆变换函数。
    volume: (C, D, H, W)
    """
    transforms = [
        (volume, lambda x: x),                                     # 原始
        (torch.flip(volume, [2]), lambda x: torch.flip(x, [2])),   # D 翻转
        (torch.flip(volume, [3]), lambda x: torch.flip(x, [3])),   # H 翻转
        (torch.flip(volume, [4]), lambda x: torch.flip(x, [4])),   # W 翻转
    ]
    return transforms


# ═══════════════════════════════════════════════════════════════
#  单样本推理 (含 TTA)
# ═══════════════════════════════════════════════════════════════


@torch.no_grad()
def inference_single(
    model: nn.Module,
    volume: torch.Tensor,
    patch_size: Tuple[int, int, int],
    overlap: float,
    device: str,
    use_amp: bool,
    use_tta: bool = False,
) -> torch.Tensor:
    """
    对单个 3D 体积做推理, 返回 (num_classes, D, H, W) 概率图。
    """
    model.eval()

    if not use_amp:
        # 不用 autocast 时直接调用滑窗推理
        return _sliding_window_inference(
            model, volume, patch_size, overlap, device, use_amp=False
        )

    if not use_tta:
        with torch.amp.autocast('cuda'):
            prob = _sliding_window_inference(
                model, volume, patch_size, overlap, device, use_amp=True
            )
        return prob

    # TTA: 对每种变换做推理, 再逆变换求平均
    tta_pairs = _tta_transforms(volume)
    prob_sum = None

    for vol_transformed, inv_fn in tta_pairs:
        with torch.amp.autocast('cuda'):
            prob = _sliding_window_inference(
                model, vol_transformed, patch_size, overlap, device, use_amp=True
            )
        prob_inv = inv_fn(prob)
        if prob_sum is None:
            prob_sum = prob_inv
        else:
            prob_sum += prob_inv

    prob_avg = prob_sum / len(tta_pairs)
    return prob_avg


# ═══════════════════════════════════════════════════════════════
#  完整测试流程
# ═══════════════════════════════════════════════════════════════


def test(args=None) -> Dict[str, float]:
    """
    在测试集上评估模型, 打印并返回指标结果。

    Parameters
    ----------
    args : argparse.Namespace | None
        若为 None 则自动解析命令行参数。

    Returns
    -------
    metrics : dict, 包含 per-class 和 mean 的 Dice / IoU / Sensitivity
    """
    if args is None:
        args = parse_args()

    device = torch.device(args.device)

    # ── 数据集 ──
    print("=" * 70)
    print("  加载测试数据集...")
    print("=" * 70)

    test_ds = BrainMRIDataset(
        split=args.test_split,
        target_spacing=args.target_spacing,
        use_cache=bool(args.use_cache),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
    )
    print(f"  测试集: {len(test_ds)} 样本 (split={args.test_split})")

    # ── 模型 ──
    print(f"\n  构建模型: {args.model_name}")
    model = build_model(args).to(device)

    # ── 加载权重 ──
    ckpt_dir = os.path.join(args.output_dir, args.model_name)
    ckpt_path = os.path.join(ckpt_dir, "best_model.pth")

    if not os.path.isfile(ckpt_path):
        # 尝试找其他权重文件
        alt_ckpts = [
            f for f in os.listdir(ckpt_dir)
            if f.endswith(".pth")
        ] if os.path.isdir(ckpt_dir) else []

        if alt_ckpts:
            ckpt_path = os.path.join(ckpt_dir, sorted(alt_ckpts)[-1])
            print(f"  best_model.pth 未找到, 使用: {ckpt_path}")
        else:
            raise FileNotFoundError(
                f"未找到检查点: {ckpt_path}  "
                f"请先运行训练, 或通过 --resume_ckpt 指定路径"
            )

    print(f"  加载权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if "val_metrics" in ckpt:
        print(f"  训练时最优 Dice(Mean): {ckpt['val_metrics'].get('dice_mean', 'N/A')}")

    # ── 推理参数 ──
    patch_size = tuple(args.patch_size)
    overlap = args.patch_overlap
    use_amp = args.amp == 1
    use_tta = args.tta

    # ── 评估 ──
    print("\n" + "=" * 70)
    print("  开始测试评估")
    print("=" * 70)

    all_metrics: Dict[str, float] = {}
    n_samples = 0

    # 输出目录
    pred_dir = os.path.join(args.output_dir, args.model_name, "predictions")
    if args.save_pred:
        os.makedirs(pred_dir, exist_ok=True)

    for idx, (volume, seg, info) in enumerate(test_loader):
        volume = volume.to(device, non_blocking=True)
        seg = seg.to(device, non_blocking=True)

        # volume: (1, C, D, H, W), seg: (1, D, H, W)
        vol_i = volume[0]   # (C, D, H, W)
        seg_i = seg[0]       # (D, H, W)

        # 推理
        prob_map = inference_single(
            model, vol_i, patch_size, overlap, device, use_amp, use_tta
        )  # (C, D, H, W)

        # 计算指标
        pred_batch = prob_map.unsqueeze(0)   # (1, C, D, H, W)
        seg_batch = seg_i.unsqueeze(0)        # (1, D, H, W)

        metrics = compute_metrics(pred_batch, seg_batch, args.num_classes)

        for k, v in metrics.items():
            all_metrics[k] = all_metrics.get(k, 0.0) + v
        n_samples += 1

        # 打印单样本结果
        patient_id = info["patient_id"][0] if isinstance(info["patient_id"], (list, tuple)) else info["patient_id"]
        print(
            f"  [{idx+1}/{len(test_ds)}] {patient_id}: "
            f"Dice={metrics['dice_mean']:.4f}  "
            f"IoU={metrics['iou_mean']:.4f}  "
            f"Sens={metrics['sens_mean']:.4f}"
        )

        # 保存预测 NIfTI
        if args.save_pred:
            _save_prediction_nifti(prob_map, info, pred_dir)

    # ── 平均指标 ──
    avg_metrics = {k: v / n_samples for k, v in all_metrics.items()}

    # ── 打印结果 ──
    print("\n" + "=" * 70)
    print("  测试评估结果")
    print("=" * 70)
    print(f"  模型: {args.model_name}")
    print(f"  数据集: {args.test_split} ({n_samples} 样本)")
    print(f"  TTA: {'启用' if use_tta else '关闭'}")
    print()

    # per-class
    label_names = ["Background", "NCR (Necrotic core)", "ED (Edema)", "ET (Enhancing tumor)"]
    for c in range(args.num_classes):
        name = label_names[c] if c < len(label_names) else f"Class {c}"
        print(
            f"  {name}:\n"
            f"    Dice: {avg_metrics[f'dice_c{c}']:.4f}\n"
            f"    IoU:  {avg_metrics[f'iou_c{c}']:.4f}\n"
            f"    Sens: {avg_metrics[f'sens_c{c}']:.4f}"
        )

    print()
    print(
        f"  ── Mean (不含背景) ──\n"
        f"    Dice: {avg_metrics['dice_mean']:.4f}\n"
        f"    IoU:  {avg_metrics['iou_mean']:.4f}\n"
        f"    Sens: {avg_metrics['sens_mean']:.4f}"
    )
    print("=" * 70)

    # ── 保存指标到文件 ──
    metrics_path = os.path.join(args.output_dir, args.model_name, "test_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Dataset: {args.test_split} ({n_samples} samples)\n")
        f.write(f"TTA: {use_tta}\n\n")
        for c in range(args.num_classes):
            name = label_names[c] if c < len(label_names) else f"Class {c}"
            f.write(f"{name}:\n")
            f.write(f"  Dice: {avg_metrics[f'dice_c{c}']:.4f}\n")
            f.write(f"  IoU:  {avg_metrics[f'iou_c{c}']:.4f}\n")
            f.write(f"  Sens: {avg_metrics[f'sens_c{c}']:.4f}\n")
        f.write(f"\nMean (excl. bg):\n")
        f.write(f"  Dice: {avg_metrics['dice_mean']:.4f}\n")
        f.write(f"  IoU:  {avg_metrics['iou_mean']:.4f}\n")
        f.write(f"  Sens: {avg_metrics['sens_mean']:.4f}\n")

    print(f"  指标已保存至: {metrics_path}")

    return avg_metrics


# ═══════════════════════════════════════════════════════════════
#  保存 NIfTI 预测
# ═══════════════════════════════════════════════════════════════


def _save_prediction_nifti(
    prob_map: torch.Tensor,
    info: dict,
    pred_dir: str,
) -> None:
    """
    将预测概率图保存为 NIfTI 文件。
    prob_map: (C, D, H, W) on any device
    """
    try:
        import nibabel as nib
    except ImportError:
        print("    [警告] nibabel 未安装, 跳过 NIfTI 保存")
        return

    pred_labels = prob_map.argmax(dim=0).cpu().numpy().astype(np.uint8)  # (D, H, W)

    patient_id = info["patient_id"]
    if isinstance(patient_id, (list, tuple)):
        patient_id = patient_id[0]

    out_path = os.path.join(pred_dir, f"{patient_id}_pred.nii.gz")
    nib.save(nib.Nifti1Image(pred_labels, affine=np.eye(4)), out_path)
