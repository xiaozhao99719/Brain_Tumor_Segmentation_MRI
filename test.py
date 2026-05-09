"""
test.py -- BrainMRI Model Testing & Evaluation
==============================================
Load a trained model checkpoint and evaluate on the test set:
    - Sliding window inference (consistent with training / validation)
    - Metrics: Dice / IoU / Sensitivity (per-class + mean)
    - Optional TTA (4 flip augmentations)
    - Optional save predictions as NIfTI files
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_load import BrainMRIDataset
from model import build_model
from param_set import parse_args
from train import compute_metrics, _sliding_window_inference


# ============================================================================
#  TTA Helpers
# ============================================================================


def _tta_transforms(volume: torch.Tensor):
    """
    Generate 4 flip variants of the volume and their inverse transform functions.
    volume: (C, D, H, W)
    """
    transforms = [
        (volume, lambda x: x),                                     # Original
        (torch.flip(volume, [2]), lambda x: torch.flip(x, [2])),   # D flip
        (torch.flip(volume, [3]), lambda x: torch.flip(x, [3])),   # H flip
        (torch.flip(volume, [4]), lambda x: torch.flip(x, [4])),   # W flip
    ]
    return transforms


# ============================================================================
#  Single-Sample Inference (with TTA)
# ============================================================================


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
    Run inference on a single 3D volume. Returns (num_classes, D, H, W) probability map.
    """
    model.eval()

    if not use_amp:
        return _sliding_window_inference(
            model, volume, patch_size, overlap, device, use_amp=False
        )

    if not use_tta:
        with torch.amp.autocast('cuda'):
            prob = _sliding_window_inference(
                model, volume, patch_size, overlap, device, use_amp=True
            )
        return prob

    # TTA: average predictions over all flip variants
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


# ============================================================================
#  Full Test Pipeline
# ============================================================================


def test(args=None) -> Dict[str, float]:
    """
    Evaluate the model on the test set. Prints and returns metrics.

    Parameters
    ----------
    args : argparse.Namespace | None
        If None, parses from command line automatically.

    Returns
    -------
    metrics : dict with per-class and mean Dice / IoU / Sensitivity
    """
    if args is None:
        args = parse_args()

    device = torch.device(args.device)

    # -- Dataset --
    print("=" * 70)
    print("  Loading test dataset...")
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
    print(f"  Test set: {len(test_ds)} samples (split={args.test_split})")

    # -- Model --
    print(f"\n  Building model: {args.model_name}")
    model = build_model(args).to(device)

    # -- Load weights --
    ckpt_dir = os.path.join(args.output_dir, args.model_name)
    ckpt_path = os.path.join(ckpt_dir, "best_model.pth")

    if not os.path.isfile(ckpt_path):
        # Try to find an alternative checkpoint
        alt_ckpts = [
            f for f in os.listdir(ckpt_dir)
            if f.endswith(".pth")
        ] if os.path.isdir(ckpt_dir) else []

        if alt_ckpts:
            ckpt_path = os.path.join(ckpt_dir, sorted(alt_ckpts)[-1])
            print(f"  best_model.pth not found, using: {ckpt_path}")
        else:
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}.  "
                f"Run training first, or specify --resume_ckpt"
            )

    print(f"  Loading weights: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if "val_metrics" in ckpt:
        print(f"  Training best Dice(Mean): {ckpt['val_metrics'].get('dice_mean', 'N/A')}")

    # -- Inference parameters --
    patch_size = tuple(args.patch_size)
    overlap = args.patch_overlap
    use_amp = args.amp == 1
    use_tta = args.tta

    # -- Evaluation --
    print("\n" + "=" * 70)
    print("  Starting test evaluation")
    print("=" * 70)

    all_metrics: Dict[str, float] = {}
    n_samples = 0

    pred_dir = os.path.join(args.output_dir, args.model_name, "predictions")
    if args.save_pred:
        os.makedirs(pred_dir, exist_ok=True)

    for idx, (volume, seg, info) in enumerate(test_loader):
        volume = volume.to(device, non_blocking=True)
        seg = seg.to(device, non_blocking=True)

        # volume: (1, C, D, H, W), seg: (1, D, H, W)
        vol_i = volume[0]   # (C, D, H, W)
        seg_i = seg[0]       # (D, H, W)

        # Inference
        prob_map = inference_single(
            model, vol_i, patch_size, overlap, device, use_amp, use_tta
        )  # (C, D, H, W)

        # Compute metrics
        pred_batch = prob_map.unsqueeze(0)   # (1, C, D, H, W)
        seg_batch = seg_i.unsqueeze(0)        # (1, D, H, W)

        metrics = compute_metrics(pred_batch, seg_batch, args.num_classes)

        for k, v in metrics.items():
            all_metrics[k] = all_metrics.get(k, 0.0) + v
        n_samples += 1

        # Print per-sample result
        patient_id = info["patient_id"][0] if isinstance(info["patient_id"], (list, tuple)) else info["patient_id"]
        print(
            f"  [{idx+1}/{len(test_ds)}] {patient_id}: "
            f"Dice={metrics['dice_mean']:.4f}  "
            f"IoU={metrics['iou_mean']:.4f}  "
            f"Sens={metrics['sens_mean']:.4f}"
        )

        # Save prediction NIfTI
        if args.save_pred:
            _save_prediction_nifti(prob_map, info, pred_dir)

    # -- Average metrics --
    avg_metrics = {k: v / n_samples for k, v in all_metrics.items()}

    # -- Print results --
    print("\n" + "=" * 70)
    print("  Test Evaluation Results")
    print("=" * 70)
    print(f"  Model   : {args.model_name}")
    print(f"  Dataset : {args.test_split} ({n_samples} samples)")
    print(f"  TTA     : {'ON' if use_tta else 'OFF'}")
    print()

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
        f"  -- Mean (excluding background) --\n"
        f"    Dice: {avg_metrics['dice_mean']:.4f}\n"
        f"    IoU:  {avg_metrics['iou_mean']:.4f}\n"
        f"    Sens: {avg_metrics['sens_mean']:.4f}"
    )
    print("=" * 70)

    # -- Save metrics to file --
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

    print(f"  Metrics saved to: {metrics_path}")

    return avg_metrics


# ============================================================================
#  Save NIfTI Prediction
# ============================================================================


def _save_prediction_nifti(
    prob_map: torch.Tensor,
    info: dict,
    pred_dir: str,
) -> None:
    """
    Save prediction probability map as a NIfTI file.
    prob_map: (C, D, H, W) on any device
    """
    try:
        import nibabel as nib
    except ImportError:
        print("    [WARNING] nibabel not installed, skipping NIfTI save")
        return

    pred_labels = prob_map.argmax(dim=0).cpu().numpy().astype(np.uint8)  # (D, H, W)

    patient_id = info["patient_id"]
    if isinstance(patient_id, (list, tuple)):
        patient_id = patient_id[0]

    out_path = os.path.join(pred_dir, f"{patient_id}_pred.nii.gz")
    nib.save(nib.Nifti1Image(pred_labels, affine=np.eye(4)), out_path)
