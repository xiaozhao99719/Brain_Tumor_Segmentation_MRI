"""
param_set.py -- BrainMRI Global Parameter Configuration
========================================================
Centralized argument management for data_load / model / train / test.
All other modules import and use args from this file.
"""

import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    """Build the global argument parser and return the ArgumentParser object."""
    parser = argparse.ArgumentParser(
        description="BrainMRI Segmentation -- Global Parameter Configuration"
    )

    # -- Data --
    group_data = parser.add_argument_group("Data")
    group_data.add_argument(
        "--data_root", type=str, default="E:/python/BrainMRI",
        help="Raw data root directory",
    )
    group_data.add_argument(
        "--cache_root", type=str, default=None,
        help="Preprocessed cache directory (default: <data_root>/preprocessed)",
    )
    group_data.add_argument(
        "--target_spacing", type=float, default=1.0,
        help="Isotropic resampling target spacing (mm)",
    )
    group_data.add_argument(
        "--n4_enabled", type=int, default=1, choices=[0, 1],
        help="Enable N4 bias field correction (1=yes, 0=no)",
    )
    group_data.add_argument(
        "--use_cache", type=int, default=1, choices=[0, 1],
        help="Use disk cache for preprocessed data (1=yes, 0=no)",
    )
    group_data.add_argument(
        "--preprocess_workers", type=int, default=4,
        help="Number of parallel workers for offline preprocessing",
    )
    group_data.add_argument(
        "--preprocess_force", action="store_true",
        help="Force re-preprocessing, ignoring existing cache",
    )
    group_data.add_argument(
        "--num_classes", type=int, default=4,
        help="Number of segmentation classes (incl. background: 0=BG, 1=NCR, 2=ED, 3=ET)",
    )

    # -- Model --
    group_model = parser.add_argument_group("Model")
    group_model.add_argument(
        "--model_name", type=str, default="uk_an",
        choices=["nnunet", "attention_unet", "transunet", "uk_an"],
        help="Model architecture",
    )
    group_model.add_argument(
        "--in_channels", type=int, default=4,
        help="Input channel count (MRI modality count)",
    )
    group_model.add_argument(
        "--base_filters", type=int, default=32,
        help="U-Net / nnU-Net base filter count",
    )
    group_model.add_argument(
        "--attention_gate_channels", type=int, default=128,
        help="Attention U-Net gate intermediate channel count",
    )
    # TransUNet-specific
    group_model.add_argument(
        "--vit_img_size", type=int, default=128,
        help="TransUNet ViT input image size (must match patch size divisibility)",
    )
    group_model.add_argument(
        "--vit_patch_size", type=int, default=16,
        help="TransUNet ViT patch size",
    )
    group_model.add_argument(
        "--vit_hidden_size", type=int, default=768,
        help="TransUNet ViT hidden dimension",
    )
    group_model.add_argument(
        "--vit_num_heads", type=int, default=12,
        help="TransUNet ViT attention head count",
    )
    group_model.add_argument(
        "--vit_num_layers", type=int, default=12,
        help="TransUNet ViT Transformer layer count",
    )
    group_model.add_argument(
        "--vit_mlp_dim", type=int, default=3072,
        help="TransUNet ViT MLP intermediate dimension",
    )
    group_model.add_argument(
        "--vit_dropout", type=float, default=0.1,
        help="TransUNet ViT dropout rate",
    )
    # U-KAN-specific
    group_model.add_argument(
        "--uk_an_grid_size", type=int, default=5,
        help="U-KAN B-spline grid size (number of intervals)",
    )
    group_model.add_argument(
        "--uk_an_spline_order", type=int, default=3,
        help="U-KAN B-spline polynomial order (3=cubic)",
    )

    # -- Training --
    group_train = parser.add_argument_group("Training")
    group_train.add_argument(
        "--epochs", type=int, default=500,
        help="Number of training epochs",
    )
    group_train.add_argument(
        "--batch_size", type=int, default=2,
        help="Training batch size",
    )
    group_train.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="Optimizer weight decay",
    )
    group_train.add_argument(
        "--step_lr_step_size", type=int, default=30,
        help="StepLR step size (epochs)",
    )
    group_train.add_argument(
        "--step_lr_gamma", type=float, default=0.5,
        help="StepLR gamma",
    )
    group_train.add_argument(
        "--loss_fn", type=str, default="dice_ce",
        choices=["dice_ce", "dice", "ce", "focal"],
        help="Loss function",
    )
    group_train.add_argument(
        "--focal_gamma", type=float, default=2.0,
        help="Focal loss gamma",
    )
    group_train.add_argument(
        "--dice_smooth", type=float, default=0.5,
        help="Dice loss smoothing term",
    )
    group_train.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader num_workers",
    )
    group_train.add_argument(
        "--pin_memory", type=int, default=0, choices=[0, 1],
        help="DataLoader pin_memory",
    )
    group_train.add_argument(
        "--prefetch_factor", type=int, default=2,
        help="DataLoader prefetch_factor",
    )
    group_train.add_argument(
        "--grad_accum_steps", type=int, default=8,
        help="Gradient accumulation steps",
    )
    group_train.add_argument(
        "--effective_lr", type=float, default=1e-4,
        help="Effective learning rate",
    )
    group_train.add_argument(
        "--lr_scheduler", type=str, default="cosine",
        choices=["cosine", "step", "none"],
        help="Learning rate scheduler",
    )

    # -- 3D Patch --
    group_patch = parser.add_argument_group("3D Patch")
    group_patch.add_argument(
        "--patch_size", type=int, nargs=3, default=[64, 64, 64],
        help="3D training patch size (D H W)",
    )
    group_patch.add_argument(
        "--patch_overlap", type=float, default=0.25,
        help="Sliding window inference overlap ratio (0~0.5)",
    )
    group_patch.add_argument(
        "--random_patch", type=int, default=1, choices=[0, 1],
        help="Random patch cropping during training (1=yes)",
    )

    # -- Data Augmentation --
    group_aug = parser.add_argument_group("Data Augmentation")
    group_aug.add_argument(
        "--aug_enabled", type=int, default=0, choices=[0, 1],
        help="Enable data augmentation during training (1=yes, 0=no)",
    )
    # Geometric augmentations
    group_aug.add_argument(
        "--aug_flip", type=int, default=1, choices=[0, 1],
        help="Random axis flipping (D/H/W) during training",
    )
    group_aug.add_argument(
        "--aug_rotate", type=float, default=15.0,
        help="Max random rotation angle (degrees, 0=disabled)",
    )
    group_aug.add_argument(
        "--aug_scale", type=float, default=0.1,
        help="Max random scale factor (0.1=+/-10%, 0=disabled)",
    )
    group_aug.add_argument(
        "--aug_translate", type=float, default=0.0,
        help="Max random translation in voxels (0=disabled)",
    )
    group_aug.add_argument(
        "--aug_elastic", type=float, default=0.0,
        help="Elastic deformation alpha (0=disabled, see scipy.ndimage)",
    )
    # Intensity augmentations
    group_aug.add_argument(
        "--aug_noise", type=float, default=0.0,
        help="Gaussian noise std (0=disabled)",
    )
    group_aug.add_argument(
        "--aug_brightness", type=float, default=0.0,
        help="Brightness scaling range (+/- factor, 0=disabled)",
    )
    group_aug.add_argument(
        "--aug_contrast", type=float, default=0.0,
        help="Contrast scaling range (+/- factor, 0=disabled)",
    )
    group_aug.add_argument(
        "--aug_gamma", type=float, default=0.0,
        help="Gamma correction range (0=disabled, typical 0.2-2.0)",
    )
    group_aug.add_argument(
        "--aug_blur", type=float, default=0.0,
        help="Gaussian blur sigma (0=disabled)",
    )
    # Augmentation probability
    group_aug.add_argument(
        "--aug_prob", type=float, default=0.5,
        help="Probability of applying each augmentation",
    )

    # -- Test / Evaluation --
    group_test = parser.add_argument_group("Test / Evaluation")
    group_test.add_argument(
        "--test_split", type=str, default="test",
        help="Dataset split for evaluation (train/val/test)",
    )
    group_test.add_argument(
        "--save_pred", action="store_true",
        help="Save prediction results as NIfTI files",
    )
    group_test.add_argument(
        "--tta", action="store_true",
        help="Test-time augmentation (4 flip variants)",
    )

    # -- Output / Logging --
    group_out = parser.add_argument_group("Output / Logging")
    group_out.add_argument(
        "--output_dir", type=str, default="E:/python/BrainMRI/output",
        help="Output directory (checkpoints / logs / predictions)",
    )
    group_out.add_argument(
        "--save_every", type=int, default=10,
        help="Save checkpoint every N epochs",
    )
    group_out.add_argument(
        "--log_every", type=int, default=10,
        help="Print training log every N batches",
    )
    group_out.add_argument(
        "--resume_ckpt", type=str, default=None,
        help="Checkpoint path to resume training from",
    )
    group_out.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )

    # -- Hardware --
    group_hw = parser.add_argument_group("Hardware")
    group_hw.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device index (-1 means CPU)",
    )
    group_hw.add_argument(
        "--amp", type=int, default=1, choices=[0, 1],
        help="Enable mixed-precision training (AMP)",
    )

    return parser


def parse_args(argv=None) -> argparse.Namespace:
    """Parse arguments and perform post-processing."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Auto-derive cache_root if not specified
    if args.cache_root is None:
        args.cache_root = os.path.join(args.data_root, "preprocessed")

    # AMP + GPU compatibility
    if args.gpu < 0:
        args.device = "cpu"
        args.amp = 0
    else:
        import torch
        args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        if args.device == "cpu":
            args.amp = 0

    return args
